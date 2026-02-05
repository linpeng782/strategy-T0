#!/usr/bin/env python3
"""
回测数据准备脚本
================
一次性准备好所有回测需要的数据，保存到本地缓存，避免每次回测/训练都重新计算。

设计目标：
1. 支持全量中证2000成分股（约2000只）
2. 统一缓存，供 train_lgbm.py 和 backtest_ml.py 共用
3. 内存优化：float64 -> float32

缓存内容：
1. df_5m.pkl - 5分钟Bar数据（包含所有特征、Z-Score、未来价格等）
2. daily_returns.pkl - 每日收益率
3. stock_list.pkl - 股票列表
4. metadata.json - 元数据（参数配置等）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import json
import time as time_module
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.data_loader import DataLoader


class SimpleConfig:
    """简单配置类"""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)


# ==================== 配置参数 ====================
# 原始数据目录
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)

# 缓存输出目录
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 回测参数
YEAR = 2025
ZSCORE_WINDOW = 20
ROLLING_WINDOW = 12
HOLDING_PERIOD = 60  # 最大持有时间（分钟）


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """内存优化：float64 -> float32"""
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


def get_csi2000_stocks() -> list:
    """获取中证2000全部成分股"""
    logger.info("获取中证2000全部成分股...")

    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    data_loader = DataLoader(config)
    all_stocks = data_loader.get_index_components("932000.INDX")

    if len(all_stocks) == 0:
        raise ValueError("未获取到中证2000成分股")

    all_stocks = sorted(all_stocks)
    logger.success(f"获取到 {len(all_stocks)} 只中证2000成分股")
    return all_stocks


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    logger.info(f"加载 {year} 年数据: {pkl_path}")

    df = pd.read_pickle(pkl_path)

    # 获取数据中实际存在的股票
    available_stocks = set(df["SecuCode"].unique())
    target_stocks = set(stock_codes)
    intersection = available_stocks & target_stocks

    logger.info(
        f"目标股票: {len(target_stocks)}, 数据中可用: {len(available_stocks)}, 交集: {len(intersection)}"
    )

    df = df[df["SecuCode"].isin(intersection)]
    df["date"] = pd.to_datetime(df["TradingDay"]).dt.date

    logger.success(f"加载完成: {len(df):,} 条, {df['SecuCode'].nunique()} 只股票")
    return df, list(intersection)


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """聚合为5分钟Bar"""
    logger.info("聚合为5分钟Bar...")

    df["bar_time"] = pd.to_datetime(df["TradingDay"]).dt.floor("5min")

    agg_df = (
        df.groupby(["SecuCode", "date", "bar_time"])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values(["SecuCode", "date", "bar_time"])
    agg_df["vwap_5m"] = agg_df["amount"] / agg_df["volume"]

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """向量化计算特征"""
    logger.info("计算特征...")

    df = df.sort_values(["SecuCode", "date", "bar_time"]).copy()
    group_keys = ["SecuCode", "date"]

    def rolling_sum(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()

    def rolling_mean(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    df["rolling_amount"] = df.groupby(group_keys)["amount"].transform(rolling_sum)
    df["rolling_volume"] = df.groupby(group_keys)["volume"].transform(rolling_sum)
    df["cum_vwap"] = df["rolling_amount"] / df["rolling_volume"]
    df["cum_twap"] = df.groupby(group_keys)["close"].transform(rolling_mean)

    df["X1"] = df["close"] / df["cum_vwap"] - 1
    df["X2"] = df["cum_vwap"] / df["cum_twap"] - 1

    df["day_high"] = df.groupby(group_keys)["close"].transform(
        lambda x: x.expanding().max()
    )
    df = df.drop(columns=["rolling_amount", "rolling_volume"])

    logger.success("特征计算完成")
    return df


def calculate_zscore_vectorized(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """向量化计算Z-Score"""
    logger.info(f"计算 Z-Score (窗口={window}天)...")

    df = df.sort_values(["SecuCode", "bar_time"]).copy()
    df["time_str"] = df["bar_time"].dt.strftime("%H:%M")

    group_keys = ["SecuCode", "time_str"]

    def rolling_mean_shifted(x):
        return x.rolling(window=window, min_periods=5).mean().shift(1)

    def rolling_std_shifted(x):
        return x.rolling(window=window, min_periods=5).std().shift(1)

    df["X2_mean"] = df.groupby(group_keys)["X2"].transform(rolling_mean_shifted)
    df["X2_std"] = df.groupby(group_keys)["X2"].transform(rolling_std_shifted)
    df["X1_mean"] = df.groupby(group_keys)["X1"].transform(rolling_mean_shifted)
    df["X1_std"] = df.groupby(group_keys)["X1"].transform(rolling_std_shifted)

    df["X2_zscore"] = (df["X2"] - df["X2_mean"]) / df["X2_std"]
    df["X1_zscore"] = (df["X1"] - df["X1_mean"]) / df["X1_std"]
    df["Z_final"] = df["X2_zscore"] - 0.5 * df["X1_zscore"]

    logger.success("Z-Score 计算完成")
    return df


def calculate_yesterday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算昨日波动率"""
    logger.info("计算昨日波动率...")

    daily_summary = (
        df.groupby(["SecuCode", "date"])["close"]
        .agg(["max", "min", "mean"])
        .reset_index()
    )
    daily_summary["daily_range"] = (
        daily_summary["max"] - daily_summary["min"]
    ) / daily_summary["mean"]
    daily_summary["yesterday_range"] = daily_summary.groupby("SecuCode")[
        "daily_range"
    ].shift(1)

    df = df.merge(
        daily_summary[["SecuCode", "date", "yesterday_range"]],
        on=["SecuCode", "date"],
        how="left",
    )

    logger.success("昨日波动率计算完成")
    return df


def prepare_future_prices(df: pd.DataFrame) -> pd.DataFrame:
    """预计算未来价格"""
    logger.info("预计算未来价格...")

    df = df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    n_bars = HOLDING_PERIOD // 5

    df["close_exit"] = df.groupby(["SecuCode", "date"])["close"].shift(-n_bars)

    def calc_future_high_low(group):
        high_rev = group["high"].iloc[::-1]
        low_rev = group["low"].iloc[::-1]
        future_high_rev = high_rev.rolling(window=n_bars, min_periods=1).max()
        future_low_rev = low_rev.rolling(window=n_bars, min_periods=1).min()
        future_high = future_high_rev.iloc[::-1].shift(-1)
        future_low = future_low_rev.iloc[::-1].shift(-1)
        return pd.DataFrame(
            {"future_high": future_high.values, "future_low": future_low.values},
            index=group.index,
        )

    future_hl = df.groupby(["SecuCode", "date"], group_keys=False).apply(
        calc_future_high_low
    )
    df["future_high"] = future_hl["future_high"]
    df["future_low"] = future_hl["future_low"]

    logger.success(f"未来价格计算完成 (持有期={HOLDING_PERIOD}分钟)")
    return df


def add_time_val(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间数值特征（供模型使用）"""
    logger.info("添加时间数值特征...")
    df["time_val"] = df["bar_time"].dt.hour + df["bar_time"].dt.minute / 60.0
    logger.success("时间数值特征添加完成")
    return df


def calculate_ml_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    计算机器学习增强特征

    新增特征：
    1. vol_burst - 成交量爆发力：当前成交量 / 过去N天同期平均成交量
    2. z_final_slope - Z值变化斜率：当日Z_final的一阶差分
    3. x1_slope - 价格动量斜率：当日X1_zscore的一阶差分
    """
    logger.info(f"计算机器学习增强特征 (vol_window={vol_window})...")

    df = df.sort_values(["SecuCode", "date", "bar_time"]).copy()

    # 1. 成交量爆发力 (Volume Burst)
    # 计算每只股票在每个时间点的历史平均成交量（过去N天同期）
    logger.info("  - 计算成交量爆发力 (vol_burst)...")
    df["vol_hist_mean"] = df.groupby(["SecuCode", "time_str"])["volume"].transform(
        lambda x: x.rolling(window=vol_window, min_periods=5).mean().shift(1)
    )
    df["vol_burst"] = df["volume"] / (df["vol_hist_mean"] + 1e-9)
    df = df.drop(columns=["vol_hist_mean"])

    # 2. Z值变化斜率 (Z_final_slope)
    # 当日内Z_final的一阶差分，表示橡皮筋是在拉长还是收缩
    logger.info("  - 计算Z值变化斜率 (z_final_slope)...")
    df["z_final_slope"] = df.groupby(["SecuCode", "date"])["Z_final"].diff()

    # 3. 价格动量斜率 (X1_slope)
    # 当日内X1_zscore的一阶差分，表示价格动量变化
    logger.info("  - 计算价格动量斜率 (x1_slope)...")
    df["x1_slope"] = df.groupby(["SecuCode", "date"])["X1_zscore"].diff()

    logger.success("机器学习增强特征计算完成: vol_burst, z_final_slope, x1_slope")
    return df


def calculate_market_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全市场环境特征：市场超跌占比和大盘引力中枢

    新增特征：
    1. mkt_oversold_ratio - 全市场超跌占比：当前时刻有多少比例的股票 Z_final < -1.0
    2. mkt_avg_z - 大盘整体的引力中枢：全市场 Z_final 的平均值
    """
    logger.info("计算全市场环境特征 (Market Sentiment)...")

    # 1. 标记哪些样本处于"引力拉伸"状态 (Z < -1.0)
    df["is_oversold_signal"] = (df["Z_final"] < -1.0).astype(int)

    # 2. 按时间点 (date + time_str) 分组，计算全市场超跌股票的比例
    logger.info("  - 计算全市场超跌占比 (mkt_oversold_ratio)...")
    df["mkt_oversold_ratio"] = df.groupby(["date", "time_str"])[
        "is_oversold_signal"
    ].transform("mean")

    # 3. 计算大盘整体的引力中枢 (全市场 Z_final 的平均值)
    logger.info("  - 计算大盘引力中枢 (mkt_avg_z)...")
    df["mkt_avg_z"] = df.groupby(["date", "time_str"])["Z_final"].transform("mean")

    # 4. 清理中间变量
    df = df.drop(columns=["is_oversold_signal"])

    logger.success("大盘环境特征计算完成: mkt_oversold_ratio, mkt_avg_z")
    return df


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日收益率"""
    logger.info("计算每日基础收益率...")

    daily_close = (
        df.groupby(["SecuCode", "date"])["close"]
        .last()
        .reset_index()
        .rename(columns={"close": "daily_close"})
    )
    daily_close = daily_close.sort_values(["SecuCode", "date"])
    daily_close["daily_return"] = daily_close.groupby("SecuCode")[
        "daily_close"
    ].pct_change()

    logger.success("每日基础收益率计算完成")
    return daily_close


def save_cache(df_5m: pd.DataFrame, daily_returns: pd.DataFrame, stock_list: list):
    """保存缓存数据"""
    logger.info(f"保存缓存数据到: {CACHE_DIR}")

    # 内存优化
    logger.info("执行内存优化...")
    df_5m = optimize_memory(df_5m)
    daily_returns = optimize_memory(daily_returns)

    # 保存5分钟数据
    df_5m_path = CACHE_DIR / "df_5m.pkl"
    df_5m.to_pickle(df_5m_path)
    logger.info(
        f"  - df_5m.pkl: {len(df_5m):,} 行, {df_5m_path.stat().st_size / 1024 / 1024:.1f} MB"
    )

    # 保存每日收益率
    daily_returns_path = CACHE_DIR / "daily_returns.pkl"
    daily_returns.to_pickle(daily_returns_path)
    logger.info(f"  - daily_returns.pkl: {len(daily_returns):,} 行")

    # 保存股票列表
    stock_list_path = CACHE_DIR / "stock_list.pkl"
    pd.to_pickle(stock_list, stock_list_path)
    logger.info(f"  - stock_list.pkl: {len(stock_list)} 只股票")

    # 保存元数据
    metadata = {
        "year": YEAR,
        "zscore_window": ZSCORE_WINDOW,
        "rolling_window": ROLLING_WINDOW,
        "holding_period": HOLDING_PERIOD,
        "df_5m_rows": len(df_5m),
        "df_5m_cols": list(df_5m.columns),
        "daily_returns_rows": len(daily_returns),
        "n_stocks": len(stock_list),
        "created_at": pd.Timestamp.now().isoformat(),
    }
    metadata_path = CACHE_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  - metadata.json: 配置参数已保存")

    logger.success("缓存数据保存完成!")


def main():
    """主函数"""
    total_start = time_module.time()

    logger.info("=" * 80)
    logger.info("回测数据准备脚本 (全量中证2000)")
    logger.info("=" * 80)

    # 1. 获取中证2000全部成分股
    csi2000_stocks = get_csi2000_stocks()

    # 2. 加载数据（返回实际可用的股票列表）
    df_1m, stock_list = load_minute_data(csi2000_stocks, YEAR)

    # 3. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)
    del df_1m  # 释放内存

    # 4. 计算特征
    df_5m = calculate_features_vectorized(df_5m)

    # 5. 计算 Z-Score
    df_5m = calculate_zscore_vectorized(df_5m, window=ZSCORE_WINDOW)

    # 6. 计算昨日波动率
    df_5m = calculate_yesterday_volatility(df_5m)

    # 7. 预计算未来价格
    df_5m = prepare_future_prices(df_5m)

    # 8. 添加时间数值特征
    df_5m = add_time_val(df_5m)

    # 9. 计算机器学习增强特征 (vol_burst, z_final_slope, x1_slope)
    df_5m = calculate_ml_features(df_5m, vol_window=ZSCORE_WINDOW)

    # 10. 计算全市场环境特征 (mkt_oversold_ratio, mkt_avg_z)
    df_5m = calculate_market_sentiment(df_5m)

    # 11. 计算每日基础收益率
    daily_returns = calculate_daily_returns(df_5m)

    # 12. 保存缓存
    save_cache(df_5m, daily_returns, stock_list)

    total_time = time_module.time() - total_start
    logger.success(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")
    logger.info(f"缓存目录: {CACHE_DIR}")
    logger.info("后续训练/回测可直接调用缓存数据，无需重新计算")


if __name__ == "__main__":
    main()
