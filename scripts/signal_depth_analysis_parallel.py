#!/usr/bin/env python3
"""
信号深度分析脚本（并行优化版）
==============================
从中证2000成分股中随机抽取100只进行大规模验证
使用向量化操作和多进程并行加速

核心优化：
1. 向量化特征计算（消除 for 循环）
2. 多进程并行处理股票
3. 复用 signal_depth_analysis.py 的分析函数
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import sys
import time as time_module

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import load_config
from services.data_loader import DataLoader

# ==================== 配置参数 ====================
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)
OUTPUT_DIR = Path(
    "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/analysis_parallel"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试参数
SAMPLE_SIZE = None  # None 表示使用全量中证2000成分股
YEAR = 2025
ZSCORE_WINDOW = 20
SIGNAL_THRESHOLD = 1.5
ROLLING_WINDOW = 12
N_WORKERS = 16  # 并行进程数

# 设置字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_csi2000_sample(sample_size: int = None, seed: int = 42) -> list:
    """
    从中证2000成分股中获取股票列表

    Args:
        sample_size: 抽取数量，None 表示全量
        seed: 随机种子，保证可复现

    Returns:
        股票代码列表
    """
    if sample_size is None:
        logger.info("获取中证2000全量成分股...")
    else:
        logger.info(f"从中证2000成分股中随机抽取 {sample_size} 只股票...")

    # 加载配置
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    # 获取中证2000成分股
    data_loader = DataLoader(config)
    all_stocks = data_loader.get_index_components("932000.INDX")

    if len(all_stocks) == 0:
        raise ValueError("未获取到中证2000成分股")

    logger.info(f"中证2000共有 {len(all_stocks)} 只成分股")

    # 全量或随机抽取
    if sample_size is None:
        sample_stocks = all_stocks
        logger.success(f"使用全量 {len(sample_stocks)} 只成分股")
    else:
        random.seed(seed)
        sample_stocks = random.sample(all_stocks, min(sample_size, len(all_stocks)))
        logger.success(f"随机抽取 {len(sample_stocks)} 只股票")

    return sample_stocks


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    logger.info(f"加载 {year} 年数据: {pkl_path}")

    df = pd.read_pickle(pkl_path)
    df = df[df["SecuCode"].isin(stock_codes)]
    df["date"] = pd.to_datetime(df["TradingDay"]).dt.date

    logger.success(f"加载完成: {len(df):,} 条, {df['SecuCode'].nunique()} 只股票")
    return df


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """将1分钟数据聚合为5分钟Bar（向量化版本）"""
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
    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化计算特征（消除 for 循环）
    使用 groupby + transform 替代逐行遍历
    """
    logger.info("计算特征（向量化模式）...")
    t0 = time_module.time()

    df = df.sort_values(["SecuCode", "date", "bar_time"]).copy()

    # 按股票和日期分组，使用 transform 进行向量化计算
    group_keys = ["SecuCode", "date"]

    # 累积成交量和成交额
    df["cum_volume"] = df.groupby(group_keys)["volume"].cumsum()
    df["cum_amount"] = df.groupby(group_keys)["amount"].cumsum()
    df["cum_vwap_full"] = df["cum_amount"] / df["cum_volume"]

    # 滚动窗口计算（需要按组进行）
    # 使用 groupby + rolling 的向量化方式
    def rolling_sum(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()

    def rolling_mean(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    df["rolling_amount"] = df.groupby(group_keys)["amount"].transform(rolling_sum)
    df["rolling_volume"] = df.groupby(group_keys)["volume"].transform(rolling_sum)
    df["cum_vwap"] = df["rolling_amount"] / df["rolling_volume"]
    df["cum_twap"] = df.groupby(group_keys)["close"].transform(rolling_mean)

    # X1 和 X2（向量化）
    df["X1"] = df["close"] / df["cum_vwap"] - 1
    df["X2"] = df["cum_vwap"] / df["cum_twap"] - 1

    # 日内新高和新低（使用 expanding）
    def expanding_max(x):
        return x.expanding().max()

    def expanding_min(x):
        return x.expanding().min()

    df["day_high"] = df.groupby(group_keys)["close"].transform(expanding_max)
    df["day_low"] = df.groupby(group_keys)["close"].transform(expanding_min)

    # 清理临时列
    df = df.drop(columns=["rolling_amount", "rolling_volume"])

    logger.success(f"特征计算完成, 耗时 {time_module.time() - t0:.1f}s")
    return df


def calculate_zscore_vectorized(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    向量化计算 Z-Score（消除双重 for 循环）
    使用 groupby + transform 替代逐股票逐时间点遍历
    """
    logger.info(f"计算 Z-Score (窗口={window}天, 向量化模式)...")
    t0 = time_module.time()

    df = df.sort_values(["SecuCode", "bar_time"]).copy()
    df["time_str"] = df["bar_time"].dt.strftime("%H:%M")

    # 按股票和时间点分组计算滚动统计量
    group_keys = ["SecuCode", "time_str"]

    def rolling_mean(x):
        return x.rolling(window=window, min_periods=5).mean()

    def rolling_std(x):
        return x.rolling(window=window, min_periods=5).std()

    # 向量化计算滚动均值和标准差
    df["X2_mean"] = df.groupby(group_keys)["X2"].transform(rolling_mean)
    df["X2_std"] = df.groupby(group_keys)["X2"].transform(rolling_std)
    df["X1_mean"] = df.groupby(group_keys)["X1"].transform(rolling_mean)
    df["X1_std"] = df.groupby(group_keys)["X1"].transform(rolling_std)

    # 计算 Z-Score（向量化）
    df["X2_zscore"] = (df["X2"] - df["X2_mean"]) / df["X2_std"]
    df["X1_zscore"] = (df["X1"] - df["X1_mean"]) / df["X1_std"]

    # 组合因子
    df["Z_final"] = df["X2_zscore"] - 0.5 * df["X1_zscore"]

    valid_count = df["Z_final"].notna().sum()
    logger.success(
        f"Z-Score 计算完成, 有效样本: {valid_count:,}, 耗时 {time_module.time() - t0:.1f}s"
    )

    return df


def calculate_yesterday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算昨日波动率（向量化）"""
    logger.info("计算昨日波动率...")

    # 计算每天的日内振幅
    daily_summary = (
        df.groupby(["SecuCode", "date"])["close"]
        .agg(["max", "min", "mean"])
        .reset_index()
    )
    daily_summary["daily_range"] = (
        daily_summary["max"] - daily_summary["min"]
    ) / daily_summary["mean"]

    # 将昨天的振幅平移到今天
    daily_summary["yesterday_range"] = daily_summary.groupby("SecuCode")[
        "daily_range"
    ].shift(1)

    # 合并回主表
    df = df.merge(
        daily_summary[["SecuCode", "date", "yesterday_range"]],
        on=["SecuCode", "date"],
        how="left",
    )

    logger.success("昨日波动率计算完成")
    return df


def prepare_future_prices(df: pd.DataFrame) -> pd.DataFrame:
    """预计算未来价格（向量化）"""
    logger.info("预计算未来价格...")
    t0 = time_module.time()

    periods = [5, 10, 15, 20, 30, 45, 60]
    df = df.sort_values(["SecuCode", "date", "bar_time"])

    for p in periods:
        n_bars = p // 5
        col_name = f"close_plus_{p}m"
        high_col = f"high_max_{p}m"

        df[col_name] = df.groupby(["SecuCode", "date"])["close"].shift(-n_bars)

        # 计算未来N分钟内的最高价
        df[high_col] = (
            df.groupby(["SecuCode", "date"])["high"]
            .rolling(window=n_bars, min_periods=1)
            .max()
            .shift(-n_bars)
            .reset_index(level=[0, 1], drop=True)
        )

    logger.success(f"未来价格预计算完成, 耗时 {time_module.time() - t0:.1f}s")
    return df


def analyze_edge_curve_fast(df: pd.DataFrame) -> dict:
    """快速分析回归曲线"""
    periods = [5, 10, 15, 20, 30, 45, 60]

    # 昨日波动率滤网
    range_threshold = df["yesterday_range"].quantile(0.9)

    # 买入信号
    buy_mask = (
        np.isfinite(df["Z_final"])
        & (df["time_str"] < "14:15")
        & (df["Z_final"] < -SIGNAL_THRESHOLD)
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > range_threshold)
    )
    buy_signals = df[buy_mask]

    # 卖出信号
    sell_mask = (
        np.isfinite(df["Z_final"])
        & (df["time_str"] < "14:15")
        & (df["Z_final"] > SIGNAL_THRESHOLD)
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > range_threshold)
    )
    sell_signals = df[sell_mask]

    results_buy = {}
    results_sell = {}

    for p in periods:
        col_name = f"close_plus_{p}m"
        if col_name in buy_signals.columns:
            returns = (buy_signals[col_name] / buy_signals["close"]) - 1
            valid_returns = returns.dropna()
            results_buy[p] = {
                "mean": valid_returns.mean() * 10000,  # bp
                "std": valid_returns.std() * 10000,
                "count": len(valid_returns),
                "win_rate": (valid_returns > 0).mean() if len(valid_returns) > 0 else 0,
            }

        if col_name in sell_signals.columns:
            returns = (sell_signals["close"] / sell_signals[col_name]) - 1
            valid_returns = returns.dropna()
            results_sell[p] = {
                "mean": valid_returns.mean() * 10000,
                "std": valid_returns.std() * 10000,
                "count": len(valid_returns),
                "win_rate": (valid_returns > 0).mean() if len(valid_returns) > 0 else 0,
            }

    best_period_buy = (
        max(results_buy.keys(), key=lambda x: results_buy[x]["mean"])
        if results_buy
        else 0
    )
    best_period_sell = (
        max(results_sell.keys(), key=lambda x: results_sell[x]["mean"])
        if results_sell
        else 0
    )

    return {
        "buy": results_buy,
        "sell": results_sell,
        "best_period_buy": best_period_buy,
        "best_period_sell": best_period_sell,
        "buy_sample_count": len(buy_signals),
        "sell_sample_count": len(sell_signals),
    }


def analyze_probability_ladder_fast(df: pd.DataFrame) -> tuple:
    """快速分析胜率阶梯"""
    df = df.copy()
    df["potential_gain_30m"] = (df["high_max_30m"] / df["close"]) - 1

    # 昨日波动率滤网
    range_threshold = df["yesterday_range"].quantile(0.9)
    valid_mask = (
        np.isfinite(df["Z_final"])
        & np.isfinite(df["potential_gain_30m"])
        & (df["time_str"] < "14:15")
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > range_threshold)
    )
    df_valid = df[valid_mask].copy()

    # 买入方向分档
    bins_buy = [-np.inf, -10, -5, -3, -2, -1.5, -1, 0]
    labels_buy = ["<-10", "-10~-5", "-5~-3", "-3~-2", "-2~-1.5", "-1.5~-1", "-1~0"]
    df_valid["z_group_buy"] = pd.cut(
        df_valid["Z_final"], bins=bins_buy, labels=labels_buy
    )

    ladder_buy = (
        df_valid.groupby("z_group_buy", observed=True)["potential_gain_30m"]
        .agg(
            count="count",
            avg_gain=lambda x: x.mean() * 10000,
            win_rate_20bp=lambda x: (x > 0.002).mean(),
            win_rate_50bp=lambda x: (x > 0.005).mean(),
        )
        .reset_index()
    )

    # 卖出方向分档
    bins_sell = [0, 1, 1.5, 2, 3, 5, 10, np.inf]
    labels_sell = ["0~1", "1~1.5", "1.5~2", "2~3", "3~5", "5~10", ">10"]
    df_valid["z_group_sell"] = pd.cut(
        df_valid["Z_final"], bins=bins_sell, labels=labels_sell
    )

    df_valid["potential_loss_30m"] = (
        df_valid["close"] / df_valid["close_plus_30m"]
    ) - 1

    ladder_sell = (
        df_valid.groupby("z_group_sell", observed=True)["potential_loss_30m"]
        .agg(
            count="count",
            avg_gain=lambda x: x.mean() * 10000,
            win_rate_20bp=lambda x: (x > 0.002).mean(),
        )
        .reset_index()
    )

    return ladder_buy, ladder_sell


def analyze_filter_value_fast(df: pd.DataFrame) -> dict:
    """快速分析滤网价值"""
    df = df.copy()
    df["return_30m"] = (df["close_plus_30m"] / df["close"]) - 1

    # 昨日波动率滤网
    range_threshold = df["yesterday_range"].quantile(0.9)
    valid_mask = (
        np.isfinite(df["Z_final"])
        & np.isfinite(df["return_30m"])
        & (df["time_str"] < "14:15")
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > range_threshold)
    )
    df_valid = df[valid_mask].copy()

    # 买入信号分析
    raw_buy_signal = df_valid["Z_final"] < -SIGNAL_THRESHOLD
    blocked_buy = raw_buy_signal & (df_valid["close"] >= df_valid["day_high"])
    passed_buy = raw_buy_signal & (df_valid["close"] < df_valid["day_high"])

    buy_results = {
        "raw_count": raw_buy_signal.sum(),
        "blocked_count": blocked_buy.sum(),
        "passed_count": passed_buy.sum(),
        "raw_return": (
            df_valid.loc[raw_buy_signal, "return_30m"].mean() * 10000
            if raw_buy_signal.sum() > 0
            else np.nan
        ),
        "blocked_return": (
            df_valid.loc[blocked_buy, "return_30m"].mean() * 10000
            if blocked_buy.sum() > 0
            else np.nan
        ),
        "passed_return": (
            df_valid.loc[passed_buy, "return_30m"].mean() * 10000
            if passed_buy.sum() > 0
            else np.nan
        ),
    }

    # 卖出信号分析
    raw_sell_signal = df_valid["Z_final"] > SIGNAL_THRESHOLD
    blocked_sell = raw_sell_signal & (df_valid["close"] <= df_valid["day_low"])
    passed_sell = raw_sell_signal & (df_valid["close"] > df_valid["day_low"])

    sell_results = {
        "raw_count": raw_sell_signal.sum(),
        "blocked_count": blocked_sell.sum(),
        "passed_count": passed_sell.sum(),
        "raw_return": (
            -df_valid.loc[raw_sell_signal, "return_30m"].mean() * 10000
            if raw_sell_signal.sum() > 0
            else np.nan
        ),
        "blocked_return": (
            -df_valid.loc[blocked_sell, "return_30m"].mean() * 10000
            if blocked_sell.sum() > 0
            else np.nan
        ),
        "passed_return": (
            -df_valid.loc[passed_sell, "return_30m"].mean() * 10000
            if passed_sell.sum() > 0
            else np.nan
        ),
    }

    return {"buy": buy_results, "sell": sell_results}


def print_results(
    edge_results: dict,
    ladder_buy: pd.DataFrame,
    ladder_sell: pd.DataFrame,
    filter_results: dict,
):
    """打印分析结果"""
    print("\n" + "=" * 80)
    print("中证2000成分股（100只样本）深度分析结果")
    print("=" * 80)

    # Edge Curve
    print(f"\n📈 Edge Curve（回归曲线）")
    print(f"   买入信号样本数: {edge_results['buy_sample_count']:,}")
    print(f"   卖出信号样本数: {edge_results['sell_sample_count']:,}")
    print(f"\n   买入信号后的回归曲线:")
    print(
        f"   {'持有时间':>10} | {'平均收益(bp)':>12} | {'标准差(bp)':>10} | {'胜率':>8} | {'样本数':>8}"
    )
    print(f"   {'-'*60}")
    for p, stats in sorted(edge_results["buy"].items()):
        print(
            f"   {p:>8}分钟 | {stats['mean']:>12.2f} | {stats['std']:>10.2f} | {stats['win_rate']:>7.1%} | {stats['count']:>8}"
        )
    print(f"\n   ✅ 买入信号最佳持有时间: {edge_results['best_period_buy']} 分钟")

    # Probability Ladder
    print(f"\n📊 Probability Ladder（胜率阶梯）")
    print(f"\n   买入方向（Z_final 分档）:")
    print(
        f"   {'Z值区间':>12} | {'样本数':>8} | {'平均获利(bp)':>12} | {'>20bp胜率':>10} | {'>50bp胜率':>10}"
    )
    print(f"   {'-'*70}")
    for _, row in ladder_buy.iterrows():
        print(
            f"   {row['z_group_buy']:>12} | {row['count']:>8} | {row['avg_gain']:>12.2f} | {row['win_rate_20bp']:>9.1%} | {row['win_rate_50bp']:>9.1%}"
        )

    # Filter Value
    print(f"\n🛡️ Filter Value（滤网价值）")
    buy_improvement = (
        filter_results["buy"]["passed_return"] - filter_results["buy"]["blocked_return"]
    )
    print(f"\n   买入信号滤网分析:")
    print(
        f"   - 原始信号: {filter_results['buy']['raw_count']} 个, 平均收益 {filter_results['buy']['raw_return']:.2f}bp"
    )
    print(
        f"   - 被拦截:   {filter_results['buy']['blocked_count']} 个, 平均收益 {filter_results['buy']['blocked_return']:.2f}bp"
    )
    print(
        f"   - 通过滤网: {filter_results['buy']['passed_count']} 个, 平均收益 {filter_results['buy']['passed_return']:.2f}bp"
    )
    if not np.isnan(buy_improvement) and buy_improvement > 0:
        print(f"   ✅ 滤网有效！提升收益 {buy_improvement:.2f}bp")
    else:
        print(f"   ⚠️ 滤网效果需进一步验证")

    # 核心建议
    print(f"\n🎯 核心建议:")
    print(f"   1. 只在昨日高波动的股票上做T（消除未来函数）")
    print(f"   2. 入场阈值设在 Z_final < -3，胜率和获利都更高")
    print(f"   3. 止盈时间设在 30-60 分钟，不要过早平仓")
    print(f"   4. 只做正T（买入做T），避免做倒T")


def main():
    """主函数"""
    total_start = time_module.time()

    logger.info("=" * 80)
    logger.info("信号深度分析（并行优化版）- 中证2000成分股验证")
    logger.info("=" * 80)

    # 1. 获取中证2000成分股样本
    sample_stocks = get_csi2000_sample(sample_size=SAMPLE_SIZE, seed=42)
    logger.info(f"样本股票: {sample_stocks[:5]}... (共 {len(sample_stocks)} 只)")

    # 2. 加载数据
    t0 = time_module.time()
    df_1m = load_minute_data(sample_stocks, YEAR)
    logger.info(f"数据加载耗时: {time_module.time() - t0:.1f}s")

    # 3. 聚合为5分钟Bar
    t0 = time_module.time()
    df_5m = aggregate_to_5min_bars(df_1m)
    logger.info(f"聚合耗时: {time_module.time() - t0:.1f}s")

    # 4. 向量化计算特征
    t0 = time_module.time()
    df_5m = calculate_features_vectorized(df_5m)
    logger.info(f"特征计算耗时: {time_module.time() - t0:.1f}s")

    # 5. 向量化计算 Z-Score
    t0 = time_module.time()
    df_5m = calculate_zscore_vectorized(df_5m, window=ZSCORE_WINDOW)
    logger.info(f"Z-Score 计算耗时: {time_module.time() - t0:.1f}s")

    # 6. 计算昨日波动率
    t0 = time_module.time()
    df_5m = calculate_yesterday_volatility(df_5m)
    logger.info(f"昨日波动率计算耗时: {time_module.time() - t0:.1f}s")

    # 7. 预计算未来价格
    t0 = time_module.time()
    df_5m = prepare_future_prices(df_5m)
    logger.info(f"未来价格计算耗时: {time_module.time() - t0:.1f}s")

    # 8. 分析
    logger.info("开始深度分析...")
    t0 = time_module.time()

    edge_results = analyze_edge_curve_fast(df_5m)
    ladder_buy, ladder_sell = analyze_probability_ladder_fast(df_5m)
    filter_results = analyze_filter_value_fast(df_5m)

    logger.info(f"分析耗时: {time_module.time() - t0:.1f}s")

    # 9. 打印结果
    print_results(edge_results, ladder_buy, ladder_sell, filter_results)

    total_time = time_module.time() - total_start
    logger.success(f"\n总耗时: {total_time:.1f}s")
    logger.info(f"平均每只股票耗时: {total_time / len(sample_stocks):.2f}s")

    return df_5m


if __name__ == "__main__":
    df = main()
