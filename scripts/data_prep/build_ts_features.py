"""
时序特征构建脚本

职责：读取 calc_feature_x1_x2.py 生成的原始特征数据，构建时序模型所需的全部特征
输入：df_features_{year}.pkl（全量48bar/天）
输出：df_ts_features_{year}.pkl（仅 ENTRY_TIME 入场行，附带时序特征）

特征分5大类（共25个）：
  1. 当前bar的X1/X2（保留现有，去掉截面rank）
  2. 日内价格轨迹（lag/diff/slope/std）
  3. 量能演变（rel_vol/加速度/集中度）
  4. 日内价格形态（晨盘收益/价格位置/振幅/波动率）
  5. 跨日特征（隔夜跳空/昨日收益/5日动量/历史波动率）

依赖：calc_feature_x1_x2.py 生产的特征数据

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 处理年份（依次处理）
YEARS = [2024, 2025]

# 入场时间（后续如需调整，改这一个变量即可）
ENTRY_TIME = "10:30"

# 跨日滚动窗口
CROSS_DAY_WINDOW = 20


# ==================== 数据加载 ====================
def load_raw_features(year: int) -> pd.DataFrame:
    """加载 calc_feature_x1_x2.py 生成的原始特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 calc_feature_x1_x2.py"
        )
    df = pd.read_pickle(pkl_path)
    logger.success(
        f"加载 {year} 年数据: {len(df):,} 行, "
        f"{df['SecuCode'].nunique()} 只股票, {df['date'].nunique()} 天"
    )
    return df


# ==================== 日内特征构建 ====================
def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在全量bar上构建日内时序特征

    包含第2/3/4类特征，需在筛选入场时间之前计算（因为需要之前bar的数据）
    """
    logger.info("构建日内时序特征...")
    df = df.sort_values(["SecuCode", "date", "entry_time"]).copy()

    # ===== 第2类: 价格轨迹 =====
    # Lag 特征: 过去 k 个 bar 的 X1/X2 值
    for col, lags in [("X1", [1, 3, 6, 10]), ("X2", [1, 3, 6])]:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby(["SecuCode", "date"])[col].shift(lag)

    # Diff 特征: X1 的短期变化
    df["X1_diff1"] = df["X1"] - df.groupby(["SecuCode", "date"])["X1"].shift(1)
    df["X1_diff3"] = df["X1"] - df.groupby(["SecuCode", "date"])["X1"].shift(3)

    # Slope 特征: 过去30分钟(6bar)的近似斜率
    df["X1_slope_6bar"] = (
        df["X1"] - df.groupby(["SecuCode", "date"])["X1"].shift(6)
    ) / 6

    # 晨盘 X1 标准差: 过去12个bar的滚动std
    df["X1_morning_std"] = (
        df.groupby(["SecuCode", "date"])["X1"]
        .rolling(12, min_periods=6)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )
    logger.info("  第2类特征(价格轨迹)完成")

    # ===== 第3类: 量能演变 =====
    # 最近3bar成交量之和
    df["_vol_r3"] = (
        df.groupby(["SecuCode", "date"])["volume"]
        .rolling(3, min_periods=2)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    # 前3bar成交量之和（shift 3 位）
    df["_vol_r3_prev"] = df.groupby(["SecuCode", "date"])["_vol_r3"].shift(3)

    # 量能加速度 = 最近3bar / 前3bar
    df["vol_accel"] = df["_vol_r3"] / df["_vol_r3_prev"]

    # 成交量集中度 = 最近3bar / 累计总量
    df["_cum_vol"] = df.groupby(["SecuCode", "date"])["volume"].cumsum()
    df["vol_concentration"] = df["_vol_r3"] / df["_cum_vol"]

    df.drop(columns=["_vol_r3", "_vol_r3_prev", "_cum_vol"], inplace=True)
    logger.info("  第3类特征(量能演变)完成")

    # ===== 第4类: 日内价格形态 =====
    grp = df.groupby(["SecuCode", "date"])

    # 晨盘收益: close / 当天第一个bar的open
    first_open = grp["open"].transform("first")
    df["morning_ret"] = df["close"] / first_open - 1

    # 价格位置: (close - 日内最低) / (日内最高 - 日内最低)
    df["_cum_high"] = grp["high"].cummax()
    df["_cum_low"] = grp["low"].cummin()
    price_range = df["_cum_high"] - df["_cum_low"]
    df["price_position"] = np.where(
        price_range > 0,
        (df["close"] - df["_cum_low"]) / price_range,
        0.5,
    )

    # 晨盘振幅
    df["morning_range"] = (df["_cum_high"] - df["_cum_low"]) / first_open

    # 日内波动率: 过去12个bar的5分钟收益率标准差
    df["_ret_5m"] = grp["close"].pct_change()
    df["intraday_vol"] = (
        df.groupby(["SecuCode", "date"])["_ret_5m"]
        .rolling(12, min_periods=6)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    df.drop(columns=["_cum_high", "_cum_low", "_ret_5m"], inplace=True)
    logger.info("  第4类特征(价格形态)完成")

    return df


# ==================== 每日汇总 ====================
def build_daily_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    从全量bar数据构建每日汇总（用于计算跨日特征）

    使用 groupby 聚合，避免遍历
    """
    logger.info("构建每日汇总...")
    daily = (
        df_all.sort_values(["SecuCode", "date", "entry_time"])
        .groupby(["SecuCode", "date"])
        .agg(
            day_open=("open", "first"),
            day_close=("close", "last"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_volume=("volume", "sum"),
        )
        .reset_index()
    )
    logger.info(f"  每日汇总: {len(daily):,} 行")
    return daily


# ==================== 跨日特征构建 ====================
def build_crossday_features(
    df_entry: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    """
    在入场行上构建跨日特征

    包含第5类特征 + rel_vol
    注意：所有跨日特征只使用截至昨日的信息，避免未来函数
    """
    logger.info("构建跨日特征...")
    daily = daily.sort_values(["SecuCode", "date"]).copy()
    grp_d = daily.groupby("SecuCode")

    # 日收益率（用于计算波动率）
    daily["day_ret"] = grp_d["day_close"].pct_change()

    # 隔夜跳空: 今日开盘 / 昨日收盘 - 1
    daily["overnight_gap"] = daily["day_open"] / grp_d["day_close"].shift(1) - 1

    # 昨日涨跌幅: shift(1) 确保是昨天的收益
    daily["prev_day_ret"] = grp_d["day_ret"].shift(1)

    # 5日动量: 昨日收盘 / 6天前收盘 - 1（5个交易日的累计收益）
    daily["momentum_5d"] = grp_d["day_close"].shift(1) / grp_d["day_close"].shift(6) - 1

    # 历史波动率: 过去20日的日收益率标准差，shift(1) 避免使用当日数据
    daily["hist_vol_20d"] = grp_d["day_ret"].transform(
        lambda x: x.rolling(CROSS_DAY_WINDOW, min_periods=10).std().shift(1)
    )

    # 合并跨日特征到入场行
    merge_cols = [
        "SecuCode",
        "date",
        "overnight_gap",
        "prev_day_ret",
        "momentum_5d",
        "hist_vol_20d",
    ]
    df_entry = df_entry.merge(daily[merge_cols], on=["SecuCode", "date"], how="left")

    # 相对成交量: 10:30累计成交量 / 过去20天同时刻均值
    df_entry = df_entry.sort_values(["SecuCode", "date"])
    df_entry["rel_vol"] = df_entry["cum_volume"] / (
        df_entry.groupby("SecuCode")["cum_volume"].transform(
            lambda x: x.rolling(CROSS_DAY_WINDOW, min_periods=5).mean().shift(1)
        )
    )

    logger.info("  第5类特征(跨日) + rel_vol 完成")
    return df_entry


# ==================== 主函数 ====================
def process_year(year: int):
    """处理一个年份的特征构建"""
    logger.info(f"{'='*60}")
    logger.info(f"开始构建 {year} 年时序特征 (入场时间={ENTRY_TIME})")
    logger.info(f"{'='*60}")

    # 1. 加载原始数据
    df_all = load_raw_features(year)

    # 2. 构建每日汇总（跨日特征需要）
    daily = build_daily_summary(df_all)

    # 3. 构建日内特征（在全量bar上计算）
    df_all = build_intraday_features(df_all)

    # 4. 筛选入场时间
    df_entry = df_all[df_all["entry_time"] == ENTRY_TIME].copy()
    logger.info(f"筛选入场时间 {ENTRY_TIME}: {len(df_entry):,} 行")

    # 5. 构建跨日特征
    df_entry = build_crossday_features(df_entry, daily)

    # 6. 保存
    output_path = OUTPUT_DIR / f"df_ts_features_{year}.pkl"
    df_entry.to_pickle(output_path)
    logger.success(
        f"{year} 年时序特征已保存: {output_path}\n"
        f"  行数: {len(df_entry):,}, 列数: {df_entry.shape[1]}"
    )

    # 打印特征完整性报告
    feat_cols = [
        "X1",
        "X2",
        "X1_zscore",
        "X2_zscore",
        "X1_lag1",
        "X1_lag3",
        "X1_lag6",
        "X1_lag10",
        "X2_lag1",
        "X2_lag3",
        "X2_lag6",
        "X1_diff1",
        "X1_diff3",
        "X1_slope_6bar",
        "X1_morning_std",
        "rel_vol",
        "vol_accel",
        "vol_concentration",
        "morning_ret",
        "price_position",
        "morning_range",
        "intraday_vol",
        "overnight_gap",
        "prev_day_ret",
        "momentum_5d",
        "hist_vol_20d",
    ]
    logger.info("特征完整性:")
    for col in feat_cols:
        if col in df_entry.columns:
            valid = df_entry[col].notna().mean()
            logger.info(f"  {col:>20s}: 有效率 {valid:.1%}")
        else:
            logger.warning(f"  {col:>20s}: 缺失!")

    return df_entry


def main():
    """主函数：依次处理所有年份"""
    for year in YEARS:
        process_year(year)
    logger.success("全部年份时序特征构建完成!")


if __name__ == "__main__":
    main()
