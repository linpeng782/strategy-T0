"""
时序特征直觉分析脚本

职责：对时序特征做基础分析，建立预测力的直觉判断
输入：df_ts_features_{year}.pkl（含时序特征 + Y_120m）
输出：终端打印报告

分析维度：
  1. 逐特征 Rank IC — 每个特征与未来收益的日度 Spearman 相关性
  2. 晨盘轨迹模式 — 按 X1 趋势方向分组，看各组收益差异
  3. 关键特征分位数 — 分10组看收益单调性

依赖：build_ts_features.py 生产的时序特征数据

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
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

YEAR = 2025

# 持有期
HOLD_COL = "Y_120m"

# 需要分析IC的全部特征
IC_FEATURES = [
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

# 分位数分析的特征（选最关键的几个）
QUANTILE_FEATURES = [
    "X1_zscore",
    "X2_zscore",
    "X1_slope_6bar",
    "rel_vol",
    "morning_ret",
    "momentum_5d",
]
N_QUANTILES = 10

# 轨迹模式分析的阈值
SLOPE_THRESHOLD = 0.001  # X1_slope_6bar 的分组阈值


# ==================== 数据加载 ====================
def load_ts_features(year: int) -> pd.DataFrame:
    """加载时序特征数据"""
    pkl_path = OUTPUT_DIR / f"df_ts_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"时序特征数据不存在: {pkl_path}\n请先运行 build_ts_features.py"
        )
    df = pd.read_pickle(pkl_path)
    logger.success(f"加载 {year} 年数据: {len(df):,} 行")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """预处理：计算收益、过滤异常值"""
    # 过滤 Y_120m 异常
    mask = df[HOLD_COL].notna() & np.isfinite(df[HOLD_COL])
    mask = mask & (df[HOLD_COL] > 0.9) & (df[HOLD_COL] < 1.1)
    df = df[mask].copy()

    # 做多收益
    df["raw_ret"] = 1.0 / df[HOLD_COL] - 1.0
    df["market_ret"] = df.groupby("date")["raw_ret"].transform("mean")
    df["excess_ret"] = df["raw_ret"] - df["market_ret"]

    logger.info(
        f"有效样本: {len(df):,}, 日均: {len(df)/df['date'].nunique():.0f} 只股票"
    )
    return df


# ==================== 分析1: 逐特征 Rank IC ====================
def analyze_rank_ic(df: pd.DataFrame):
    """
    计算每个特征的日度 Rank IC（Spearman 相关性）

    Rank IC = 每天截面上 rank(feature) 与 rank(excess_ret) 的 Spearman 相关
    ICIR = IC均值 / IC标准差（越高越稳定）
    """
    print("\n" + "=" * 90)
    print("分析1: 逐特征 Rank IC（日度 Spearman 相关，目标=excess_ret）")
    print("=" * 90)

    results = []
    for feat in IC_FEATURES:
        if feat not in df.columns:
            continue

        # 按日计算 Spearman 相关
        daily_ic = (
            df.groupby("date")
            .apply(
                lambda g: (
                    g[feat].corr(g["excess_ret"], method="spearman")
                    if g[feat].notna().sum() > 30
                    else np.nan
                ),
                include_groups=False,
            )
            .dropna()
        )

        if len(daily_ic) < 10:
            continue

        results.append(
            {
                "feature": feat,
                "ic_mean": daily_ic.mean(),
                "ic_std": daily_ic.std(),
                "icir": daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0,
                "ic_pos_rate": (daily_ic > 0).mean(),
                "n_days": len(daily_ic),
            }
        )

    df_ic = pd.DataFrame(results).sort_values("icir", key=abs, ascending=False)

    # 打印报告
    header = (
        f"  {'Feature':>20s}  {'IC_Mean':>10s}  {'IC_Std':>10s}  "
        f"{'ICIR':>8s}  {'IC>0':>8s}  {'Days':>6s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 70)
    for _, row in df_ic.iterrows():
        print(
            f"  {row['feature']:>20s}  {row['ic_mean']:>10.4f}  {row['ic_std']:>10.4f}  "
            f"{row['icir']:>8.2f}  {row['ic_pos_rate']:>7.1%}  {row['n_days']:>6.0f}"
        )
    print("  " + "-" * 70)

    return df_ic


# ==================== 分析2: 晨盘轨迹模式 ====================
def analyze_trajectory_pattern(df: pd.DataFrame):
    """
    按晨盘 X1 轨迹将股票分组，比较各组的未来收益

    分组逻辑（基于 X1_slope_6bar 和 X1）：
      - 加速上涨: slope > 阈值 且 X1 > 0
      - 加速下跌: slope < -阈值 且 X1 < 0
      - 高位回落: slope < -阈值 且 X1 > 0
      - 低位反弹: slope > 阈值 且 X1 < 0
      - 震荡: |slope| <= 阈值
    """
    print("\n" + "=" * 90)
    print("分析2: 晨盘轨迹模式分析（基于 X1_slope_6bar + X1）")
    print("=" * 90)

    required = ["X1_slope_6bar", "X1"]
    if not all(c in df.columns for c in required):
        print("  缺少必要特征，跳过")
        return

    sub = df[df["X1_slope_6bar"].notna() & df["X1"].notna()].copy()

    # 分组
    conditions = [
        (sub["X1_slope_6bar"] > SLOPE_THRESHOLD) & (sub["X1"] > 0),
        (sub["X1_slope_6bar"] < -SLOPE_THRESHOLD) & (sub["X1"] < 0),
        (sub["X1_slope_6bar"] < -SLOPE_THRESHOLD) & (sub["X1"] > 0),
        (sub["X1_slope_6bar"] > SLOPE_THRESHOLD) & (sub["X1"] < 0),
    ]
    labels = ["加速上涨", "加速下跌", "高位回落", "低位反弹"]
    sub["trajectory"] = np.select(conditions, labels, default="震荡")

    # 各组统计
    header = (
        f"  {'轨迹模式':>12s}  {'样本数':>8s}  {'占比':>8s}  "
        f"{'均值bps':>10s}  {'中位bps':>10s}  {'胜率':>8s}  {'excess_bps':>12s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 80)

    for label in labels + ["震荡"]:
        grp = sub[sub["trajectory"] == label]
        if len(grp) < 50:
            continue
        print(
            f"  {label:>12s}  {len(grp):>8,}  {len(grp)/len(sub):>7.1%}  "
            f"{grp['raw_ret'].mean()*10000:>10.2f}  "
            f"{grp['raw_ret'].median()*10000:>10.2f}  "
            f"{(grp['raw_ret']>0).mean():>7.1%}  "
            f"{grp['excess_ret'].mean()*10000:>12.2f}"
        )
    print("  " + "-" * 80)

    # 全样本基准
    print(
        f"  {'全样本':>12s}  {len(sub):>8,}  {'100.0%':>8s}  "
        f"{sub['raw_ret'].mean()*10000:>10.2f}  "
        f"{sub['raw_ret'].median()*10000:>10.2f}  "
        f"{(sub['raw_ret']>0).mean():>7.1%}  "
        f"{sub['excess_ret'].mean()*10000:>12.2f}"
    )


# ==================== 分析3: 特征分位数 ====================
def analyze_quantile(df: pd.DataFrame):
    """
    对关键特征做等频分位数分析

    将特征分为 N_QUANTILES 组，计算每组的平均收益/胜率，
    观察是否存在单调关系
    """
    print("\n" + "=" * 90)
    print(f"分析3: 关键特征分位数分析 ({N_QUANTILES} 组)")
    print("=" * 90)

    for feat in QUANTILE_FEATURES:
        if feat not in df.columns:
            continue

        sub = df[df[feat].notna() & np.isfinite(df[feat])].copy()
        if len(sub) < N_QUANTILES * 100:
            continue

        sub["_grp"] = pd.qcut(sub[feat], N_QUANTILES, labels=False, duplicates="drop")

        stats = sub.groupby("_grp").agg(
            count=("raw_ret", "size"),
            feat_mean=(feat, "mean"),
            raw_bps=("raw_ret", lambda x: x.mean() * 10000),
            excess_bps=("excess_ret", lambda x: x.mean() * 10000),
            win_rate=("raw_ret", lambda x: (x > 0).mean()),
        )

        print(f"\n--- {feat} ---")
        header = (
            f"  {'组':>4s}  {'样本数':>8s}  {'特征均值':>12s}  "
            f"{'raw_bps':>10s}  {'excess_bps':>12s}  {'胜率':>8s}"
        )
        print(header)
        print("  " + "-" * 62)

        for grp_id, row in stats.iterrows():
            print(
                f"  {grp_id:>4.0f}  {row['count']:>8,.0f}  {row['feat_mean']:>12.4f}  "
                f"{row['raw_bps']:>10.2f}  {row['excess_bps']:>12.2f}  "
                f"{row['win_rate']:>7.1%}"
            )

        # 单调性指标: 头尾组差异
        if len(stats) >= 2:
            spread = stats.iloc[-1]["excess_bps"] - stats.iloc[0]["excess_bps"]
            print(f"  头尾组 excess 差: {spread:+.2f} bps")


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info(f"时序特征直觉分析 ({YEAR} 年)")
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_ts_features(YEAR)
    df = prepare_data(df)

    # 2. Rank IC 分析
    df_ic = analyze_rank_ic(df)

    # 3. 轨迹模式分析
    analyze_trajectory_pattern(df)

    # 4. 分位数分析
    analyze_quantile(df)

    print("\n" + "=" * 90)
    print("分析完成!")
    print("=" * 90)

    return df_ic


if __name__ == "__main__":
    df_ic = main()
