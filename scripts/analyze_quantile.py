"""
分层测试脚本

职责：读取特征数据，将指定特征按等频分组，计算每组的Y统计量
输入：df_features_{year}.pkl

分析维度：
  - 入场时间: ENTRY_TIMES 指定的时间点
  - 持有时间: HOLD_HORIZONS 指定的标签列
  - 分层特征: QUANTILE_FEATURE 指定的特征列
  - 分层区间: QUANTILE_RANGE 指定的特征值范围

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

YEAR = 2025

# 入场时间列表
# ENTRY_TIMES = ["09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00"]
ENTRY_TIMES = ["10:30"]

# 持有时间列表: (标签列名, 显示名)
# HOLD_HORIZONS = [
#     ("Y_15m", "15m"),
#     ("Y_30m", "30m"),
#     ("Y_60m", "60m"),
#     ("Y_90m", "90m"),
#     ("Y_120m", "120m"),
#     ("Y", "rest"),
# ]
HOLD_HORIZONS = [("Y_120m", "120m")]

# 分层测试配置
QUANTILE_FEATURE = "X1_zscore"  # 分层依据的特征
QUANTILE_FEATURE_NAME = "X1"  # 显示名
N_QUANTILES = 10  # 分组数
QUANTILE_RANGE = (4, None)  # 分层区间 (min, max)，None表示不限制
# 示例: (None, -1.5) 表示 Zf < -1.5; (2.0, None) 表示 Zf > 2.0; (-2, 2) 表示 -2 < Zf < 2


# ==================== 数据加载 ====================
def load_features(year: int) -> pd.DataFrame:
    """加载特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 calc_feature_x1_x2.py"
        )

    logger.info(f"加载特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    n_stocks = df["SecuCode"].nunique()
    n_days = df["date"].nunique()
    logger.success(f"加载完成: {len(df):,} 行, {n_stocks} 只股票, {n_days} 个交易日")
    return df


def prefilter(df: pd.DataFrame) -> pd.DataFrame:
    """预过滤：去除特征为NaN/Inf的行"""
    mask = (
        np.isfinite(df["X2_zscore"])
        & np.isfinite(df["X1_zscore"])
        & np.isfinite(df["Z_final"])
    )
    df_valid = df[mask].copy()
    logger.info(f"预过滤: {len(df):,} -> {len(df_valid):,} 行")
    return df_valid


# ==================== 核心计算 ====================
def analyze_quantile(df: pd.DataFrame) -> pd.DataFrame:
    """
    分层测试：将特征分成 N_QUANTILES 组，计算每组的 Y 统计量

    按 (入场时间, 持有时间) 组合，对 QUANTILE_FEATURE 做等频分组
    """
    lo, hi = QUANTILE_RANGE
    range_desc = ""
    if lo is not None or hi is not None:
        lo_s = f"{lo}" if lo is not None else "-inf"
        hi_s = f"{hi}" if hi is not None else "+inf"
        range_desc = f", 区间: [{lo_s}, {hi_s}]"
    logger.info(
        f"开始分层测试: {QUANTILE_FEATURE_NAME} 分 {N_QUANTILES} 组{range_desc}..."
    )
    all_rows = []

    for entry_time in ENTRY_TIMES:
        df_entry = df[df["entry_time"] == entry_time]
        if df_entry.empty:
            continue

        for label_col, horizon_name in HOLD_HORIZONS:
            if label_col not in df_entry.columns:
                continue

            # 过滤标签异常值
            mask = (
                df_entry[label_col].notna()
                & np.isfinite(df_entry[label_col])
                & (df_entry[label_col] > 0.9)
                & (df_entry[label_col] < 1.1)
            )
            ds = df_entry.loc[mask, [QUANTILE_FEATURE, label_col]].copy()

            # 按用户指定的区间过滤
            if lo is not None:
                ds = ds[ds[QUANTILE_FEATURE] >= lo]
            if hi is not None:
                ds = ds[ds[QUANTILE_FEATURE] <= hi]

            if len(ds) < N_QUANTILES * 10:
                continue

            # 等频分组
            ds["grp"] = pd.qcut(
                ds[QUANTILE_FEATURE], N_QUANTILES, labels=False, duplicates="drop"
            )

            # 向量化统计
            grp = ds.groupby("grp")[label_col]
            feat_grp = ds.groupby("grp")[QUANTILE_FEATURE]
            stats = pd.DataFrame(
                {
                    "count": grp.count(),
                    "Y_mean": grp.mean(),
                    "Y_std": grp.std(),
                    "Y_median": grp.median(),
                    "win_rate": grp.apply(lambda x: (x < 1).mean()),
                    "feat_mean": feat_grp.mean(),
                    "feat_min": feat_grp.min(),
                    "feat_max": feat_grp.max(),
                }
            ).reset_index()

            # 收益 bps = (Y_mean - 1) * 10000
            stats["ret_bps"] = (stats["Y_mean"] - 1) * 10000
            stats["entry_time"] = entry_time
            stats["hold"] = horizon_name
            all_rows.append(stats)

        logger.debug(f"  入场时间 {entry_time} 分层完成")

    if not all_rows:
        logger.warning("分层测试无有效数据")
        return pd.DataFrame()

    df_q = pd.concat(all_rows, ignore_index=True)
    logger.success(f"分层测试完成: {len(df_q)} 组")
    return df_q


# ==================== 报告打印 ====================
def print_quantile_report(df_q: pd.DataFrame):
    """打印分层测试报告"""
    if df_q.empty:
        return

    print("\n" + "=" * 120)
    lo, hi = QUANTILE_RANGE
    range_str = ""
    if lo is not None or hi is not None:
        lo_s = f"{lo}" if lo is not None else "-inf"
        hi_s = f"{hi}" if hi is not None else "+inf"
        range_str = f", 区间: [{lo_s}, {hi_s}]"
    print(f"分层测试报告: {QUANTILE_FEATURE_NAME} 分 {N_QUANTILES} 组{range_str}")
    print("=" * 120)

    # 列宽
    W = {
        "grp": 5,
        "count": 8,
        "feature_range": 22,
        "feature_mean": 10,
        "ret_bps": 10,
        "Y_mean": 12,
        "Y_std": 10,
        "win_rate": 10,
    }

    for entry_time in ENTRY_TIMES:
        for _, horizon_name in HOLD_HORIZONS:
            sub = df_q[
                (df_q["entry_time"] == entry_time) & (df_q["hold"] == horizon_name)
            ]
            if sub.empty:
                continue

            print(f"\n--- entry: {entry_time}, hold: {horizon_name} ---")
            header = (
                f"  {'grp':>{W['grp']}}  {'count':>{W['count']}}  "
                f"{'feature_range':>{W['feature_range']}}  {'feature_mean':>{W['feature_mean']}}  "
                f"{'ret_bps':>{W['ret_bps']}}  {'Y_mean':>{W['Y_mean']}}  "
                f"{'Y_std':>{W['Y_std']}}  {'win_rate':>{W['win_rate']}}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))

            for _, row in sub.sort_values("grp").iterrows():
                f_range = f"[{row['feat_min']:+.3f}, {row['feat_max']:+.3f}]"
                print(
                    f"  {int(row['grp']):>{W['grp']}}  {int(row['count']):>{W['count']},}  "
                    f"{f_range:>{W['feature_range']}}  {row['feat_mean']:>{W['feature_mean']}.4f}  "
                    f"{row['ret_bps']:>{W['ret_bps']}.2f}  {row['Y_mean']:>{W['Y_mean']}.6f}  "
                    f"{row['Y_std']:>{W['Y_std']}.6f}  {row['win_rate']:>{W['win_rate']}.4f}"
                )

    print("\n" + "=" * 120)


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("分层测试分析")
    logger.info("=" * 60)

    # 1. 加载并预过滤
    df = load_features(YEAR)
    df = prefilter(df)

    # 2. 分层测试
    df_q = analyze_quantile(df)

    # 3. 打印报告
    print_quantile_report(df_q)

    return df_q


if __name__ == "__main__":
    df_q = main()
