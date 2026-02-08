"""
分层测试 + 相对成交量过滤

在 analyze_quantile.py 基础上，增加相对成交量（rel_vol）过滤维度。
对每个成交量阈值分别做分层测试，观察特征在放量/缩量条件下的表现差异。

输入：df_features_{year}.pkl
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
ENTRY_TIMES = ["10:30"]

# 持有时间列表: (标签列名, 显示名)
HOLD_HORIZONS = [("Y_120m", "120m")]

# 分层测试配置
QUANTILE_FEATURE = "X1_zscore"  # 分层依据的特征
QUANTILE_FEATURE_NAME = "X1"  # 显示名
N_QUANTILES = 10  # 分组数
QUANTILE_RANGE = (3, None)  # 分层区间 (min, max)，None表示不限制

# 相对成交量配置
VOL_ZSCORE_WINDOW = 10  # 计算相对成交量的滚动窗口（天数）
VOL_FILTER_THRESHOLDS = [None, 1.0, 1.5, 2.0]  # None=不过滤，数值=rel_vol>阈值


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
    logger.success(
        f"加载完成: {len(df):,} 行, {df['SecuCode'].nunique()} 只股票, "
        f"{df['date'].nunique()} 个交易日"
    )
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


def calc_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算相对成交量：当前Bar的cum_volume / 过去N天同一entry_time的cum_volume均值
    shift(1) 避免使用当天数据
    """
    logger.info(f"计算相对成交量 (窗口={VOL_ZSCORE_WINDOW}天)...")
    df = df.sort_values(["SecuCode", "entry_time", "date"])
    grp = df.groupby(["SecuCode", "entry_time"])["cum_volume"]
    vol_mean = grp.rolling(VOL_ZSCORE_WINDOW, min_periods=2).mean().shift(1)
    vol_mean = vol_mean.reset_index(level=[0, 1], drop=True)
    df["rel_vol"] = df["cum_volume"] / vol_mean
    n_valid = df["rel_vol"].notna().sum()
    logger.success(f"相对成交量计算完成, 有效样本: {n_valid:,}")
    return df


# ==================== 核心计算 ====================
def analyze_quantile_with_vol(
    df: pd.DataFrame, vol_threshold=None
) -> pd.DataFrame:
    """
    分层测试：对 QUANTILE_FEATURE 做等频分组，计算每组的 Y 统计量
    可选：按 rel_vol 阈值过滤样本
    """
    lo, hi = QUANTILE_RANGE
    filter_desc = f"rel_vol>{vol_threshold}" if vol_threshold else "无过滤"

    range_desc = ""
    if lo is not None or hi is not None:
        lo_s = f"{lo}" if lo is not None else "-inf"
        hi_s = f"{hi}" if hi is not None else "+inf"
        range_desc = f", 区间: [{lo_s}, {hi_s}]"
    logger.info(
        f"分层测试: {QUANTILE_FEATURE_NAME} 分 {N_QUANTILES} 组{range_desc}, "
        f"成交量过滤: {filter_desc}"
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
            cols = [QUANTILE_FEATURE, label_col, "rel_vol"]
            ds = df_entry.loc[mask, cols].copy()

            # 成交量过滤
            if vol_threshold is not None:
                ds = ds[ds["rel_vol"] > vol_threshold]

            # 按用户指定的特征区间过滤
            if lo is not None:
                ds = ds[ds[QUANTILE_FEATURE] >= lo]
            if hi is not None:
                ds = ds[ds[QUANTILE_FEATURE] <= hi]

            if len(ds) < N_QUANTILES * 10:
                logger.warning(
                    f"样本不足: {len(ds)} (需要>={N_QUANTILES * 10}), 跳过"
                )
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

            stats["ret_bps"] = (stats["Y_mean"] - 1) * 10000
            stats["entry_time"] = entry_time
            stats["hold"] = horizon_name
            stats["vol_filter"] = filter_desc
            all_rows.append(stats)

        logger.debug(f"  入场时间 {entry_time} 分层完成")

    if not all_rows:
        return pd.DataFrame()

    df_q = pd.concat(all_rows, ignore_index=True)
    logger.success(f"分层测试完成: {len(df_q)} 组 ({filter_desc})")
    return df_q


# ==================== 报告打印 ====================
def print_quantile_report(df_q: pd.DataFrame):
    """打印分层测试报告"""
    if df_q.empty:
        return

    lo, hi = QUANTILE_RANGE
    range_str = ""
    if lo is not None or hi is not None:
        lo_s = f"{lo}" if lo is not None else "-inf"
        hi_s = f"{hi}" if hi is not None else "+inf"
        range_str = f", 区间: [{lo_s}, {hi_s}]"

    # 列宽
    W = {
        "grp": 5, "count": 8, "feature_range": 22, "feature_mean": 12,
        "ret_bps": 10, "Y_mean": 12, "Y_std": 10, "win_rate": 10,
    }

    for vol_filter in df_q["vol_filter"].unique():
        df_vf = df_q[df_q["vol_filter"] == vol_filter]

        print("\n" + "=" * 120)
        print(
            f"分层测试报告: {QUANTILE_FEATURE_NAME} 分 {N_QUANTILES} 组"
            f"{range_str} | 成交量过滤: {vol_filter}"
        )
        print("=" * 120)

        for entry_time in ENTRY_TIMES:
            for _, horizon_name in HOLD_HORIZONS:
                sub = df_vf[
                    (df_vf["entry_time"] == entry_time)
                    & (df_vf["hold"] == horizon_name)
                ]
                if sub.empty:
                    continue

                print(f"\n--- entry: {entry_time}, hold: {horizon_name} ---")
                header = (
                    f"  {'grp':>{W['grp']}}  {'count':>{W['count']}}  "
                    f"{'feature_range':>{W['feature_range']}}  "
                    f"{'feature_mean':>{W['feature_mean']}}  "
                    f"{'ret_bps':>{W['ret_bps']}}  {'Y_mean':>{W['Y_mean']}}  "
                    f"{'Y_std':>{W['Y_std']}}  {'win_rate':>{W['win_rate']}}"
                )
                print(header)
                print("  " + "-" * (len(header) - 2))

                for _, row in sub.sort_values("grp").iterrows():
                    f_range = f"[{row['feat_min']:+.3f}, {row['feat_max']:+.3f}]"
                    print(
                        f"  {int(row['grp']):>{W['grp']}}  "
                        f"{int(row['count']):>{W['count']},}  "
                        f"{f_range:>{W['feature_range']}}  "
                        f"{row['feat_mean']:>{W['feature_mean']}.4f}  "
                        f"{row['ret_bps']:>{W['ret_bps']}.2f}  "
                        f"{row['Y_mean']:>{W['Y_mean']}.6f}  "
                        f"{row['Y_std']:>{W['Y_std']}.6f}  "
                        f"{row['win_rate']:>{W['win_rate']}.4f}"
                    )

    print("\n" + "=" * 120)


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("分层测试 + 相对成交量过滤")
    logger.info("=" * 60)

    # 1. 加载并预过滤
    df = load_features(YEAR)
    df = prefilter(df)

    # 2. 计算相对成交量
    df = calc_relative_volume(df)

    # 3. 对每个成交量阈值做分层测试
    all_stats = []
    for vol_th in VOL_FILTER_THRESHOLDS:
        stats = analyze_quantile_with_vol(df, vol_threshold=vol_th)
        if not stats.empty:
            all_stats.append(stats)

    # 4. 合并打印
    if all_stats:
        df_all = pd.concat(all_stats, ignore_index=True)
        print_quantile_report(df_all)
    else:
        logger.warning("无有效结果")


if __name__ == "__main__":
    main()
