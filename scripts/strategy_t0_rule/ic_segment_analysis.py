"""
分时段IC分析脚本
职责：在全局IC分析基础上，按日内时间段（开盘/午前/午后/尾盘）分别计算IC，
      帮助判断哪个时间段的特征预测力最强，辅助确定最优交易时间窗口。

依赖：需要先运行 ic_analysis.py 生成 ic_daily_{YEAR}.pkl 和 features_{YEAR}.pkl
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

TEST_YEAR = 2024

HOLD_PERIODS = {
    "15min": 3,
    "25min": 5,
    "40min": 8,
    "60min": 12,
}

# 日内时间段划分
TIME_SEGMENTS = {
    "开盘段(09:35-10:00)": ("09:35", "10:00"),
    "午前段(10:05-11:30)": ("10:05", "11:30"),
    "午后段(13:00-14:00)": ("13:00", "14:00"),
    "尾盘段(14:05-14:55)": ("14:05", "14:55"),
}

COST_BPS = 15


# ==================== 复用 ic_analysis 的核心函数 ====================
def _agg_ic_stats(ic_vals: pd.Series) -> pd.Series:
    """对一组IC序列计算汇总统计"""
    ic_vals = ic_vals.dropna()
    n = len(ic_vals)
    if n < 2:
        return pd.Series(
            {"MeanIC": np.nan, "StdIC": np.nan, "ICIR": np.nan,
             "t_stat": np.nan, "p_value": np.nan, "IC_pos_ratio": np.nan, "N_days": n}
        )
    mean_ic = ic_vals.mean()
    std_ic = ic_vals.std()
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat, p_value = stats.ttest_1samp(ic_vals, popmean=0)
    ic_pos = (ic_vals > 0).mean()
    return pd.Series(
        {"MeanIC": mean_ic, "StdIC": std_ic, "ICIR": icir,
         "t_stat": t_stat, "p_value": p_value, "IC_pos_ratio": ic_pos, "N_days": n}
    )


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """格式化IC汇总表"""
    summary["MeanIC"] = summary["MeanIC"].round(4)
    summary["StdIC"] = summary["StdIC"].round(4)
    summary["ICIR"] = summary["ICIR"].round(3)
    summary["t_stat"] = summary["t_stat"].round(3)
    summary["p_value"] = summary["p_value"].round(4)
    summary["IC_pos_ratio"] = (summary["IC_pos_ratio"] * 100).round(1).astype(str) + "%"
    summary["N_days"] = summary["N_days"].astype(int)
    return summary


def compute_ic(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """全向量化Rank IC计算（从ic_analysis复用）"""
    ret_cols = [f"fwd_ret_{label}" for label in HOLD_PERIODS]
    grp_keys = ["SecuCode", "date"]

    work = df[grp_keys].copy()
    for col in feature_cols + ret_cols:
        if col in df.columns:
            work[col + "_r"] = df.groupby(grp_keys)[col].rank(pct=True, na_option="keep")

    grp_size = df.groupby(grp_keys)["close"].transform("count")
    work = work[grp_size >= 5].copy()

    records = []
    grp = work.groupby(grp_keys)

    for feat_col in feature_cols:
        xr = feat_col + "_r"
        if xr not in work.columns:
            continue
        mean_x = grp[xr].transform("mean")
        dx = work[xr] - mean_x
        work["_dx2"] = dx * dx
        sum_dx2 = grp["_dx2"].transform("sum")

        for period_label, ret_col in zip(HOLD_PERIODS.keys(), ret_cols):
            yr = ret_col + "_r"
            if yr not in work.columns:
                continue
            mean_y = grp[yr].transform("mean")
            dy = work[yr] - mean_y
            work["_dxdy"] = dx * dy
            work["_dy2"] = dy * dy
            sum_dxdy = grp["_dxdy"].transform("sum")
            sum_dy2 = grp["_dy2"].transform("sum")

            denom = np.sqrt(sum_dx2 * sum_dy2)
            ic_vals = np.where(denom > 0, sum_dxdy / denom, np.nan)

            ic_series = (
                pd.Series(ic_vals, index=work.index)
                .groupby([work["SecuCode"], work["date"]])
                .first()
                .reset_index()
            )
            ic_series.columns = ["SecuCode", "date", "ic"]
            ic_series["feature"] = feat_col
            ic_series["period"] = period_label
            records.append(ic_series)

    work.drop(columns=["_dxdy", "_dx2", "_dy2"], inplace=True, errors="ignore")
    if not records:
        return pd.DataFrame(columns=["SecuCode", "date", "feature", "period", "ic"])
    return pd.concat(records, ignore_index=True)[["SecuCode", "date", "feature", "period", "ic"]]


# ==================== 分时段IC计算 ====================
def compute_ic_by_time_segment(
    df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    按日内时间段分别计算IC，输出各时间段的IC汇总统计。
    返回 segment_summary_df：
        columns = [segment, feature, period, MeanIC, StdIC, ICIR, t_stat, p_value,
                   IC_pos_ratio, N_days]
    """
    if "time" not in df.columns:
        df = df.copy()
        df["time"] = df["bar_time"].dt.strftime("%H:%M")

    logger.info(f"开始分时段IC计算，共 {len(TIME_SEGMENTS)} 个时间段...")
    all_summaries = []

    fwd_ret_cols = [f"fwd_ret_{l}" for l in HOLD_PERIODS]

    for seg_name, (t_start, t_end) in TIME_SEGMENTS.items():
        mask = (df["time"] >= t_start) & (df["time"] <= t_end)
        df_seg = df[mask].copy()
        if len(df_seg) == 0:
            logger.warning(f"时间段 {seg_name} 无数据，跳过")
            continue

        logger.info(f"  时间段 {seg_name}：{len(df_seg):,} 行")

        valid_cols = [c for c in fwd_ret_cols if c in df_seg.columns]
        if not valid_cols:
            continue
        valid_mask = df_seg["buy_open"].notna() & df_seg[valid_cols].notna().all(axis=1)
        df_seg = df_seg[valid_mask]

        feat_cols_exist = [c for c in feature_cols if c in df_seg.columns]
        if not feat_cols_exist:
            continue

        ic_df_seg = compute_ic(df_seg, feat_cols_exist)
        if ic_df_seg.empty:
            continue

        ic_daily = ic_df_seg.groupby(["feature", "period", "date"])["ic"].mean().reset_index()
        seg_summary = (
            ic_daily.groupby(["feature", "period"])["ic"]
            .apply(_agg_ic_stats)
            .unstack(level=-1)
            .reset_index()
        )
        seg_summary = _format_summary(seg_summary)
        seg_summary.insert(0, "segment", seg_name)
        all_summaries.append(seg_summary)

    if not all_summaries:
        return pd.DataFrame()

    result = pd.concat(all_summaries, ignore_index=True)
    logger.success(f"分时段IC计算完成，共 {len(result)} 条汇总记录")
    return result


# ==================== 打印输出 ====================
def print_segment_summary(seg_summary: pd.DataFrame):
    """按持有期分组打印分时段IC对比表"""
    if seg_summary.empty:
        logger.warning("分时段汇总为空，跳过打印")
        return

    period_order = list(HOLD_PERIODS.keys())
    seg_order = list(TIME_SEGMENTS.keys())

    # 表1：各持有期下，行=时间段，列=特征，值=MeanIC
    logger.info(f"\n{'='*80}")
    logger.info("分时段MeanIC对比（行=时间段，列=特征）")
    logger.info(f"{'='*80}")
    for period in period_order:
        sub = seg_summary[seg_summary["period"] == period].copy()
        if sub.empty:
            continue
        pivot = sub.pivot(index="segment", columns="feature", values="MeanIC")
        pivot = pivot.reindex([s for s in seg_order if s in pivot.index])
        logger.info(f"\n持有期: {period}")
        print(pivot.round(4).to_string())

    # 表2：各特征下，行=时间段，列=持有期，值=ICIR
    logger.info(f"\n{'='*80}")
    logger.info("分时段ICIR对比（行=时间段，列=持有期）")
    logger.info(f"{'='*80}")
    for feat in seg_summary["feature"].unique():
        sub = seg_summary[seg_summary["feature"] == feat].copy()
        pivot = sub.pivot(index="segment", columns="period", values="ICIR")
        pivot = pivot.reindex(
            columns=[p for p in period_order if p in pivot.columns],
            index=[s for s in seg_order if s in pivot.index],
        )
        logger.info(f"\n特征: {feat}  (ICIR)")
        print(pivot.round(3).to_string())


# ==================== 主函数 ====================
def main():
    """
    分时段IC分析流程：
    1. 加载已缓存的特征文件（需先运行 ic_analysis.py 生成）
    2. 计算未来收益率
    3. 按时间段分别计算IC
    4. 汇总统计并输出
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ic_analysis import compute_forward_returns, NORM_FEATURE_COLS

    YEAR = TEST_YEAR

    # ---- 加载特征文件 ----
    feat_cache = OUTPUT_DIR / f"features_{YEAR}.pkl"
    if not feat_cache.exists():
        logger.error(f"特征文件不存在: {feat_cache}，请先运行 ic_analysis.py")
        return

    logger.info(f"加载特征文件: {feat_cache}")
    df_feat = pd.read_pickle(feat_cache)

    # ---- 计算未来收益率 ----
    df_feat = compute_forward_returns(df_feat)

    fwd_ret_cols = [f"fwd_ret_{l}" for l in HOLD_PERIODS]
    valid_mask = df_feat["buy_open"].notna() & df_feat[fwd_ret_cols].notna().all(axis=1)
    df_valid = df_feat[valid_mask].copy()
    logger.info(f"有效bar数: {len(df_valid):,}")

    all_feat_cols = [c for c in NORM_FEATURE_COLS if c in df_valid.columns]
    logger.info(f"分析特征({len(all_feat_cols)}个): {all_feat_cols}")

    # ---- 分时段IC计算 ----
    seg_summary = compute_ic_by_time_segment(df_valid, all_feat_cols)
    print_segment_summary(seg_summary)

    # ---- 保存结果 ----
    if not seg_summary.empty:
        seg_csv = OUTPUT_DIR / f"ic_segment_summary_{YEAR}.csv"
        seg_summary.to_csv(seg_csv, index=False)
        logger.success(f"分时段IC汇总已保存: {seg_csv}")


if __name__ == "__main__":
    main()
