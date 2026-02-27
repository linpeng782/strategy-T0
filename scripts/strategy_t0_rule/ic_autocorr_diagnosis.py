"""
自相关诊断脚本
职责：诊断5分钟数据中三层自相关问题，量化ICIR虚高程度。

三层诊断：
  1. 跨天IC自相关：每只股票的IC时间序列（跨天）是否有自相关
     → 如果低，说明跨天ICIR计算有效
  2. 日内因子bar自相关：同一只股票同一天，相邻bar的因子值是否高度相关
     → 越高说明日内独立样本数越少，ICIR越虚高
  3. 日内收益序列重叠自相关：相邻bar的未来收益是否因窗口重叠而高度相关
     → 持有期越长，重叠越严重，ICIR虚高越严重

输出：
  - 控制台打印三层诊断结果
  - 保存诊断报告到 output/autocorr_diagnosis_{YEAR}.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from ic_analysis import (
    compute_forward_returns,
    NORM_FEATURE_COLS,
    HOLD_PERIODS,
    OUTPUT_DIR,
    TEST_YEAR,
)

# ==================== 手动修改参数 ====================
YEAR = TEST_YEAR
# 诊断1/2/3 使用的样本股票数（None=全市场，建议先用小样本）
DIAG_SAMPLE_STOCKS = 20
# ======================================================


def diag_ic_cross_day_autocorr(ic_df: pd.DataFrame, feat_list: list) -> pd.DataFrame:
    """
    诊断1：IC时间序列的跨天自相关（lag-1）。
    每只股票的IC序列在时间维度上的自相关，反映IC的持续性。
    如果接近0，说明跨天ICIR计算有效（今天IC高不预测明天IC高）。
    """
    logger.info("诊断1：IC时间序列跨天自相关（lag-1）...")
    records = []
    for feat in feat_list:
        for period in HOLD_PERIODS.keys():
            sub = ic_df[(ic_df["feature"] == feat) & (ic_df["period"] == period)]
            autocorrs = []
            for stock, grp in sub.groupby("SecuCode"):
                ic_ts = grp.sort_values("date")["ic"].dropna()
                if len(ic_ts) >= 20:
                    ac = ic_ts.autocorr(lag=1)
                    if not np.isnan(ac):
                        autocorrs.append(ac)
            if autocorrs:
                records.append({
                    "诊断": "跨天IC自相关(lag-1)",
                    "feature": feat,
                    "period": period,
                    "mean": round(np.mean(autocorrs), 4),
                    "std": round(np.std(autocorrs), 4),
                    "median": round(np.median(autocorrs), 4),
                    "结论": "低(<0.1)→跨天ICIR有效" if abs(np.mean(autocorrs)) < 0.1 else "高→跨天IC有持续性",
                })
    return pd.DataFrame(records)


def diag_factor_intraday_autocorr(df: pd.DataFrame, feat_list: list) -> pd.DataFrame:
    """
    诊断2：日内因子bar自相关（lag-1）。
    同一只股票同一天，相邻bar的因子值自相关，反映因子序列的平滑程度。
    越高说明日内独立样本数越少，IC估计的有效样本数被高估。

    有效独立样本数估算：N_eff ≈ N / (1 + 2*rho/(1-rho))
    """
    logger.info("诊断2：日内因子bar自相关（lag-1）...")
    records = []
    for feat in feat_list:
        if feat not in df.columns:
            continue
        autocorrs = []
        for (stock, date), grp in df.groupby(["SecuCode", "date"]):
            vals = grp.sort_values("bar_time")[feat].dropna()
            if len(vals) >= 10:
                ac = vals.autocorr(lag=1)
                if not np.isnan(ac):
                    autocorrs.append(ac)
        if autocorrs:
            rho = np.mean(autocorrs)
            # 估算有效独立样本数（每天48根bar）
            n_total = 48
            if rho < 1:
                n_eff = n_total * (1 - rho) / (1 + rho)
            else:
                n_eff = 1.0
            records.append({
                "诊断": "日内因子自相关(lag-1)",
                "feature": feat,
                "period": "all",
                "mean": round(rho, 4),
                "std": round(np.std(autocorrs), 4),
                "median": round(np.median(autocorrs), 4),
                "N_eff(每天)": round(n_eff, 1),
                "结论": f"48bar中约{n_eff:.0f}个独立样本",
            })
    return pd.DataFrame(records)


def diag_return_overlap_autocorr(df: pd.DataFrame) -> pd.DataFrame:
    """
    诊断3：未来收益序列的重叠自相关（lag-1）。
    相邻bar的未来收益因窗口重叠而高度相关，直接导致IC序列方差被压缩，ICIR虚高。

    理论上限：持有N根bar → 相邻收益重叠(N-1)/N → 自相关上限=(N-1)/N
    """
    logger.info("诊断3：未来收益序列重叠自相关（lag-1）...")
    records = []
    for period, n_bars in HOLD_PERIODS.items():
        col = f"fwd_ret_{period}"
        if col not in df.columns:
            continue
        autocorrs = []
        for (stock, date), grp in df.groupby(["SecuCode", "date"]):
            vals = grp.sort_values("bar_time")[col].dropna()
            if len(vals) >= 10:
                ac = vals.autocorr(lag=1)
                if not np.isnan(ac):
                    autocorrs.append(ac)
        if autocorrs:
            theory_upper = (n_bars - 1) / n_bars
            actual = np.mean(autocorrs)
            # ICIR虚高倍数估算：std被压缩 → ICIR被高估
            # 近似：ICIR_true ≈ ICIR_observed * sqrt(1 - actual^2) / sqrt(1 - 0)
            icir_inflation = 1.0 / np.sqrt(max(1 - actual**2, 0.01))
            records.append({
                "诊断": "收益序列重叠自相关(lag-1)",
                "feature": "fwd_ret",
                "period": period,
                "持有bar数": n_bars,
                "理论上限": round(theory_upper, 3),
                "实测mean": round(actual, 4),
                "实测/理论": round(actual / theory_upper, 2),
                "ICIR虚高估计": f"约{icir_inflation:.2f}x",
                "结论": f"60min最严重({actual:.2f})，15min相对轻({np.mean(autocorrs):.2f})" if period == "60min" else "",
            })
    return pd.DataFrame(records)


def print_diagnosis(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    """打印三层诊断结果"""
    logger.info(f"\n{'='*80}")
    logger.info("诊断1：IC时间序列跨天自相关（lag-1）")
    logger.info("结论：接近0说明跨天ICIR计算有效，今天IC高不预测明天IC高")
    logger.info(f"{'='*80}")
    if not df1.empty:
        # 只打印15min和60min
        sub = df1[df1["period"].isin(["15min", "60min"])][
            ["feature", "period", "mean", "median", "结论"]
        ]
        print(sub.to_string(index=False))

    logger.info(f"\n{'='*80}")
    logger.info("诊断2：日内因子bar自相关（lag-1）")
    logger.info("结论：越高说明日内独立样本越少，IC估计偏差越大")
    logger.info(f"{'='*80}")
    if not df2.empty:
        print(df2[["feature", "mean", "median", "N_eff(每天)", "结论"]].to_string(index=False))

    logger.info(f"\n{'='*80}")
    logger.info("诊断3：未来收益序列重叠自相关（lag-1）")
    logger.info("结论：持有期越长重叠越严重，ICIR虚高越严重，这是60min ICIR虚高的根本原因")
    logger.info(f"{'='*80}")
    if not df3.empty:
        print(df3[["period", "持有bar数", "理论上限", "实测mean", "实测/理论", "ICIR虚高估计"]].to_string(index=False))


def main():
    # ---- 加载特征数据 ----
    feat_cache = OUTPUT_DIR / f"features_{YEAR}.pkl"
    if not feat_cache.exists():
        logger.error(f"未找到特征缓存文件: {feat_cache}，请先运行 ic_analysis.py")
        return

    logger.info(f"加载特征文件: {feat_cache}")
    df_feat = pd.read_pickle(feat_cache)

    # 取样本股票做诊断2/3（避免全市场耗时过长）
    if DIAG_SAMPLE_STOCKS is not None:
        sample_stocks = df_feat["SecuCode"].unique()[:DIAG_SAMPLE_STOCKS]
        df_sample = df_feat[df_feat["SecuCode"].isin(sample_stocks)].copy()
        logger.info(f"诊断2/3使用 {len(sample_stocks)} 只样本股票")
    else:
        df_sample = df_feat.copy()
        logger.info("诊断2/3使用全市场数据")

    df_sample = compute_forward_returns(df_sample)

    # 诊断2：日内因子自相关
    feat_list = [c for c in NORM_FEATURE_COLS if c in df_sample.columns]
    df_diag2 = diag_factor_intraday_autocorr(df_sample, feat_list)

    # 诊断3：收益序列重叠自相关
    df_diag3 = diag_return_overlap_autocorr(df_sample)

    # 诊断1：需要已计算的IC序列
    ic_ts_path = OUTPUT_DIR / f"ic_daily_ts_{YEAR}.pkl"
    if ic_ts_path.exists():
        logger.info(f"加载已保存的时序IC: {ic_ts_path}")
        ic_df = pd.read_pickle(ic_ts_path)
        # 只用样本股票的IC
        ic_sample = ic_df[ic_df["SecuCode"].isin(df_sample["SecuCode"].unique())]
        diag_feats = ["vwap_bias_norm", "range_pos_norm", "dif_norm", "ret_1bar_norm", "cord20_norm"]
        df_diag1 = diag_ic_cross_day_autocorr(ic_sample, diag_feats)
    else:
        logger.warning(f"未找到时序IC文件 {ic_ts_path}，跳过诊断1（请先运行 ic_analysis.py）")
        df_diag1 = pd.DataFrame()

    # ---- 打印结果 ----
    print_diagnosis(df_diag1, df_diag2, df_diag3)

    # ---- 保存诊断报告 ----
    out_path = OUTPUT_DIR / f"autocorr_diagnosis_{YEAR}.csv"
    all_diag = pd.concat(
        [df for df in [df_diag1, df_diag2, df_diag3] if not df.empty],
        ignore_index=True,
    )
    all_diag.to_csv(out_path, index=False)
    logger.success(f"诊断报告已保存: {out_path}")


if __name__ == "__main__":
    main()
