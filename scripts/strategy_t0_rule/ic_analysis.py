"""
IC分析脚本
职责：基于模块化特征工程输出的特征数据，计算各特征对多个未来收益率周期的IC统计。

IC计算逻辑：
  - 未来收益率周期：15min / 30min / 45min / 60min（对3/6/9/12根分钟bar）
  - 买入价：信号bar的下一根bar的bar_vwap
  - 卖出价：往后推N根bar的bar_vwap
  - 每日每只股票：每日独立随机非重叠降采样，计算特征与未来收益的 Rank IC（Spearman相关）
  - 汇总统计：Mean IC / ICIR / p-value / IC>0比例

命令行使用：
  python ic_analysis.py                    # 分析全部因子
  python ic_analysis.py --group macd       # 只分析MACD类因子
  python ic_analysis.py --group volume     # 只分析量价类因子
  python ic_analysis.py --group alpha158   # 只分析alpha158类因子
  python ic_analysis.py --rebuild macd     # 重新构造MACD特征缓存并分析

输出：
  - 控制台打印汇总表
  - 保存CSV到output目录
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试年份
TEST_YEAR = 2024

# 未来收益率周期（单位：根5分钟bar）
# 15min=3根, 30min=6根, 45min=9根, 60min=12根
HOLD_PERIODS = {
    "15min": 3,
    "30min": 6,
    "45min": 9,
    "60min": 12,
}

# 交易成本（bps）—— 仅用于回测层，fwd_ret本身不扣除成本
# COST_BPS = 15  # 已移至回测脚本

# 特征分组（标准化后）
NORM_FEATURE_GROUPS = {
    # 经典技术指标：MACD双参数、KDJ、布林带、RSV
    "classic": [
        "dif_norm",
        "dea_norm",
        "macd_norm",
        "macd_area_norm",
        "dif_slope3_norm",
        "dif_std_norm",
        "dea_std_norm",
        "macd_std_norm",
        "kdj_k_norm",
        "kdj_d_norm",
        "kdj_j_norm",
        "kdj_kd_norm",
        "boll_pct_b_norm",
        "boll_bw_norm",
        "rsv10_norm",
        "rsv20_norm",
    ],
    "price_volume": [
        "vol_ratio_norm",
        "buy_pressure_norm",
        "amplitude_norm",
        "ret_1bar_norm",
        "ret_3bar_norm",
        "ret_5bar_norm",
        "range_pos_norm",
    ],
    "alpha158": [
        "vwap_bias_norm",
        "x2_norm",
        "kmid_norm",
        "kup2_norm",
        "klow2_norm",
        "cord20_norm",
        "wvma10_norm",
        "vsumd10_norm",
    ],
}

# 全部特征列（合并所有组）
NORM_FEATURE_COLS = (
    NORM_FEATURE_GROUPS["classic"]
    + NORM_FEATURE_GROUPS["price_volume"]
    + NORM_FEATURE_GROUPS["alpha158"]
)


# ==================== 未来收益率计算 ====================
def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    向量化计算多个周期的未来原始收益率（基于bar_vwap，不扣成本）。

    bar_vwap = amount / volume，即每根5分钟bar内的成交量加权均价。
    相比open价格，vwap更难被滑点影响，更接近实盘平均成交价。

    买入价 = 下一根bar(signal+1)的vwap
    卖出价 = signal+1+N 根bar的vwap
    按日内分组，避免跨日计算。

    注意：fwd_ret 为纯价格收益率，不含成本扣除。
    成本需在回测层（t0_backtest.py）独立处理。
    """
    logger.info("计算多周期未来收益率（基于bar_vwap）...")

    # 计算每根bar的vwap = amount / volume（若缓存中已有则跳过）
    if "bar_vwap" not in df.columns:
        df["bar_vwap"] = (df["amount"] / df["volume"]).replace(
            [np.inf, -np.inf], np.nan
        )

    # 按股票+日期分组，避免跨日
    grp = df.groupby(["SecuCode", "date"])

    # 买入价：下一根bar的vwap
    df["buy_open"] = grp["bar_vwap"].shift(-1)

    for label, n_bars in HOLD_PERIODS.items():
        sell_shift = -(1 + n_bars)
        sell_vwap = grp["bar_vwap"].shift(sell_shift)
        raw_ret = sell_vwap / df["buy_open"] - 1
        df[f"fwd_ret_{label}"] = raw_ret  # 纯价格收益率，不扣成本

    logger.success(f"未来收益率列: {[f'fwd_ret_{l}' for l in HOLD_PERIODS]}")
    return df


# ==================== IC计算主函数（全向量化） ====================
def compute_ic(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    全向量化计算时序Rank IC（Spearman）。

    时序IC定义：
      分组键 = (SecuCode, date)，即一只股票在某一交易日内的所有bar（48根）。
      在这48根bar内，计算因子序列与未来收益序列的Spearman相关。
      含义：对该股票该天，因子值高的时刻能否预测该股票自身的未来涨跌。
      这是T0策略的核心问题，与截面IC（跨股票比较）完全不同。

    计算方法：
      1. 组内rank（pct=True）→ rank后Pearson = Spearman
      2. Pearson = sum(dx*dy) / sqrt(sum(dx²)*sum(dy²))
         通过groupby.transform(sum)全向量化，无需apply循环。

    返回 ic_df：
        columns = [SecuCode, date, feature, period, ic]
        每行 = 一只股票在某一交易日的时序IC值
    """
    logger.info(
        f"开始计算IC（全向量化），特征数={len(feature_cols)}，周期数={len(HOLD_PERIODS)}..."
    )

    ret_cols = [f"fwd_ret_{label}" for label in HOLD_PERIODS]
    grp_keys = ["SecuCode", "date"]

    # ---- 组内rank（pct=True，等价于归一化到[0,1]的均匀分布） ----
    work = df[grp_keys].copy()
    for col in feature_cols + ret_cols:
        work[col + "_r"] = df.groupby(grp_keys)[col].rank(pct=True, na_option="keep")

    # 过滤组内样本数<5的行
    grp_size = df.groupby(grp_keys)["close"].transform("count")
    work = work[grp_size >= 5].copy()

    # ---- 向量化Pearson相关 = sum(dx*dy) / sqrt(sum(dx²)*sum(dy²)) ----
    # 标准Pearson公式，与n-1无关，无需均值/std修正
    records = []
    grp = work.groupby(grp_keys)

    for feat_col in feature_cols:
        xr = feat_col + "_r"
        mean_x = grp[xr].transform("mean")
        dx = work[xr] - mean_x
        work["_dx2"] = dx * dx
        sum_dx2 = grp["_dx2"].transform("sum")

        for period_label, ret_col in zip(HOLD_PERIODS.keys(), ret_cols):
            yr = ret_col + "_r"
            mean_y = grp[yr].transform("mean")
            dy = work[yr] - mean_y

            work["_dxdy"] = dx * dy
            work["_dy2"] = dy * dy
            sum_dxdy = grp["_dxdy"].transform("sum")
            sum_dy2 = grp["_dy2"].transform("sum")

            denom = np.sqrt(sum_dx2 * sum_dy2)
            ic_vals = np.where(denom > 0, sum_dxdy / denom, np.nan)

            # 每组只取第一行（组内IC值相同）
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
    ic_df = pd.concat(records, ignore_index=True)[
        ["SecuCode", "date", "feature", "period", "ic"]
    ]
    logger.success(f"IC计算完成，共 {len(ic_df):,} 条记录")
    return ic_df


def compute_ic_daily_random(
    df: pd.DataFrame,
    feature_cols: list,
    hold_periods: dict = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    每日独立随机非重叠IC（推荐方案，替代多次迭代平均）。

    核心思路：
      对每个持有期N，每个(SecuCode, date)组独立随机选取起始偏移（0~N-1），
      然后每隔N根bar取1个样本，确保收益序列零重叠。

      相比方案A（5次迭代平均）的优势：
        - 计算效率高5倍（只跑1次）
        - 统计等价：全市场24万个(股票,日期)组，每组独立随机，
          长期来看所有时段均匀覆盖，期望值与5次平均完全相同
        - 更贴近实盘：实盘信号触发时间本身就是随机的

    参数：
      seed: 随机种子，保证可复现

    返回 ic_df：columns = [SecuCode, date, feature, period, ic]
    """
    if hold_periods is None:
        hold_periods = HOLD_PERIODS

    grp_keys = ["SecuCode", "date"]
    records = []
    rng = np.random.default_rng(seed)

    for period_label, n_bars in hold_periods.items():
        ret_col = f"fwd_ret_{period_label}"
        if ret_col not in df.columns:
            continue

        # 所有持有期统一做非重叠降采样，保证结果可比
        # 每个(SecuCode, date)组独立随机offset，批量生成
        groups = df.groupby(grp_keys)
        group_keys = list(groups.groups.keys())
        offsets = rng.integers(0, n_bars, size=len(group_keys))
        offset_map = dict(zip(group_keys, offsets.tolist()))

        df_temp = df.copy()
        df_temp["_bar_idx"] = groups.cumcount()
        df_temp["_offset"] = df_temp.set_index(grp_keys).index.map(offset_map).values
        mask = (df_temp["_bar_idx"] - df_temp["_offset"]) % n_bars == 0
        df_s = df_temp[mask].copy()
        df_s.drop(columns=["_bar_idx", "_offset"], inplace=True)

        grp_size = df_s.groupby(grp_keys)["close"].transform("count")
        n_per_day = grp_size.median()
        logger.info(
            f"  持有期 {period_label}：非重叠降采样，间隔={n_bars}根bar，每天约{n_per_day:.0f}个样本"
        )
        df_s = df_s[grp_size >= 2].copy()

        if len(df_s) == 0:
            continue

        # 组内rank
        work = df_s[grp_keys].copy()
        cols_needed = [c for c in feature_cols if c in df_s.columns] + [ret_col]
        for col in cols_needed:
            work[col + "_r"] = df_s.groupby(grp_keys)[col].rank(
                pct=True, na_option="keep"
            )

        grp = work.groupby(grp_keys)
        yr = ret_col + "_r"
        mean_y = grp[yr].transform("mean")
        dy = work[yr] - mean_y
        work["_dy2"] = dy * dy
        sum_dy2 = grp["_dy2"].transform("sum")

        for feat_col in feature_cols:
            if feat_col not in df_s.columns:
                continue
            xr = feat_col + "_r"
            mean_x = grp[xr].transform("mean")
            dx = work[xr] - mean_x
            work["_dx2"] = dx * dx
            work["_dxdy"] = dx * dy
            sum_dx2 = grp["_dx2"].transform("sum")
            sum_dxdy = grp["_dxdy"].transform("sum")

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

        logger.info(f"  持有期 {period_label} 完成，采样后 {len(df_s):,} 行")

    if not records:
        return pd.DataFrame(columns=["SecuCode", "date", "feature", "period", "ic"])

    ic_df = pd.concat(records, ignore_index=True)[
        ["SecuCode", "date", "feature", "period", "ic"]
    ]
    logger.success(f"每日独立随机IC计算完成，共 {len(ic_df):,} 条记录")
    return ic_df


# ==================== IC汇总统计 ====================
def _agg_ic_stats(ic_vals: pd.Series) -> pd.Series:
    """对一组IC序列计算汇总统计，供多处复用"""
    ic_vals = ic_vals.dropna()
    n = len(ic_vals)
    if n < 2:
        return pd.Series(
            {
                "MeanIC": np.nan,
                "StdIC": np.nan,
                "ICIR": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "IC_pos_ratio": np.nan,
                "N_samples": n,
            }
        )
    mean_ic = ic_vals.mean()
    std_ic = ic_vals.std()
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat, p_value = stats.ttest_1samp(ic_vals, popmean=0)
    ic_pos = (ic_vals > 0).mean()
    return pd.Series(
        {
            "MeanIC": mean_ic,
            "StdIC": std_ic,
            "ICIR": icir,
            "t_stat": t_stat,
            "p_value": p_value,
            "IC_pos_ratio": ic_pos,
            "N_samples": n,
        }
    )


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    对时序IC序列计算汇总统计。

    时序IC汇总逻辑：
      - ic_df中每行 = 一只股票在某一交易日的IC值
      - 直接将所有(SecuCode, date)的IC值展平，计算全局ICIR
      - 这样ICIR的分母是所有股票所有天的IC横截面方差，反映真实预测稳定性
      - N_samples = 有效的(SecuCode, date)对数量

    返回 summary_df：index=(feature, period)
    """
    logger.info("汇总时序IC统计...")

    # 直接展平所有(SecuCode, date)的IC，不再按日期平均
    summary = (
        ic_df.groupby(["feature", "period"])["ic"]
        .apply(_agg_ic_stats)
        .unstack(level=-1)
        .reset_index()
    )

    summary = _format_summary(summary)
    return summary


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """格式化IC汇总表"""
    summary["MeanIC"] = summary["MeanIC"].round(4)
    summary["StdIC"] = summary["StdIC"].round(4)
    summary["ICIR"] = summary["ICIR"].round(3)
    summary["t_stat"] = summary["t_stat"].round(3)
    summary["p_value"] = summary["p_value"].round(4)
    summary["IC_pos_ratio"] = (summary["IC_pos_ratio"] * 100).round(1).astype(str) + "%"
    summary["N_samples"] = summary["N_samples"].astype(int)
    return summary


# ==================== 打印输出 ====================
def print_summary(summary: pd.DataFrame):
    """按特征分组打印时序IC汇总表（N_samples=股票数×交易日数）"""
    logger.info(f"\n{'='*80}")
    logger.info("IC分析汇总（时序IC，每股票每天一个IC值）")
    logger.info(f"{'='*80}")

    period_order = list(HOLD_PERIODS.keys())

    for feat in summary["feature"].unique():
        sub = summary[summary["feature"] == feat].copy()
        sub["period"] = pd.Categorical(
            sub["period"], categories=period_order, ordered=True
        )
        sub = sub.sort_values("period")
        sub = sub.drop(columns=["feature"])

        logger.info(f"\n特征: {feat}")
        print(sub.to_string(index=False))


def print_period_comparison(summary: pd.DataFrame):
    """
    按持有期对比输出IC汇总表。
    输出两张透视表：
      1. MeanIC对比：行=特征，列=持有期（15/25/40/60min）
      2. ICIR对比：行=特征，列=持有期
    帮助判断最优持有时间。
    """
    logger.info(f"\n{'='*80}")
    logger.info("按持有期对比 - MeanIC（行=特征，列=持有期）")
    logger.info(f"{'='*80}")

    period_order = list(HOLD_PERIODS.keys())

    # 透视：行=特征，列=持有期，值=MeanIC
    pivot_ic = summary.pivot(index="feature", columns="period", values="MeanIC")
    pivot_ic = pivot_ic.reindex(
        columns=[p for p in period_order if p in pivot_ic.columns]
    )
    # 按15min的绝对值排序
    if "15min" in pivot_ic.columns:
        pivot_ic = pivot_ic.reindex(
            pivot_ic["15min"].abs().sort_values(ascending=False).index
        )
    print(pivot_ic.round(4).to_string())

    logger.info(f"\n{'='*80}")
    logger.info("按持有期对比 - ICIR（行=特征，列=持有期）")
    logger.info(f"{'='*80}")

    pivot_icir = summary.pivot(index="feature", columns="period", values="ICIR")
    pivot_icir = pivot_icir.reindex(
        columns=[p for p in period_order if p in pivot_icir.columns]
    )
    if "15min" in pivot_icir.columns:
        pivot_icir = pivot_icir.reindex(
            pivot_icir["15min"].abs().sort_values(ascending=False).index
        )
    print(pivot_icir.round(3).to_string())

    logger.info(f"\n{'='*80}")
    logger.info("按持有期对比 - IC>0比例（行=特征，列=持有期）")
    logger.info(f"{'='*80}")

    pivot_pos = summary.pivot(index="feature", columns="period", values="IC_pos_ratio")
    pivot_pos = pivot_pos.reindex(
        columns=[p for p in period_order if p in pivot_pos.columns]
    )
    if "15min" in pivot_pos.columns:
        pivot_pos = pivot_pos.reindex(pivot_ic.index)  # 保持与MeanIC相同排序
    print(pivot_pos.to_string())


# ==================== 主函数 ====================
def print_comparison(summary_ts: pd.DataFrame, summary_a: pd.DataFrame):
    """
    打印时序IC（全48bar）与方案A（非重叠5次平均）的ICIR对比表。
    帮助直观看出各因子的ICIR虚高程度及真实预测力排名。
    """
    logger.info(f"\n{'='*100}")
    logger.info("ICIR对比：时序IC（全48bar） vs 方案A非重叠IC（5次随机offset平均）")
    logger.info("A/TS比值越接近1说明原始时序IC越可靠；比值越小说明虚高越严重")
    logger.info(f"{'='*100}")

    period_order = list(HOLD_PERIODS.keys())

    def get_icir_pivot(summary):
        p = summary.pivot(index="feature", columns="period", values="ICIR")
        return p.reindex(columns=[c for c in period_order if c in p.columns])

    pts = get_icir_pivot(summary_ts)
    pa = get_icir_pivot(summary_a)

    # 按方案A的15min绝对值排序
    if "15min" in pa.columns:
        order = pa["15min"].abs().sort_values(ascending=False).index
        pts = pts.reindex(order)
        pa = pa.reindex(order)

    header = "%-25s | %7s %7s %5s | %7s %7s %5s | %7s %7s %5s | %7s %7s %5s" % (
        "feature",
        "TS_15",
        "A_15",
        "A/TS",
        "TS_25",
        "A_25",
        "A/TS",
        "TS_40",
        "A_40",
        "A/TS",
        "TS_60",
        "A_60",
        "A/TS",
    )
    logger.info(f"\n{header}")
    logger.info("-" * 100)

    for feat in pa.index:
        row_parts = [f"{feat:25s}"]
        for period in period_order:
            ts_v = (
                pts.loc[feat, period]
                if feat in pts.index and period in pts.columns
                else float("nan")
            )
            a_v = (
                pa.loc[feat, period]
                if feat in pa.index and period in pa.columns
                else float("nan")
            )
            ratio = (
                a_v / ts_v
                if (ts_v != 0 and not np.isnan(ts_v) and not np.isnan(a_v))
                else float("nan")
            )
            row_parts.append(f"| {ts_v:7.3f} {a_v:7.3f} {ratio:5.2f}")
        logger.info(" ".join(row_parts))


def main():
    """
    IC分析主函数。
    职责：加载 build_all.py 生成的因子缓存，计算IC并汇总。
    因子构造请使用 build_all.py --year <year> --index <index>。
    """
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from features.build_all import build_all_features

    parser = argparse.ArgumentParser(description="IC分析脚本")
    parser.add_argument("--year", type=int, default=TEST_YEAR, help="分析年份")
    parser.add_argument(
        "--index",
        type=str,
        default="csi1000",
        help="股票池：csi1000（默认）或 all（全市场）",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "classic", "price_volume", "alpha158"],
        help="分析的因子组（默认all）",
    )
    parser.add_argument(
        "--rebuild",
        type=str,
        default=None,
        choices=["classic", "price_volume", "alpha158", "all"],
        help="重新构造指定组特征缓存（不指定则从缓存加载）",
    )
    args = parser.parse_args()

    YEAR = args.year
    GROUP = args.group
    REBUILD = args.rebuild
    INDEX = args.index

    logger.info(
        f"开始 IC 分析，year={YEAR}，index={INDEX}，group={GROUP}，rebuild={REBUILD}"
    )

    # ---- 加载或构造特征（由 build_all.py 负责，ic_analysis 只消费缓存）----
    df_feat = build_all_features(year=YEAR, index=INDEX, rebuild=REBUILD)

    # ---- 计算未来收益率 ----
    df_feat = compute_forward_returns(df_feat)

    # 过滤：只要求买入价有效
    valid_mask = df_feat["buy_open"].notna()
    df_valid = df_feat[valid_mask].copy()
    fwd_ret_cols = [f"fwd_ret_{l}" for l in HOLD_PERIODS]
    for col in fwd_ret_cols:
        n_valid = df_valid[col].notna().sum()
        logger.info(f"  {col}: 有效bar数={n_valid:,}")
    logger.info(f"总有效bar数（buy_open有效）: {len(df_valid):,}")

    # ---- 根据--group选择分析的特征列 ----
    if GROUP == "all":
        feat_cols_to_use = NORM_FEATURE_COLS
    else:
        feat_cols_to_use = NORM_FEATURE_GROUPS[GROUP]
    all_feat_cols = [c for c in feat_cols_to_use if c in df_valid.columns]
    logger.info(f"分析因子组={GROUP}，特征数={len(all_feat_cols)}")

    # ---- 计算每日独立随机非重叠IC ----
    logger.info("计算每日独立随机非重叠IC...")
    ic_dr = compute_ic_daily_random(df_valid, all_feat_cols, seed=42)
    summary_dr = summarize_ic(ic_dr)

    # ---- 打印汇总 ----
    logger.info(f"\n{'='*80}")
    logger.info(f"【IC分析结果】因子组={GROUP}，年份={YEAR}，股票池={INDEX}")
    logger.info(f"{'='*80}")
    print_summary(summary_dr)
    print_period_comparison(summary_dr)

    # ---- 保存结果 ----
    suffix = f"{INDEX}_{GROUP}_{YEAR}"
    summary_dr.to_csv(OUTPUT_DIR / f"ic_summary_{suffix}.csv", index=False)
    ic_dr.to_pickle(OUTPUT_DIR / f"ic_daily_{suffix}.pkl")
    logger.success(f"IC结果已保存: ic_summary_{suffix}.csv")


if __name__ == "__main__":
    main()
