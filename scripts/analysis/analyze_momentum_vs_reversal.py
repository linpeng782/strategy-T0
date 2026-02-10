"""
动量 vs 反转分析脚本

核心问题：模型到底学会了"强势动量"还是"超跌反转"？
方法：按 X1_zscore 正负将信号拆分为两组，分别分析收益表现

分组逻辑：
  - 动量组 (X1 > 0)：股价高于VWAP，强势股继续做多
  - 反转组 (X1 < 0)：股价低于VWAP，超跌股反弹做多

进一步细分：
  - 按 X1_zscore_rank 分位数（高/低）
  - 按 rel_vol 分位数（放量/缩量）
  - 按 X2_zscore 正负（成交分布偏向）

输入：output/lgbm_test_predictions.pkl
输出：终端报告 + 分析图表

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"

COST_BPS = 15
COST_RATE = COST_BPS / 10000
PROB_THRESHOLD = 0.45  # 做多模型信号门槛
START_TIME = "09:50"
END_TIME = "14:50"
COOLDOWN_BARS = 24  # 冷却期 = 持有期（24bars=120min）


def time_to_bar_index(t: str) -> int:
    """将 entry_time 转为 bar 序号"""
    h, m = map(int, t.split(":"))
    total_min = h * 60 + m
    if total_min <= 11 * 60 + 30:
        return (total_min - 9 * 60 - 35) // 5
    else:
        return 24 + (total_min - 13 * 60) // 5


def apply_cooldown(signals: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """冷却期去重：同一股票在 cooldown_bars 个bar内不再重复发信号"""
    signals = signals.sort_values(["SecuCode", "date", "entry_time"]).copy()
    signals["bar_idx"] = signals["entry_time"].apply(time_to_bar_index)

    def filter_group(g):
        g = g.sort_values("bar_idx")
        keep = [True]
        last_bar = g.iloc[0]["bar_idx"]
        for i in range(1, len(g)):
            if g.iloc[i]["bar_idx"] - last_bar >= cooldown_bars:
                keep.append(True)
                last_bar = g.iloc[i]["bar_idx"]
            else:
                keep.append(False)
        return pd.Series(keep, index=g.index)

    keep_mask = signals.groupby(["SecuCode", "date"], group_keys=False).apply(
        filter_group, include_groups=False
    )
    return signals[keep_mask].copy()


def load_signals() -> pd.DataFrame:
    """加载做多模型的预测结果，筛选出信号"""
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(df):,} 行, {df['date'].nunique()} 个交易日")

    # 全量数据保留用于对比
    df["net_ret"] = df["raw_ret"] - COST_RATE
    df["net_ret_bps"] = df["net_ret"] * 10000

    return df


def calc_group_stats(group: pd.DataFrame, label: str) -> dict:
    """计算一组交易的统计指标"""
    if len(group) == 0:
        return {"label": label, "n_trades": 0}

    net_ret = group["net_ret"]
    daily = group.groupby("date")["net_ret"].mean()
    n_days = len(daily)

    sharpe = 0.0
    if n_days > 1 and daily.std() > 0:
        sharpe = daily.mean() / daily.std() * np.sqrt(252)

    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0

    gross_profit = net_ret[net_ret > 0].sum()
    gross_loss = abs(net_ret[net_ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "label": label,
        "n_trades": len(group),
        "n_days": n_days,
        "win_rate": (net_ret > 0).mean(),
        "day_win_rate": (daily > 0).mean(),
        "avg_net_bps": group["net_ret_bps"].mean(),
        "daily_bps": daily.mean() * 10000,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "pf": pf,
        "daily_ret": daily,
    }


def print_group_table(results: list, title: str):
    """打印分组对比表"""
    print(f"\n{'=' * 130}")
    print(title)
    print(f"{'=' * 130}")

    header = (
        f"{'Group':>24s}  {'Trades':>8s}  {'Days':>6s}  {'WR':>8s}  {'DayWR':>8s}  "
        f"{'NetBps':>10s}  {'DailyBps':>10s}  {'Sharpe':>8s}  {'MaxDD':>10s}  {'PF':>8s}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        if r["n_trades"] == 0:
            print(f"{r['label']:>24s}  {'N/A':>8s}")
            continue
        print(
            f"{r['label']:>24s}  {r['n_trades']:>8,}  {r['n_days']:>6}  "
            f"{r['win_rate']:>7.1%}  {r['day_win_rate']:>7.1%}  "
            f"{r['avg_net_bps']:>10.1f}  {r['daily_bps']:>10.1f}  "
            f"{r['sharpe']:>8.2f}  {r['max_dd']:>9.2%}  "
            f"{r['pf']:>8.2f}"
        )
    print("-" * 130)


# ==================== 分析1: X1 正负拆分（动量 vs 反转） ====================
def analyze_x1_split(signals: pd.DataFrame):
    """按 X1_zscore 正负拆分信号"""
    momentum = signals[signals["X1_zscore"] > 0]
    reversal = signals[signals["X1_zscore"] <= 0]

    results = [
        calc_group_stats(signals, "ALL signals"),
        calc_group_stats(momentum, "Momentum (X1>0)"),
        calc_group_stats(reversal, "Reversal (X1<=0)"),
    ]
    print_group_table(results, "Analysis 1: X1 Split -- Momentum vs Reversal")

    # 进一步按 X1 分位细分
    print(f"\n--- X1_zscore Quantile Breakdown ---")
    bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
    labels = ["<-2", "-2~-1", "-1~-0.5", "-0.5~0", "0~0.5", "0.5~1", "1~2", ">2"]
    signals["x1_bin"] = pd.cut(signals["X1_zscore"], bins=bins, labels=labels)

    detail_results = []
    for label in labels:
        subset = signals[signals["x1_bin"] == label]
        detail_results.append(calc_group_stats(subset, f"X1: {label}"))

    print_group_table(detail_results, "Analysis 1b: X1_zscore Quantile Breakdown")
    return results


# ==================== 分析2: X2 正负拆分 ====================
def analyze_x2_split(signals: pd.DataFrame):
    """按 X2_zscore 正负拆分"""
    x2_high = signals[signals["X2_zscore"] > 0]
    x2_low = signals[signals["X2_zscore"] <= 0]

    results = [
        calc_group_stats(signals, "ALL signals"),
        calc_group_stats(x2_high, "X2>0 (high side)"),
        calc_group_stats(x2_low, "X2<=0 (low side)"),
    ]
    print_group_table(results, "Analysis 2: X2 Split")
    return results


# ==================== 分析3: 组合逻辑拆分 ====================
def analyze_combined_logic(signals: pd.DataFrame):
    """按 notebook 提出的两套逻辑拆分"""
    # 逻辑A：强势动量 — X1_zscore_rank 高 + rel_vol 高
    mask_a = (signals["X1_zscore_rank"] > 0.7) & (signals["rel_vol"] > 1.2)

    # 逻辑B：超跌反转 — X1_zscore 极低 + X2_zscore 极低 + rel_vol 高
    mask_b = (
        (signals["X1_zscore"] < -1)
        & (signals["X2_zscore"] < -1)
        & (signals["rel_vol"] > 1.2)
    )

    # 重叠信号优先归逻辑B（超跌反转更符合其特征本质），从A中剔除
    overlap_count = (mask_a & mask_b).sum()
    logger.info(f"逻辑A与逻辑B重叠: {overlap_count} 笔（优先归逻辑B）")
    mask_a_exclusive = mask_a & ~mask_b  # A中剔除与B重叠的部分

    logic_a = signals[mask_a_exclusive]
    logic_b = signals[mask_b]
    logic_other = signals[~mask_a_exclusive & ~mask_b]

    # 校验三组互斥且总和=ALL
    assert len(logic_a) + len(logic_b) + len(logic_other) == len(
        signals
    ), f"分组不互斥: {len(logic_a)}+{len(logic_b)}+{len(logic_other)} != {len(signals)}"

    results = [
        calc_group_stats(signals, "ALL signals"),
        calc_group_stats(logic_a, "Logic-A: Momentum"),
        calc_group_stats(logic_b, "Logic-B: Reversal"),
        calc_group_stats(logic_other, "Others"),
    ]
    print_group_table(
        results, "Analysis 3: Combined Logic Split (Momentum vs Reversal vs Others)"
    )

    # 特征均值对比
    print(f"\n--- Feature Mean Comparison ---")
    feat_cols = [
        "X1_zscore",
        "X2_zscore",
        "X1_zscore_rank",
        "X2_zscore_rank",
        "rel_vol",
        "X1_delta_15m",
    ]
    header = f"{'Group':>20s}" + "".join(f"{c:>18s}" for c in feat_cols)
    print(header)
    print("-" * (20 + 18 * len(feat_cols)))
    for name, subset in [
        ("ALL signals", signals),
        ("Logic-A: Momentum", logic_a),
        ("Logic-B: Reversal", logic_b),
        ("Others", logic_other),
    ]:
        vals = "".join(f"{subset[c].mean():>18.4f}" for c in feat_cols)
        print(f"{name:>20s}{vals}")

    return results


# ==================== 分析4: rel_vol 拆分（量的作用） ====================
def analyze_volume_split(signals: pd.DataFrame):
    """按 rel_vol 拆分，验证成交量对信号质量的影响"""
    bins = [0, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf]
    labels = ["<0.8", "0.8~1.0", "1.0~1.2", "1.2~1.5", "1.5~2.0", ">2.0"]
    signals["vol_bin"] = pd.cut(signals["rel_vol"], bins=bins, labels=labels)

    results = [calc_group_stats(signals, "ALL signals")]
    for label in labels:
        subset = signals[signals["vol_bin"] == label]
        results.append(calc_group_stats(subset, f"rel_vol: {label}"))

    print_group_table(results, "Analysis 4: Volume Quantile Split")
    return results


# ==================== 绘图 ====================
def plot_equity_curves(analysis_results: dict):
    """绘制 Combined Logic 资金曲线"""
    results = analysis_results.get("combined", [])
    valid = [r for r in results if r["n_trades"] > 0 and "daily_ret" in r]
    if not valid:
        logger.warning("无有效数据，跳过绘图")
        return

    # 找到覆盖天数最多的系列，用它的日期做 X 轴
    ref = max(valid, key=lambda r: len(r["daily_ret"]))
    ref_dates = ref["daily_ret"].index
    date_strs = [str(d) for d in ref_dates]
    n_points = len(ref_dates)

    # 颜色映射
    colors = {
        "ALL signals": "#1f77b4",
        "Logic-A: Momentum": "#ff7f0e",
        "Logic-B: Reversal": "#2ca02c",
        "Others": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(18, 8))

    for r in valid:
        daily = r["daily_ret"]
        # 对齐到参考日期（缺失日填0）
        aligned = daily.reindex(ref_dates, fill_value=0.0)
        cum_bps = aligned.cumsum() * 10000
        lbl = r["label"]
        color = colors.get(lbl, None)
        ax.plot(
            range(n_points),
            cum_bps.values,
            label=f"{lbl} (Sharpe={r['sharpe']:.2f}, N={r['n_trades']})",
            color=color,
            linewidth=1.8,
            alpha=0.85,
        )

    # 日期刻度：每月第一个交易日
    tick_pos, tick_labels = [], []
    seen_months = set()
    for i, d in enumerate(date_strs):
        month_key = d[:7]  # "2025-01"
        if month_key not in seen_months:
            seen_months.add(month_key)
            tick_pos.append(i)
            tick_labels.append(d[:10])  # "2025-01-06"

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=9, ha="right")

    ax.set_title(
        f"Combined Logic: Momentum vs Reversal vs Others\n"
        f"(prob > {PROB_THRESHOLD}, cost={COST_BPS}bps, hold=120min)",
        fontsize=13,
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Cumulative Net Return (bps)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "analyze_momentum_vs_reversal.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("动量 vs 反转分析")
    logger.info("=" * 60)

    # 加载数据
    df = load_signals()

    # 筛选信号（与做多模型回测一致：时间+概率+冷却期去重）
    mask = (
        (df["entry_time"] >= START_TIME)
        & (df["entry_time"] <= END_TIME)
        & (df["pred_prob"] > PROB_THRESHOLD)
    )
    signals = df[mask].copy()
    logger.info(
        f"初筛信号: {len(signals):,} 笔 (prob>{PROB_THRESHOLD}, {START_TIME}~{END_TIME})"
    )

    # 冷却期去重
    signals = apply_cooldown(signals, COOLDOWN_BARS)
    logger.info(
        f"去重后信号: {len(signals):,} 笔 (冷却期={COOLDOWN_BARS}bars={COOLDOWN_BARS*5}min)"
    )
    logger.info(
        f"信号特征均值: X1={signals['X1_zscore'].mean():.4f}, "
        f"X2={signals['X2_zscore'].mean():.4f}, "
        f"rel_vol={signals['rel_vol'].mean():.4f}"
    )

    # 分析
    analysis_results = {}
    analysis_results["x1_split"] = analyze_x1_split(signals)
    analysis_results["x2_split"] = analyze_x2_split(signals)
    analysis_results["combined"] = analyze_combined_logic(signals)
    analysis_results["volume"] = analyze_volume_split(signals)

    # 绘图
    plot_equity_curves(analysis_results)

    logger.success("动量 vs 反转分析完成!")


if __name__ == "__main__":
    main()
