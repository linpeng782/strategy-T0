"""
LGBM 信号分析脚本

读取 lgbm_classifier.py 保存的测试集预测结果，分析信号分布：
  1. 按 entry_time 分布：各时间点的信号数量、胜率、净收益
  2. 按日期分布：每日信号数量变化
  3. 按概率区间分布：不同概率区间的信号质量
  4. 信号的特征分布：被选中信号 vs 全量样本的特征对比

输入：output/lgbm_test_predictions.pkl（由 lgbm_classifier.py 生成）
输出：终端报告 + 图表

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

# 交易成本
COST_BPS = 15
COST_RATE = COST_BPS / 10000

# 信号概率阈值（用于筛选 trades）
SIGNAL_THRESHOLD = 0.45
START_TIME = "09:50"
END_TIME = "14:50"
COOLDOWN_BARS = 24  # 冷却期 = 持有期（24bars=120min）

# 特征列表
FEATURES = [
    "X1_zscore",
    "X2_zscore",
    "X1_zscore_rank",
    "X2_zscore_rank",
    "rel_vol",
    "X1_delta_15m",
    "time_index",
]


# ==================== 冷却期去重 ====================
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


# ==================== 数据加载 ====================
def load_predictions() -> pd.DataFrame:
    """加载测试集预测结果"""
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"预测结果文件不存在: {PRED_PATH}\n请先运行 lgbm_classifier.py 生成预测结果"
        )
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(df):,} 行, {df['date'].nunique()} 个交易日")
    return df


# ==================== 分析函数 ====================
def analyze_entry_time_distribution(df: pd.DataFrame, trades: pd.DataFrame):
    """按 entry_time 分析信号分布"""
    print("\n" + "=" * 100)
    print(f"1. Signal Distribution by Entry Time (threshold={SIGNAL_THRESHOLD})")
    print("=" * 100)

    # 全量样本按 entry_time 的分布
    all_by_time = df.groupby("entry_time").agg(
        total=("pred_prob", "size"),
        avg_prob=("pred_prob", "mean"),
    )

    # 信号按 entry_time 的分布
    if len(trades) == 0:
        print("  No trades found at this threshold.")
        return pd.DataFrame()

    trades_net = trades.copy()
    trades_net["net_ret"] = trades_net["raw_ret"] - COST_RATE

    stats = trades_net.groupby("entry_time").agg(
        n_signals=("pred_prob", "size"),
        avg_prob=("pred_prob", "mean"),
        win_rate=("net_ret", lambda x: (x > 0).mean()),
        avg_net_bps=("net_ret", lambda x: x.mean() * 10000),
        avg_excess_bps=("excess_ret", lambda x: x.mean() * 10000),
        precision=("is_profitable", "mean"),
    )

    # 合并全量信息
    stats = stats.join(all_by_time[["total"]], how="left")
    stats["signal_rate"] = stats["n_signals"] / stats["total"]

    header = (
        f"{'EntryTime':>10s}  {'Total':>10s}  {'Signals':>8s}  {'SigRate':>8s}  "
        f"{'AvgProb':>8s}  {'WinRate':>8s}  {'Prec':>8s}  "
        f"{'NetBps':>10s}  {'ExcessBps':>10s}"
    )
    print(f"\n{header}")
    print("-" * 96)

    for t, row in stats.iterrows():
        print(
            f"{t:>10s}  {row['total']:>10.0f}  {row['n_signals']:>8.0f}  "
            f"{row['signal_rate']:>7.2%}  {row['avg_prob']:>8.4f}  "
            f"{row['win_rate']:>7.1%}  {row['precision']:>7.1%}  "
            f"{row['avg_net_bps']:>10.2f}  {row['avg_excess_bps']:>10.2f}"
        )

    print("-" * 96)

    # 汇总行
    total_signals = len(trades_net)
    total_wr = (trades_net["net_ret"] > 0).mean()
    total_net = trades_net["net_ret"].mean() * 10000
    print(
        f"{'TOTAL':>10s}  {len(df):>10}  {total_signals:>8}  "
        f"{total_signals/len(df):>7.2%}  {trades_net['pred_prob'].mean():>8.4f}  "
        f"{total_wr:>7.1%}  {trades_net['is_profitable'].mean():>7.1%}  "
        f"{total_net:>10.2f}  {trades_net['excess_ret'].mean()*10000:>10.2f}"
    )

    return stats


def analyze_date_distribution(trades: pd.DataFrame):
    """按日期分析信号分布"""
    print("\n" + "=" * 100)
    print(f"2. Signal Distribution by Date (threshold={SIGNAL_THRESHOLD})")
    print("=" * 100)

    if len(trades) == 0:
        print("  No trades found.")
        return pd.DataFrame()

    trades_net = trades.copy()
    trades_net["net_ret"] = trades_net["raw_ret"] - COST_RATE

    daily = trades_net.groupby("date").agg(
        n_signals=("pred_prob", "size"),
        avg_prob=("pred_prob", "mean"),
        win_rate=("net_ret", lambda x: (x > 0).mean()),
        avg_net_bps=("net_ret", lambda x: x.mean() * 10000),
    )

    # 汇总统计
    print(f"\n  总共 {len(daily)} 个交易日有信号")
    print(f"  日均信号数: {daily['n_signals'].mean():.1f}")
    print(f"  信号数范围: [{daily['n_signals'].min()}, {daily['n_signals'].max()}]")

    # 信号最多的5天
    top5 = daily.nlargest(5, "n_signals")
    print(f"\n  信号最多的5天:")
    for d, row in top5.iterrows():
        print(
            f"    {d}: {row['n_signals']:.0f} signals, "
            f"WinRate={row['win_rate']:.1%}, NetBps={row['avg_net_bps']:.1f}"
        )

    # 单日收益明细：信号爆炸日 vs 正常日
    daily["total_net_bps"] = daily["avg_net_bps"] * daily["n_signals"]
    outlier_dates = daily[daily["n_signals"] > 100].index.tolist()
    if outlier_dates:
        print(f"\n  === 信号爆炸日（>100笔）对总收益的影响 ===")
        total_bps_all = daily["avg_net_bps"].sum()
        for od in outlier_dates:
            row = daily.loc[od]
            print(
                f"    {od}: {row['n_signals']:.0f} signals, "
                f"DayNetBps={row['avg_net_bps']:.1f}, "
                f"WinRate={row['win_rate']:.1%}"
            )
        outlier_bps = daily.loc[outlier_dates, "avg_net_bps"].sum()
        normal_bps = total_bps_all - outlier_bps
        normal_days = len(daily) - len(outlier_dates)
        print(
            f"    剔除异常日后: 累计DayNetBps={normal_bps:.1f} ({normal_days}天), "
            f"含异常日: 累计DayNetBps={total_bps_all:.1f} ({len(daily)}天)"
        )

    return daily


def analyze_prob_distribution(df: pd.DataFrame):
    """按概率区间分析信号质量"""
    print("\n" + "=" * 100)
    print("3. Signal Quality by Probability Bin")
    print("=" * 100)

    df_copy = df.copy()
    df_copy["net_ret"] = df_copy["raw_ret"] - COST_RATE

    # 概率分箱
    bins = [0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 1.0]
    labels = [
        "<0.30",
        "0.30-0.35",
        "0.35-0.40",
        "0.40-0.45",
        "0.45-0.50",
        "0.50-0.55",
        ">0.55",
    ]
    df_copy["prob_bin"] = pd.cut(
        df_copy["pred_prob"], bins=bins, labels=labels, right=False
    )

    stats = df_copy.groupby("prob_bin", observed=False).agg(
        count=("pred_prob", "size"),
        avg_prob=("pred_prob", "mean"),
        win_rate=("net_ret", lambda x: (x > 0).mean()),
        avg_net_bps=("net_ret", lambda x: x.mean() * 10000),
        avg_excess_bps=("excess_ret", lambda x: x.mean() * 10000),
        precision=("is_profitable", "mean"),
    )

    header = (
        f"{'ProbBin':>12}  {'Count':>10}  {'AvgProb':>8}  "
        f"{'WinRate':>8}  {'Prec':>8}  {'NetBps':>10}  {'ExcessBps':>10}"
    )
    print(f"\n{header}")
    print("-" * 80)

    for b, row in stats.iterrows():
        if row["count"] == 0:
            continue
        print(
            f"{b:>12}  {row['count']:>10.0f}  {row['avg_prob']:>7.4f}  "
            f"{row['win_rate']:>7.1%}  {row['precision']:>7.1%}  "
            f"{row['avg_net_bps']:>10.2f}  {row['avg_excess_bps']:>10.2f}"
        )

    print("-" * 80)
    return stats


def analyze_feature_comparison(df: pd.DataFrame, trades: pd.DataFrame):
    """对比信号样本 vs 全量样本的特征分布"""
    print("\n" + "=" * 100)
    print(f"4. Feature Comparison: Signals vs All (threshold={SIGNAL_THRESHOLD})")
    print("=" * 100)

    if len(trades) == 0:
        print("  No trades found.")
        return

    header = (
        f"{'Feature':>18}  {'All_Mean':>10}  {'All_Std':>10}  "
        f"{'Sig_Mean':>10}  {'Sig_Std':>10}  {'Diff':>10}"
    )
    print(f"\n{header}")
    print("-" * 80)

    for feat in FEATURES:
        all_mean = df[feat].mean()
        all_std = df[feat].std()
        sig_mean = trades[feat].mean()
        sig_std = trades[feat].std()
        diff = sig_mean - all_mean
        print(
            f"{feat:>18}  {all_mean:>10.4f}  {all_std:>10.4f}  "
            f"{sig_mean:>10.4f}  {sig_std:>10.4f}  {diff:>+10.4f}"
        )

    print("-" * 80)


# ==================== 绘图 ====================
def plot_signal_analysis(
    time_stats: pd.DataFrame,
    daily_stats: pd.DataFrame,
    prob_stats: pd.DataFrame,
    trades: pd.DataFrame,
):
    """绘制信号分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 按 entry_time 的信号数量和净收益
    if len(time_stats) > 0:
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        x_pos = range(len(time_stats))
        ax1.bar(
            x_pos,
            time_stats["n_signals"],
            alpha=0.4,
            color="steelblue",
            label="Signals",
        )
        ax1_twin.plot(
            x_pos, time_stats["avg_net_bps"], "ro-", markersize=5, label="NetBps"
        )
        ax1_twin.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(time_stats.index, rotation=45, fontsize=7)
        ax1.set_ylabel("Num Signals", color="steelblue")
        ax1_twin.set_ylabel("Avg Net Return (bps)")
        ax1_twin.legend(fontsize=8, loc="upper right")
        ax1.set_title(f"Signals by Entry Time (P>{SIGNAL_THRESHOLD})", fontsize=11)
        ax1.grid(True, alpha=0.3)

    # 2. 按日期的信号数量（x轴显示MM-DD标签）
    if len(daily_stats) > 0:
        ax2 = axes[0, 1]
        ax2.bar(
            range(len(daily_stats)),
            daily_stats["n_signals"],
            alpha=0.6,
            color="steelblue",
        )
        # 每月首个交易日标注日期
        date_strs = [str(d) for d in daily_stats.index]
        tick_pos, tick_labels = [], []
        seen_months = set()
        for i, d in enumerate(date_strs):
            month_key = d[:7]
            if month_key not in seen_months:
                seen_months.add(month_key)
                tick_pos.append(i)
                tick_labels.append(d[5:10])  # "MM-DD"
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Num Signals")
        ax2.set_title(f"Daily Signal Count (P>{SIGNAL_THRESHOLD})", fontsize=11)
        ax2.grid(True, alpha=0.3)

    # 3. 概率区间的信号质量
    if len(prob_stats) > 0:
        valid_prob = prob_stats[prob_stats["count"] > 0]
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        x_pos = range(len(valid_prob))
        ax3.bar(x_pos, valid_prob["count"], alpha=0.4, color="gray", label="Count")
        ax3_twin.plot(
            x_pos, valid_prob["avg_net_bps"], "rs-", markersize=6, label="NetBps"
        )
        ax3_twin.plot(
            x_pos, valid_prob["avg_excess_bps"], "bo-", markersize=6, label="ExcessBps"
        )
        ax3_twin.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(valid_prob.index, rotation=30, fontsize=8)
        ax3.set_ylabel("Count", color="gray")
        ax3_twin.set_ylabel("Avg Return (bps)")
        ax3_twin.legend(fontsize=8, loc="upper left")
        ax3.set_title("Signal Quality by Prob Bin", fontsize=11)
        ax3.grid(True, alpha=0.3)

    # 4. 信号的 pred_prob 直方图
    if len(trades) > 0:
        ax4 = axes[1, 1]
        ax4.hist(
            trades["pred_prob"],
            bins=30,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax4.axvline(
            SIGNAL_THRESHOLD,
            color="red",
            linestyle="--",
            label=f"Threshold={SIGNAL_THRESHOLD}",
        )
        ax4.set_xlabel("Predicted Probability")
        ax4.set_ylabel("Count")
        ax4.set_title(f"Prob Distribution of Signals (N={len(trades)})", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "lgbm_signal_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("LGBM 信号分析")
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_predictions()
    logger.info(
        f"预测概率分布: min={df['pred_prob'].min():.4f}, "
        f"median={df['pred_prob'].median():.4f}, max={df['pred_prob'].max():.4f}"
    )

    # 2. 筛选信号（时间+概率+冷却期去重）
    mask = (
        (df["entry_time"] >= START_TIME)
        & (df["entry_time"] <= END_TIME)
        & (df["pred_prob"] > SIGNAL_THRESHOLD)
    )
    trades = df[mask].copy()
    logger.info(
        f"初筛信号: {len(trades):,} 笔 (prob>{SIGNAL_THRESHOLD}, {START_TIME}~{END_TIME})"
    )

    trades = apply_cooldown(trades, COOLDOWN_BARS)
    logger.info(
        f"去重后信号: {len(trades):,} 笔 (冷却期={COOLDOWN_BARS}bars={COOLDOWN_BARS*5}min)"
    )

    # 3. 分析
    time_stats = analyze_entry_time_distribution(df, trades)
    daily_stats = analyze_date_distribution(trades)
    prob_stats = analyze_prob_distribution(df)
    analyze_feature_comparison(df, trades)

    # 4. 绘图
    plot_signal_analysis(time_stats, daily_stats, prob_stats, trades)

    logger.success("信号分析完成!")


if __name__ == "__main__":
    main()
