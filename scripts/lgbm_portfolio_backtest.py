"""
LGBM 组合回测脚本 (Top-N 模式)

基于 design_scheme_3.ipynb 的方案：
  1. 时间过滤：只做 10:30 后的信号（早盘容易被骗线）
  2. 容量控制：每个 5min Bar 仅取 Prob 最高的 N 只票
  3. 计算组合层面的净值曲线
  4. 对比剔除异常日前后的表现

输入：output/lgbm_test_predictions.pkl（由 lgbm_classifier.py 生成）
输出：终端报告 + 图表
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

# ==================== 配置 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"

COST_BPS = 15
COST_RATE = COST_BPS / 10000

# 策略约束参数
TOP_N_LIST = [5, 10, 20, 50]  # 多组 Top-N 对比
MIN_PROB = 0.45  # 概率门槛
START_TIME = "09:50"  # 最早入场时间（不过滤早盘）
END_TIME = "14:50"  # 最晚入场时间

# 异常日剔除列表（信号爆炸日）
OUTLIER_DATES = ["2025-04-09", "2025-04-07"]


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


# ==================== 核心回测 ====================
def run_portfolio_backtest(
    df: pd.DataFrame,
    top_n: int,
    min_prob: float = MIN_PROB,
    start_time: str = START_TIME,
    end_time: str = END_TIME,
    exclude_dates: list = None,
) -> dict:
    """Top-N 组合回测

    逻辑：
      1. 时间+概率筛选
      2. 每个 (date, entry_time) 取 pred_prob 最高的 top_n 只
      3. Bar 内等权 → 日内 Bar 均值 → 日收益
    """
    # 1. 时间和门槛初筛
    mask = (
        (df["entry_time"] >= start_time)
        & (df["entry_time"] <= end_time)
        & (df["pred_prob"] >= min_prob)
    )
    if exclude_dates:
        exclude_set = set(exclude_dates)
        mask = mask & (~df["date"].astype(str).isin(exclude_set))

    df_active = df[mask].copy()

    if len(df_active) == 0:
        logger.warning(f"Top-{top_n}: 筛选后无信号")
        return {"top_n": top_n, "n_trades": 0}

    # 2. Top-N 筛选（向量化：排名后过滤）
    df_active["rank_in_bar"] = df_active.groupby(["date", "entry_time"])[
        "pred_prob"
    ].rank(ascending=False, method="first")
    portfolio = df_active[df_active["rank_in_bar"] <= top_n].copy()

    # 3. 计算净收益
    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE
    portfolio["net_ret_bps"] = portfolio["net_ret"] * 10000

    n_trades = len(portfolio)

    # 4. 日收益：Bar 等权均值 → 日均值
    bar_ret = portfolio.groupby(["date", "entry_time"])["net_ret"].mean()
    daily_ret = bar_ret.groupby("date").mean()
    n_days = len(daily_ret)

    # 5. 绩效统计
    win_rate = (portfolio["net_ret"] > 0).mean()
    day_win_rate = (daily_ret > 0).mean()
    avg_net_bps = portfolio["net_ret_bps"].mean()
    avg_daily_bps = daily_ret.mean() * 10000

    sharpe = 0.0
    if n_days > 1 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    cum = daily_ret.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0

    # 获利因子
    gross_profit = portfolio.loc[portfolio["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(portfolio.loc[portfolio["net_ret"] < 0, "net_ret"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # 日均选股数
    avg_daily_trades = n_trades / n_days if n_days > 0 else 0

    # 每个 bar 实际平均选了几只
    bar_counts = portfolio.groupby(["date", "entry_time"]).size()
    avg_bar_count = bar_counts.mean()

    logger.info(
        f"  Top-{top_n:>2}: {n_trades:>6} trades/{n_days} days, "
        f"AvgBarN={avg_bar_count:.1f}, "
        f"WR={win_rate:.1%}, DayWR={day_win_rate:.1%}, "
        f"NetBps={avg_net_bps:.1f}, DailyBps={avg_daily_bps:.1f}, "
        f"Sharpe={sharpe:.2f}, MaxDD={max_dd*10000:.0f}bps"
    )

    return {
        "top_n": top_n,
        "n_trades": n_trades,
        "n_days": n_days,
        "avg_daily_trades": avg_daily_trades,
        "avg_bar_count": avg_bar_count,
        "win_rate": win_rate,
        "day_win_rate": day_win_rate,
        "avg_net_bps": avg_net_bps,
        "avg_daily_bps": avg_daily_bps,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "daily_ret": daily_ret,
    }


# ==================== 报告 ====================
def print_report(results_normal: list, results_excl: list):
    """打印对比报告"""
    print("\n" + "=" * 130)
    print("LGBM 组合回测报告 (Top-N)")
    print(
        f"  时间范围: {START_TIME} ~ {END_TIME} | 最低概率: {MIN_PROB} | 成本: {COST_BPS}bps"
    )
    print("=" * 130)

    def _print_table(results, label):
        print(f"\n  --- {label} ---")
        header = (
            f"{'TopN':>6}  {'Trades':>8}  {'Days':>6}  {'AvgBarN':>8}  "
            f"{'WinRate':>8}  {'DayWR':>8}  "
            f"{'NetBps':>10}  {'DailyBps':>10}  {'Sharpe':>8}  "
            f"{'MaxDD':>10}  {'PF':>8}"
        )
        print(f"\n{header}")
        print("-" * 110)
        for r in results:
            if r["n_trades"] == 0:
                print(f"{r['top_n']:>6}  {'N/A':>8}")
                continue
            print(
                f"{r['top_n']:>6}  {r['n_trades']:>8,}  {r['n_days']:>6}  "
                f"{r['avg_bar_count']:>8.1f}  "
                f"{r['win_rate']:>7.1%}  {r['day_win_rate']:>7.1%}  "
                f"{r['avg_net_bps']:>10.1f}  {r['avg_daily_bps']:>10.1f}  "
                f"{r['sharpe']:>8.2f}  "
                f"{r['max_drawdown']*10000:>9.0f}bps  "
                f"{r['profit_factor']:>8.2f}"
            )
        print("-" * 110)

    _print_table(results_normal, "含全部日期")
    _print_table(results_excl, f"剔除异常日 {OUTLIER_DATES}")
    print("=" * 130)


# ==================== 绘图 ====================
def plot_results(results_normal: list, results_excl: list):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    def _plot_equity(ax, results, title):
        for r in results:
            if r["n_trades"] == 0 or "daily_ret" not in r:
                continue
            daily = r["daily_ret"]
            cum_bps = daily.cumsum() * 10000
            label = f"Top-{r['top_n']} (S={r['sharpe']:.2f}, N={r['n_trades']})"
            ax.plot(
                range(len(cum_bps)),
                cum_bps.values,
                label=label,
                alpha=0.8,
                linewidth=1.5,
            )

            # 日期刻度
            date_strs = [str(d) for d in daily.index]
            tick_pos, tick_labels = [], []
            seen = set()
            for i, d in enumerate(date_strs):
                mk = d[:7]
                if mk not in seen:
                    seen.add(mk)
                    tick_pos.append(i)
                    tick_labels.append(d[5:10])
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Net Return (bps)")
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    _plot_equity(axes[0, 0], results_normal, "All Dates")
    _plot_equity(axes[0, 1], results_excl, f"Excl. Outliers {OUTLIER_DATES}")

    # Top-N vs Sharpe 柱状图
    valid_n = [r for r in results_normal if r["n_trades"] > 0]
    valid_e = [r for r in results_excl if r["n_trades"] > 0]

    if valid_n:
        ax3 = axes[1, 0]
        x = range(len(valid_n))
        labels = [f"Top-{r['top_n']}" for r in valid_n]
        sharpes_n = [r["sharpe"] for r in valid_n]
        sharpes_e = [r["sharpe"] for r in valid_e] if valid_e else [0] * len(valid_n)
        w = 0.35
        ax3.bar(
            [i - w / 2 for i in x],
            sharpes_n,
            w,
            label="All Dates",
            alpha=0.7,
            color="steelblue",
        )
        ax3.bar(
            [i + w / 2 for i in x],
            sharpes_e,
            w,
            label="Excl. Outliers",
            alpha=0.7,
            color="coral",
        )
        ax3.set_xticks(list(x))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Sharpe by Top-N", fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

    # Top-N vs NetBps 柱状图
    if valid_n:
        ax4 = axes[1, 1]
        net_n = [r["avg_net_bps"] for r in valid_n]
        net_e = [r["avg_net_bps"] for r in valid_e] if valid_e else [0] * len(valid_n)
        ax4.bar(
            [i - w / 2 for i in x],
            net_n,
            w,
            label="All Dates",
            alpha=0.7,
            color="steelblue",
        )
        ax4.bar(
            [i + w / 2 for i in x],
            net_e,
            w,
            label="Excl. Outliers",
            alpha=0.7,
            color="coral",
        )
        ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(labels)
        ax4.set_ylabel("Avg Net Return (bps)")
        ax4.set_title("Net bps by Top-N", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "lgbm_portfolio_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("LGBM 组合回测 (Top-N)")
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_predictions()

    # 2. 含所有日期的 Top-N 回测
    logger.info(f"--- 含所有日期 ({START_TIME}~{END_TIME}, MinProb={MIN_PROB}) ---")
    results_normal = []
    for n in TOP_N_LIST:
        r = run_portfolio_backtest(df, top_n=n)
        results_normal.append(r)

    # 3. 剔除异常日的 Top-N 回测
    logger.info(f"--- 剔除异常日 {OUTLIER_DATES} ---")
    results_excl = []
    for n in TOP_N_LIST:
        r = run_portfolio_backtest(df, top_n=n, exclude_dates=OUTLIER_DATES)
        results_excl.append(r)

    # 4. 报告
    print_report(results_normal, results_excl)

    # 5. 绘图
    plot_results(results_normal, results_excl)

    logger.success("组合回测完成!")


if __name__ == "__main__":
    main()
