"""
时序模型组合回测脚本 (Top-N 排名选股)

基于时序 LightGBM 模型的预测结果进行回测：
  1. 每日 10:30 对全市场 ~1000 只股票排名
  2. 选取 pred_prob 最高的 Top-N 只等权买入
  3. 买入价 = 10:35 bar VWAP, 卖出价 = 后续24bar VWAP (约到14:05)
  4. 计算扣费后净值曲线、月度表现、风险指标

输入：output/ts_lgbm_predictions.pkl
输出：终端报告 + output/ts_portfolio_backtest.png

依赖：ts_lgbm_train.py 生产的预测数据

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
PRED_PATH = OUTPUT_DIR / "ts_lgbm_predictions.pkl"

# 交易成本 (单边, 含佣金+滑点)
COST_BPS = 15
COST_RATE = COST_BPS / 10000

# Top-N 策略参数
TOP_N_LIST = [20, 50, 100, 200]

# 持有期说明
ENTRY_TIME = "10:30"
# 买入: 10:35 bar VWAP, 卖出: 后续24bar VWAP (~14:05), 跨午休


# ==================== 数据加载 ====================
def load_predictions() -> pd.DataFrame:
    """加载测试集预测结果"""
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"预测结果不存在: {PRED_PATH}\n请先运行 ts_lgbm_train.py"
        )
    df = pd.read_pickle(PRED_PATH)
    logger.success(
        f"加载预测数据: {len(df):,} 行, "
        f"{df['date'].nunique()} 天, {df['SecuCode'].nunique()} 只股票"
    )
    return df


# ==================== 核心回测 ====================
def run_topn_backtest(
    df: pd.DataFrame,
    top_n: int,
    exclude_dates: list = None,
) -> dict:
    """
    Top-N 排名选股回测

    逻辑:
      1. 每日按 pred_prob 降序排名
      2. 取前 top_n 只等权买入
      3. 扣除双边成本
      4. 计算日度净值序列和绩效指标
    """
    df_active = df.copy()

    # 剔除异常日
    if exclude_dates:
        exclude_set = set(exclude_dates)
        df_active = df_active[~df_active["date"].astype(str).isin(exclude_set)]

    # 每日排名并取 Top-N
    df_active["rank"] = df_active.groupby("date")["pred_prob"].rank(
        ascending=False, method="first"
    )
    portfolio = df_active[df_active["rank"] <= top_n].copy()

    # 净收益 = raw_ret - 双边成本
    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE

    n_trades = len(portfolio)

    # 日度汇总（等权）
    daily_ret = portfolio.groupby("date").agg(
        raw_ret=("raw_ret", "mean"),
        excess_ret=("excess_ret", "mean"),
        net_ret=("net_ret", "mean"),
        n_stocks=("SecuCode", "size"),
        avg_prob=("pred_prob", "mean"),
        market_ret=("market_ret", "first"),
    )
    daily_ret.index = pd.to_datetime(daily_ret.index)
    daily_ret = daily_ret.sort_index()

    n_days = len(daily_ret)
    if n_days == 0:
        return {"top_n": top_n, "n_trades": 0}

    # ===== 绩效指标 =====
    # 胜率
    win_rate = (portfolio["net_ret"] > 0).mean()
    day_win_rate = (daily_ret["net_ret"] > 0).mean()
    excess_day_win_rate = (daily_ret["excess_ret"] > 0).mean()

    # 收益
    avg_net_bps = daily_ret["net_ret"].mean() * 10000
    avg_excess_bps = daily_ret["excess_ret"].mean() * 10000
    avg_raw_bps = daily_ret["raw_ret"].mean() * 10000

    # 年化
    annual_net = daily_ret["net_ret"].mean() * 242 * 100
    annual_excess = daily_ret["excess_ret"].mean() * 242 * 100

    # Sharpe
    sharpe_net = (
        daily_ret["net_ret"].mean() / daily_ret["net_ret"].std() * np.sqrt(242)
        if daily_ret["net_ret"].std() > 0
        else 0
    )
    sharpe_excess = (
        daily_ret["excess_ret"].mean() / daily_ret["excess_ret"].std() * np.sqrt(242)
        if daily_ret["excess_ret"].std() > 0
        else 0
    )

    # 最大回撤
    cum_net = daily_ret["net_ret"].cumsum()
    max_dd_net = (cum_net - cum_net.cummax()).min()

    cum_excess = daily_ret["excess_ret"].cumsum()
    max_dd_excess = (cum_excess - cum_excess.cummax()).min()

    # 获利因子
    gross_profit = portfolio.loc[portfolio["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(portfolio.loc[portfolio["net_ret"] < 0, "net_ret"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar
    calmar = annual_net / abs(max_dd_net * 100) if max_dd_net < 0 else np.inf

    return {
        "top_n": top_n,
        "n_trades": n_trades,
        "n_days": n_days,
        "avg_stocks_per_day": n_trades / n_days,
        "win_rate": win_rate,
        "day_win_rate": day_win_rate,
        "excess_day_wr": excess_day_win_rate,
        "avg_raw_bps": avg_raw_bps,
        "avg_net_bps": avg_net_bps,
        "avg_excess_bps": avg_excess_bps,
        "annual_net": annual_net,
        "annual_excess": annual_excess,
        "sharpe_net": sharpe_net,
        "sharpe_excess": sharpe_excess,
        "max_dd_net": max_dd_net,
        "max_dd_excess": max_dd_excess,
        "profit_factor": pf,
        "calmar": calmar,
        "daily_ret": daily_ret,
    }


# ==================== 动态换手控制版回测 ====================
def run_topn_backtest_turnover(
    df: pd.DataFrame,
    top_n: int,
    buffer_ratio: float = 1.0,
    exclude_dates: list = None,
) -> dict:
    """
    Top-N 排名选股回测（动态换手控制版）

    与原版区别：
      - 与昨日持仓对比，重叠股票不收费
      - buffer_ratio > 1 时，昨日持仓排名在 Top-(N*buffer_ratio) 内可保留
      - 日成本 = 换手率 × COST_RATE

    参数:
      buffer_ratio: 缓冲比例。1.0 = 仅重叠免费，2.0 = Top-2N 内保留
    """
    buffer_n = int(top_n * buffer_ratio)

    df_active = df.copy()
    if exclude_dates:
        exclude_set = set(exclude_dates)
        df_active = df_active[~df_active["date"].astype(str).isin(exclude_set)]

    # 每日排名
    df_active["rank"] = df_active.groupby("date")["pred_prob"].rank(
        ascending=False, method="first"
    )

    dates = sorted(df_active["date"].unique())
    prev_holdings = set()
    daily_records = []
    trade_count = 0

    for date in dates:
        day_data = df_active[df_active["date"] == date]

        if buffer_ratio > 1.0 and len(prev_holdings) > 0:
            # 昨日持仓中排名在 buffer_n 以内的可保留
            kept_mask = day_data["SecuCode"].isin(prev_holdings) & (
                day_data["rank"] <= buffer_n
            )
            kept_stocks = set(day_data.loc[kept_mask, "SecuCode"])

            # 保留的超过 top_n 时按排名截断
            if len(kept_stocks) > top_n:
                kept_data = day_data[day_data["SecuCode"].isin(kept_stocks)].nsmallest(
                    top_n, "rank"
                )
                kept_stocks = set(kept_data["SecuCode"])

            # 剩余名额从 Top-N 新候选中补
            n_fill = top_n - len(kept_stocks)
            if n_fill > 0:
                fill_mask = (day_data["rank"] <= top_n) & (
                    ~day_data["SecuCode"].isin(kept_stocks)
                )
                new_fills = set(
                    day_data[fill_mask].nsmallest(n_fill, "rank")["SecuCode"]
                )
            else:
                new_fills = set()

            today_holdings = kept_stocks | new_fills
            new_stocks = today_holdings - prev_holdings
        else:
            # 无缓冲 或 第一天
            today_holdings = set(day_data[day_data["rank"] <= top_n]["SecuCode"])
            new_stocks = today_holdings - prev_holdings

        # 计算收益
        portfolio = day_data[day_data["SecuCode"].isin(today_holdings)]
        if len(portfolio) == 0:
            prev_holdings = set()
            continue

        n_new = len(new_stocks)
        n_total = len(today_holdings)
        turnover = n_new / n_total if n_total > 0 else 1.0
        daily_cost = turnover * COST_RATE

        raw_mean = portfolio["raw_ret"].mean()
        excess_mean = portfolio["excess_ret"].mean()
        net_mean = raw_mean - daily_cost

        daily_records.append(
            {
                "date": date,
                "raw_ret": raw_mean,
                "excess_ret": excess_mean,
                "net_ret": net_mean,
                "n_stocks": n_total,
                "n_new": n_new,
                "turnover": turnover,
                "market_ret": portfolio["market_ret"].iloc[0],
                "cost_bps": daily_cost * 10000,
            }
        )
        trade_count += n_total
        prev_holdings = today_holdings

    if not daily_records:
        return {"top_n": top_n, "n_trades": 0}

    daily_ret = pd.DataFrame(daily_records)
    daily_ret.index = pd.to_datetime(daily_ret["date"])
    daily_ret = daily_ret.sort_index()
    n_days = len(daily_ret)

    # ===== 绩效指标 =====
    day_win_rate = (daily_ret["net_ret"] > 0).mean()
    excess_day_wr = (daily_ret["excess_ret"] > 0).mean()

    avg_net_bps = daily_ret["net_ret"].mean() * 10000
    avg_excess_bps = daily_ret["excess_ret"].mean() * 10000
    avg_raw_bps = daily_ret["raw_ret"].mean() * 10000
    avg_turnover = daily_ret["turnover"].mean()
    avg_cost_bps = daily_ret["cost_bps"].mean()

    annual_net = daily_ret["net_ret"].mean() * 242 * 100
    annual_excess = daily_ret["excess_ret"].mean() * 242 * 100

    sharpe_net = (
        daily_ret["net_ret"].mean() / daily_ret["net_ret"].std() * np.sqrt(242)
        if daily_ret["net_ret"].std() > 0
        else 0
    )
    sharpe_excess = (
        daily_ret["excess_ret"].mean() / daily_ret["excess_ret"].std() * np.sqrt(242)
        if daily_ret["excess_ret"].std() > 0
        else 0
    )

    cum_net = daily_ret["net_ret"].cumsum()
    max_dd_net = (cum_net - cum_net.cummax()).min()

    gross_profit = daily_ret.loc[daily_ret["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(daily_ret.loc[daily_ret["net_ret"] < 0, "net_ret"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "top_n": top_n,
        "n_trades": trade_count,
        "n_days": n_days,
        "avg_stocks_per_day": trade_count / n_days,
        "win_rate": day_win_rate,
        "day_win_rate": day_win_rate,
        "excess_day_wr": excess_day_wr,
        "avg_raw_bps": avg_raw_bps,
        "avg_net_bps": avg_net_bps,
        "avg_excess_bps": avg_excess_bps,
        "annual_net": annual_net,
        "annual_excess": annual_excess,
        "sharpe_net": sharpe_net,
        "sharpe_excess": sharpe_excess,
        "max_dd_net": max_dd_net,
        "profit_factor": pf,
        "daily_ret": daily_ret,
        "avg_turnover": avg_turnover,
        "avg_cost_bps": avg_cost_bps,
    }


# ==================== 报告 ====================
def print_report(results: list):
    """打印回测报告"""
    print("\n" + "=" * 130)
    print(
        f"时序模型组合回测报告 | "
        f"入场: {ENTRY_TIME}, 持有: 24bar (~10:35→14:05跨午休) | "
        f"成本: {COST_BPS}bps×2"
    )
    print("=" * 130)

    # 表1: 核心绩效
    print("\n  --- Core Performance ---\n")
    print(
        f"  {'TopN':>5} {'Days':>5} {'Stk':>5}"
        f" {'WinR':>6} {'DayWR':>6}"
        f" {'NetBps':>8} {'ExcBps':>8}"
        f" {'AnnNet%':>7} {'AnnExc%':>7}"
        f" {'ShpN':>6} {'ShpE':>6}"
        f" {'MaxDD':>8} {'PF':>5}"
    )
    print("  " + "-" * 95)

    for r in results:
        if r["n_trades"] == 0:
            print(f"  {r['top_n']:>5} N/A")
            continue
        print(
            f"  {r['top_n']:>5} {r['n_days']:>5} {r['avg_stocks_per_day']:>5.0f}"
            f" {r['win_rate']:>5.1%} {r['day_win_rate']:>5.1%}"
            f" {r['avg_net_bps']:>8.2f} {r['avg_excess_bps']:>8.2f}"
            f" {r['annual_net']:>6.1f}% {r['annual_excess']:>6.1f}%"
            f" {r['sharpe_net']:>6.2f} {r['sharpe_excess']:>6.2f}"
            f" {r['max_dd_net']*10000:>7.0f}bp {r['profit_factor']:>5.2f}"
        )
    print("  " + "-" * 95)

    # 表2: 月度明细（用 Top-50）
    for r in results:
        if r["top_n"] == 50 and r["n_trades"] > 0:
            print_monthly_detail(r)
            break


def print_monthly_detail(r: dict):
    """打印月度明细"""
    daily = r["daily_ret"]
    daily["month"] = daily.index.to_period("M")

    monthly = daily.groupby("month").agg(
        n_days=("net_ret", "size"),
        net_bps=("net_ret", lambda x: x.mean() * 10000),
        excess_bps=("excess_ret", lambda x: x.mean() * 10000),
        cum_net=("net_ret", lambda x: x.sum() * 10000),
        cum_excess=("excess_ret", lambda x: x.sum() * 10000),
        win_rate=("net_ret", lambda x: (x > 0).mean()),
    )

    print(f"\n  --- Monthly Detail (Top-{r['top_n']}) ---\n")
    print(
        f"  {'Month':>10} {'Days':>5}"
        f" {'NetBps':>8} {'ExcBps':>8}"
        f" {'CumNet':>8} {'CumExc':>8} {'DayWR':>6}"
    )
    print("  " + "-" * 60)

    for month, row in monthly.iterrows():
        print(
            f"  {str(month):>10} {row['n_days']:>5.0f}"
            f" {row['net_bps']:>8.2f} {row['excess_bps']:>8.2f}"
            f" {row['cum_net']:>8.1f} {row['cum_excess']:>8.1f}"
            f" {row['win_rate']:>5.1%}"
        )

    total_net = monthly["cum_net"].sum()
    total_excess = monthly["cum_excess"].sum()
    print("  " + "-" * 60)
    print(
        f"  {'Total':>10} {r['n_days']:>5}"
        f" {r['avg_net_bps']:>8.2f} {r['avg_excess_bps']:>8.2f}"
        f" {total_net:>8.1f} {total_excess:>8.1f}"
        f" {r['day_win_rate']:>5.1%}"
    )


# ==================== 绘图 ====================
def plot_results(results: list):
    """绘制回测结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f"Time-Series Model Portfolio Backtest\n"
        f"Entry: {ENTRY_TIME} | Hold: 24 bars (~10:35->14:05) | Cost: {COST_BPS}bps x2",
        fontsize=13,
    )

    # (1) 净值累计曲线 (net)
    ax1 = axes[0, 0]
    for r in results:
        if r["n_trades"] == 0 or "daily_ret" not in r:
            continue
        daily = r["daily_ret"]
        cum_bps = daily["net_ret"].cumsum() * 10000
        label = f"Top-{r['top_n']} (Sharpe={r['sharpe_net']:.2f})"
        ax1.plot(
            range(len(cum_bps)), cum_bps.values, label=label, alpha=0.8, linewidth=1.5
        )

        # 月度刻度
        date_strs = [str(d.date()) for d in daily.index]
        tick_pos, tick_labels = [], []
        seen = set()
        for i, d in enumerate(date_strs):
            mk = d[:7]
            if mk not in seen:
                seen.add(mk)
                tick_pos.append(i)
                tick_labels.append(d[5:10])
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax1.set_title("Cumulative Net Return (after cost)", fontsize=11)
    ax1.set_ylabel("Cumulative (bps)")
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # (2) 超额累计曲线
    ax2 = axes[0, 1]
    for r in results:
        if r["n_trades"] == 0 or "daily_ret" not in r:
            continue
        daily = r["daily_ret"]
        cum_bps = daily["excess_ret"].cumsum() * 10000
        label = f"Top-{r['top_n']} (Sharpe={r['sharpe_excess']:.2f})"
        ax2.plot(
            range(len(cum_bps)), cum_bps.values, label=label, alpha=0.8, linewidth=1.5
        )

        date_strs = [str(d.date()) for d in daily.index]
        tick_pos, tick_labels = [], []
        seen = set()
        for i, d in enumerate(date_strs):
            mk = d[:7]
            if mk not in seen:
                seen.add(mk)
                tick_pos.append(i)
                tick_labels.append(d[5:10])
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax2.set_title("Cumulative Excess Return (vs market)", fontsize=11)
    ax2.set_ylabel("Cumulative (bps)")
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # (3) Sharpe 对比柱状图
    ax3 = axes[1, 0]
    valid = [r for r in results if r["n_trades"] > 0]
    if valid:
        x = range(len(valid))
        labels = [f"Top-{r['top_n']}" for r in valid]
        sharpe_n = [r["sharpe_net"] for r in valid]
        sharpe_e = [r["sharpe_excess"] for r in valid]
        w = 0.35
        ax3.bar(
            [i - w / 2 for i in x],
            sharpe_n,
            w,
            label="Net Sharpe",
            alpha=0.7,
            color="steelblue",
        )
        ax3.bar(
            [i + w / 2 for i in x],
            sharpe_e,
            w,
            label="Excess Sharpe",
            alpha=0.7,
            color="coral",
        )
        ax3.set_xticks(list(x))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel("Sharpe Ratio")
        ax3.set_title("Sharpe by Top-N", fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

    # (4) 日均收益柱状图
    ax4 = axes[1, 1]
    if valid:
        net_bps = [r["avg_net_bps"] for r in valid]
        exc_bps = [r["avg_excess_bps"] for r in valid]
        ax4.bar(
            [i - w / 2 for i in x],
            net_bps,
            w,
            label="Net bps",
            alpha=0.7,
            color="steelblue",
        )
        ax4.bar(
            [i + w / 2 for i in x],
            exc_bps,
            w,
            label="Excess bps",
            alpha=0.7,
            color="coral",
        )
        ax4.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax4.set_xticks(list(x))
        ax4.set_xticklabels(labels)
        ax4.set_ylabel("Avg Daily Return (bps)")
        ax4.set_title("Daily Returns by Top-N", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "ts_portfolio_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("时序模型组合回测")
    logger.info(
        f"入场: {ENTRY_TIME} | 持有: 24bar (~10:35→14:05跨午休) | 成本: {COST_BPS}bps×2"
    )
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_predictions()

    # 2. 各 Top-N 回测（原版：全换手）
    results = []
    for n in TOP_N_LIST:
        logger.info(f"回测 Top-{n}...")
        r = run_topn_backtest(df, top_n=n)
        results.append(r)

    # 3. 报告
    print_report(results)

    # 4. 绘图
    plot_results(results)

    # 5. 动态换手控制版回测
    print("\n\n" + "=" * 130)
    print("动态换手控制版 | 重叠持仓不换仓，只对换手部分收取成本")
    print("=" * 130)

    for buf_label, buf_ratio in [
        ("无缓冲(仅重叠免费)", 1.0),
        ("2x缓冲(Top-2N保留)", 2.0),
    ]:
        print(f"\n  --- {buf_label}, buffer_ratio={buf_ratio} ---\n")
        print(
            f"  {'TopN':>5} {'Days':>5}"
            f" {'RawBps':>8} {'NetBps':>8} {'ExcBps':>8}"
            f" {'Turnov':>7} {'CostBps':>8}"
            f" {'AnnNet%':>7} {'ShpN':>6} {'ShpE':>6}"
            f" {'MaxDD':>8} {'DayWR':>6}"
        )
        print("  " + "-" * 100)

        for n in TOP_N_LIST:
            r = run_topn_backtest_turnover(df, top_n=n, buffer_ratio=buf_ratio)
            if r["n_trades"] == 0:
                print(f"  {n:>5} N/A")
                continue
            print(
                f"  {r['top_n']:>5} {r['n_days']:>5}"
                f" {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f} {r['avg_excess_bps']:>8.2f}"
                f" {r['avg_turnover']:>6.1%} {r['avg_cost_bps']:>8.2f}"
                f" {r['annual_net']:>6.1f}% {r['sharpe_net']:>6.2f} {r['sharpe_excess']:>6.2f}"
                f" {r['max_dd_net']*10000:>7.0f}bp {r['day_win_rate']:>5.1%}"
            )
        print("  " + "-" * 100)

    # 6. 原版 vs 换手控制版对比（Top-50）
    r_orig = [r for r in results if r["top_n"] == 50][0]
    r_no_buf = run_topn_backtest_turnover(df, top_n=50, buffer_ratio=1.0)
    r_buf2x = run_topn_backtest_turnover(df, top_n=50, buffer_ratio=2.0)

    print("\n\n" + "=" * 80)
    print("Top-50 三版对比")
    print("=" * 80)
    print(
        f"  {'版本':>20} {'NetBps':>8} {'ExcBps':>8} {'Turnov':>7} {'CostBps':>8} {'ShpN':>6} {'ShpE':>6}"
    )
    print("  " + "-" * 65)
    print(
        f"  {'原版(全换手)':>20} {r_orig['avg_net_bps']:>8.2f} {r_orig['avg_excess_bps']:>8.2f}"
        f" {'100.0%':>7} {COST_BPS:>8.2f}"
        f" {r_orig['sharpe_net']:>6.2f} {r_orig['sharpe_excess']:>6.2f}"
    )
    print(
        f"  {'换手控制(无缓冲)':>20} {r_no_buf['avg_net_bps']:>8.2f} {r_no_buf['avg_excess_bps']:>8.2f}"
        f" {r_no_buf['avg_turnover']:>6.1%} {r_no_buf['avg_cost_bps']:>8.2f}"
        f" {r_no_buf['sharpe_net']:>6.2f} {r_no_buf['sharpe_excess']:>6.2f}"
    )
    print(
        f"  {'换手控制(2x缓冲)':>20} {r_buf2x['avg_net_bps']:>8.2f} {r_buf2x['avg_excess_bps']:>8.2f}"
        f" {r_buf2x['avg_turnover']:>6.1%} {r_buf2x['avg_cost_bps']:>8.2f}"
        f" {r_buf2x['sharpe_net']:>6.2f} {r_buf2x['sharpe_excess']:>6.2f}"
    )
    print("  " + "-" * 65)
    saved_no_buf = COST_BPS - r_no_buf["avg_cost_bps"]
    saved_buf2x = COST_BPS - r_buf2x["avg_cost_bps"]
    print(
        f"\n  成本节省: 无缓冲 {saved_no_buf:.2f}bps/天, 2x缓冲 {saved_buf2x:.2f}bps/天"
    )

    logger.success("回测完成!")


if __name__ == "__main__":
    main()
