"""
LGBM T0 仓位约束回测

在信号去重（冷却期）基础上，引入 T0 仓位比例约束：
  - 实际 T0 可操作仓位 = 总仓位 × POSITION_RATIO
  - 策略日收益 = 信号日收益 × POSITION_RATIO
  - 持仓池 = 中证1000（已在预测数据中，不额外过滤）

用户可调参数：
  - POSITION_RATIO: T0可操作仓位比例（默认0.25，即1/4）
  - COOLDOWN_BARS: 冷却期（默认24bars=120分钟=持有期）
  - TOP_N: 每个bar最多选几只（默认50）
  - MIN_PROB: 概率门槛（默认0.45）

输入：output/lgbm_test_predictions.pkl
输出：终端报告 + 图表 lgbm_t0_backtest.png

作者：量化研究
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 用户可调参数 ====================
POSITION_RATIO = 0.25  # T0可操作仓位比例（1/4仓位做T0）

COOLDOWN_BARS = 24  # 冷却期（24bars=120分钟=持有期）

TOP_N_LIST = [10, 20, 50]  # 多组 Top-N 对比
MIN_PROB = 0.45
START_TIME = "09:50"
END_TIME = "14:50"
COST_BPS = 15
COST_RATE = COST_BPS / 10000

# 异常日剔除
OUTLIER_DATES = ["2025-04-09", "2025-04-07"]

# ==================== 路径 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"


# ==================== 工具函数 ====================
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


# ==================== 核心回测 ====================
def run_t0_backtest(
    df: pd.DataFrame,
    top_n: int,
    position_ratio: float = POSITION_RATIO,
    cooldown_bars: int = COOLDOWN_BARS,
    exclude_dates: list = None,
    label_prefix: str = "",
) -> dict:
    """T0 仓位约束回测

    流程：
      1. 时间+概率初筛
      2. 冷却期去重
      3. Top-N 筛选
      4. 计算信号层面收益
      5. 乘以仓位比例得到策略真实收益
    """
    # 1. 初筛
    mask = (
        (df["entry_time"] >= START_TIME)
        & (df["entry_time"] <= END_TIME)
        & (df["pred_prob"] >= MIN_PROB)
    )
    if exclude_dates:
        mask = mask & (~df["date"].astype(str).isin(set(exclude_dates)))
    signals = df[mask].copy()

    if len(signals) == 0:
        return {"label": f"{label_prefix}Top-{top_n}", "n_trades": 0}

    # 2. 冷却期去重
    if cooldown_bars > 0:
        signals = apply_cooldown(signals, cooldown_bars)

    # 3. Top-N 筛选
    signals["rank_in_bar"] = signals.groupby(["date", "entry_time"])["pred_prob"].rank(
        ascending=False, method="first"
    )
    portfolio = signals[signals["rank_in_bar"] <= top_n].copy()

    if len(portfolio) == 0:
        return {"label": f"{label_prefix}Top-{top_n}", "n_trades": 0}

    # 4. 信号层面收益
    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE

    n_trades = len(portfolio)

    # Bar等权 → 日均值 = 信号日收益
    bar_ret = portfolio.groupby(["date", "entry_time"])["net_ret"].mean()
    signal_daily_ret = bar_ret.groupby("date").mean()

    # 5. 策略真实日收益 = 信号日收益 × 仓位比例
    strategy_daily_ret = signal_daily_ret * position_ratio
    n_days = len(strategy_daily_ret)

    # 6. 绩效统计
    # 信号层面指标
    sig_win_rate = (portfolio["net_ret"] > 0).mean()
    sig_day_wr = (signal_daily_ret > 0).mean()
    sig_net_bps = portfolio["net_ret"].mean() * 10000
    sig_daily_bps = signal_daily_ret.mean() * 10000

    # 策略层面指标（考虑仓位比例）
    strat_daily_bps = strategy_daily_ret.mean() * 10000
    strat_annual_ret = strategy_daily_ret.mean() * 252
    strat_sharpe = (
        strategy_daily_ret.mean() / strategy_daily_ret.std() * np.sqrt(252)
        if strategy_daily_ret.std() > 0
        else 0
    )
    strat_cum = strategy_daily_ret.cumsum()
    strat_max_dd = (strat_cum - strat_cum.cummax()).min()

    # 每日独立股票数
    unique_stocks_day = portfolio.groupby("date")["SecuCode"].nunique().mean()

    label = f"{label_prefix}Top-{top_n}"

    return {
        "label": label,
        "n_trades": n_trades,
        "n_days": n_days,
        "trades_per_day": n_trades / n_days,
        "unique_stocks_day": unique_stocks_day,
        # 信号层面
        "sig_win_rate": sig_win_rate,
        "sig_day_wr": sig_day_wr,
        "sig_net_bps": sig_net_bps,
        "sig_daily_bps": sig_daily_bps,
        # 策略层面（含仓位比例）
        "strat_daily_bps": strat_daily_bps,
        "strat_annual_ret": strat_annual_ret,
        "strat_sharpe": strat_sharpe,
        "strat_max_dd": strat_max_dd,
        # 时间序列
        "signal_daily_ret": signal_daily_ret,
        "strategy_daily_ret": strategy_daily_ret,
    }


# ==================== 报告 ====================
def print_report(results: list, position_ratio: float, cooldown_bars: int):
    """打印回测报告"""
    print("\n" + "=" * 150)
    print(f"LGBM T0 仓位约束回测报告")
    print(
        f"  仓位比例: {position_ratio:.0%} | 冷却期: {cooldown_bars}bars ({cooldown_bars*5}min)"
    )
    print(
        f"  时间范围: {START_TIME}~{END_TIME} | 概率门槛: {MIN_PROB} | 成本: {COST_BPS}bps"
    )
    print("=" * 150)

    header = (
        f"{'Label':>25s}  {'Trades':>7s}  {'Days':>5s}  {'T/Day':>6s}  {'Stk/D':>6s}  "
        f"{'SigWR':>6s}  {'DayWR':>6s}  {'SigBps':>7s}  {'SigDly':>7s}  "
        f"{'StratDly':>8s}  {'AnnRet':>8s}  {'Sharpe':>7s}  {'MaxDD':>7s}"
    )
    print(header)
    print("-" * 150)

    for r in results:
        if r["n_trades"] == 0:
            print(f"{r['label']:>25s}  {'N/A':>7s}")
            continue
        print(
            f"{r['label']:>25s}  {r['n_trades']:>7,}  {r['n_days']:>5}  "
            f"{r['trades_per_day']:>6.1f}  {r['unique_stocks_day']:>6.1f}  "
            f"{r['sig_win_rate']:>5.1%}  {r['sig_day_wr']:>5.1%}  "
            f"{r['sig_net_bps']:>7.1f}  {r['sig_daily_bps']:>7.1f}  "
            f"{r['strat_daily_bps']:>8.2f}  {r['strat_annual_ret']:>7.2%}  "
            f"{r['strat_sharpe']:>7.2f}  {r['strat_max_dd']:>6.2%}"
        )
    print("-" * 150)


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info(
        f"T0 仓位约束回测 (仓位比例={POSITION_RATIO:.0%}, 冷却={COOLDOWN_BARS}bars)"
    )
    logger.info("=" * 60)

    # 加载数据
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(df):,} 行, {df['date'].nunique()} 个交易日")

    # ========== 场景1: 含异常日 ==========
    results_all = []
    for top_n in TOP_N_LIST:
        r = run_t0_backtest(df, top_n, POSITION_RATIO, COOLDOWN_BARS)
        results_all.append(r)

    print_report(results_all, POSITION_RATIO, COOLDOWN_BARS)

    # ========== 场景2: 剔除异常日 ==========
    results_excl = []
    for top_n in TOP_N_LIST:
        r = run_t0_backtest(
            df,
            top_n,
            POSITION_RATIO,
            COOLDOWN_BARS,
            exclude_dates=OUTLIER_DATES,
            label_prefix="excl|",
        )
        results_excl.append(r)

    print_report(results_excl, POSITION_RATIO, COOLDOWN_BARS)

    # ========== 场景3: 不同仓位比例敏感性 ==========
    print("\n" + "=" * 100)
    print("仓位比例敏感性分析 (Top-50, 含异常日)")
    print("=" * 100)
    header = f"{'Ratio':>8s}  {'StratDlyBps':>12s}  {'AnnualRet':>10s}  {'Sharpe':>8s}  {'MaxDD':>8s}"
    print(header)
    print("-" * 100)
    for ratio in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]:
        r = run_t0_backtest(
            df, top_n=50, position_ratio=ratio, cooldown_bars=COOLDOWN_BARS
        )
        if r["n_trades"] > 0:
            print(
                f"{ratio:>7.0%}  {r['strat_daily_bps']:>12.2f}  "
                f"{r['strat_annual_ret']:>9.2%}  {r['strat_sharpe']:>8.2f}  "
                f"{r['strat_max_dd']:>7.2%}"
            )
    print("-" * 100)

    logger.success("T0 仓位约束回测完成!")


if __name__ == "__main__":
    main()
