"""
组合回测诊断脚本

诊断 lgbm_portfolio_backtest.py 的回测结果，包括：
  1. 信号稀疏度分析：每个Bar有多少信号，Top-N实际选了几只
  2. NetBps vs DailyBps 差异分析：笔级均值 vs 日级均值的差异来源
  3. 信号爆炸日 vs 正常日的对比
  4. 单日笔级 vs 日级差异的详细拆解

输入：output/lgbm_test_predictions.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# ==================== 配置（与 lgbm_portfolio_backtest.py 保持一致） ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"

COST_BPS = 15
COST_RATE = COST_BPS / 10000

# 策略约束参数
TOP_N = 5             # 诊断哪个 Top-N
MIN_PROB = 0.45       # 概率门槛
START_TIME = "09:50"  # 最早入场时间
END_TIME = "14:50"    # 最晚入场时间


def load_and_filter() -> pd.DataFrame:
    """加载并筛选信号"""
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(df):,} 行, {df['date'].nunique()} 个交易日")

    mask = (
        (df["entry_time"] >= START_TIME)
        & (df["entry_time"] <= END_TIME)
        & (df["pred_prob"] >= MIN_PROB)
    )
    df_active = df[mask].copy()
    logger.info(
        f"筛选后信号: {len(df_active):,} 笔, "
        f"{df_active['date'].nunique()} 个交易日"
    )
    return df_active


def diagnose_signal_sparsity(df_active: pd.DataFrame):
    """诊断1: 信号稀疏度"""
    print("\n" + "=" * 80)
    print("诊断1: 信号稀疏度分析")
    print("=" * 80)

    bar_signal_count = df_active.groupby(["date", "entry_time"]).size()
    print(f"\n总共 {len(bar_signal_count)} 个 (date, entry_time) Bar 有信号")
    print(f"每个 Bar 的信号数统计:")
    print(bar_signal_count.describe().to_string())

    print(f"\n信号数分布:")
    for threshold in [1, 2, 3, 5, 10, 20, 50, 100]:
        cnt = (bar_signal_count >= threshold).sum()
        print(f"  >= {threshold:>3} 只: {cnt:>4} 个Bar ({cnt/len(bar_signal_count):.1%})")

    # Top-N 筛选
    df_active["rank_in_bar"] = df_active.groupby(["date", "entry_time"])[
        "pred_prob"
    ].rank(ascending=False, method="first")
    portfolio = df_active[df_active["rank_in_bar"] <= TOP_N].copy()

    bar_counts = portfolio.groupby(["date", "entry_time"]).size()
    print(f"\nTop-{TOP_N} 筛选后: {len(portfolio)} 笔交易, {len(bar_counts)} 个Bar")
    print(f"每个Bar实际选股数分布:")
    vc = bar_counts.value_counts().sort_index()
    for n, cnt in vc.items():
        print(f"  选了 {n} 只: {cnt:>4} 个Bar ({cnt/len(bar_counts):.1%})")

    return portfolio


def diagnose_netbps_vs_dailybps(portfolio: pd.DataFrame):
    """诊断2: NetBps vs DailyBps 差异"""
    print("\n" + "=" * 80)
    print("诊断2: NetBps vs DailyBps 差异分析")
    print("=" * 80)

    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE
    portfolio["net_ret_bps"] = portfolio["net_ret"] * 10000

    # ---- 笔级 ----
    avg_net_bps = portfolio["net_ret_bps"].mean()

    # ---- 日级（与回测脚本一致的计算方式） ----
    bar_ret = portfolio.groupby(["date", "entry_time"])["net_ret"].mean()
    daily_ret = bar_ret.groupby("date").mean()
    daily_bps = daily_ret.mean() * 10000

    print(f"\n笔级 NetBps (所有交易简单平均): {avg_net_bps:.2f} bps")
    print(f"日级 DailyBps (Bar均值→日均值→总均值): {daily_bps:.2f} bps")
    print(f"差异: {daily_bps - avg_net_bps:.2f} bps")

    print(f"\n--- 差异原因 ---")
    print(f"笔级: 每笔交易权重相等 → 信号多的天贡献更多权重")
    print(f"日级: 每天权重相等 → 不论当天有1笔还是78笔，权重都是1/N_days")

    # ---- 按天拆解 ----
    daily_stats = portfolio.groupby("date").agg(
        n_trades=("net_ret_bps", "size"),
        trade_mean_bps=("net_ret_bps", "mean"),
    )
    daily_stats["daily_bps"] = daily_ret * 10000

    print(f"\n--- 信号最多的10天: 笔级 vs 日级对比 ---")
    top_days = daily_stats.nlargest(10, "n_trades")
    print(f"{'日期':>12}  {'笔数':>6}  {'笔级NetBps':>12}  {'日级DailyBps':>14}  {'差异':>8}")
    print("-" * 60)
    for d, row in top_days.iterrows():
        diff = row["daily_bps"] - row["trade_mean_bps"]
        print(
            f"{str(d):>12}  {row['n_trades']:>6.0f}  "
            f"{row['trade_mean_bps']:>12.1f}  {row['daily_bps']:>14.1f}  "
            f"{diff:>8.1f}"
        )

    print(f"\n--- 笔级 vs 日级差异的来源拆解 ---")
    print(f"同一天内，笔级和日级可能不等，原因:")
    print(f"  笔级 = 当天所有交易的简单平均")
    print(f"  日级 = 先按Bar等权平均，再对所有Bar取均值")
    print(f"  当同一天不同Bar的交易数量不同时，两者就会不等")
    print(f"  例: Bar_A有1笔(+100bps), Bar_B有10笔(平均-10bps)")
    print(f"    笔级 = (100 + 10*(-10)) / 11 = 0 bps")
    print(f"    日级 = (100 + (-10)) / 2 = 45 bps")

    return daily_stats, daily_ret


def diagnose_explosion_vs_normal(daily_stats: pd.DataFrame, daily_ret: pd.Series):
    """诊断3: 爆炸日 vs 正常日"""
    print("\n" + "=" * 80)
    print("诊断3: 信号爆炸日 vs 正常日")
    print("=" * 80)

    daily_stats["daily_bps"] = daily_ret * 10000
    explosion_mask = daily_stats["n_trades"] > 20

    normal = daily_stats[~explosion_mask]
    explosion = daily_stats[explosion_mask]

    print(f"\n正常日 (<=20笔): {len(normal)} 天")
    print(f"  笔级均值: {normal['trade_mean_bps'].mean():.2f} bps")
    print(f"  日级均值: {normal['daily_bps'].mean():.2f} bps")

    print(f"\n爆炸日 (>20笔): {len(explosion)} 天")
    print(f"  笔级均值: {explosion['trade_mean_bps'].mean():.2f} bps")
    print(f"  日级均值: {explosion['daily_bps'].mean():.2f} bps")

    total_trades = daily_stats["n_trades"].sum()
    explosion_trades = explosion["n_trades"].sum()
    print(f"\n爆炸日总笔数占比: {explosion_trades}/{total_trades} = {explosion_trades/total_trades:.1%}")


def diagnose_dailybps_vs_cost(portfolio: pd.DataFrame, daily_ret: pd.Series):
    """诊断4: DailyBps vs 成本的关系"""
    print("\n" + "=" * 80)
    print("诊断4: DailyBps=12-13 vs 成本=15bps，为什么还能盈利？")
    print("=" * 80)

    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE
    portfolio["net_ret_bps"] = portfolio["net_ret"] * 10000
    portfolio["raw_ret_bps"] = portfolio["raw_ret"] * 10000

    # 毛收益
    avg_raw_bps = portfolio["raw_ret_bps"].mean()
    # 日级毛收益
    bar_raw = portfolio.groupby(["date", "entry_time"])["raw_ret"].mean()
    daily_raw = bar_raw.groupby("date").mean()
    daily_raw_bps = daily_raw.mean() * 10000

    # 日级净收益
    daily_net_bps = daily_ret.mean() * 10000

    print(f"\n--- 收益拆解 ---")
    print(f"笔级毛收益 (GrossBps):  {avg_raw_bps:.2f} bps")
    print(f"交易成本:               {COST_BPS:.0f} bps")
    print(f"笔级净收益 (NetBps):    {avg_raw_bps - COST_BPS:.2f} bps")
    print()
    print(f"日级毛收益:             {daily_raw_bps:.2f} bps")
    print(f"交易成本:               {COST_BPS:.0f} bps")
    print(f"日级净收益 (DailyBps):  {daily_net_bps:.2f} bps")
    print()
    print(f"--- 关键解释 ---")
    print(f"DailyBps = {daily_net_bps:.1f} bps 是【已经扣除成本后】的净收益！")
    print(f"日级毛收益 = {daily_raw_bps:.1f} bps，远大于成本 {COST_BPS} bps")
    print(f"所以策略是盈利的: 毛收益 {daily_raw_bps:.1f} - 成本 {COST_BPS} = 净收益 {daily_net_bps:.1f}")
    print()
    print(f"DailyBps 不是毛收益，不需要 > 15bps 才能盈利")
    print(f"DailyBps > 0 就说明策略在扣除成本后仍然赚钱")

    # 累计收益
    cum_net = daily_ret.cumsum()
    total_ret_bps = cum_net.iloc[-1] * 10000
    n_days = len(daily_ret)
    print(f"\n--- 累计表现 ---")
    print(f"交易天数: {n_days}")
    print(f"日均净收益: {daily_net_bps:.2f} bps")
    print(f"累计净收益: {total_ret_bps:.0f} bps ({total_ret_bps/100:.1f}%)")
    print(f"年化收益估算: {daily_net_bps * 252:.0f} bps ({daily_net_bps * 252 / 100:.1f}%)")


def main():
    logger.info("=" * 60)
    logger.info(f"组合回测诊断 (Top-{TOP_N}, MinProb={MIN_PROB})")
    logger.info("=" * 60)

    # 加载数据
    df_active = load_and_filter()

    # 诊断1: 信号稀疏度
    portfolio = diagnose_signal_sparsity(df_active)

    # 诊断2: NetBps vs DailyBps
    daily_stats, daily_ret = diagnose_netbps_vs_dailybps(portfolio)

    # 诊断3: 爆炸日 vs 正常日
    diagnose_explosion_vs_normal(daily_stats, daily_ret)

    # 诊断4: DailyBps vs 成本
    diagnose_dailybps_vs_cost(portfolio, daily_ret)

    print("\n" + "=" * 80)
    logger.success("诊断完成!")


if __name__ == "__main__":
    main()
