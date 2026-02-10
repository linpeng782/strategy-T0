"""
信号重叠检查

诊断同一天同一只股票是否重复发信号，导致回测收益虚高。

检查内容：
  1. 每天每只股票发信号次数分布
  2. 信号持有期重叠度（120分钟 = 24个bar，连续信号大量重叠）
  3. 对比：原始 vs 去重（冷却期约束）后的回测表现

输入：output/lgbm_test_predictions.pkl
输出：终端诊断报告

作者：量化研究
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# ==================== 配置 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"

COST_BPS = 15
COST_RATE = COST_BPS / 10000
MIN_PROB = 0.45
START_TIME = "09:50"
END_TIME = "14:50"
TOP_N = 50  # 与 portfolio backtest 一致
HOLD_BARS = 24  # 120分钟 / 5分钟
COOLDOWN_BARS = 24  # 冷却期 = 持有期（同一股票发信号后，24个bar内不再重复）


def load_and_filter() -> pd.DataFrame:
    """加载并初筛信号"""
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(df):,} 行")

    mask = (
        (df["entry_time"] >= START_TIME)
        & (df["entry_time"] <= END_TIME)
        & (df["pred_prob"] >= MIN_PROB)
    )
    signals = df[mask].copy()
    logger.info(f"筛选后信号: {len(signals):,} 笔 (prob>={MIN_PROB}, {START_TIME}~{END_TIME})")
    return signals


def time_to_bar_index(t: str) -> int:
    """将 entry_time 转为 bar 序号（09:35=0, 09:40=1, ...）"""
    h, m = map(int, t.split(":"))
    total_min = h * 60 + m
    # 09:35 开始，每5分钟一个bar
    # 上午: 09:35 ~ 11:30 (24 bars: 0~23)
    # 下午: 13:00 ~ 14:55 (24 bars: 24~47)
    if total_min <= 11 * 60 + 30:
        return (total_min - 9 * 60 - 35) // 5
    else:
        return 24 + (total_min - 13 * 60) // 5


def diagnose_overlap(signals: pd.DataFrame):
    """诊断信号重叠情况"""
    print("\n" + "=" * 100)
    print("诊断1: 每天每只股票信号次数分布")
    print("=" * 100)

    # 每天每只股票的信号数量
    sig_counts = signals.groupby(["date", "SecuCode"]).size().reset_index(name="n_signals")

    print(f"\n  总(date, stock)对数: {len(sig_counts):,}")
    print(f"  信号次数分布:")
    dist = sig_counts["n_signals"].value_counts().sort_index()
    total_pairs = len(sig_counts)
    for n, cnt in dist.items():
        pct = cnt / total_pairs * 100
        print(f"    发了 {n:>2} 次信号: {cnt:>6,} 对 ({pct:>5.1f}%)")

    multi = sig_counts[sig_counts["n_signals"] > 1]
    print(f"\n  重复发信号的(date,stock): {len(multi):,} / {total_pairs:,} = {len(multi)/total_pairs:.1%}")
    print(f"  重复信号贡献的交易笔数: {multi['n_signals'].sum() - len(multi):,} (冗余)")
    print(f"  去重后独立(date,stock)对: {total_pairs:,}")
    print(f"  当前总信号数: {len(signals):,}")
    print(f"  去重后最多保留: {total_pairs:,} (每只股票每天最多1次)")
    redundancy = 1 - total_pairs / len(signals)
    print(f"  信号冗余率: {redundancy:.1%}")

    # 热门重复案例
    print(f"\n  重复次数最多的案例 (Top 10):")
    top_multi = sig_counts.nlargest(10, "n_signals")
    for _, row in top_multi.iterrows():
        print(f"    {row['date']} | {row['SecuCode']} | {row['n_signals']} 次")


def apply_cooldown(signals: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """应用冷却期去重：同一股票在 cooldown_bars 个bar内不再重复发信号"""
    signals = signals.sort_values(["SecuCode", "date", "entry_time"]).copy()
    signals["bar_idx"] = signals["entry_time"].apply(time_to_bar_index)

    keep_mask = []
    last_signal = {}  # (SecuCode, date) -> last bar_idx

    for idx, row in signals.iterrows():
        key = (row["SecuCode"], row["date"])
        bar = row["bar_idx"]

        if key not in last_signal or (bar - last_signal[key]) >= cooldown_bars:
            keep_mask.append(True)
            last_signal[key] = bar
        else:
            keep_mask.append(False)

    deduped = signals[keep_mask].copy()
    logger.info(
        f"冷却期去重: {len(signals):,} → {len(deduped):,} "
        f"(去除 {len(signals)-len(deduped):,} 笔, 冷却={cooldown_bars}bars={cooldown_bars*5}min)"
    )
    return deduped


def apply_cooldown_vectorized(signals: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """向量化版冷却期去重（快速）"""
    signals = signals.sort_values(["SecuCode", "date", "entry_time"]).copy()
    signals["bar_idx"] = signals["entry_time"].apply(time_to_bar_index)

    # 同一(SecuCode, date)内，计算与前一个信号的bar间距
    grp = signals.groupby(["SecuCode", "date"])["bar_idx"]
    signals["bar_diff"] = grp.diff()

    # 第一个信号保留；后续信号只有间距 >= cooldown 才保留
    # 但这只处理了相邻信号的情况，连续密集信号需要迭代
    # 用 groupby + apply 更准确
    def filter_group(g):
        g = g.sort_values("bar_idx")
        keep = [True]  # 第一个总是保留
        last_bar = g.iloc[0]["bar_idx"]
        for i in range(1, len(g)):
            if g.iloc[i]["bar_idx"] - last_bar >= cooldown_bars:
                keep.append(True)
                last_bar = g.iloc[i]["bar_idx"]
            else:
                keep.append(False)
        return pd.Series(keep, index=g.index)

    keep_mask = signals.groupby(["SecuCode", "date"], group_keys=False).apply(filter_group)
    deduped = signals[keep_mask].copy()
    logger.info(
        f"冷却期去重: {len(signals):,} → {len(deduped):,} "
        f"(去除 {len(signals)-len(deduped):,} 笔, 冷却={cooldown_bars}bars={cooldown_bars*5}min)"
    )
    return deduped


def run_backtest(signals: pd.DataFrame, label: str, top_n: int = TOP_N) -> dict:
    """简易回测（与 portfolio backtest 逻辑一致）"""
    # Top-N 筛选
    signals["rank_in_bar"] = signals.groupby(["date", "entry_time"])[
        "pred_prob"
    ].rank(ascending=False, method="first")
    portfolio = signals[signals["rank_in_bar"] <= top_n].copy()

    portfolio["net_ret"] = portfolio["raw_ret"] - COST_RATE

    n_trades = len(portfolio)
    if n_trades == 0:
        return {"label": label, "n_trades": 0}

    bar_ret = portfolio.groupby(["date", "entry_time"])["net_ret"].mean()
    daily_ret = bar_ret.groupby("date").mean()
    n_days = len(daily_ret)

    win_rate = (portfolio["net_ret"] > 0).mean()
    day_wr = (daily_ret > 0).mean()
    avg_net_bps = portfolio["net_ret"].mean() * 10000
    daily_bps = daily_ret.mean() * 10000
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    cum = daily_ret.cumsum()
    max_dd = (cum - cum.cummax()).min()

    # 每天独立(date,stock)对数
    n_unique_stocks_day = portfolio.groupby("date")["SecuCode"].nunique().mean()

    return {
        "label": label,
        "n_trades": n_trades,
        "n_days": n_days,
        "trades_per_day": n_trades / n_days,
        "unique_stocks_per_day": n_unique_stocks_day,
        "win_rate": win_rate,
        "day_wr": day_wr,
        "avg_net_bps": avg_net_bps,
        "daily_bps": daily_bps,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "daily_ret": daily_ret,
    }


def print_comparison(results: list):
    """打印对比报告"""
    print("\n" + "=" * 130)
    print(f"诊断2: 去重前后回测对比 (Top-{TOP_N})")
    print("=" * 130)

    header = (
        f"{'Label':>30s}  {'Trades':>8s}  {'Days':>5s}  {'T/Day':>7s}  {'Stk/Day':>7s}  "
        f"{'WR':>7s}  {'DayWR':>7s}  {'NetBps':>8s}  {'DlyBps':>8s}  "
        f"{'Sharpe':>7s}  {'MaxDD':>8s}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        if r["n_trades"] == 0:
            print(f"{r['label']:>30s}  {'N/A':>8s}")
            continue
        print(
            f"{r['label']:>30s}  {r['n_trades']:>8,}  {r['n_days']:>5}  "
            f"{r['trades_per_day']:>7.1f}  {r['unique_stocks_per_day']:>7.1f}  "
            f"{r['win_rate']:>6.1%}  {r['day_wr']:>6.1%}  "
            f"{r['avg_net_bps']:>8.1f}  {r['daily_bps']:>8.1f}  "
            f"{r['sharpe']:>7.2f}  {r['max_dd']:>7.2%}"
        )
    print("-" * 130)


def main():
    logger.info("=" * 60)
    logger.info("信号重叠诊断")
    logger.info("=" * 60)

    signals = load_and_filter()

    # 诊断1: 重叠情况
    diagnose_overlap(signals)

    # 诊断2: 对比不同冷却期
    results = []

    # 原始（无去重）
    r_original = run_backtest(signals, "Original (no dedup)")
    results.append(r_original)

    # 冷却期 = 6bars (30min)
    deduped_6 = apply_cooldown_vectorized(signals, cooldown_bars=6)
    r_6 = run_backtest(deduped_6, "Cooldown=30min (6bars)")
    results.append(r_6)

    # 冷却期 = 12bars (60min)
    deduped_12 = apply_cooldown_vectorized(signals, cooldown_bars=12)
    r_12 = run_backtest(deduped_12, "Cooldown=60min (12bars)")
    results.append(r_12)

    # 冷却期 = 24bars (120min = 持有期)
    deduped_24 = apply_cooldown_vectorized(signals, cooldown_bars=24)
    r_24 = run_backtest(deduped_24, "Cooldown=120min (24bars)")
    results.append(r_24)

    # 极端去重：每天每只股票只保留第一个信号
    first_only = signals.sort_values("entry_time").drop_duplicates(
        subset=["date", "SecuCode"], keep="first"
    )
    r_first = run_backtest(first_only, "First signal only")
    results.append(r_first)

    print_comparison(results)

    logger.success("信号重叠诊断完成!")


if __name__ == "__main__":
    main()
