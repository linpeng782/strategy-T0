"""
F1: 冲高过滤测试
在 Top-N 选股前，剔除 morning_ret 超过阈值的股票
测试不同阈值对 ExcBps / NetBps / Sharpe 的影响
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ts_portfolio_backtest import COST_BPS, COST_RATE

OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")


def run_filtered_backtest(df, top_n, morning_ret_cap=None, buffer_ratio=1.0):
    """
    带 morning_ret 过滤的 Top-N 回测（含换手控制）

    morning_ret_cap: 过滤阈值，None=不过滤，0.01=剔除早盘涨幅>1%的
    """
    df_active = df.copy()

    # 过滤高 morning_ret
    if morning_ret_cap is not None:
        before = len(df_active)
        df_active = df_active[df_active["morning_ret"] <= morning_ret_cap]
        pct_removed = 1 - len(df_active) / before
    else:
        pct_removed = 0.0

    buffer_n = int(top_n * buffer_ratio)

    # 重新排名（过滤后）
    df_active["rank"] = df_active.groupby("date")["pred_prob"].rank(
        ascending=False, method="first"
    )

    dates = sorted(df_active["date"].unique())
    prev_holdings = set()
    daily_records = []

    for date in dates:
        day_data = df_active[df_active["date"] == date]

        if buffer_ratio > 1.0 and len(prev_holdings) > 0:
            kept_mask = day_data["SecuCode"].isin(prev_holdings) & (
                day_data["rank"] <= buffer_n
            )
            kept_stocks = set(day_data.loc[kept_mask, "SecuCode"])
            if len(kept_stocks) > top_n:
                kept_data = day_data[day_data["SecuCode"].isin(kept_stocks)].nsmallest(
                    top_n, "rank"
                )
                kept_stocks = set(kept_data["SecuCode"])
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
            today_holdings = set(day_data[day_data["rank"] <= top_n]["SecuCode"])
            new_stocks = today_holdings - prev_holdings

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

        daily_records.append({
            "date": date, "raw_ret": raw_mean, "excess_ret": excess_mean,
            "net_ret": net_mean, "turnover": turnover, "n_stocks": n_total,
        })
        prev_holdings = today_holdings

    if not daily_records:
        return None

    dr = pd.DataFrame(daily_records)
    n_days = len(dr)
    ann = 242

    return {
        "avg_raw_bps": dr["raw_ret"].mean() * 10000,
        "avg_net_bps": dr["net_ret"].mean() * 10000,
        "avg_excess_bps": dr["excess_ret"].mean() * 10000,
        "avg_turnover": dr["turnover"].mean(),
        "sharpe_net": dr["net_ret"].mean() / dr["net_ret"].std() * np.sqrt(ann) if dr["net_ret"].std() > 0 else 0,
        "sharpe_excess": dr["excess_ret"].mean() / dr["excess_ret"].std() * np.sqrt(ann) if dr["excess_ret"].std() > 0 else 0,
        "annual_net": dr["net_ret"].mean() * ann * 100,
        "n_days": n_days,
        "pct_removed": pct_removed,
    }


def main():
    # 加载回归模型预测
    df = pd.read_pickle(OUTPUT_DIR / "ts_lgbm_predictions.pkl")
    logger.success(f"加载数据: {len(df):,} 行")

    # morning_ret 分布
    mr = df["morning_ret"]
    print(f"\nmorning_ret 分布:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {mr.quantile(p/100)*100:.2f}%")

    # 测试阈值
    thresholds = [None, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005]
    threshold_labels = ["无过滤", "3%", "2%", "1.5%", "1%", "0.8%", "0.5%"]

    # ===== 表1: 全换手 =====
    print("\n" + "=" * 130)
    print("表1: morning_ret 过滤 + 全换手 (buffer_ratio=1.0)")
    print("=" * 130)

    for top_n in [20, 50, 100]:
        print(f"\n  --- Top-{top_n} ---")
        print(f"  {'阈值':>8s} {'剔除%':>6s} {'RawBps':>8s} {'NetBps':>8s} {'ExcBps':>8s}"
              f" {'换手率':>7s} {'AnnNet%':>8s} {'ShpNet':>7s} {'ShpExc':>7s}")
        print("  " + "-" * 90)

        for th, label in zip(thresholds, threshold_labels):
            r = run_filtered_backtest(df, top_n=top_n, morning_ret_cap=th, buffer_ratio=1.0)
            if r is None:
                continue
            print(f"  {label:>8s} {r['pct_removed']:>5.1%} {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
                  f" {r['avg_excess_bps']:>8.2f} {r['avg_turnover']:>6.1%} {r['annual_net']:>7.1f}%"
                  f" {r['sharpe_net']:>7.2f} {r['sharpe_excess']:>7.2f}")
        print("  " + "-" * 90)

    # ===== 表2: 2x缓冲换手控制 =====
    print("\n\n" + "=" * 130)
    print("表2: morning_ret 过滤 + 2x缓冲换手控制")
    print("=" * 130)

    for top_n in [20, 50, 100]:
        print(f"\n  --- Top-{top_n} ---")
        print(f"  {'阈值':>8s} {'剔除%':>6s} {'RawBps':>8s} {'NetBps':>8s} {'ExcBps':>8s}"
              f" {'换手率':>7s} {'AnnNet%':>8s} {'ShpNet':>7s} {'ShpExc':>7s}")
        print("  " + "-" * 90)

        for th, label in zip(thresholds, threshold_labels):
            r = run_filtered_backtest(df, top_n=top_n, morning_ret_cap=th, buffer_ratio=2.0)
            if r is None:
                continue
            print(f"  {label:>8s} {r['pct_removed']:>5.1%} {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
                  f" {r['avg_excess_bps']:>8.2f} {r['avg_turnover']:>6.1%} {r['annual_net']:>7.1f}%"
                  f" {r['sharpe_net']:>7.2f} {r['sharpe_excess']:>7.2f}")
        print("  " + "-" * 90)

    # ===== 最佳配置 close_ret 测试 =====
    print("\n\n" + "=" * 130)
    print("表3: 最佳过滤阈值下的 VWAP vs Close 离场对比")
    print("=" * 130)

    # 加载含close的数据
    df_close_data = pd.read_pickle(OUTPUT_DIR / "ts_lgbm_predictions_with_close.pkl")

    for th, label in [(None, "无过滤"), (0.015, "1.5%"), (0.01, "1%")]:
        print(f"\n  --- 过滤阈值: {label} ---")
        # VWAP
        r_vwap = run_filtered_backtest(df, top_n=50, morning_ret_cap=th, buffer_ratio=2.0)
        # Close: 替换收益
        df_c = df_close_data.copy()
        close_market = df_c.groupby("date")["close_ret"].transform("mean")
        df_c["raw_ret"] = df_c["close_ret"]
        df_c["excess_ret"] = df_c["close_ret"] - close_market
        r_close = run_filtered_backtest(df_c, top_n=50, morning_ret_cap=th, buffer_ratio=2.0)

        if r_vwap and r_close:
            print(f"    VWAP离场:  Net={r_vwap['avg_net_bps']:>6.2f}bps, Exc={r_vwap['avg_excess_bps']:>6.2f}bps, Shp={r_vwap['sharpe_net']:.2f}")
            print(f"    Close离场: Net={r_close['avg_net_bps']:>6.2f}bps, Exc={r_close['avg_excess_bps']:>6.2f}bps, Shp={r_close['sharpe_net']:.2f}")
            print(f"    差(VWAP-Close): Net={r_vwap['avg_net_bps']-r_close['avg_net_bps']:>+6.2f}bps")

    logger.success("F1 冲高过滤测试完成!")


if __name__ == "__main__":
    main()
