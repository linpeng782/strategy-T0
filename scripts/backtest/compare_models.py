"""
多模型对比回测脚本

对比 V2-Regression / S2-Huber / S4-LambdaRank 三个模型
在不同 Top-N 和换手控制策略下的表现

输入：output/ 下的三个预测文件
输出：终端对比报告

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# 复用回测函数
sys.path.insert(0, str(Path(__file__).parent))
from ts_portfolio_backtest import (
    run_topn_backtest,
    run_topn_backtest_turnover,
    COST_BPS,
    COST_RATE,
)

OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

# 模型配置
MODELS = {
    "V2-Reg(MSE)": OUTPUT_DIR / "ts_lgbm_predictions.pkl",
    "S2-Huber": OUTPUT_DIR / "ts_lgbm_predictions_huber.pkl",
    "S4-LambdaRank": OUTPUT_DIR / "ts_lgbm_predictions_lambdarank.pkl",
}

TOP_N_LIST = [20, 50, 100, 200]


def main():
    print("\n" + "=" * 140)
    print("多模型对比回测 | 成本: 15bps | Top-N + 换手控制")
    print("=" * 140)

    # 加载所有预测
    preds = {}
    for name, path in MODELS.items():
        if path.exists():
            preds[name] = pd.read_pickle(path)
            logger.success(f"加载 {name}: {len(preds[name]):,} 行")
        else:
            logger.warning(f"跳过 {name}: 文件不存在 {path}")

    # ===== 表1: 原版回测（全换手）=====
    print("\n\n" + "=" * 140)
    print("表1: 原版回测（全换手，每日100%换手率，成本15bps/天）")
    print("=" * 140)

    for top_n in TOP_N_LIST:
        print(f"\n  --- Top-{top_n} ---")
        print(
            f"  {'模型':>18} {'RawBps':>8} {'NetBps':>8} {'ExcBps':>8}"
            f" {'AnnNet%':>8} {'ShpNet':>7} {'ShpExc':>7} {'DayWR':>6}"
        )
        print("  " + "-" * 80)
        for name, df in preds.items():
            r = run_topn_backtest(df, top_n=top_n)
            if r["n_trades"] == 0:
                continue
            print(
                f"  {name:>18} {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
                f" {r['avg_excess_bps']:>8.2f}"
                f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
                f" {r['sharpe_excess']:>7.2f} {r['day_win_rate']:>5.1%}"
            )
        print("  " + "-" * 80)

    # ===== 表2: 换手控制版（无缓冲）=====
    print("\n\n" + "=" * 140)
    print("表2: 换手控制（无缓冲，重叠持仓免费）")
    print("=" * 140)

    for top_n in TOP_N_LIST:
        print(f"\n  --- Top-{top_n} ---")
        print(
            f"  {'模型':>18} {'RawBps':>8} {'NetBps':>8} {'ExcBps':>8}"
            f" {'Turnov':>7} {'CostBps':>8}"
            f" {'AnnNet%':>8} {'ShpNet':>7} {'ShpExc':>7}"
        )
        print("  " + "-" * 95)
        for name, df in preds.items():
            r = run_topn_backtest_turnover(df, top_n=top_n, buffer_ratio=1.0)
            if r["n_trades"] == 0:
                continue
            print(
                f"  {name:>18} {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
                f" {r['avg_excess_bps']:>8.2f}"
                f" {r['avg_turnover']:>6.1%} {r['avg_cost_bps']:>8.2f}"
                f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
                f" {r['sharpe_excess']:>7.2f}"
            )
        print("  " + "-" * 95)

    # ===== 表3: 换手控制版（2x缓冲）=====
    print("\n\n" + "=" * 140)
    print("表3: 换手控制（2x缓冲，昨日持仓在Top-2N内保留）")
    print("=" * 140)

    for top_n in TOP_N_LIST:
        print(f"\n  --- Top-{top_n} ---")
        print(
            f"  {'模型':>18} {'RawBps':>8} {'NetBps':>8} {'ExcBps':>8}"
            f" {'Turnov':>7} {'CostBps':>8}"
            f" {'AnnNet%':>8} {'ShpNet':>7} {'ShpExc':>7}"
        )
        print("  " + "-" * 95)
        for name, df in preds.items():
            r = run_topn_backtest_turnover(df, top_n=top_n, buffer_ratio=2.0)
            if r["n_trades"] == 0:
                continue
            print(
                f"  {name:>18} {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
                f" {r['avg_excess_bps']:>8.2f}"
                f" {r['avg_turnover']:>6.1%} {r['avg_cost_bps']:>8.2f}"
                f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
                f" {r['sharpe_excess']:>7.2f}"
            )
        print("  " + "-" * 95)

    # ===== 总结: Top-50 三模型 × 三策略 =====
    print("\n\n" + "=" * 140)
    print("总结: Top-50 三模型 × 三策略")
    print("=" * 140)
    print(
        f"  {'模型':>18} {'策略':>18}"
        f" {'RawBps':>8} {'NetBps':>8} {'ExcBps':>8}"
        f" {'Turnov':>7} {'CostBps':>8}"
        f" {'AnnNet%':>8} {'ShpNet':>7}"
    )
    print("  " + "-" * 110)

    for name, df in preds.items():
        # 原版
        r = run_topn_backtest(df, top_n=50)
        print(
            f"  {name:>18} {'全换手':>18}"
            f" {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
            f" {r['avg_excess_bps']:>8.2f}"
            f" {'100.0%':>7} {COST_BPS:>8.2f}"
            f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
        )
        # 无缓冲
        r = run_topn_backtest_turnover(df, top_n=50, buffer_ratio=1.0)
        print(
            f"  {'':>18} {'换手控制(无缓冲)':>18}"
            f" {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
            f" {r['avg_excess_bps']:>8.2f}"
            f" {r['avg_turnover']:>6.1%} {r['avg_cost_bps']:>8.2f}"
            f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
        )
        # 2x缓冲
        r = run_topn_backtest_turnover(df, top_n=50, buffer_ratio=2.0)
        print(
            f"  {'':>18} {'换手控制(2x缓冲)':>18}"
            f" {r['avg_raw_bps']:>8.2f} {r['avg_net_bps']:>8.2f}"
            f" {r['avg_excess_bps']:>8.2f}"
            f" {r['avg_turnover']:>6.1%} {r['avg_cost_bps']:>8.2f}"
            f" {r['annual_net']:>7.1f}% {r['sharpe_net']:>7.2f}"
        )
        print("  " + "." * 110)

    print("  " + "-" * 110)
    logger.success("多模型对比回测完成!")


if __name__ == "__main__":
    main()
