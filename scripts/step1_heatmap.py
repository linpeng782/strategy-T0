#!/usr/bin/env python3
"""
Step 1 (修正版): 物理定律检查 (高波动池限定)
===========================================
加上 "Yesterday Volatility Top 10%" 滤网后，
重新暴力扫描最佳 TP/SL 组合。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt

# 绘图设置
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# 路径配置
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/step1_eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter_data():
    logger.info("加载缓存数据...")
    df = pd.read_pickle(CACHE_DIR / "df_5m.pkl")

    # 1. 计算波动率阈值 (Top 10%)
    # 注意：应该按天计算阈值，还是全局计算？
    # 为了回测严谨，严格来说应该按天 groupby 计算，或者简单点按全局算也行（分布通常稳定）
    # 这里我们采用你代码里的逻辑：全局 quantile(0.9) 作为简单近似，或者更严谨点按日计算

    # 严谨做法：按日期计算当天的 90% 分位阈值
    daily_thresholds = df.groupby("date")["yesterday_range"].quantile(0.9).to_dict()
    df["vol_threshold"] = df["date"].map(daily_thresholds)

    logger.info("应用核心滤网: Z < -1.5 且 昨日波动率 > Top 10% ...")

    mask = (
        (df["Z_final"] < -1.5)  # 超跌
        & (df["Z_final"] > -5.0)  # 别跌太死 (参考你的代码 Z > -5.0)
        & (df["yesterday_range"] > df["vol_threshold"])  # 核心：只做高波动的票
        & (df["close"] < df["day_high"])  # 不追高
        & (df["time_str"] > "09:35")
        & (df["time_str"] < "14:00")
    )

    df_filtered = df[mask].copy()
    logger.info(
        f"筛选后样本数: {len(df_filtered):,} (原始占比 {len(df_filtered)/len(df):.2%})"
    )

    return df_filtered


def simulate_trades(df, tp_bp_list, sl_bp_list):
    results = []

    for tp in tp_bp_list:
        for sl in sl_bp_list:
            tp_ratio = tp / 10000.0
            sl_ratio = sl / 10000.0

            tp_price = df["close"] * (1 + tp_ratio)
            sl_price = df["close"] * (1 - sl_ratio)

            # 判定逻辑
            win_mask = (df["future_high"] >= tp_price) & (df["future_low"] > sl_price)
            win_rate = win_mask.mean()

            # 期望收益 (扣除 15bp 手续费，参考你的配置)
            cost = 15.0
            expectancy = (win_rate * tp - (1 - win_rate) * sl) - cost

            results.append(
                {"TP": tp, "SL": sl, "WinRate": win_rate, "Expectancy": expectancy}
            )

    return pd.DataFrame(results)


def main():
    df = load_and_filter_data()

    if len(df) == 0:
        logger.error("筛选后没有数据！请检查 yesterday_range 是否在数据中。")
        return

    # 扫描范围：既然是高波动票，TP 可以看高一点
    tps = range(40, 150, 10)  # 40 ~ 140bp
    sls = range(30, 100, 10)  # 30 ~ 90bp

    logger.info(f"开始扫描...")
    res_df = simulate_trades(df, tps, sls)

    # 寻找最佳配置
    best = res_df.sort_values("Expectancy", ascending=False).iloc[0]

    print("\n" + "=" * 50)
    print(f"🏆 最佳参数组合 (高波动池 + 扣除15bp成本):")
    print(f"  - 止盈 (TP): {best['TP']:.0f} bp")
    print(f"  - 止损 (SL): {best['SL']:.0f} bp")
    print(f"  - 基础胜率 : {best['WinRate']:.2%}")
    print(f"  - 期望收益 : {best['Expectancy']:.2f} bp/trade (净)")
    print("=" * 50 + "\n")

    # 绘图
    pivot_exp = res_df.pivot(index="TP", columns="SL", values="Expectancy")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_exp, annot=True, fmt=".0f", cmap="RdYlGn", center=0)
    plt.title("Net Expectancy (bp) - High Volatility Only")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_high_vol.png")
    logger.success(f"结果已保存: {OUTPUT_DIR / 'heatmap_high_vol.png'}")


if __name__ == "__main__":
    main()
