#!/usr/bin/env python3
"""
Step 2 (严谨版): 特征有效性验证 (Alpha Check)
===========================================
严格使用 "每日" 波动率 Top 10% 滤网，避免未来函数。
固定 TP=80bp / SL=40bp，观察核心特征区分度。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# 绘图设置
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# 路径配置
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/step2_eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_TO_CHECK = [
 "mkt_oversold_ratio",  # 大盘超跌占比
    "mkt_avg_x1",  # 大盘价格偏离
    "mkt_avg_z",  # 大盘引力中枢
    "mkt_ret_15m",  # 大盘15分钟动量
    "time_val",  # 时间特征
    "yesterday_range",  # 昨日波动率
    "X1_zscore",  # 个股价格偏离
    "vol_burst",  # 成交量爆发力
    "x1_slope",  # 价格动量斜率
    "Z_final",  # 个股引力
    "relative_z",  # 个股相对强度（独立超跌）
    "volatility_ratio",  # 波动率放大比
]


def load_and_label_data_strict():
    logger.info("加载缓存数据...")
    df = pd.read_pickle(CACHE_DIR / "df_5m.pkl")

    # 确保有 date 列
    if "date" not in df.columns:
        df["date"] = df["bar_time"].dt.date

    # ==========================================
    # 核心改动：严格按日期计算每日阈值 (Daily Cross-Section)
    # ==========================================
    logger.info("正在计算每日波动率阈值 (Top 10%)...")

    # 1. 计算每一天的 90% 分位数
    daily_thresholds = df.groupby("date")["yesterday_range"].quantile(0.9)

    # 2. 将阈值映射回原数据表
    df["vol_threshold"] = df["date"].map(daily_thresholds)

    # 3. 检查是否有 NaN (防呆设计)
    if df["vol_threshold"].isna().any():
        logger.warning("注意：部分日期无法计算阈值，将被剔除")
        df = df.dropna(subset=["vol_threshold"])

    logger.info("应用严谨滤网...")
    mask = (
        (df["Z_final"] < -1.5)
        & (df["Z_final"] > -5.0)
        & (df["yesterday_range"] > df["vol_threshold"])  # 使用每日动态阈值
        & (df["close"] < df["day_high"])
        & (df["time_str"] > "09:35")
        & (df["time_str"] < "14:00")
    )

    df_filtered = df[mask].copy()
    logger.info(f"筛选样本数: {len(df_filtered):,}")

    # 打标 (TP=80bp, SL=40bp)
    tp_ratio = 80 / 10000.0
    sl_ratio = 40 / 10000.0

    tp_price = df_filtered["close"] * (1 + tp_ratio)
    sl_price = df_filtered["close"] * (1 - sl_ratio)

    df_filtered["target"] = (df_filtered["future_high"] >= tp_price) & (
        df_filtered["future_low"] > sl_price
    )

    win_rate = df_filtered["target"].mean()
    logger.info(f"严谨滤网下的基础胜率: {win_rate:.2%}")

    return df_filtered


def plot_feature_separation(df):
    n_features = len(FEATURES_TO_CHECK)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(FEATURES_TO_CHECK):
        ax = axes[i]
        if feature not in df.columns:
            continue

        data_win = df[df["target"] == True][feature]
        data_loss = df[df["target"] == False][feature]

        # 简单清洗极端值
        q_low, q_high = df[feature].quantile([0.01, 0.99])
        data_win = data_win.clip(q_low, q_high)
        data_loss = data_loss.clip(q_low, q_high)

        sns.kdeplot(data_win, ax=ax, color="red", fill=True, alpha=0.3, label="Win")
        sns.kdeplot(data_loss, ax=ax, color="blue", fill=True, alpha=0.3, label="Loss")

        ic = df[feature].corr(df["target"].astype(int))
        ax.set_title(f"{feature} (IC={ic:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "feature_separation_strict.png"
    plt.savefig(save_path)
    logger.success(f"特征分布图已保存: {save_path}")


if __name__ == "__main__":
    df = load_and_label_data_strict()
    plot_feature_separation(df)
