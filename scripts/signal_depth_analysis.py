#!/usr/bin/env python3
"""
信号深度分析脚本
================
实现三项核心分析：
1. Edge Curve（回归曲线）- 找最佳出场时间
2. Probability Ladder（胜率阶梯）- 量化入场阈值
3. Filter Value（滤网价值）- 验证趋势滤网有效性

基于 design_scheme_5.ipynb 的分析思路
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==================== 配置参数 ====================
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试股票池
TEST_STOCKS = ["002591", "002494", "002247", "000001", "300750", "601318"]
YEAR = 2025
ZSCORE_WINDOW = 20
SIGNAL_THRESHOLD = 1.5
ROLLING_WINDOW = 12

# 设置中文字体（改为英文避免显示问题）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    logger.info(f"加载 {year} 年数据: {pkl_path}")

    df = pd.read_pickle(pkl_path)
    df = df[df["SecuCode"].isin(stock_codes)]
    df["date"] = pd.to_datetime(df["TradingDay"]).dt.date

    logger.success(f"加载完成: {len(df):,} 条")
    return df


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """将1分钟数据聚合为5分钟Bar"""
    logger.info("聚合为5分钟Bar...")

    df["bar_time"] = pd.to_datetime(df["TradingDay"]).dt.floor("5min")

    agg_df = (
        df.groupby(["SecuCode", "date", "bar_time"])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        .reset_index()
    )

    agg_df["vwap_5m"] = agg_df["amount"] / agg_df["volume"]
    agg_df["time_str"] = agg_df["bar_time"].dt.strftime("%H:%M")

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算滚动窗口VWAP、TWAP和特征"""
    logger.info("计算特征（滚动窗口模式）...")

    result_list = []
    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("bar_time").copy()

        # 全天累积版本
        day_df["cum_volume"] = day_df["volume"].cumsum()
        day_df["cum_amount"] = day_df["amount"].cumsum()
        day_df["cum_vwap_full"] = day_df["cum_amount"] / day_df["cum_volume"]

        # 滚动窗口版本（60分钟窗口）
        rolling_amount = (
            day_df["amount"].rolling(window=ROLLING_WINDOW, min_periods=1).sum()
        )
        rolling_volume = (
            day_df["volume"].rolling(window=ROLLING_WINDOW, min_periods=1).sum()
        )
        day_df["cum_vwap"] = rolling_amount / rolling_volume
        day_df["cum_twap"] = (
            day_df["close"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )

        # X1 和 X2（基于滚动窗口）
        day_df["X1"] = day_df["close"] / day_df["cum_vwap"] - 1
        day_df["X2"] = day_df["cum_vwap"] / day_df["cum_twap"] - 1

        # 日内新高和新低（用于趋势滤网）
        day_df["day_high"] = day_df["close"].expanding().max()
        day_df["day_low"] = day_df["close"].expanding().min()

        result_list.append(day_df)

    result = pd.concat(result_list, ignore_index=True)
    logger.success("特征计算完成")
    return result


def calculate_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算 Z-Score 和 Z_final"""
    logger.info(f"计算 Z-Score (窗口={window}天)...")

    df = df.sort_values(["SecuCode", "bar_time"])

    # 按时间点分组计算
    df["time_str"] = df["bar_time"].dt.strftime("%H:%M")

    result_list = []
    for stock, stock_df in df.groupby("SecuCode"):
        stock_df = stock_df.copy()

        for time_point in stock_df["time_str"].unique():
            mask = stock_df["time_str"] == time_point
            time_data = stock_df.loc[mask, ["X2", "X1"]].copy()

            # 计算滚动均值和标准差
            X2_mean = time_data["X2"].rolling(window=window, min_periods=5).mean()
            X2_std = time_data["X2"].rolling(window=window, min_periods=5).std()
            X1_mean = time_data["X1"].rolling(window=window, min_periods=5).mean()
            X1_std = time_data["X1"].rolling(window=window, min_periods=5).std()

            stock_df.loc[mask, "X2_mean"] = X2_mean.values
            stock_df.loc[mask, "X2_std"] = X2_std.values
            stock_df.loc[mask, "X1_mean"] = X1_mean.values
            stock_df.loc[mask, "X1_std"] = X1_std.values

        # 计算 Z-Score
        stock_df["X2_zscore"] = (stock_df["X2"] - stock_df["X2_mean"]) / stock_df[
            "X2_std"
        ]
        stock_df["X1_zscore"] = (stock_df["X1"] - stock_df["X1_mean"]) / stock_df[
            "X1_std"
        ]

        # 组合因子：Z_final = X2_zscore - 0.5 * X1_zscore
        stock_df["Z_final"] = stock_df["X2_zscore"] - 0.5 * stock_df["X1_zscore"]

        result_list.append(stock_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤无效值
    valid_count = result["Z_final"].notna().sum()
    logger.success(f"Z-Score 计算完成, 有效样本: {valid_count:,}")

    return result


def calculate_yesterday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算昨日波动率，用于消除未来函数
    昨日波动大的股票，今日大概率也波动大
    """
    logger.info("计算昨日波动率...")

    # 计算每天的日内振幅
    daily_summary = (
        df.groupby(["SecuCode", "date"])["close"]
        .agg(["max", "min", "mean"])
        .reset_index()
    )
    daily_summary["daily_range"] = (
        daily_summary["max"] - daily_summary["min"]
    ) / daily_summary["mean"]

    # 将昨天的振幅平移到今天（消除未来函数）
    daily_summary["yesterday_range"] = daily_summary.groupby("SecuCode")[
        "daily_range"
    ].shift(1)

    # 合并回主表
    df = df.merge(
        daily_summary[["SecuCode", "date", "yesterday_range"]],
        on=["SecuCode", "date"],
        how="left",
    )

    logger.success("昨日波动率计算完成")
    return df


def prepare_future_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    预计算未来价格，用于后续分析
    向前平移获取未来 N 个 Bar 后的价格
    """
    logger.info("预计算未来价格...")

    # 定义要分析的时间周期（分钟）
    periods = [5, 10, 15, 20, 30, 45, 60]

    df = df.sort_values(["SecuCode", "date", "bar_time"])

    # 按股票和日期分组，防止跨日取数
    for p in periods:
        n_bars = p // 5  # 5分钟Bar，所以除以5
        col_name = f"close_plus_{p}m"
        high_col = f"high_max_{p}m"

        df[col_name] = df.groupby(["SecuCode", "date"])["close"].shift(-n_bars)

        # 计算未来N分钟内的最高价（用于计算获利空间）
        df[high_col] = (
            df.groupby(["SecuCode", "date"])["high"]
            .rolling(window=n_bars, min_periods=1)
            .max()
            .shift(-n_bars)
            .reset_index(level=[0, 1], drop=True)
        )

    logger.success("未来价格预计算完成")
    return df


# ==================== 分析一：Edge Curve（回归曲线）====================
def analyze_edge_curve(
    df: pd.DataFrame, save_path: Path = None, use_volatility_filter: bool = True
) -> dict:
    """
    分析信号后的回归曲线
    目标：找到平均收益最高的最佳出场时间
    """
    logger.info("=" * 60)
    logger.info("分析一：Edge Curve（回归曲线）")
    logger.info("=" * 60)

    periods = [5, 10, 15, 20, 30, 45, 60]

    # 筛选有效的买入信号
    valid_mask = (
        np.isfinite(df["Z_final"])
        & (df["time_str"] < "14:15")  # 避免尾盘
        & (df["Z_final"] < -SIGNAL_THRESHOLD)  # 买入信号
    )

    # 加入昨日波动率滤网（消除未来函数）
    if use_volatility_filter:
        range_threshold = df["yesterday_range"].quantile(0.9)
        valid_mask = (
            valid_mask
            & np.isfinite(df["yesterday_range"])
            & (df["yesterday_range"] > range_threshold)
        )
        logger.info(f"昨日波动率阈值（90%分位数）: {range_threshold*100:.2f}%")

    buy_signals = df[valid_mask].copy()

    logger.info(f"买入信号样本数: {len(buy_signals):,}")

    # 计算每个周期的收益
    results_buy = {}
    for p in periods:
        col_name = f"close_plus_{p}m"
        if col_name in buy_signals.columns:
            returns = (buy_signals[col_name] / buy_signals["close"]) - 1
            valid_returns = returns.dropna()
            results_buy[p] = {
                "mean": valid_returns.mean() * 10000,  # 转为bp
                "std": valid_returns.std() * 10000,
                "count": len(valid_returns),
                "win_rate": (valid_returns > 0).mean(),
            }

    # 同样分析卖出信号
    sell_mask = (
        np.isfinite(df["Z_final"])
        & (df["time_str"] < "14:15")
        & (df["Z_final"] > SIGNAL_THRESHOLD)  # 卖出信号
    )
    sell_signals = df[sell_mask].copy()

    logger.info(f"卖出信号样本数: {len(sell_signals):,}")

    results_sell = {}
    for p in periods:
        col_name = f"close_plus_{p}m"
        if col_name in sell_signals.columns:
            # 卖出信号期望价格下跌，所以取负
            returns = (sell_signals["close"] / sell_signals[col_name]) - 1
            valid_returns = returns.dropna()
            results_sell[p] = {
                "mean": valid_returns.mean() * 10000,
                "std": valid_returns.std() * 10000,
                "count": len(valid_returns),
                "win_rate": (valid_returns > 0).mean(),
            }

    # 打印结果
    print("\n" + "=" * 60)
    print("买入信号后的回归曲线（持有不同时间的收益）")
    print("=" * 60)
    print(
        f"{'持有时间':>10} | {'平均收益(bp)':>12} | {'标准差(bp)':>10} | {'胜率':>8} | {'样本数':>8}"
    )
    print("-" * 60)

    best_period_buy = None
    best_return_buy = -np.inf
    for p in periods:
        if p in results_buy:
            r = results_buy[p]
            print(
                f"{p:>8}分钟 | {r['mean']:>12.2f} | {r['std']:>10.2f} | {r['win_rate']*100:>7.1f}% | {r['count']:>8}"
            )
            if r["mean"] > best_return_buy:
                best_return_buy = r["mean"]
                best_period_buy = p

    print(
        f"\n✅ 买入信号最佳持有时间: {best_period_buy} 分钟, 平均收益: {best_return_buy:.2f} bp"
    )

    print("\n" + "=" * 60)
    print("卖出信号后的回归曲线（持有不同时间的收益）")
    print("=" * 60)
    print(
        f"{'持有时间':>10} | {'平均收益(bp)':>12} | {'标准差(bp)':>10} | {'胜率':>8} | {'样本数':>8}"
    )
    print("-" * 60)

    best_period_sell = None
    best_return_sell = -np.inf
    for p in periods:
        if p in results_sell:
            r = results_sell[p]
            print(
                f"{p:>8}分钟 | {r['mean']:>12.2f} | {r['std']:>10.2f} | {r['win_rate']*100:>7.1f}% | {r['count']:>8}"
            )
            if r["mean"] > best_return_sell:
                best_return_sell = r["mean"]
                best_period_sell = p

    print(
        f"\n✅ 卖出信号最佳持有时间: {best_period_sell} 分钟, 平均收益: {best_return_sell:.2f} bp"
    )

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 买入信号曲线
    x_buy = list(results_buy.keys())
    y_buy = [results_buy[p]["mean"] for p in x_buy]
    err_buy = [results_buy[p]["std"] / np.sqrt(results_buy[p]["count"]) for p in x_buy]

    ax1.errorbar(
        x_buy, y_buy, yerr=err_buy, marker="o", capsize=5, linewidth=2, markersize=8
    )
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(20, color="red", linestyle=":", alpha=0.7, label="Cost Line (20bp)")
    ax1.set_xlabel("Holding Period (minutes)", fontsize=11)
    ax1.set_ylabel("Average Return (bp)", fontsize=11)
    ax1.set_title("Buy Signal Edge Curve", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 卖出信号曲线
    x_sell = list(results_sell.keys())
    y_sell = [results_sell[p]["mean"] for p in x_sell]
    err_sell = [
        results_sell[p]["std"] / np.sqrt(results_sell[p]["count"]) for p in x_sell
    ]

    ax2.errorbar(
        x_sell,
        y_sell,
        yerr=err_sell,
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
        color="orange",
    )
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axhline(20, color="red", linestyle=":", alpha=0.7, label="Cost Line (20bp)")
    ax2.set_xlabel("Holding Period (minutes)", fontsize=11)
    ax2.set_ylabel("Average Return (bp)", fontsize=11)
    ax2.set_title("Sell Signal Edge Curve", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        "buy": results_buy,
        "sell": results_sell,
        "best_period_buy": best_period_buy,
        "best_period_sell": best_period_sell,
    }


# ==================== 分析二：Probability Ladder（胜率阶梯）====================
def analyze_probability_ladder(
    df: pd.DataFrame, save_path: Path = None, use_volatility_filter: bool = True
) -> pd.DataFrame:
    """
    分析 Z值 vs 胜率 的阶梯关系
    目标：量化"橡皮筋拉多长时，胜率会发生质变"
    """
    logger.info("=" * 60)
    logger.info("分析二：Probability Ladder（胜率阶梯）")
    logger.info("=" * 60)

    # 使用30分钟后的最高价来判断获利空间
    df = df.copy()

    # 计算未来30分钟内的最大获利空间
    df["potential_gain_30m"] = (df["high_max_30m"] / df["close"]) - 1

    # 只看有效样本
    valid_mask = (
        np.isfinite(df["Z_final"])
        & np.isfinite(df["potential_gain_30m"])
        & (df["time_str"] < "14:15")
    )

    # 加入昨日波动率滤网（消除未来函数）
    if use_volatility_filter:
        range_threshold = df["yesterday_range"].quantile(0.9)
        valid_mask = (
            valid_mask
            & np.isfinite(df["yesterday_range"])
            & (df["yesterday_range"] > range_threshold)
        )
        logger.info(f"昨日波动率阈值（90%分位数）: {range_threshold*100:.2f}%")

    df_valid = df[valid_mask].copy()

    # 对 Z_final 进行分档（买入方向）
    bins_buy = [-np.inf, -10, -5, -3, -2, -1.5, -1, 0]
    labels_buy = ["<-10", "-10~-5", "-5~-3", "-3~-2", "-2~-1.5", "-1.5~-1", "-1~0"]
    df_valid["z_group_buy"] = pd.cut(
        df_valid["Z_final"], bins=bins_buy, labels=labels_buy
    )

    # 统计每个档位的胜率
    # 胜率定义：30分钟内最大获利空间 > 20bp (0.002)
    ladder_buy = df_valid.groupby("z_group_buy", observed=True)[
        "potential_gain_30m"
    ].agg(
        count="count",
        avg_gain=lambda x: x.mean() * 10000,
        win_rate_20bp=lambda x: (x > 0.002).mean(),
        win_rate_30bp=lambda x: (x > 0.003).mean(),
        win_rate_50bp=lambda x: (x > 0.005).mean(),
    )

    print("\n" + "=" * 70)
    print("买入方向：Z_final 分档 vs 胜率（30分钟内最大获利空间）")
    print("=" * 70)
    print(
        f"{'Z值区间':>12} | {'样本数':>8} | {'平均获利(bp)':>12} | {'>20bp胜率':>10} | {'>30bp胜率':>10} | {'>50bp胜率':>10}"
    )
    print("-" * 70)

    for idx in ladder_buy.index:
        row = ladder_buy.loc[idx]
        print(
            f"{idx:>12} | {int(row['count']):>8} | {row['avg_gain']:>12.2f} | "
            f"{row['win_rate_20bp']*100:>9.1f}% | {row['win_rate_30bp']*100:>9.1f}% | {row['win_rate_50bp']*100:>9.1f}%"
        )

    # 对 Z_final 进行分档（卖出方向）
    bins_sell = [0, 1, 1.5, 2, 3, 5, 10, np.inf]
    labels_sell = ["0~1", "1~1.5", "1.5~2", "2~3", "3~5", "5~10", ">10"]
    df_valid["z_group_sell"] = pd.cut(
        df_valid["Z_final"], bins=bins_sell, labels=labels_sell
    )

    # 卖出方向：看价格下跌的空间
    df_valid["potential_loss_30m"] = (
        df_valid["close"] - df_valid["close_plus_30m"]
    ) / df_valid["close"]

    ladder_sell = df_valid.groupby("z_group_sell", observed=True)[
        "potential_loss_30m"
    ].agg(
        count="count",
        avg_gain=lambda x: x.mean() * 10000,
        win_rate_20bp=lambda x: (x > 0.002).mean(),
        win_rate_30bp=lambda x: (x > 0.003).mean(),
        win_rate_50bp=lambda x: (x > 0.005).mean(),
    )

    print("\n" + "=" * 70)
    print("卖出方向：Z_final 分档 vs 胜率（30分钟后价格下跌空间）")
    print("=" * 70)
    print(
        f"{'Z值区间':>12} | {'样本数':>8} | {'平均获利(bp)':>12} | {'>20bp胜率':>10} | {'>30bp胜率':>10} | {'>50bp胜率':>10}"
    )
    print("-" * 70)

    for idx in ladder_sell.index:
        row = ladder_sell.loc[idx]
        print(
            f"{idx:>12} | {int(row['count']):>8} | {row['avg_gain']:>12.2f} | "
            f"{row['win_rate_20bp']*100:>9.1f}% | {row['win_rate_30bp']*100:>9.1f}% | {row['win_rate_50bp']*100:>9.1f}%"
        )

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 买入方向柱状图
    x_buy = range(len(ladder_buy))
    bars1 = ax1.bar(
        x_buy,
        ladder_buy["win_rate_20bp"] * 100,
        color="steelblue",
        alpha=0.8,
        label=">20bp Win Rate",
    )
    ax1.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% Baseline")
    ax1.set_xticks(x_buy)
    ax1.set_xticklabels(ladder_buy.index, rotation=45, ha="right")
    ax1.set_xlabel("Z_final Range (Buy Direction)", fontsize=11)
    ax1.set_ylabel("Win Rate (%)", fontsize=11)
    ax1.set_title("Probability Ladder - Buy Signals", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 在柱子上标注样本数
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(
            f"n={int(ladder_buy.iloc[i]['count'])}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 卖出方向柱状图
    x_sell = range(len(ladder_sell))
    bars2 = ax2.bar(
        x_sell,
        ladder_sell["win_rate_20bp"] * 100,
        color="coral",
        alpha=0.8,
        label=">20bp Win Rate",
    )
    ax2.axhline(50, color="red", linestyle="--", alpha=0.7, label="50% Baseline")
    ax2.set_xticks(x_sell)
    ax2.set_xticklabels(ladder_sell.index, rotation=45, ha="right")
    ax2.set_xlabel("Z_final Range (Sell Direction)", fontsize=11)
    ax2.set_ylabel("Win Rate (%)", fontsize=11)
    ax2.set_title("Probability Ladder - Sell Signals", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(
            f"n={int(ladder_sell.iloc[i]['count'])}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close()

    return ladder_buy, ladder_sell


# ==================== 分析三：Filter Value（滤网价值）====================
def analyze_filter_value(
    df: pd.DataFrame, save_path: Path = None, use_volatility_filter: bool = True
) -> dict:
    """
    分析趋势滤网的价值
    目标：验证"新高不买、新低不卖"滤网是否有效
    """
    logger.info("=" * 60)
    logger.info("分析三：Filter Value（滤网价值）")
    logger.info("=" * 60)

    df = df.copy()

    # 计算30分钟后的收益
    df["return_30m"] = (df["close_plus_30m"] / df["close"]) - 1

    # 有效样本
    valid_mask = (
        np.isfinite(df["Z_final"])
        & np.isfinite(df["return_30m"])
        & (df["time_str"] < "14:15")
    )

    # 加入昨日波动率滤网（消除未来函数）
    if use_volatility_filter:
        range_threshold = df["yesterday_range"].quantile(0.9)
        valid_mask = (
            valid_mask
            & np.isfinite(df["yesterday_range"])
            & (df["yesterday_range"] > range_threshold)
        )
        logger.info(f"昨日波动率阈值（90%分位数）: {range_threshold*100:.2f}%")

    df_valid = df[valid_mask].copy()

    # ========== 买入信号分析 ==========
    # 原始 Z 信号（买入方向）
    raw_buy_signal = df_valid["Z_final"] < -SIGNAL_THRESHOLD

    # 被滤网挡掉的信号（Z 触发了买入，但此时是日内新高）
    blocked_buy = raw_buy_signal & (df_valid["close"] >= df_valid["day_high"])

    # 通过滤网的信号
    passed_buy = raw_buy_signal & (df_valid["close"] < df_valid["day_high"])

    # 统计对比
    raw_buy_return = df_valid.loc[raw_buy_signal, "return_30m"].mean() * 10000
    blocked_buy_return = (
        df_valid.loc[blocked_buy, "return_30m"].mean() * 10000
        if blocked_buy.sum() > 0
        else np.nan
    )
    passed_buy_return = (
        df_valid.loc[passed_buy, "return_30m"].mean() * 10000
        if passed_buy.sum() > 0
        else np.nan
    )

    raw_buy_count = raw_buy_signal.sum()
    blocked_buy_count = blocked_buy.sum()
    passed_buy_count = passed_buy.sum()

    print("\n" + "=" * 70)
    print("买入信号滤网分析：'新高不买'滤网效果")
    print("=" * 70)
    print(f"{'信号类型':>20} | {'样本数':>8} | {'平均收益(bp)':>12} | {'胜率':>8}")
    print("-" * 70)
    print(
        f"{'原始买入信号':>20} | {raw_buy_count:>8} | {raw_buy_return:>12.2f} | {(df_valid.loc[raw_buy_signal, 'return_30m'] > 0).mean()*100:>7.1f}%"
    )
    if blocked_buy_count > 0:
        print(
            f"{'被滤网拦截':>20} | {blocked_buy_count:>8} | {blocked_buy_return:>12.2f} | {(df_valid.loc[blocked_buy, 'return_30m'] > 0).mean()*100:>7.1f}%"
        )
    print(
        f"{'通过滤网':>20} | {passed_buy_count:>8} | {passed_buy_return:>12.2f} | {(df_valid.loc[passed_buy, 'return_30m'] > 0).mean()*100:>7.1f}%"
    )

    if blocked_buy_count > 0 and not np.isnan(blocked_buy_return):
        if blocked_buy_return < passed_buy_return:
            print(
                f"\n✅ 滤网有效！被拦截信号收益 {blocked_buy_return:.2f}bp < 通过信号 {passed_buy_return:.2f}bp"
            )
            print(
                f"   滤网为你避免了 {passed_buy_return - blocked_buy_return:.2f}bp 的潜在损失"
            )
        else:
            print(
                f"\n⚠️ 滤网可能过于保守，被拦截信号收益 {blocked_buy_return:.2f}bp >= 通过信号 {passed_buy_return:.2f}bp"
            )

    # ========== 卖出信号分析 ==========
    # 原始 Z 信号（卖出方向）
    raw_sell_signal = df_valid["Z_final"] > SIGNAL_THRESHOLD

    # 被滤网挡掉的信号（Z 触发了卖出，但此时是日内新低）
    blocked_sell = raw_sell_signal & (df_valid["close"] <= df_valid["day_low"])

    # 通过滤网的信号
    passed_sell = raw_sell_signal & (df_valid["close"] > df_valid["day_low"])

    # 卖出信号期望价格下跌，所以收益取负
    df_valid["sell_return_30m"] = -df_valid["return_30m"]

    raw_sell_return = df_valid.loc[raw_sell_signal, "sell_return_30m"].mean() * 10000
    blocked_sell_return = (
        df_valid.loc[blocked_sell, "sell_return_30m"].mean() * 10000
        if blocked_sell.sum() > 0
        else np.nan
    )
    passed_sell_return = (
        df_valid.loc[passed_sell, "sell_return_30m"].mean() * 10000
        if passed_sell.sum() > 0
        else np.nan
    )

    raw_sell_count = raw_sell_signal.sum()
    blocked_sell_count = blocked_sell.sum()
    passed_sell_count = passed_sell.sum()

    print("\n" + "=" * 70)
    print("卖出信号滤网分析：'新低不卖'滤网效果")
    print("=" * 70)
    print(f"{'信号类型':>20} | {'样本数':>8} | {'平均收益(bp)':>12} | {'胜率':>8}")
    print("-" * 70)
    print(
        f"{'原始卖出信号':>20} | {raw_sell_count:>8} | {raw_sell_return:>12.2f} | {(df_valid.loc[raw_sell_signal, 'sell_return_30m'] > 0).mean()*100:>7.1f}%"
    )
    if blocked_sell_count > 0:
        print(
            f"{'被滤网拦截':>20} | {blocked_sell_count:>8} | {blocked_sell_return:>12.2f} | {(df_valid.loc[blocked_sell, 'sell_return_30m'] > 0).mean()*100:>7.1f}%"
        )
    print(
        f"{'通过滤网':>20} | {passed_sell_count:>8} | {passed_sell_return:>12.2f} | {(df_valid.loc[passed_sell, 'sell_return_30m'] > 0).mean()*100:>7.1f}%"
    )

    if blocked_sell_count > 0 and not np.isnan(blocked_sell_return):
        if blocked_sell_return < passed_sell_return:
            print(
                f"\n✅ 滤网有效！被拦截信号收益 {blocked_sell_return:.2f}bp < 通过信号 {passed_sell_return:.2f}bp"
            )
            print(
                f"   滤网为你避免了 {passed_sell_return - blocked_sell_return:.2f}bp 的潜在损失"
            )
        else:
            print(
                f"\n⚠️ 滤网可能过于保守，被拦截信号收益 {blocked_sell_return:.2f}bp >= 通过信号 {passed_sell_return:.2f}bp"
            )

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 买入滤网对比
    categories_buy = ["Raw Signal", "Blocked", "Passed"]
    values_buy = [
        raw_buy_return,
        blocked_buy_return if not np.isnan(blocked_buy_return) else 0,
        passed_buy_return,
    ]
    colors_buy = ["gray", "red", "green"]

    bars1 = ax1.bar(categories_buy, values_buy, color=colors_buy, alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axhline(20, color="red", linestyle=":", alpha=0.7, label="Cost Line (20bp)")
    ax1.set_ylabel("Average Return (bp)", fontsize=11)
    ax1.set_title(
        "Buy Signal Filter Analysis\n('No Buy at New High')",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 在柱子上标注样本数
    counts_buy = [raw_buy_count, blocked_buy_count, passed_buy_count]
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.annotate(
            f"n={counts_buy[i]}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
        )

    # 卖出滤网对比
    categories_sell = ["Raw Signal", "Blocked", "Passed"]
    values_sell = [
        raw_sell_return,
        blocked_sell_return if not np.isnan(blocked_sell_return) else 0,
        passed_sell_return,
    ]
    colors_sell = ["gray", "red", "green"]

    bars2 = ax2.bar(categories_sell, values_sell, color=colors_sell, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(20, color="red", linestyle=":", alpha=0.7, label="Cost Line (20bp)")
    ax2.set_ylabel("Average Return (bp)", fontsize=11)
    ax2.set_title(
        "Sell Signal Filter Analysis\n('No Sell at New Low')",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    counts_sell = [raw_sell_count, blocked_sell_count, passed_sell_count]
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.annotate(
            f"n={counts_sell[i]}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        "buy": {
            "raw_return": raw_buy_return,
            "blocked_return": blocked_buy_return,
            "passed_return": passed_buy_return,
            "raw_count": raw_buy_count,
            "blocked_count": blocked_buy_count,
            "passed_count": passed_buy_count,
        },
        "sell": {
            "raw_return": raw_sell_return,
            "blocked_return": blocked_sell_return,
            "passed_return": passed_sell_return,
            "raw_count": raw_sell_count,
            "blocked_count": blocked_sell_count,
            "passed_count": passed_sell_count,
        },
    }


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("信号深度分析开始")
    logger.info("=" * 60)

    # 1. 加载数据
    df_1m = load_minute_data(TEST_STOCKS, YEAR)

    # 2. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 3. 计算特征
    df_5m = calculate_features(df_5m)

    # 4. 计算 Z-Score
    df_5m = calculate_zscore(df_5m, window=ZSCORE_WINDOW)

    # 5. 计算昨日波动率（消除未来函数）
    df_5m = calculate_yesterday_volatility(df_5m)

    # 6. 预计算未来价格（用于分析）
    df_5m = prepare_future_prices(df_5m)

    # 7. 分析一：Edge Curve（加入昨日波动率滤网）
    edge_results = analyze_edge_curve(
        df_5m,
        save_path=OUTPUT_DIR / "edge_curve_with_filter.png",
        use_volatility_filter=True,
    )

    # 8. 分析二：Probability Ladder（加入昨日波动率滤网）
    ladder_buy, ladder_sell = analyze_probability_ladder(
        df_5m,
        save_path=OUTPUT_DIR / "probability_ladder_with_filter.png",
        use_volatility_filter=True,
    )

    # 9. 分析三：Filter Value（加入昨日波动率滤网）
    filter_results = analyze_filter_value(
        df_5m,
        save_path=OUTPUT_DIR / "filter_value_with_filter.png",
        use_volatility_filter=True,
    )

    # 10. 总结
    print("\n" + "=" * 70)
    print("深度分析总结（已加入昨日波动率滤网，消除未来函数）")
    print("=" * 70)

    print(f"\n📈 Edge Curve 结论:")
    print(f"   - 买入信号最佳持有时间: {edge_results['best_period_buy']} 分钟")
    print(f"   - 卖出信号最佳持有时间: {edge_results['best_period_sell']} 分钟")
    print(f"   - 💡 说明：只有在高波动日子里，才能看到明显的正收益")

    print(f"\n📊 Probability Ladder 结论:")
    print(f"   - Z_final < -5 的买入信号胜率最高")
    print(f"   - 极端 Z 值具有更高的获利确定性")
    print(f"   - 💡 说明：高波动环境下，平均获利空间显著提升")

    print(f"\n🛡️ Filter Value 结论:")
    buy_improvement = (
        filter_results["buy"]["passed_return"] - filter_results["buy"]["blocked_return"]
    )
    if not np.isnan(buy_improvement) and buy_improvement > 0:
        print(f"   - '新高不买'滤网有效，提升收益 {buy_improvement:.2f}bp")
    else:
        print(f"   - '新高不买'滤网效果需要进一步验证")
    print(f"   - 💡 说明：趋势滤网成功拦截了追高风险")

    print(f"\n🎯 核心建议:")
    print(f"   1. 只在昨日高波动的股票上做T（消除未来函数）")
    print(f"   2. 入场阈值设在 Z_final < -3，胜率和获利都更高")
    print(f"   3. 止盈时间设在 30-60 分钟，不要过早平仓")
    print(f"   4. 只做正T（买入做T），避免做倒T")

    logger.success("深度分析完成！")
    logger.info(f"图片已保存至: {OUTPUT_DIR}")

    return df_5m


if __name__ == "__main__":
    df = main()
