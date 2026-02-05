"""
日内做T信号可视化脚本

功能：
1. 主图：价格线 + VWAP线 + 买卖信号标注
2. 副图：Z_final、X2_zscore、X1_zscore 曲线 + 阈值线
3. 支持多日期、多股票批量绘图

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# Use default font (English only)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 配置参数 ====================
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试股票池
TEST_STOCKS = ["002591", "002494", "002247", "000001", "300750", "601318"]
YEAR = 2025
ZSCORE_WINDOW = 20
SIGNAL_THRESHOLD = 1.5  # 信号触发阈值
ROLLING_WINDOW = 12  # 滚动窗口大小（12个5分钟Bar = 60分钟）


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载指定股票的1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    logger.info(f"加载 {year} 年数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    result = df[df["SecuCode"].isin(stock_codes)].copy()
    result = result.sort_values(["SecuCode", "TradingDay"]).reset_index(drop=True)
    result["date"] = result["TradingDay"].dt.date
    result["time"] = result["TradingDay"].dt.strftime("%H:%M")

    logger.success(f"加载完成: {len(result):,} 条")
    return result


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
    """
    计算滚动窗口VWAP、TWAP和特征
    改动：使用滚动窗口替代全天累积，让VWAP跟着趋势走
    """
    logger.info("计算特征（滚动窗口模式）...")

    result_list = []
    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("bar_time").copy()

        # 保留全天累积版本（用于计算 V_rest）
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

        result_list.append(day_df)

    result = pd.concat(result_list, ignore_index=True)
    logger.success("特征计算完成（滚动窗口模式）")
    return result


def calculate_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算 Z-Score 和 Z_final"""
    logger.info(f"计算 Z-Score (窗口={window}天)...")

    result_list = []

    for (stock, time_str), group_df in df.groupby(["SecuCode", "time_str"]):
        group_df = group_df.sort_values("date").copy()

        # X2 Z-Score
        group_df["X2_mean"] = (
            group_df["X2"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X2_std"] = (
            group_df["X2"].rolling(window, min_periods=5).std().shift(1)
        )
        group_df["X2_zscore"] = (group_df["X2"] - group_df["X2_mean"]) / group_df[
            "X2_std"
        ]

        # X1 Z-Score
        group_df["X1_mean"] = (
            group_df["X1"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X1_std"] = (
            group_df["X1"].rolling(window, min_periods=5).std().shift(1)
        )
        group_df["X1_zscore"] = (group_df["X1"] - group_df["X1_mean"]) / group_df[
            "X1_std"
        ]

        # Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore
        group_df["Z_final"] = 2.5 * group_df["X2_zscore"] - 0.5 * group_df["X1_zscore"]

        result_list.append(group_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤无效样本
    valid_mask = result["X2_zscore"].notna() & result["X1_zscore"].notna()
    result = result[valid_mask]

    logger.success(f"Z-Score 计算完成, 有效样本: {len(result):,}")
    return result


def plot_intraday_signals(
    df: pd.DataFrame, ticker: str, target_date, save_path: Path = None
):
    """
    绘制单只股票一天的分时图及做T信号

    参数:
        df: 包含所有特征的 DataFrame
        ticker: 股票代码
        target_date: 目标日期
        save_path: 保存路径（可选）
    """
    # 筛选数据
    day_data = df[(df["SecuCode"] == ticker) & (df["date"] == target_date)].copy()
    if len(day_data) == 0:
        logger.warning(f"未找到 {ticker} 在 {target_date} 的数据")
        return

    day_data = day_data.sort_values("bar_time").reset_index(drop=True)

    # ==================== 趋势滤网：新高不买入，新低不卖出 ====================
    day_data["day_high"] = day_data["close"].expanding().max()
    day_data["day_low"] = day_data["close"].expanding().min()
    is_not_new_high = day_data["close"] < day_data["day_high"]
    is_not_new_low = day_data["close"] > day_data["day_low"]

    # ==================== 信号去重：只显示首次触发的信号 ====================
    day_data["is_sell_trigger"] = day_data["Z_final"] > SIGNAL_THRESHOLD
    day_data["is_buy_trigger"] = day_data["Z_final"] < -SIGNAL_THRESHOLD
    # 只有从非触发状态变为触发状态时才显示信号
    day_data["is_first_sell"] = day_data["is_sell_trigger"] & (
        day_data["is_sell_trigger"] != day_data["is_sell_trigger"].shift(1)
    )
    day_data["is_first_buy"] = day_data["is_buy_trigger"] & (
        day_data["is_buy_trigger"] != day_data["is_buy_trigger"].shift(1)
    )

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.05},
    )

    # ==================== 主图：价格与 VWAP ====================
    times = day_data["bar_time"]
    prices = day_data["close"]
    vwap = day_data["cum_vwap"]

    # 价格线
    ax1.plot(
        times, prices, label="Close Price", color="black", linewidth=1.5, alpha=0.9
    )
    # VWAP 线
    ax1.plot(
        times,
        vwap,
        label="Cum VWAP",
        color="blue",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    # 卖出信号：Z_final > threshold + 趋势滤网（非新低） + 信号去重（首次触发）
    sell_signals = day_data[day_data["is_first_sell"] & is_not_new_low]
    if len(sell_signals) > 0:
        ax1.scatter(
            sell_signals["bar_time"],
            sell_signals["close"],
            color="red",
            marker="v",
            s=200,
            label=f"Sell Signal (Z>{SIGNAL_THRESHOLD})",
            zorder=5,
            edgecolors="darkred",
            linewidths=2,
        )

    # 买入信号：Z_final < -threshold + 趋势滤网（非新高） + 信号去重（首次触发）
    buy_signals = day_data[day_data["is_first_buy"] & is_not_new_high]
    if len(buy_signals) > 0:
        ax1.scatter(
            buy_signals["bar_time"],
            buy_signals["close"],
            color="lime",
            marker="^",
            s=200,
            label=f"Buy Signal (Z<-{SIGNAL_THRESHOLD})",
            zorder=5,
            edgecolors="darkgreen",
            linewidths=2,
        )

    # 计算当日收益空间
    price_range = prices.max() - prices.min()
    price_range_pct = price_range / prices.mean() * 100

    ax1.set_title(
        f"Intraday T+0 Analysis: {ticker} | {target_date} | Range: {price_range_pct:.2f}% | Rolling VWAP (60min)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 填充价格与VWAP之间的区域
    ax1.fill_between(
        times,
        prices,
        vwap,
        where=(prices > vwap),
        color="red",
        alpha=0.1,
        label="Price > VWAP",
    )
    ax1.fill_between(
        times,
        prices,
        vwap,
        where=(prices < vwap),
        color="green",
        alpha=0.1,
        label="Price < VWAP",
    )

    # ==================== 副图：Z-Score 指标 ====================
    ax2.plot(
        times,
        day_data["Z_final"],
        label="Z_final (Combined)",
        color="purple",
        linewidth=2,
    )
    ax2.plot(
        times,
        day_data["X2_zscore"],
        label="X2_zscore (Energy)",
        color="orange",
        linewidth=1,
        alpha=0.7,
    )
    ax2.plot(
        times,
        day_data["X1_zscore"],
        label="X1_zscore (Price)",
        color="gray",
        linewidth=1,
        alpha=0.5,
    )

    # 阈值线
    ax2.axhline(
        SIGNAL_THRESHOLD,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Sell Threshold (+{SIGNAL_THRESHOLD})",
    )
    ax2.axhline(
        -SIGNAL_THRESHOLD,
        color="green",
        linestyle=":",
        alpha=0.7,
        label=f"Buy Threshold (-{SIGNAL_THRESHOLD})",
    )
    ax2.axhline(0, color="black", linewidth=0.5)

    # 填充极值区域
    ax2.fill_between(
        times,
        SIGNAL_THRESHOLD,
        day_data["Z_final"],
        where=(day_data["Z_final"] > SIGNAL_THRESHOLD),
        color="red",
        alpha=0.3,
    )
    ax2.fill_between(
        times,
        -SIGNAL_THRESHOLD,
        day_data["Z_final"],
        where=(day_data["Z_final"] < -SIGNAL_THRESHOLD),
        color="green",
        alpha=0.3,
    )

    ax2.set_ylabel("Z-Score", fontsize=11)
    ax2.set_ylim(-100, 100)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # 时间轴格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel("Time", fontsize=11)
    plt.xticks(rotation=45)

    # 添加统计信息
    n_buy = len(buy_signals)
    n_sell = len(sell_signals)
    max_z = day_data["Z_final"].max()
    min_z = day_data["Z_final"].min()

    stats_text = f"Buy Signals: {n_buy} | Sell Signals: {n_sell} | Z_final Range: [{min_z:.2f}, {max_z:.2f}] | Filters: Trend + Dedup"
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout()

    # 保存或显示
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def find_interesting_days(df: pd.DataFrame, n_days: int = 5) -> list:
    """
    找出最有趣的交易日（信号最多或Z值最极端的日子）
    """
    logger.info("寻找最有趣的交易日...")

    day_stats = []
    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        if len(day_df) < 40:  # 跳过数据不完整的日子
            continue

        n_buy = (day_df["Z_final"] < -SIGNAL_THRESHOLD).sum()
        n_sell = (day_df["Z_final"] > SIGNAL_THRESHOLD).sum()
        max_z = day_df["Z_final"].max()
        min_z = day_df["Z_final"].min()
        z_range = max_z - min_z

        day_stats.append(
            {
                "SecuCode": stock,
                "date": date,
                "n_buy": n_buy,
                "n_sell": n_sell,
                "n_signals": n_buy + n_sell,
                "max_z": max_z,
                "min_z": min_z,
                "z_range": z_range,
            }
        )

    stats_df = pd.DataFrame(day_stats)

    # 按信号数量和Z值范围排序
    stats_df["score"] = stats_df["n_signals"] + stats_df["z_range"]
    top_days = stats_df.nlargest(n_days, "score")

    logger.success(f"找到 {len(top_days)} 个有趣的交易日")
    return top_days.to_dict("records")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Path = None,
    use_z_final: bool = True,
    volatility_filter: bool = True,
):
    """
    绘制因子分位数 vs Y 均值的热力图

    参数:
        use_z_final: True 使用 Z_final（组合因子），False 使用 X2_zscore（单因子）
        volatility_filter: True 只看日内波动剧烈的样本（方向B：日内弹性滤网）
    """
    factor_name = "Z_final" if use_z_final else "X2_zscore"
    filter_desc = " + 昨日高波动滤网（无未来函数）" if volatility_filter else ""
    logger.info(f"绘制相关性热力图（因子: {factor_name}{filter_desc}）...")

    # 计算每日波动率（日内振幅）
    daily_volatility = (
        df.groupby(["SecuCode", "date"])
        .apply(lambda x: (x["close"].max() - x["close"].min()) / x["close"].mean())
        .reset_index(name="daily_range")
    )

    # 【消除未来函数】将昨天的振幅平移到今天 (Shift)
    # 实盘中，我们只能用"昨日已知"的信息来筛选今天的"活鱼"
    daily_volatility["yesterday_range"] = daily_volatility.groupby("SecuCode")[
        "daily_range"
    ].shift(1)
    df = df.merge(
        daily_volatility[["SecuCode", "date", "yesterday_range"]],
        on=["SecuCode", "date"],
        how="left",
    )

    # 过滤有效数据
    valid_mask = (
        np.isfinite(df["Z_final"])
        & np.isfinite(df["X2_zscore"])
        & np.isfinite(df["Y"])
        & (df["Y"] > 0.9)
        & (df["Y"] < 1.1)
        & (df["time_str"] < "14:15")
        & np.isfinite(df["yesterday_range"])  # 排除没有昨日数据的样本
    )

    # 方向B升级（无未来函数版）：使用昨日振幅筛选 - 只保留昨日振幅前10%的活跃股票样本
    if volatility_filter:
        # 计算昨日振幅的90%分位数阈值（只保留前10%活跃的样本）
        range_threshold = df["yesterday_range"].quantile(0.9)
        valid_mask = valid_mask & (df["yesterday_range"] > range_threshold)
        logger.info(f"昨日振幅阈值（90%分位数）: {range_threshold*100:.2f}%")

    df_valid = df[valid_mask].copy()

    logger.info(f"有效样本数: {len(df_valid):,}（活跃股筛选: {volatility_filter}）")

    # 选择因子进行分箱
    factor_col = "Z_final" if use_z_final else "X2_zscore"
    df_valid["factor_bin"] = pd.qcut(
        df_valid[factor_col], q=50, labels=False, duplicates="drop"
    )

    # 计算每个分位数的 Y 均值
    bin_stats = (
        df_valid.groupby("factor_bin")
        .agg(
            {
                "Y": ["mean", "std", "count"],
                factor_col: ["min", "max", "mean"],
            }
        )
        .reset_index()
    )

    bin_stats.columns = [
        "bin",
        "y_mean",
        "y_std",
        "count",
        "factor_min",
        "factor_max",
        "factor_mean",
    ]

    # 打印极端组的收益空间（notebook 新增要求）
    extreme_left = bin_stats.iloc[0]  # 最超跌组
    extreme_right = bin_stats.iloc[-1]  # 最超涨组
    print("\n" + "=" * 70)
    print(f"📊 极端组收益空间分析（因子: {factor_name}, q=50 分位数）")
    print("=" * 70)
    print(
        f"最超跌组 (Q1):  {factor_name}={extreme_left['factor_mean']:.2f}, Y偏离={abs(extreme_left['y_mean']-1)*10000:.2f} bp, 样本数={int(extreme_left['count'])}"
    )
    print(
        f"最超涨组 (Q50): {factor_name}={extreme_right['factor_mean']:.2f}, Y偏离={abs(extreme_right['y_mean']-1)*10000:.2f} bp, 样本数={int(extreme_right['count'])}"
    )

    # 判断是否突破盈利生死线
    max_profit = max(
        abs(extreme_left["y_mean"] - 1) * 10000,
        abs(extreme_right["y_mean"] - 1) * 10000,
    )
    if max_profit >= 20:
        print(f"\n✅ 突破20bp盈利生死线！最大收益空间: {max_profit:.2f} bp")
    else:
        print(f"\n⚠️ 未突破20bp盈利生死线，最大收益空间: {max_profit:.2f} bp")
    print("=" * 70 + "\n")

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 8))  # 加宽画布以适应更多柱子

    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(bin_stats)))

    # 动态计算柱宽（根据分位数数量调整）
    x_range = bin_stats["factor_mean"].max() - bin_stats["factor_mean"].min()
    bar_width = x_range / len(bin_stats) * 0.8

    bars = ax.bar(
        bin_stats["factor_mean"],
        (bin_stats["y_mean"] - 1) * 10000,  # 转换为 bp
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # 添加误差线
    ax.errorbar(
        bin_stats["factor_mean"],
        (bin_stats["y_mean"] - 1) * 10000,
        yerr=bin_stats["y_std"] * 10000,
        fmt="none",
        color="gray",
        capsize=3,
        alpha=0.5,
    )

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(
        20,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Profit Line (+20bp)",
    )
    ax.axhline(
        -20,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label="Profit Line (-20bp)",
    )

    ax.set_xlabel(f"{factor_name} Quantile Mean", fontsize=12)
    ax.set_ylabel("Y Deviation (bp)", fontsize=12)
    title_suffix = " [Yesterday High Vol, No Look-ahead]" if volatility_filter else ""
    ax.set_title(
        f"{factor_name} vs V_5m/V_rest Relationship (50 Quantiles){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 只标注极端组的样本数（避免图面过于拥挤）
    for idx in [0, len(bin_stats) - 1]:  # 只标注第一个和最后一个
        row = bin_stats.iloc[idx]
        ax.annotate(
            f"n={int(row['count'])}\n{abs(row['y_mean']-1)*10000:.1f}bp",
            (row["factor_mean"], (row["y_mean"] - 1) * 10000),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="darkblue",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"热力图已保存: {save_path}")
    else:
        plt.show()

    plt.close()


def calculate_remaining_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """计算 V_rest 和 Y"""
    logger.info("计算 V_rest 和 Y...")

    result_list = []

    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("bar_time").copy()
        n_bars = len(day_df)

        total_volume = day_df["volume"].sum()
        total_amount = day_df["amount"].sum()

        day_df["V_rest"] = np.nan
        day_df["Y"] = np.nan

        for i in range(n_bars):
            rest_volume = total_volume - day_df["cum_volume"].iloc[i]
            rest_amount = total_amount - day_df["cum_amount"].iloc[i]
            if rest_volume > 0:
                day_df.iloc[i, day_df.columns.get_loc("V_rest")] = (
                    rest_amount / rest_volume
                )
                day_df.iloc[i, day_df.columns.get_loc("Y")] = day_df["vwap_5m"].iloc[
                    i
                ] / (rest_amount / rest_volume)

        result_list.append(day_df)

    result = pd.concat(result_list, ignore_index=True)
    logger.success("V_rest 和 Y 计算完成")
    return result


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始绘制日内做T信号图")
    logger.info("=" * 60)

    # 1. 加载数据
    df_1m = load_minute_data(TEST_STOCKS, YEAR)

    # 2. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 3. 计算特征
    df_5m = calculate_features(df_5m)

    # 4. 计算 V_rest 和 Y
    df_5m = calculate_remaining_vwap(df_5m)

    # 5. 计算 Z-Score
    df_5m = calculate_zscore(df_5m, window=ZSCORE_WINDOW)

    # 6. 找出最有趣的交易日
    interesting_days = find_interesting_days(df_5m, n_days=10)

    print("\n" + "=" * 60)
    print("📊 最有趣的交易日（信号最多 / Z值最极端）")
    print("=" * 60)
    for i, day in enumerate(interesting_days, 1):
        print(
            f"{i}. {day['SecuCode']} | {day['date']} | "
            f"买入信号: {day['n_buy']} | 卖出信号: {day['n_sell']} | "
            f"Z范围: [{day['min_z']:.2f}, {day['max_z']:.2f}]"
        )

    # 7. 绘制 Top 5 的分时图
    print("\n" + "=" * 60)
    print("📈 正在绘制分时图...")
    print("=" * 60)

    for i, day in enumerate(interesting_days[:5], 1):
        ticker = day["SecuCode"]
        target_date = day["date"]
        save_path = OUTPUT_DIR / f"intraday_{ticker}_{target_date}.png"
        plot_intraday_signals(df_5m, ticker, target_date, save_path)
        print(f"  [{i}/5] {ticker} - {target_date} ✅")

    # 8. 绘制相关性热力图
    print("\n📊 正在绘制相关性热力图...")
    heatmap_path = OUTPUT_DIR / "x2_y_correlation_heatmap.png"
    plot_correlation_heatmap(df_5m, heatmap_path)

    print("\n" + "=" * 60)
    print(f"✅ 所有图片已保存至: {OUTPUT_DIR}")
    print("=" * 60)

    return df_5m


if __name__ == "__main__":
    df = main()
