"""
日内做T时序模型 - 离线相关性验证脚本

目标：验证价格偏离(X1)和能量偏离(X2)与下午收益(Y)的相关性
股票：平安银行 000001.XSHE
时间：2025-01-01 ~ 2025-12-31

特征定义：
- X1 (价格偏离): Price_1030 / VWAP_1030 - 1
- X2 (能量偏离): VWAP_1030 / TWAP_1030 - 1

标签定义：
- Y: Price_1450 / Price_1030 - 1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
DATA_DIR = Path("/nfs/ofs-prediction/changxi/data/temp_1m_chunks")
STOCK_CODE = "000001"  # 平安银行
YEAR = 2025
OBSERVE_TIME = "10:30"  # 观察时间点
EXIT_TIME = "14:50"  # 出场时间点


def load_stock_data(stock_code: str, year: int) -> pd.DataFrame:
    """
    加载指定股票全年的分钟数据

    Args:
        stock_code: 股票代码（如 "000001"）
        year: 年份

    Returns:
        DataFrame: 该股票全年的分钟数据
    """
    logger.info(f"开始加载 {stock_code} {year}年分钟数据...")

    dfs = []
    for month in range(1, 13):
        pkl_path = DATA_DIR / f"{year}-{month:02d}.pkl"
        if not pkl_path.exists():
            logger.warning(f"文件不存在: {pkl_path}")
            continue

        logger.debug(f"读取: {pkl_path.name}")
        df = pd.read_pickle(pkl_path)

        # 筛选目标股票
        df_stock = df[df["SecuCode"] == stock_code].copy()
        if len(df_stock) > 0:
            dfs.append(df_stock)

    if not dfs:
        raise ValueError(f"未找到 {stock_code} 的数据")

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("TradingDay").reset_index(drop=True)

    logger.success(
        f"加载完成: {len(result):,} 条分钟数据, 覆盖 {result['TradingDay'].dt.date.nunique()} 个交易日"
    )
    return result


def calculate_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日的特征和标签

    Args:
        df: 分钟数据 DataFrame

    Returns:
        DataFrame: 每日特征和标签
    """
    logger.info("开始计算每日特征和标签...")

    # 提取日期和时间
    df["date"] = df["TradingDay"].dt.date
    df["time"] = df["TradingDay"].dt.strftime("%H:%M")

    # 按日期分组计算
    results = []

    for date, day_df in df.groupby("date"):
        day_df = day_df.sort_values("TradingDay")

        # 获取10:30之前的数据（09:31 ~ 10:30）
        morning_df = day_df[day_df["time"] <= OBSERVE_TIME]
        if len(morning_df) == 0:
            continue

        # 获取10:30时刻的数据
        obs_df = day_df[day_df["time"] == OBSERVE_TIME]
        if len(obs_df) == 0:
            continue

        # 获取14:50时刻的数据
        exit_df = day_df[day_df["time"] == EXIT_TIME]
        if len(exit_df) == 0:
            continue

        # 计算 VWAP (成交量加权平均价) - 09:31 ~ 10:30
        total_amount = morning_df["amount"].sum()
        total_volume = morning_df["volume"].sum()
        if total_volume == 0:
            continue
        vwap_1030 = total_amount / total_volume

        # 计算 TWAP (时间加权平均价) - 09:31 ~ 10:30 每分钟close的简单平均
        twap_1030 = morning_df["close"].mean()

        # 获取10:30的当前价格
        price_1030 = obs_df["close"].values[0]

        # 获取14:50的价格
        price_1450 = exit_df["close"].values[0]

        # 计算特征
        x1 = price_1030 / vwap_1030 - 1  # 价格偏离
        x2 = vwap_1030 / twap_1030 - 1  # 能量偏离

        # 计算标签
        y = price_1450 / price_1030 - 1  # 下午收益

        results.append(
            {
                "date": date,
                "price_1030": price_1030,
                "price_1450": price_1450,
                "vwap_1030": vwap_1030,
                "twap_1030": twap_1030,
                "X1_price_deviation": x1,
                "X2_energy_deviation": x2,
                "Y_afternoon_return": y,
                "morning_volume": total_volume,
                "morning_amount": total_amount,
            }
        )

    result_df = pd.DataFrame(results)
    logger.success(f"特征计算完成: {len(result_df)} 个交易日")
    return result_df


def analyze_correlation(df: pd.DataFrame) -> dict:
    """
    分析特征与标签的相关性

    Args:
        df: 包含特征和标签的 DataFrame

    Returns:
        dict: 相关性分析结果
    """
    logger.info("开始相关性分析...")

    # 计算相关系数
    corr_x1_y = df["X1_price_deviation"].corr(df["Y_afternoon_return"])
    corr_x2_y = df["X2_energy_deviation"].corr(df["Y_afternoon_return"])
    corr_x1_x2 = df["X1_price_deviation"].corr(df["X2_energy_deviation"])

    # 计算统计量
    stats = {
        "样本数": len(df),
        "X1_Y相关系数": corr_x1_y,
        "X2_Y相关系数": corr_x2_y,
        "X1_X2相关系数": corr_x1_x2,
        "X1均值": df["X1_price_deviation"].mean(),
        "X1标准差": df["X1_price_deviation"].std(),
        "X2均值": df["X2_energy_deviation"].mean(),
        "X2标准差": df["X2_energy_deviation"].std(),
        "Y均值": df["Y_afternoon_return"].mean(),
        "Y标准差": df["Y_afternoon_return"].std(),
    }

    return stats


def analyze_by_quantile(df: pd.DataFrame) -> pd.DataFrame:
    """
    分位数分析：按X1/X2分组，查看各组的平均Y

    Args:
        df: 包含特征和标签的 DataFrame

    Returns:
        DataFrame: 分位数分析结果
    """
    logger.info("开始分位数分析...")

    results = []

    # X1 分位数分析
    df["X1_quantile"] = pd.qcut(
        df["X1_price_deviation"], q=5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"]
    )
    x1_analysis = (
        df.groupby("X1_quantile")
        .agg(
            {
                "Y_afternoon_return": ["mean", "std", "count"],
                "X1_price_deviation": "mean",
            }
        )
        .round(6)
    )
    x1_analysis.columns = ["Y均值", "Y标准差", "样本数", "X1均值"]
    x1_analysis["特征"] = "X1_价格偏离"
    results.append(x1_analysis.reset_index().rename(columns={"X1_quantile": "分位组"}))

    # X2 分位数分析
    df["X2_quantile"] = pd.qcut(
        df["X2_energy_deviation"],
        q=5,
        labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"],
    )
    x2_analysis = (
        df.groupby("X2_quantile")
        .agg(
            {
                "Y_afternoon_return": ["mean", "std", "count"],
                "X2_energy_deviation": "mean",
            }
        )
        .round(6)
    )
    x2_analysis.columns = ["Y均值", "Y标准差", "样本数", "X2均值"]
    x2_analysis["特征"] = "X2_能量偏离"
    results.append(x2_analysis.reset_index().rename(columns={"X2_quantile": "分位组"}))

    return pd.concat(results, ignore_index=True)


def print_report(stats: dict, quantile_df: pd.DataFrame, feature_df: pd.DataFrame):
    """
    打印分析报告
    """
    print("\n" + "=" * 70)
    print("📊 日内做T时序模型 - 离线相关性验证报告")
    print("=" * 70)
    print(f"股票: 平安银行 (000001.XSHE)")
    print(f"时间范围: 2025-01-01 ~ 2025-12-31")
    print(f"观察时点: 10:30 | 出场时点: 14:50")
    print(f"样本数: {stats['样本数']} 个交易日")

    print("\n" + "-" * 70)
    print("📈 特征定义")
    print("-" * 70)
    print("X1 (价格偏离) = Price_1030 / VWAP_1030 - 1")
    print("X2 (能量偏离) = VWAP_1030 / TWAP_1030 - 1")
    print("Y  (下午收益) = Price_1450 / Price_1030 - 1")

    print("\n" + "-" * 70)
    print("📊 相关性分析")
    print("-" * 70)
    print(
        f"X1 与 Y 相关系数: {stats['X1_Y相关系数']:>10.4f}  {'✅ 负相关(符合预期)' if stats['X1_Y相关系数'] < 0 else '⚠️ 正相关(不符合预期)'}"
    )
    print(f"X2 与 Y 相关系数: {stats['X2_Y相关系数']:>10.4f}")
    print(f"X1 与 X2 相关系数: {stats['X1_X2相关系数']:>10.4f}")

    print("\n" + "-" * 70)
    print("📊 特征统计")
    print("-" * 70)
    print(
        f"X1 均值: {stats['X1均值']*100:>8.4f}%  标准差: {stats['X1标准差']*100:>8.4f}%"
    )
    print(
        f"X2 均值: {stats['X2均值']*100:>8.4f}%  标准差: {stats['X2标准差']*100:>8.4f}%"
    )
    print(
        f"Y  均值: {stats['Y均值']*100:>8.4f}%  标准差: {stats['Y标准差']*100:>8.4f}%"
    )

    print("\n" + "-" * 70)
    print("📊 X1 分位数分析 (价格偏离)")
    print("-" * 70)
    x1_df = quantile_df[quantile_df["特征"] == "X1_价格偏离"]
    print(f"{'分位组':<12} {'X1均值':>12} {'Y均值':>12} {'样本数':>8}")
    for _, row in x1_df.iterrows():
        print(
            f"{row['分位组']:<12} {row['X1均值']*100:>11.4f}% {row['Y均值']*100:>11.4f}% {int(row['样本数']):>8}"
        )

    print("\n" + "-" * 70)
    print("📊 X2 分位数分析 (能量偏离)")
    print("-" * 70)
    x2_df = quantile_df[quantile_df["特征"] == "X2_能量偏离"]
    print(f"{'分位组':<12} {'X2均值':>12} {'Y均值':>12} {'样本数':>8}")
    for _, row in x2_df.iterrows():
        print(
            f"{row['分位组']:<12} {row['X2均值']*100:>11.4f}% {row['Y均值']*100:>11.4f}% {int(row['样本数']):>8}"
        )

    # 计算单调性
    x1_y_means = x1_df["Y均值"].values
    x2_y_means = x2_df["Y均值"].values

    x1_monotonic = (
        "单调递减 ✅"
        if all(x1_y_means[i] >= x1_y_means[i + 1] for i in range(len(x1_y_means) - 1))
        else (
            "单调递增"
            if all(
                x1_y_means[i] <= x1_y_means[i + 1] for i in range(len(x1_y_means) - 1)
            )
            else "非单调"
        )
    )

    print("\n" + "-" * 70)
    print("📊 结论")
    print("-" * 70)
    print(f"X1 分位数单调性: {x1_monotonic}")

    if stats["X1_Y相关系数"] < -0.05:
        print("✅ X1与Y呈负相关，符合均值回归假设：价格越低于VWAP，下午回升概率越大")
    elif stats["X1_Y相关系数"] > 0.05:
        print("⚠️ X1与Y呈正相关，不符合均值回归假设，可能存在动量效应")
    else:
        print("⚠️ X1与Y相关性较弱，需要进一步分析")

    print("=" * 70)


def main():
    """主函数"""
    # 1. 加载数据
    df = load_stock_data(STOCK_CODE, YEAR)

    # 2. 计算特征和标签
    feature_df = calculate_daily_features(df)

    # 3. 相关性分析
    stats = analyze_correlation(feature_df)

    # 4. 分位数分析
    quantile_df = analyze_by_quantile(feature_df)

    # 5. 打印报告
    print_report(stats, quantile_df, feature_df)

    # 6. 保存结果
    output_dir = Path(
        "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/scripts/output"
    )
    output_dir.mkdir(exist_ok=True)

    feature_df.to_csv(output_dir / "intraday_features_000001_2025.csv", index=False)
    quantile_df.to_csv(output_dir / "quantile_analysis_000001_2025.csv", index=False)

    logger.success(f"结果已保存至: {output_dir}")

    return feature_df, stats, quantile_df


if __name__ == "__main__":
    feature_df, stats, quantile_df = main()
