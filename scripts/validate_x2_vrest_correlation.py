"""
验证脚本：X2_zscore 与 V_5m/V_rest 的相关性

目标：验证老板的直觉 —— X2（能量偏差）能否预测"现在买入是否划算"

决策流程（以站在10:00为例）：
  1. bar_time=10:00（ceil）表示 09:56~10:00 的数据，在10:00时刻全部可用
  2. 特征(X1, X2, Z-Score)用截至当前Bar（含）的累积数据 → 无未来函数
  3. 决定是否在下一个Bar(10:01~10:05)以VWAP买入
  4. V_5m = 下一个Bar的VWAP（买入成本，上帝视角）
  5. V_rest = 下一个Bar之后到收盘的VWAP（上帝视角）
  6. Y = V_5m / V_rest，Y < 1 说明买便宜了

逻辑：
- X2_zscore 越大（高位放量） → V_5m 应该比 V_rest 贵 → Y > 1
- X2_zscore 越小（低位放量） → V_5m 应该比 V_rest 便宜 → Y < 1
- 如果相关性稳定且为正，说明 X2 有预测力

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# 测试股票池（先用小样本验证）
TEST_STOCKS = ["002591", "002494", "002247", "000001", "300750", "601318"]
YEAR = 2025
ZSCORE_WINDOW = 20  # Z-Score 滚动窗口


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """
    加载指定股票的1分钟数据
    """
    pkl_path = DATA_DIR / f"{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")

    logger.info(f"加载 {year} 年数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # 筛选目标股票
    result = df[df["SecuCode"].isin(stock_codes)].copy()
    if len(result) == 0:
        raise ValueError(f"未找到 {stock_codes} 的数据")

    result = result.sort_values(["SecuCode", "TradingDay"]).reset_index(drop=True)

    # 添加辅助列
    result["date"] = result["TradingDay"].dt.date
    result["time"] = result["TradingDay"].dt.strftime("%H:%M")

    n_stocks = result["SecuCode"].nunique()
    n_days = result["date"].nunique()
    logger.success(
        f"加载完成: {len(result):,} 条, {n_stocks} 只股票, {n_days} 个交易日"
    )

    return result


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    将1分钟数据聚合为5分钟Bar，显式输出两套列：

    当前Bar数据（用于特征，在bar_time时刻全部可用）：
      - volume, amount, close, cum_volume, cum_amount, cum_vwap, cum_twap
    下一个Bar数据（用于标签，上帝视角）：
      - next_vwap_5m → 下一个Bar的VWAP（买入成本）

    bar_time 使用 ceil（Bar的结束时间），与米筐get_price()一致
    语义：bar_time=10:00 表示 09:56~10:00 的数据，在10:00时刻全部可用
    """
    logger.info("聚合为5分钟Bar...")

    # 创建5分钟时间标签（向上取整到5分钟边界，与米筐一致）
    df["bar_time"] = pd.to_datetime(df["TradingDay"]).dt.ceil("5min")

    # 按 股票 + 日期 + 5分钟Bar 聚合
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

    # 先确保按股票+日期+时间排序
    agg_df = agg_df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    grp_keys = ["SecuCode", "date"]

    # 当前Bar的VWAP
    agg_df["vwap_5m"] = agg_df["amount"] / agg_df["volume"]
    agg_df["vwap_5m"] = agg_df["vwap_5m"].replace([np.inf, -np.inf], np.nan)

    # ========== 当前Bar的累积数据（用于特征，在bar_time时刻全部可用）==========
    # 截至当前Bar（含）的累积成交量/成交额
    agg_df["cum_volume"] = agg_df.groupby(grp_keys)["volume"].cumsum()
    agg_df["cum_amount"] = agg_df.groupby(grp_keys)["amount"].cumsum()

    # 截至当前Bar的累积VWAP
    agg_df["cum_vwap"] = agg_df["cum_amount"] / agg_df["cum_volume"]
    agg_df["cum_vwap"] = agg_df["cum_vwap"].replace([np.inf, -np.inf], np.nan)

    # 截至当前Bar的累积TWAP（close的累积均值）
    agg_df["cum_twap"] = (
        agg_df.groupby(grp_keys)["close"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # ========== 下一个Bar的VWAP（用于标签，上帝视角）==========
    agg_df["next_vwap_5m"] = agg_df.groupby(grp_keys)["vwap_5m"].shift(-1)

    # 提取时间字符串（用于后续筛选）
    agg_df["time_str"] = agg_df["bar_time"].dt.strftime("%H:%M")

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征和标签

    特征（无未来函数，ceil语义下当前Bar数据在bar_time时刻已全部可用）：
      - X1 = close / cum_vwap - 1（当前Bar收盘价 vs 截至当前Bar的累积VWAP）
      - X2 = cum_vwap / cum_twap - 1（截至当前Bar的累积VWAP vs 累积TWAP）
    标签（上帝视角）：
      - next_vwap_5m: 下一个Bar的VWAP（决策后买入的成本）
      - V_rest: 下一个Bar之后到收盘的VWAP
      - Y = next_vwap_5m / V_rest
    """
    logger.info("计算特征和标签...")

    # ========== 特征部分（当前Bar数据在bar_time时刻已全部可用，无未来函数）==========
    df["X1"] = df["close"] / df["cum_vwap"] - 1
    df["X2"] = df["cum_vwap"] / df["cum_twap"] - 1

    # ========== 标签部分（上帝视角）==========
    # V_rest: 下一个Bar之后到收盘的VWAP
    # 先算截至下一个Bar（含）的累积量，然后用全天总量减去
    grp_keys = ["SecuCode", "date"]
    # 截至下一个Bar的累积量 = 截至当前Bar的累积量 + 下一个Bar的量
    next_cum_amount = df["cum_amount"] + df.groupby(grp_keys)["amount"].shift(
        -1
    ).fillna(0)
    next_cum_volume = df["cum_volume"] + df.groupby(grp_keys)["volume"].shift(
        -1
    ).fillna(0)
    total_amount = df.groupby(grp_keys)["amount"].transform("sum")
    total_volume = df.groupby(grp_keys)["volume"].transform("sum")

    rest_amount = total_amount - next_cum_amount
    rest_volume = total_volume - next_cum_volume
    df["V_rest"] = (rest_amount / rest_volume).replace([np.inf, -np.inf], np.nan)

    # Y = next_vwap_5m / V_rest（下一个Bar买入成本 vs 之后的均价）
    df["Y"] = df["next_vwap_5m"] / df["V_rest"]

    # 过滤：每天第一个Bar的cum_twap可能不稳定，每天最后两个Bar的标签无意义
    valid = (
        df["cum_vwap"].notna()
        & df["cum_twap"].notna()
        & df["next_vwap_5m"].notna()
        & df["V_rest"].notna()
        & df["Y"].notna()
        & (rest_volume > 0)
    )
    result = df[valid].copy()

    logger.success(f"特征和标签计算完成, 有效样本: {len(result):,}")
    return result


def calculate_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算 X2 的 Z-Score

    关键：按 (股票, 时间点) 分组，使用历史同时间点的数据计算
    例如：10:35 的 Z-Score 使用过去20天在 10:35 的 X2 数据
    """
    logger.info(f"计算 Z-Score (窗口={window}天)...")

    result_list = []

    for (stock, time_str), group_df in df.groupby(["SecuCode", "time_str"]):
        group_df = group_df.sort_values("date").copy()

        # 滚动均值和标准差（shift(1) 避免数据泄露）
        group_df["X2_mean"] = (
            group_df["X2"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X2_std"] = (
            group_df["X2"].rolling(window, min_periods=5).std().shift(1)
        )

        # Z-Score
        group_df["X2_zscore"] = (group_df["X2"] - group_df["X2_mean"]) / group_df[
            "X2_std"
        ]

        # 同样处理 X1
        group_df["X1_mean"] = (
            group_df["X1"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X1_std"] = (
            group_df["X1"].rolling(window, min_periods=5).std().shift(1)
        )
        group_df["X1_zscore"] = (group_df["X1"] - group_df["X1_mean"]) / group_df[
            "X1_std"
        ]

        result_list.append(group_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤掉 Z-Score 为空的行（窗口期不足）
    valid_mask = result["X2_zscore"].notna() & result["Y"].notna()
    result = result[valid_mask]

    logger.success(f"Z-Score 计算完成, 有效样本: {len(result):,}")
    return result


def analyze_correlation(df: pd.DataFrame) -> dict:
    """
    分析 X2_zscore 与 Y 的相关性
    """
    logger.info("分析相关性...")

    # 过滤掉 inf 和 nan 值
    valid_mask = (
        np.isfinite(df["X2_zscore"])
        & np.isfinite(df["X1_zscore"])
        & np.isfinite(df["Y"])
        & (df["Y"] > 0.9)  # 过滤异常Y值
        & (df["Y"] < 1.1)
    )
    df_valid = df[valid_mask].copy()
    logger.info(f"过滤异常值后有效样本: {len(df_valid):,} (原 {len(df):,})")

    # 整体相关性
    corr_x2_y = df_valid["X2_zscore"].corr(df_valid["Y"])
    corr_x1_y = df_valid["X1_zscore"].corr(df_valid["Y"])

    # 按股票分组的相关性（使用过滤后的数据）
    stock_corrs = df_valid.groupby("SecuCode").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "corr_X1_Y": x["X1_zscore"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )

    # 按时间段分组的相关性（使用过滤后的数据）
    time_corrs = df_valid.groupby("time_str").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )

    stats = {
        "total_samples": len(df_valid),
        "corr_X2_Y_overall": corr_x2_y,
        "corr_X1_Y_overall": corr_x1_y,
        "stock_corrs": stock_corrs,
        "time_corrs": time_corrs,
    }

    return stats


def print_report(stats: dict, df: pd.DataFrame):
    """
    打印验证报告
    """
    # 过滤有效数据用于统计
    valid_mask = np.isfinite(df["Y"]) & (df["Y"] > 0.9) & (df["Y"] < 1.1)
    df_valid = df[valid_mask]

    print("\n" + "=" * 70)
    print("📊 X2_zscore 与 V_5m/V_rest 相关性验证报告")
    print("=" * 70)

    print(f"\n样本总数: {stats['total_samples']:,}")
    print(f"股票数量: {df_valid['SecuCode'].nunique()}")
    print(f"交易日数: {df_valid['date'].nunique()}")

    print("\n" + "-" * 70)
    print("📈 整体相关性")
    print("-" * 70)
    corr_x2 = stats["corr_X2_Y_overall"]
    corr_x1 = stats["corr_X1_Y_overall"]

    status_x2 = (
        "✅ 正相关(符合预期)"
        if corr_x2 > 0.05
        else ("⚠️ 相关性弱" if abs(corr_x2) < 0.05 else "❌ 负相关")
    )
    status_x1 = (
        "✅ 正相关"
        if corr_x1 > 0.05
        else ("⚠️ 相关性弱" if abs(corr_x1) < 0.05 else "❌ 负相关")
    )

    print(f"X2_zscore 与 Y 相关系数: {corr_x2:>8.4f}  {status_x2}")
    print(f"X1_zscore 与 Y 相关系数: {corr_x1:>8.4f}  {status_x1}")

    print("\n" + "-" * 70)
    print("📊 按股票分组的相关性")
    print("-" * 70)
    stock_corrs = stats["stock_corrs"]
    print(f"{'股票代码':<10} {'X2与Y相关性':>12} {'X1与Y相关性':>12} {'样本数':>10}")
    for stock, row in stock_corrs.iterrows():
        print(
            f"{stock:<10} {row['corr_X2_Y']:>12.4f} {row['corr_X1_Y']:>12.4f} {int(row['sample_count']):>10}"
        )

    print("\n" + "-" * 70)
    print("📊 按时间段分组的相关性（部分）")
    print("-" * 70)
    time_corrs = stats["time_corrs"].sort_index()
    # 只显示部分关键时间点
    key_times = ["09:40", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00", "14:30"]
    print(f"{'时间点':<10} {'X2与Y相关性':>12} {'样本数':>10}")
    for t in key_times:
        if t in time_corrs.index:
            row = time_corrs.loc[t]
            print(f"{t:<10} {row['corr_X2_Y']:>12.4f} {int(row['sample_count']):>10}")

    print("\n" + "-" * 70)
    print("📊 Y 值统计")
    print("-" * 70)
    print(f"Y 均值: {df_valid['Y'].mean():.6f}")
    print(f"Y 标准差: {df_valid['Y'].std():.6f}")
    print(f"Y < 0.995 (买入机会) 占比: {(df_valid['Y'] < 0.995).mean()*100:.2f}%")
    print(f"Y > 1.005 (卖出机会) 占比: {(df_valid['Y'] > 1.005).mean()*100:.2f}%")
    print(
        f"0.995 <= Y <= 1.005 (观望) 占比: {((df_valid['Y'] >= 0.995) & (df_valid['Y'] <= 1.005)).mean()*100:.2f}%"
    )

    print("\n" + "-" * 70)
    print("📊 结论")
    print("-" * 70)
    if corr_x2 > 0.1:
        print("✅ X2_zscore 与 Y 呈显著正相关，老板的直觉是正确的！")
        print("   特征有预测力，可以继续推进模型训练。")
    elif corr_x2 > 0.05:
        print("⚠️ X2_zscore 与 Y 呈弱正相关，有一定预测力，但需要进一步优化特征。")
    elif corr_x2 > 0:
        print("⚠️ X2_zscore 与 Y 相关性很弱，可能需要调整特征或标签定义。")
    else:
        print("❌ X2_zscore 与 Y 呈负相关，与预期不符，需要重新审视逻辑。")

    print("=" * 70)


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("开始验证 X2_zscore 与 V_5m/V_rest 的相关性")
    logger.info("=" * 50)

    # 1. 加载1分钟数据
    df_1m = load_minute_data(TEST_STOCKS, YEAR)

    # 2. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 3. 计算特征（无未来函数）和标签（上帝视角）
    df_5m = calculate_features_and_labels(df_5m)

    # 4. 计算 Z-Score
    df_5m = calculate_zscore(df_5m, window=ZSCORE_WINDOW)

    # 5. 分析相关性
    stats = analyze_correlation(df_5m)

    # 6. 打印报告
    print_report(stats, df_5m)

    # 7. 保存结果
    output_file = OUTPUT_DIR / "x2_vrest_correlation_validation.csv"
    df_5m.to_csv(output_file, index=False)
    logger.success(f"结果已保存至: {output_file}")

    return df_5m, stats


if __name__ == "__main__":
    df, stats = main()
