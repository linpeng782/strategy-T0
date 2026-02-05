"""
验证脚本 V2：X2_zscore 与 V_5m/V_rest 相关性验证（优化版）

优化内容：
1. 限制交易时间窗口：剔除 14:15 以后的样本（尾盘信号失效）
2. 重新构建组合因子：Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore
3. 引入波动率滤网：单独分析 |X2_zscore| > 1.0 的极值样本
4. 多时段标签：增加 Y_15m、Y_30m 对比

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

# 测试股票池
TEST_STOCKS = ["002591", "002494", "002247", "000001", "300750", "601318"]
YEAR = 2025
ZSCORE_WINDOW = 20  # Z-Score 滚动窗口
TIME_CUTOFF = "14:15"  # 时间截止点（剔除尾盘）
EXTREME_THRESHOLD = 1.0  # 极值阈值


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载指定股票的1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")

    logger.info(f"加载 {year} 年数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    result = df[df["SecuCode"].isin(stock_codes)].copy()
    if len(result) == 0:
        raise ValueError(f"未找到 {stock_codes} 的数据")

    result = result.sort_values(["SecuCode", "TradingDay"]).reset_index(drop=True)
    result["date"] = result["TradingDay"].dt.date
    result["time"] = result["TradingDay"].dt.strftime("%H:%M")

    n_stocks = result["SecuCode"].nunique()
    n_days = result["date"].nunique()
    logger.success(
        f"加载完成: {len(result):,} 条, {n_stocks} 只股票, {n_days} 个交易日"
    )
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
    agg_df["vwap_5m"] = agg_df["vwap_5m"].replace([np.inf, -np.inf], np.nan)
    agg_df["time_str"] = agg_df["bar_time"].dt.strftime("%H:%M")

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_cumulative_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个5分钟Bar的累积VWAP和TWAP"""
    logger.info("计算累积VWAP和TWAP...")

    result_list = []
    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("bar_time").copy()

        day_df["cum_volume"] = day_df["volume"].cumsum()
        day_df["cum_amount"] = day_df["amount"].cumsum()
        day_df["cum_vwap"] = day_df["cum_amount"] / day_df["cum_volume"]
        day_df["cum_twap"] = day_df["close"].expanding().mean()

        # 计算 X1 和 X2
        day_df["X1"] = day_df["close"] / day_df["cum_vwap"] - 1
        day_df["X2"] = day_df["cum_vwap"] / day_df["cum_twap"] - 1

        result_list.append(day_df)

    result = pd.concat(result_list, ignore_index=True)
    logger.success("累积VWAP计算完成")
    return result


def calculate_multi_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算多时间维度的标签：
    - Y_5m: V_5m / V_rest（原始标签）
    - Y_15m: V_5m / V_next_15m（15分钟内的回归）
    - Y_30m: V_5m / V_next_30m（30分钟内的回归）
    """
    logger.info("计算多时间维度标签 (Y_5m, Y_15m, Y_30m)...")

    result_list = []

    for (stock, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("bar_time").copy()
        n_bars = len(day_df)

        # 全天总成交
        total_volume = day_df["volume"].sum()
        total_amount = day_df["amount"].sum()

        # 初始化标签列
        day_df["V_rest"] = np.nan
        day_df["V_next_15m"] = np.nan
        day_df["V_next_30m"] = np.nan
        day_df["Y"] = np.nan
        day_df["Y_15m"] = np.nan
        day_df["Y_30m"] = np.nan

        for i in range(n_bars):
            # V_rest: 从当前Bar之后到收盘
            rest_volume = total_volume - day_df["cum_volume"].iloc[i]
            rest_amount = total_amount - day_df["cum_amount"].iloc[i]
            if rest_volume > 0:
                day_df.iloc[i, day_df.columns.get_loc("V_rest")] = (
                    rest_amount / rest_volume
                )
                day_df.iloc[i, day_df.columns.get_loc("Y")] = day_df["vwap_5m"].iloc[
                    i
                ] / (rest_amount / rest_volume)

            # V_next_15m: 未来15分钟（3个Bar）
            if i + 3 < n_bars:
                next_15m_slice = day_df.iloc[i + 1 : i + 4]
                vol_15m = next_15m_slice["volume"].sum()
                amt_15m = next_15m_slice["amount"].sum()
                if vol_15m > 0:
                    v_next_15m = amt_15m / vol_15m
                    day_df.iloc[i, day_df.columns.get_loc("V_next_15m")] = v_next_15m
                    day_df.iloc[i, day_df.columns.get_loc("Y_15m")] = (
                        day_df["vwap_5m"].iloc[i] / v_next_15m
                    )

            # V_next_30m: 未来30分钟（6个Bar）
            if i + 6 < n_bars:
                next_30m_slice = day_df.iloc[i + 1 : i + 7]
                vol_30m = next_30m_slice["volume"].sum()
                amt_30m = next_30m_slice["amount"].sum()
                if vol_30m > 0:
                    v_next_30m = amt_30m / vol_30m
                    day_df.iloc[i, day_df.columns.get_loc("V_next_30m")] = v_next_30m
                    day_df.iloc[i, day_df.columns.get_loc("Y_30m")] = (
                        day_df["vwap_5m"].iloc[i] / v_next_30m
                    )

        result_list.append(day_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤无效样本
    valid_mask = result["Y"].notna() & (result["Y"] > 0.9) & (result["Y"] < 1.1)
    result = result[valid_mask]

    logger.success(f"多时间维度标签计算完成, 有效样本: {len(result):,}")
    return result


def calculate_zscore_and_zfinal(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算 Z-Score 和组合因子 Z_final
    Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore
    """
    logger.info(f"计算 Z-Score 和 Z_final (窗口={window}天)...")

    result_list = []

    for (stock, time_str), group_df in df.groupby(["SecuCode", "time_str"]):
        group_df = group_df.sort_values("date").copy()

        # X2 的 Z-Score
        group_df["X2_mean"] = (
            group_df["X2"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X2_std"] = (
            group_df["X2"].rolling(window, min_periods=5).std().shift(1)
        )
        group_df["X2_zscore"] = (group_df["X2"] - group_df["X2_mean"]) / group_df[
            "X2_std"
        ]

        # X1 的 Z-Score
        group_df["X1_mean"] = (
            group_df["X1"].rolling(window, min_periods=5).mean().shift(1)
        )
        group_df["X1_std"] = (
            group_df["X1"].rolling(window, min_periods=5).std().shift(1)
        )
        group_df["X1_zscore"] = (group_df["X1"] - group_df["X1_mean"]) / group_df[
            "X1_std"
        ]

        # 组合因子 Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore
        group_df["Z_final"] = 2.5 * group_df["X2_zscore"] - 0.5 * group_df["X1_zscore"]

        result_list.append(group_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤无效样本
    valid_mask = result["X2_zscore"].notna() & result["X1_zscore"].notna()
    result = result[valid_mask]

    logger.success(f"Z-Score 和 Z_final 计算完成, 有效样本: {len(result):,}")
    return result


def analyze_correlation_v2(df: pd.DataFrame) -> dict:
    """
    V2 版相关性分析：
    1. 剔除尾盘（14:15之后）
    2. 分析全样本 + 极值样本
    3. 分析多时间维度标签
    """
    logger.info("=" * 50)
    logger.info("V2 版相关性分析")
    logger.info("=" * 50)

    # 1. 剔除尾盘
    df_filtered = df[df["time_str"] < TIME_CUTOFF].copy()
    logger.info(f"剔除 {TIME_CUTOFF} 之后的样本: {len(df):,} -> {len(df_filtered):,}")

    # 2. 过滤异常值
    valid_mask = (
        np.isfinite(df_filtered["X2_zscore"])
        & np.isfinite(df_filtered["X1_zscore"])
        & np.isfinite(df_filtered["Z_final"])
        & np.isfinite(df_filtered["Y"])
    )
    df_valid = df_filtered[valid_mask].copy()
    logger.info(f"过滤异常值后: {len(df_valid):,}")

    # 3. 极值样本（|X2_zscore| > 1.0）
    extreme_mask = np.abs(df_valid["X2_zscore"]) > EXTREME_THRESHOLD
    df_extreme = df_valid[extreme_mask].copy()
    logger.info(f"极值样本 (|X2_zscore| > {EXTREME_THRESHOLD}): {len(df_extreme):,}")

    # 4. 计算相关性
    results = {}

    # 4.1 全样本相关性
    results["full_sample"] = {
        "n_samples": len(df_valid),
        "corr_X2_Y": df_valid["X2_zscore"].corr(df_valid["Y"]),
        "corr_X1_Y": df_valid["X1_zscore"].corr(df_valid["Y"]),
        "corr_Zfinal_Y": df_valid["Z_final"].corr(df_valid["Y"]),
    }

    # 4.2 极值样本相关性
    if len(df_extreme) > 100:
        results["extreme_sample"] = {
            "n_samples": len(df_extreme),
            "corr_X2_Y": df_extreme["X2_zscore"].corr(df_extreme["Y"]),
            "corr_X1_Y": df_extreme["X1_zscore"].corr(df_extreme["Y"]),
            "corr_Zfinal_Y": df_extreme["Z_final"].corr(df_extreme["Y"]),
        }

    # 4.3 多时间维度标签相关性
    for label_col, label_name in [
        ("Y", "V_rest"),
        ("Y_15m", "V_15m"),
        ("Y_30m", "V_30m"),
    ]:
        mask = df_valid[label_col].notna() & np.isfinite(df_valid[label_col])
        df_label = df_valid[mask]
        if len(df_label) > 100:
            results[f"horizon_{label_name}"] = {
                "n_samples": len(df_label),
                "corr_X2": df_label["X2_zscore"].corr(df_label[label_col]),
                "corr_Zfinal": df_label["Z_final"].corr(df_label[label_col]),
            }

    # 4.4 按股票分组
    stock_corrs = df_valid.groupby("SecuCode").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "corr_Zfinal_Y": x["Z_final"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )
    results["by_stock"] = stock_corrs

    # 4.5 按时间段分组
    time_corrs = df_valid.groupby("time_str").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "corr_Zfinal_Y": x["Z_final"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )
    results["by_time"] = time_corrs

    # 4.6 Y 值统计
    results["y_stats"] = {
        "mean": df_valid["Y"].mean(),
        "std": df_valid["Y"].std(),
        "buy_ratio": (df_valid["Y"] < 0.995).mean(),
        "sell_ratio": (df_valid["Y"] > 1.005).mean(),
        "hold_ratio": ((df_valid["Y"] >= 0.995) & (df_valid["Y"] <= 1.005)).mean(),
    }

    return results


def print_report_v2(results: dict):
    """打印 V2 版验证报告"""
    print("\n" + "=" * 80)
    print("📊 X2_zscore 与 Y 相关性验证报告 V2（优化版）")
    print("=" * 80)

    # 全样本
    full = results["full_sample"]
    print(f"\n📈 全样本分析（剔除 {TIME_CUTOFF} 之后）")
    print("-" * 80)
    print(f"样本数: {full['n_samples']:,}")
    print(f"{'特征':<15} {'与 Y 相关性':>15} {'说明':>20}")
    print(
        f"{'X2_zscore':<15} {full['corr_X2_Y']:>15.4f} {'✅ 正相关' if full['corr_X2_Y'] > 0.05 else '⚠️ 弱'}"
    )
    print(
        f"{'X1_zscore':<15} {full['corr_X1_Y']:>15.4f} {'动量效应' if full['corr_X1_Y'] < 0 else ''}"
    )
    print(
        f"{'Z_final':<15} {full['corr_Zfinal_Y']:>15.4f} {'✅ 组合优化' if full['corr_Zfinal_Y'] > full['corr_X2_Y'] else ''}"
    )

    # 极值样本
    if "extreme_sample" in results:
        ext = results["extreme_sample"]
        print(f"\n📈 极值样本分析（|X2_zscore| > {EXTREME_THRESHOLD}）")
        print("-" * 80)
        print(
            f"样本数: {ext['n_samples']:,} ({ext['n_samples']/full['n_samples']*100:.1f}%)"
        )
        print(f"{'特征':<15} {'与 Y 相关性':>15} {'相比全样本':>20}")
        improve_x2 = ext["corr_X2_Y"] - full["corr_X2_Y"]
        improve_zf = ext["corr_Zfinal_Y"] - full["corr_Zfinal_Y"]
        print(
            f"{'X2_zscore':<15} {ext['corr_X2_Y']:>15.4f} {'+' if improve_x2 > 0 else ''}{improve_x2:.4f}"
        )
        print(
            f"{'Z_final':<15} {ext['corr_Zfinal_Y']:>15.4f} {'+' if improve_zf > 0 else ''}{improve_zf:.4f}"
        )

    # 多时间维度
    print(f"\n📈 多时间维度标签对比")
    print("-" * 80)
    print(f"{'时间维度':<15} {'样本数':>10} {'X2相关性':>12} {'Z_final相关性':>15}")
    for key in ["horizon_V_rest", "horizon_V_15m", "horizon_V_30m"]:
        if key in results:
            h = results[key]
            name = key.replace("horizon_", "")
            print(
                f"{name:<15} {h['n_samples']:>10,} {h['corr_X2']:>12.4f} {h['corr_Zfinal']:>15.4f}"
            )

    # 按股票分组
    print(f"\n📈 按股票分组")
    print("-" * 80)
    stock_corrs = results["by_stock"]
    print(
        f"{'股票代码':<10} {'X2与Y相关性':>15} {'Z_final与Y相关性':>18} {'样本数':>10}"
    )
    for stock, row in stock_corrs.iterrows():
        print(
            f"{stock:<10} {row['corr_X2_Y']:>15.4f} {row['corr_Zfinal_Y']:>18.4f} {int(row['sample_count']):>10}"
        )

    # 按时间段分组（关键时间点）
    print(f"\n📈 按时间段分组（关键时间点）")
    print("-" * 80)
    time_corrs = results["by_time"].sort_index()
    key_times = ["09:35", "09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00"]
    print(f"{'时间点':<10} {'X2与Y相关性':>15} {'Z_final与Y相关性':>18} {'样本数':>10}")
    for t in key_times:
        if t in time_corrs.index:
            row = time_corrs.loc[t]
            print(
                f"{t:<10} {row['corr_X2_Y']:>15.4f} {row['corr_Zfinal_Y']:>18.4f} {int(row['sample_count']):>10}"
            )

    # Y 值统计
    y_stats = results["y_stats"]
    print(f"\n📈 Y 值分布")
    print("-" * 80)
    print(f"Y 均值: {y_stats['mean']:.6f}")
    print(f"Y 标准差: {y_stats['std']:.6f}")
    print(f"买入机会 (Y < 0.995): {y_stats['buy_ratio']*100:.2f}%")
    print(f"卖出机会 (Y > 1.005): {y_stats['sell_ratio']*100:.2f}%")
    print(f"观望 (0.995 ≤ Y ≤ 1.005): {y_stats['hold_ratio']*100:.2f}%")

    # 结论
    print("\n" + "=" * 80)
    print("📊 结论")
    print("=" * 80)

    zfinal_corr = full["corr_Zfinal_Y"]
    x2_corr = full["corr_X2_Y"]

    if zfinal_corr > x2_corr:
        print(f"✅ Z_final 组合因子（{zfinal_corr:.4f}）优于单一 X2（{x2_corr:.4f}）")
    else:
        print(
            f"⚠️ Z_final 组合因子（{zfinal_corr:.4f}）未能优于 X2（{x2_corr:.4f}），需调整权重"
        )

    if "extreme_sample" in results:
        ext_corr = results["extreme_sample"]["corr_Zfinal_Y"]
        if ext_corr > zfinal_corr:
            print(f"✅ 极值样本相关性（{ext_corr:.4f}）显著提升，波动率滤网有效")
        else:
            print(f"⚠️ 极值样本相关性未提升，可能需要调整阈值")

    print("=" * 80)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("开始 V2 版验证：X2_zscore 与 V_5m/V_rest 相关性")
    logger.info("=" * 60)

    # 1. 加载数据
    df_1m = load_minute_data(TEST_STOCKS, YEAR)

    # 2. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 3. 计算累积VWAP和特征
    df_5m = calculate_cumulative_vwap(df_5m)

    # 4. 计算多时间维度标签
    df_5m = calculate_multi_horizon_labels(df_5m)

    # 5. 计算 Z-Score 和 Z_final
    df_5m = calculate_zscore_and_zfinal(df_5m, window=ZSCORE_WINDOW)

    # 6. 分析相关性
    results = analyze_correlation_v2(df_5m)

    # 7. 打印报告
    print_report_v2(results)

    # 8. 保存结果
    output_file = OUTPUT_DIR / "x2_vrest_correlation_validation_v2.csv"
    df_5m.to_csv(output_file, index=False)
    logger.success(f"结果已保存至: {output_file}")

    return df_5m, results


if __name__ == "__main__":
    df, results = main()
