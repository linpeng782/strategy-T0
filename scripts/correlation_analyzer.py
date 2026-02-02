"""
相关性分析模块

负责计算特征与标签的相关性，输出统计报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


def calculate_correlation(df: pd.DataFrame, features: List[str] = None) -> Dict:
    """
    计算特征与Y的相关系数

    Args:
        df: 特征数据 DataFrame，需包含 X1, X2, Z, Y 列
        features: 要计算相关性的特征列表，默认 ["X1", "X2", "Z"]

    Returns:
        Dict: 相关性统计结果
    """
    if features is None:
        features = ["X1", "X2", "Z"]

    result = {
        "n_samples": len(df),
        "n_stocks": df["stock_code"].nunique() if "stock_code" in df.columns else 1,
        "n_days": df["date"].nunique() if "date" in df.columns else len(df),
    }

    # 计算各特征与Y的相关系数
    for feat in features:
        if feat in df.columns:
            corr = df[feat].corr(df["Y"])
            result[f"corr_{feat}_Y"] = corr

    # 计算特征之间的相关系数
    if "X1" in df.columns and "X2" in df.columns:
        result["corr_X1_X2"] = df["X1"].corr(df["X2"])

    # 计算统计量
    for col in features + ["Y"]:
        if col in df.columns:
            result[f"{col}_mean"] = df[col].mean()
            result[f"{col}_std"] = df[col].std()

    return result


def analyze_by_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    按股票分组分析相关性

    Args:
        df: 特征数据 DataFrame

    Returns:
        DataFrame: 每只股票的相关性分析结果
    """
    results = []

    for stock_code, stock_df in df.groupby("stock_code"):
        if len(stock_df) < 10:  # 样本太少跳过
            continue

        stats = calculate_correlation(stock_df)
        stats["stock_code"] = stock_code
        results.append(stats)

    result_df = pd.DataFrame(results)

    # 调整列顺序
    cols = [
        "stock_code",
        "n_samples",
        "corr_X1_Y",
        "corr_X2_Y",
        "corr_Z_Y",
        "X1_std",
        "X2_std",
        "Y_std",
        "Y_mean",
    ]
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]

    return result_df.sort_values("corr_Z_Y")


def print_correlation_report(
    stats: Dict,
    stock_code: str = None,
    quantile_df: pd.DataFrame = None,
    resonance_stats: Dict = None,
):
    """
    打印相关性分析报告

    Args:
        stats: 相关性统计结果
        stock_code: 股票代码
        quantile_df: 分位数分析结果
        resonance_stats: 共振信号统计结果
    """
    print("\n" + "=" * 70)
    print("📊 日内做T时序模型 - 相关性验证报告")
    print("=" * 70)

    if stock_code:
        print(f"股票: {stock_code}")
    print(f"样本数: {stats['n_samples']} 条记录")
    if stats.get("n_stocks", 1) > 1:
        print(f"股票数: {stats['n_stocks']} 只")
    print(f"交易日: {stats['n_days']} 天")

    print("\n" + "-" * 70)
    print("📈 特征定义")
    print("-" * 70)
    print("X1 (价格偏离) = Price / VWAP - 1")
    print("X2 (能量偏离) = VWAP / TWAP - 1")
    print("Z  (组合因子) = X1 + 2 * X2")
    print("Y  (下午收益) = Price_exit / Price_obs - 1")

    print("\n" + "-" * 70)
    print("📊 相关性分析")
    print("-" * 70)

    for feat in ["X1", "X2", "Z"]:
        key = f"corr_{feat}_Y"
        if key in stats:
            corr = stats[key]
            sign = "✅ 负相关" if corr < 0 else "⚠️ 正相关"
            print(f"{feat} 与 Y 相关系数: {corr:>10.4f}  {sign}")

    if "corr_X1_X2" in stats:
        print(f"X1 与 X2 相关系数: {stats['corr_X1_X2']:>10.4f}")

    print("\n" + "-" * 70)
    print("📊 特征统计")
    print("-" * 70)
    for feat in ["X1", "X2", "Z", "Y"]:
        mean_key = f"{feat}_mean"
        std_key = f"{feat}_std"
        if mean_key in stats and std_key in stats:
            print(
                f"{feat} 均值: {stats[mean_key]*100:>8.4f}%  标准差: {stats[std_key]*100:>8.4f}%"
            )

    # 共振信号分析
    if resonance_stats:
        print("\n" + "-" * 70)
        print("📊 共振信号分析")
        print("-" * 70)
        print(f"超跌共振 (X1<-1% 且 X2<-0.5%):")
        print(
            f"  触发次数: {resonance_stats['resonance_count']} ({resonance_stats['resonance_ratio']*100:.1f}%)"
        )
        print(f"  胜率(Y>0.3%): {resonance_stats['resonance_win_rate']*100:.1f}%")
        print(f"  平均收益: {resonance_stats['resonance_avg_return']*100:.4f}%")
        print(f"超涨共振 (X1>1% 且 X2>0.5%):")
        print(
            f"  触发次数: {resonance_stats['reverse_count']} ({resonance_stats['reverse_ratio']*100:.1f}%)"
        )
        print(f"  胜率(Y<-0.3%): {resonance_stats['reverse_win_rate']*100:.1f}%")
        print(f"  平均收益: {resonance_stats['reverse_avg_return']*100:.4f}%")

    # 分位数分析
    if quantile_df is not None and len(quantile_df) > 0:
        print("\n" + "-" * 70)
        print("📊 Z因子分位数分析")
        print("-" * 70)
        print(f"{'分位组':<8} {'Z均值':>12} {'Y均值':>12} {'样本数':>8}")
        for _, row in quantile_df.iterrows():
            z_col = (
                [c for c in row.index if c.endswith("_mean") and c != "Y_mean"][0]
                if any(c.endswith("_mean") and c != "Y_mean" for c in row.index)
                else None
            )
            z_val = row[z_col] if z_col else 0
            print(
                f"{row['quantile']:<8} {z_val*100:>11.4f}% {row['Y_mean']*100:>11.4f}% {int(row['count']):>8}"
            )

    print("=" * 70)


def print_batch_summary(stock_results: pd.DataFrame):
    """
    打印批量测试汇总

    Args:
        stock_results: 按股票分组的分析结果
    """
    print("\n" + "=" * 70)
    print("📊 批量测试汇总")
    print("=" * 70)
    print(
        f"{'股票代码':<10} {'X1-Y相关':>10} {'X2-Y相关':>10} {'Z-Y相关':>10} {'Y波动':>10} {'样本数':>8}"
    )
    print("-" * 70)

    for _, row in stock_results.iterrows():
        corr_x1 = row.get("corr_X1_Y", 0)
        corr_x2 = row.get("corr_X2_Y", 0)
        corr_z = row.get("corr_Z_Y", 0)
        y_std = row.get("Y_std", 0)
        print(
            f"{row['stock_code']:<10} {corr_x1:>10.4f} {corr_x2:>10.4f} {corr_z:>10.4f} {y_std*100:>9.2f}% {int(row['n_samples']):>8}"
        )

    print("-" * 70)

    # 统计负相关的股票数量
    n_x1_neg = (stock_results["corr_X1_Y"] < 0).sum()
    n_x2_neg = (stock_results["corr_X2_Y"] < 0).sum()
    n_z_neg = (
        (stock_results["corr_Z_Y"] < 0).sum()
        if "corr_Z_Y" in stock_results.columns
        else 0
    )
    n_total = len(stock_results)

    print(f"X1负相关股票: {n_x1_neg}/{n_total} ({n_x1_neg/n_total*100:.1f}%)")
    print(f"X2负相关股票: {n_x2_neg}/{n_total} ({n_x2_neg/n_total*100:.1f}%)")
    print(f"Z负相关股票:  {n_z_neg}/{n_total} ({n_z_neg/n_total*100:.1f}%)")

    # 平均相关系数
    print(f"\n平均相关系数:")
    print(f"  X1-Y: {stock_results['corr_X1_Y'].mean():.4f}")
    print(f"  X2-Y: {stock_results['corr_X2_Y'].mean():.4f}")
    if "corr_Z_Y" in stock_results.columns:
        print(f"  Z-Y:  {stock_results['corr_Z_Y'].mean():.4f}")

    print("=" * 70)
