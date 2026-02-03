"""相关性分析核心逻辑"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger


def calculate_correlation(
    df: pd.DataFrame,
    features: List[str] = None,
) -> Dict:
    """
    计算特征与收益的相关性

    Args:
        df: 特征数据 DataFrame
        features: 要分析的特征列表，默认 ["X1", "X2", "Z"]

    Returns:
        Dict: 相关性统计结果
    """
    if features is None:
        features = ["X1", "X2", "Z"]

    result = {"n_samples": len(df)}

    # 计算相关系数
    for col in features:
        if col in df.columns and "Y" in df.columns:
            valid_mask = df[col].notna() & df["Y"].notna()
            if valid_mask.sum() > 2:
                corr = df.loc[valid_mask, col].corr(df.loc[valid_mask, "Y"])
                result[f"corr_{col}_Y"] = corr
            else:
                result[f"corr_{col}_Y"] = np.nan

    # 计算统计量
    for col in features + ["Y"]:
        if col in df.columns:
            result[f"{col}_mean"] = df[col].mean()
            result[f"{col}_std"] = df[col].std()

    return result


def analyze_by_stock(df: pd.DataFrame, use_zscore: bool = False) -> pd.DataFrame:
    """
    按股票分组分析相关性

    Args:
        df: 特征数据 DataFrame
        use_zscore: 是否使用 Z-Score 特征

    Returns:
        DataFrame: 每只股票的相关性分析结果
    """
    results = []

    # 确定要分析的特征列
    if use_zscore:
        features = ["Z_X1", "Z_X2", "Z_final"]
    else:
        features = ["X1", "X2", "Z"]

    for stock_code, stock_df in df.groupby("stock_code"):
        # 过滤有效样本
        if use_zscore:
            valid_df = stock_df[stock_df["Z_X1"].notna() & stock_df["Z_X2"].notna()]
        else:
            valid_df = stock_df

        if len(valid_df) < 10:  # 样本太少跳过
            continue

        stats = calculate_correlation(valid_df, features=features)
        stats["stock_code"] = stock_code
        results.append(stats)

    result_df = pd.DataFrame(results)

    # 调整列顺序
    if use_zscore:
        cols = [
            "stock_code",
            "n_samples",
            "corr_Z_X1_Y",
            "corr_Z_X2_Y",
            "corr_Z_final_Y",
            "Z_X1_std",
            "Z_X2_std",
            "Y_std",
            "Y_mean",
        ]
        sort_col = "corr_Z_final_Y"
    else:
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
        sort_col = "corr_Z_Y"

    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]

    return (
        result_df.sort_values(sort_col) if sort_col in result_df.columns else result_df
    )


def print_batch_summary(
    stock_results: pd.DataFrame, use_zscore: bool = False, z_threshold: float = -1.0
):
    """
    打印批量测试汇总

    Args:
        stock_results: 按股票分组的分析结果
        use_zscore: 是否使用 Z-Score 特征
        z_threshold: Z-Score 阈值
    """
    print("\n" + "=" * 70)
    if use_zscore:
        print("📊 批量测试汇总（Z-Score 标准化）")
    else:
        print("📊 批量测试汇总")
    print("=" * 70)

    if use_zscore:
        # Z-Score 模式
        print(
            f"{'股票代码':<10} {'Z_X1-Y':>10} {'Z_X2-Y':>10} {'Z_final-Y':>10} {'Y波动':>10} {'样本数':>8}"
        )
        print("-" * 70)

        for _, row in stock_results.iterrows():
            corr_x1 = row.get("corr_Z_X1_Y", 0)
            corr_x2 = row.get("corr_Z_X2_Y", 0)
            corr_z = row.get("corr_Z_final_Y", 0)
            y_std = row.get("Y_std", 0)
            print(
                f"{row['stock_code']:<10} {corr_x1:>10.4f} {corr_x2:>10.4f} {corr_z:>10.4f} {y_std*100:>9.2f}% {int(row['n_samples']):>8}"
            )

        print("-" * 70)

        # 统计负相关的股票数量
        n_x1_neg = (
            (stock_results["corr_Z_X1_Y"] < 0).sum()
            if "corr_Z_X1_Y" in stock_results.columns
            else 0
        )
        n_x2_neg = (
            (stock_results["corr_Z_X2_Y"] < 0).sum()
            if "corr_Z_X2_Y" in stock_results.columns
            else 0
        )
        n_z_neg = (
            (stock_results["corr_Z_final_Y"] < 0).sum()
            if "corr_Z_final_Y" in stock_results.columns
            else 0
        )
        n_total = len(stock_results)

        print(f"Z_X1负相关股票: {n_x1_neg}/{n_total} ({n_x1_neg/n_total*100:.1f}%)")
        print(f"Z_X2负相关股票: {n_x2_neg}/{n_total} ({n_x2_neg/n_total*100:.1f}%)")
        print(f"Z_final负相关股票: {n_z_neg}/{n_total} ({n_z_neg/n_total*100:.1f}%)")

        # 平均相关系数
        print(f"\n平均相关系数:")
        if "corr_Z_X1_Y" in stock_results.columns:
            print(f"  Z_X1-Y: {stock_results['corr_Z_X1_Y'].mean():.4f}")
        if "corr_Z_X2_Y" in stock_results.columns:
            print(f"  Z_X2-Y: {stock_results['corr_Z_X2_Y'].mean():.4f}")
        if "corr_Z_final_Y" in stock_results.columns:
            print(f"  Z_final-Y: {stock_results['corr_Z_final_Y'].mean():.4f}")
    else:
        # 原始特征模式
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


def print_resonance_summary(resonance_stats: Dict, z_threshold: float):
    """
    打印共振信号汇总

    Args:
        resonance_stats: 共振信号统计结果
        z_threshold: Z-Score 阈值
    """
    print("\n" + "-" * 70)
    print("📊 整体共振信号分析（Z-Score 模式）")
    print("-" * 70)
    print(f"有效样本数: {resonance_stats['total_samples']}")
    print(f"Z-Score 阈值: {resonance_stats['z_threshold']} 个标准差")
    print(f"超跌共振 (Z_X1 < {z_threshold} 且 Z_X2 < {z_threshold}):")
    print(
        f"  触发次数: {resonance_stats['resonance_count']} ({resonance_stats['resonance_ratio']*100:.1f}%)"
    )
    print(f"  胜率(Y>0.3%): {resonance_stats['resonance_win_rate']*100:.1f}%")
    print(f"  平均收益: {resonance_stats['resonance_avg_return']*100:.4f}%")
    if resonance_stats["resonance_count"] > 0:
        print(f"  收益标准差: {resonance_stats['resonance_std_return']*100:.4f}%")
        print("  触发详情:")
        for detail in resonance_stats["resonance_details"][:5]:  # 最多显示5条
            print(
                f"    {detail['date']} {detail['stock_code']}: Z_X1={detail['Z_X1']:.2f}, Z_X2={detail['Z_X2']:.2f}, Y={detail['Y']*100:.2f}%"
            )
    print(f"超涨共振 (Z_X1 > {-z_threshold} 且 Z_X2 > {-z_threshold}):")
    print(
        f"  触发次数: {resonance_stats['reverse_count']} ({resonance_stats['reverse_ratio']*100:.1f}%)"
    )
    print(f"  胜率(Y<-0.3%): {resonance_stats['reverse_win_rate']*100:.1f}%")
    print(f"  平均收益: {resonance_stats['reverse_avg_return']*100:.4f}%")
