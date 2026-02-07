"""
相关性测试脚本：X1/X2特征与Y标签的相关性分析

职责：读取特征生产脚本(validate_x1_x2.py)输出的特征数据，进行相关性分析
输入：df_features_{year}.pkl（包含特征+标签的完整数据）

分析内容：
1. 基础分析：X2_zscore / X1_zscore 与 Y 的整体相关性
2. 按股票分组的相关性
3. 按入场时间分组的相关性
4. Z_final 组合因子分析
5. 极值样本分析（|X2_zscore| > 阈值）
6. 多时间维度标签对比（Y, Y_15m ~ Y_120m）
7. Y 值分布统计

依赖：validate_x1_x2.py 生产的特征数据

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
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR = 2025
TIME_CUTOFF = "14:15"  # 时间截止点（剔除尾盘）
EXTREME_THRESHOLD = 1.0  # 极值阈值


def load_features(year: int) -> pd.DataFrame:
    """加载特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 validate_x1_x2.py"
        )

    logger.info(f"加载特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    n_stocks = df["SecuCode"].nunique()
    n_days = df["date"].nunique()
    logger.success(
        f"加载完成: {len(df):,} 行, {n_stocks} 只股票, {n_days} 个交易日"
    )
    return df


# ==================== 基础相关性分析 ====================
def analyze_basic_correlation(df: pd.DataFrame) -> dict:
    """基础相关性分析：X2_zscore / X1_zscore 与 Y"""
    logger.info("基础相关性分析...")

    valid_mask = (
        np.isfinite(df["X2_zscore"])
        & np.isfinite(df["X1_zscore"])
        & np.isfinite(df["Y"])
        & (df["Y"] > 0.9)
        & (df["Y"] < 1.1)
    )
    df_valid = df[valid_mask].copy()
    logger.info(f"过滤异常值后有效样本: {len(df_valid):,} (原 {len(df):,})")

    return {
        "n_samples": len(df_valid),
        "corr_X2_Y": df_valid["X2_zscore"].corr(df_valid["Y"]),
        "corr_X1_Y": df_valid["X1_zscore"].corr(df_valid["Y"]),
        "corr_Zfinal_Y": df_valid["Z_final"].corr(df_valid["Y"]),
        "df_valid": df_valid,
    }


# ==================== 高级相关性分析 ====================
def analyze_advanced_correlation(df_valid: pd.DataFrame) -> dict:
    """
    高级相关性分析：
    1. 剔除尾盘
    2. 极值样本
    3. 多时间维度标签
    4. 按股票/入场时间分组
    """
    logger.info("高级相关性分析...")
    results = {}

    # 1. 剔除尾盘
    df_filtered = df_valid[df_valid["entry_time"] < TIME_CUTOFF].copy()
    logger.info(
        f"剔除 {TIME_CUTOFF} 之后: {len(df_valid):,} -> {len(df_filtered):,}"
    )

    # 2. 全样本相关性（剔除尾盘后）
    results["full_sample"] = {
        "n_samples": len(df_filtered),
        "corr_X2_Y": df_filtered["X2_zscore"].corr(df_filtered["Y"]),
        "corr_X1_Y": df_filtered["X1_zscore"].corr(df_filtered["Y"]),
        "corr_Zfinal_Y": df_filtered["Z_final"].corr(df_filtered["Y"]),
    }

    # 3. 极值样本（|X2_zscore| > 阈值）
    extreme_mask = np.abs(df_filtered["X2_zscore"]) > EXTREME_THRESHOLD
    df_extreme = df_filtered[extreme_mask]
    logger.info(
        f"极值样本 (|X2_zscore| > {EXTREME_THRESHOLD}): {len(df_extreme):,}"
    )
    if len(df_extreme) > 100:
        results["extreme_sample"] = {
            "n_samples": len(df_extreme),
            "corr_X2_Y": df_extreme["X2_zscore"].corr(df_extreme["Y"]),
            "corr_X1_Y": df_extreme["X1_zscore"].corr(df_extreme["Y"]),
            "corr_Zfinal_Y": df_extreme["Z_final"].corr(df_extreme["Y"]),
        }

    # 4. 多时间维度标签相关性
    for label_col, label_name in [
        ("Y", "V_rest"),
        ("Y_15m", "V_15m"),
        ("Y_30m", "V_30m"),
        ("Y_60m", "V_60m"),
        ("Y_90m", "V_90m"),
        ("Y_120m", "V_120m"),
    ]:
        if label_col not in df_filtered.columns:
            continue
        mask = df_filtered[label_col].notna() & np.isfinite(df_filtered[label_col])
        df_label = df_filtered[mask]
        if len(df_label) > 100:
            results[f"horizon_{label_name}"] = {
                "n_samples": len(df_label),
                "corr_X2": df_label["X2_zscore"].corr(df_label[label_col]),
                "corr_Zfinal": df_label["Z_final"].corr(df_label[label_col]),
            }

    # 5. 按股票分组
    results["by_stock"] = df_filtered.groupby("SecuCode").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "corr_X1_Y": x["X1_zscore"].corr(x["Y"]),
                "corr_Zfinal_Y": x["Z_final"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )

    # 6. 按入场时间分组
    results["by_entry_time"] = df_filtered.groupby("entry_time").apply(
        lambda x: pd.Series(
            {
                "corr_X2_Y": x["X2_zscore"].corr(x["Y"]),
                "corr_Zfinal_Y": x["Z_final"].corr(x["Y"]),
                "sample_count": len(x),
            }
        )
    )

    # 7. Y 值统计
    results["y_stats"] = {
        "mean": df_filtered["Y"].mean(),
        "std": df_filtered["Y"].std(),
        "buy_ratio": (df_filtered["Y"] < 0.995).mean(),
        "sell_ratio": (df_filtered["Y"] > 1.005).mean(),
        "hold_ratio": (
            (df_filtered["Y"] >= 0.995) & (df_filtered["Y"] <= 1.005)
        ).mean(),
    }

    return results


# ==================== 报告打印 ====================
def print_report(basic: dict, advanced: dict):
    """打印完整的相关性验证报告"""
    print("\n" + "=" * 90)
    print("X2_zscore 与 V_5m/V_rest 相关性验证报告")
    print("=" * 90)

    # --- 基础分析 ---
    print(f"\n样本总数: {basic['n_samples']:,}")
    print(f"\n--- 整体相关性（全样本） ---")
    print(
        f"  {'Feature':<15} {'Corr with Y':>15} {'Status':>20}"
    )
    for feat, corr_key, desc in [
        ("X2_zscore", "corr_X2_Y", "能量偏差"),
        ("X1_zscore", "corr_X1_Y", "价格偏离"),
        ("Z_final", "corr_Zfinal_Y", "组合因子"),
    ]:
        c = basic[corr_key]
        status = "正相关" if c > 0.05 else ("弱" if abs(c) < 0.05 else "负相关")
        print(f"  {feat:<15} {c:>15.4f} {status:>20}")

    # --- 高级分析 ---
    full = advanced["full_sample"]
    print(f"\n--- 全样本分析（剔除 {TIME_CUTOFF} 之后，{full['n_samples']:,} 样本）---")
    print(f"  {'Feature':<15} {'Corr with Y':>15}")
    print(f"  {'X2_zscore':<15} {full['corr_X2_Y']:>15.4f}")
    print(f"  {'X1_zscore':<15} {full['corr_X1_Y']:>15.4f}")
    print(f"  {'Z_final':<15} {full['corr_Zfinal_Y']:>15.4f}")

    # 极值样本
    if "extreme_sample" in advanced:
        ext = advanced["extreme_sample"]
        print(
            f"\n--- 极值样本（|X2_zscore| > {EXTREME_THRESHOLD}，"
            f"{ext['n_samples']:,} 样本，"
            f"{ext['n_samples']/full['n_samples']*100:.1f}%）---"
        )
        print(f"  {'Feature':<15} {'Corr':>10} {'vs Full':>10}")
        for feat in ["corr_X2_Y", "corr_Zfinal_Y"]:
            name = feat.replace("corr_", "").replace("_Y", "")
            diff = ext[feat] - full[feat]
            print(
                f"  {name:<15} {ext[feat]:>10.4f} "
                f"{'+' if diff > 0 else ''}{diff:>9.4f}"
            )

    # 多时间维度
    print(f"\n--- 多时间维度标签对比 ---")
    print(f"  {'Horizon':<12} {'Samples':>10} {'X2 Corr':>10} {'Z_final Corr':>14}")
    for key in [
        "horizon_V_rest", "horizon_V_15m", "horizon_V_30m",
        "horizon_V_60m", "horizon_V_90m", "horizon_V_120m",
    ]:
        if key in advanced:
            h = advanced[key]
            name = key.replace("horizon_", "")
            print(
                f"  {name:<12} {h['n_samples']:>10,} "
                f"{h['corr_X2']:>10.4f} {h['corr_Zfinal']:>14.4f}"
            )

    # 按股票分组
    print(f"\n--- 按股票分组 ---")
    stock_corrs = advanced["by_stock"]
    print(
        f"  {'Stock':<10} {'X2 Corr':>10} {'X1 Corr':>10} "
        f"{'Z_final':>10} {'Samples':>10}"
    )
    for stock, row in stock_corrs.iterrows():
        print(
            f"  {stock:<10} {row['corr_X2_Y']:>10.4f} {row['corr_X1_Y']:>10.4f} "
            f"{row['corr_Zfinal_Y']:>10.4f} {int(row['sample_count']):>10}"
        )

    # 按入场时间分组（关键时间点）
    print(f"\n--- 按入场时间分组（关键时间点）---")
    time_corrs = advanced["by_entry_time"].sort_index()
    key_times = [
        "09:40", "09:45", "10:00", "10:30", "11:00",
        "13:05", "13:30", "14:00",
    ]
    print(f"  {'Time':<10} {'X2 Corr':>10} {'Z_final':>10} {'Samples':>10}")
    for t in key_times:
        if t in time_corrs.index:
            row = time_corrs.loc[t]
            print(
                f"  {t:<10} {row['corr_X2_Y']:>10.4f} "
                f"{row['corr_Zfinal_Y']:>10.4f} {int(row['sample_count']):>10}"
            )

    # Y 值统计
    y_stats = advanced["y_stats"]
    print(f"\n--- Y 值分布 ---")
    print(f"  均值: {y_stats['mean']:.6f}")
    print(f"  标准差: {y_stats['std']:.6f}")
    print(f"  买入机会 (Y < 0.995): {y_stats['buy_ratio']*100:.2f}%")
    print(f"  卖出机会 (Y > 1.005): {y_stats['sell_ratio']*100:.2f}%")
    print(f"  观望 (0.995 <= Y <= 1.005): {y_stats['hold_ratio']*100:.2f}%")

    # 结论
    print(f"\n--- 结论 ---")
    zf = full["corr_Zfinal_Y"]
    x2 = full["corr_X2_Y"]
    if zf > x2:
        print(f"  Z_final({zf:.4f}) 优于 X2({x2:.4f})，组合因子有效")
    else:
        print(f"  Z_final({zf:.4f}) 未优于 X2({x2:.4f})，需调整权重")

    if "extreme_sample" in advanced:
        ext_corr = advanced["extreme_sample"]["corr_Zfinal_Y"]
        if ext_corr > zf:
            print(f"  极值样本相关性({ext_corr:.4f})显著提升，波动率滤网有效")
        else:
            print(f"  极值样本相关性未提升，可能需要调整阈值")

    print("=" * 90)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("相关性测试：X1/X2 与 Y 的相关性分析")
    logger.info("=" * 60)

    # 1. 加载特征数据
    df = load_features(YEAR)

    # 2. 基础分析
    basic = analyze_basic_correlation(df)

    # 3. 高级分析
    advanced = analyze_advanced_correlation(basic["df_valid"])

    # 4. 打印报告
    print_report(basic, advanced)

    # 5. 保存分析结果
    output_file = OUTPUT_DIR / "x2_vrest_correlation_analysis.csv"
    basic["df_valid"].to_csv(output_file, index=False)
    logger.success(f"分析结果已保存: {output_file}")

    return basic, advanced


if __name__ == "__main__":
    basic, advanced = main()
