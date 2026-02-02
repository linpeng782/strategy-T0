"""
日内做T时序模型 - 主运行脚本

支持用户输入股票代码列表进行批量测试
包含组合因子Z和共振信号分析

使用方法:
    # 在脚本中修改 STOCK_CODES 列表
    python run_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# 导入模块
from data_loader import load_minute_data
from feature_calculator import (
    calculate_intraday_features,
    calculate_resonance_signals,
    calculate_quantile_analysis,
)
from correlation_analyzer import (
    calculate_correlation,
    analyze_by_stock,
    print_correlation_report,
    print_batch_summary,
)


# ==================== 配置参数 ====================
# 在这里修改要测试的股票代码列表
STOCK_CODES = ["002591", "002494", "002247"]

# 时间参数
YEAR = 2025
OBSERVE_TIME = "10:30"  # 观察时间点
EXIT_TIME = "14:50"  # 出场时间点

# 组合因子权重
X2_WEIGHT = 2.0  # Z = X1 + X2_WEIGHT * X2

# Z-Score 参数
ZSCORE_WINDOW = 20  # Z-Score 滚动窗口（天数）
Z_THRESHOLD = -2.0  # Z-Score 阈值（标准差倍数）
Y_PROFIT_THRESHOLD = 0.003  # Y > 0.3% 视为盈利

# 输出目录
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/scripts/output")


def run_single_stock_analysis(
    stock_code: str,
    year: int = YEAR,
    observe_time: str = OBSERVE_TIME,
    exit_time: str = EXIT_TIME,
    x2_weight: float = X2_WEIGHT,
    zscore_window: int = ZSCORE_WINDOW,
    verbose: bool = True,
) -> dict:
    """
    对单只股票进行完整分析

    Args:
        stock_code: 股票代码
        year: 年份
        observe_time: 观察时间点
        exit_time: 出场时间点
        x2_weight: X2权重
        zscore_window: Z-Score 滚动窗口
        verbose: 是否打印详细报告

    Returns:
        dict: 分析结果
    """
    # 1. 加载数据
    df = load_minute_data(stock_code, year=year)

    # 2. 计算特征（含 Z-Score 标准化）
    feature_df = calculate_intraday_features(
        df,
        observe_time=observe_time,
        exit_time=exit_time,
        x2_weight=x2_weight,
        zscore_window=zscore_window,
    )

    # 3. 相关性分析
    stats = calculate_correlation(feature_df)

    # 4. 共振信号分析（使用 Z-Score 阈值）
    resonance_stats = calculate_resonance_signals(
        feature_df,
        z_threshold=Z_THRESHOLD,
        y_profit_threshold=Y_PROFIT_THRESHOLD,
        use_zscore=True,
    )

    # 5. 分位数分析（使用标准化组合因子 Z_final）
    valid_df = feature_df[feature_df["Z_final"].notna()]
    quantile_df = calculate_quantile_analysis(
        valid_df, feature_col="Z_final", n_quantiles=5
    )

    # 6. 打印报告
    if verbose:
        print_correlation_report(stats, stock_code, quantile_df, resonance_stats)

    return {
        "stock_code": stock_code,
        "feature_df": feature_df,
        "stats": stats,
        "resonance_stats": resonance_stats,
        "quantile_df": quantile_df,
    }


def run_batch_analysis(
    stock_codes: list,
    year: int = YEAR,
    observe_time: str = OBSERVE_TIME,
    exit_time: str = EXIT_TIME,
    x2_weight: float = X2_WEIGHT,
    zscore_window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    批量分析多只股票

    Args:
        stock_codes: 股票代码列表
        year: 年份
        observe_time: 观察时间点
        exit_time: 出场时间点
        x2_weight: X2权重
        zscore_window: Z-Score 滚动窗口

    Returns:
        DataFrame: 汇总结果
    """
    logger.info(f"开始批量分析 {len(stock_codes)} 只股票...")

    # 1. 一次性加载所有股票数据
    df = load_minute_data(stock_codes, year=year)

    # 2. 计算特征（含 Z-Score 标准化）
    feature_df = calculate_intraday_features(
        df,
        observe_time=observe_time,
        exit_time=exit_time,
        x2_weight=x2_weight,
        zscore_window=zscore_window,
    )

    # 3. 按股票分组分析（使用 Z-Score 特征）
    stock_results = analyze_by_stock(feature_df, use_zscore=True)

    # 4. 整体共振信号分析（使用 Z-Score 阈值）
    resonance_stats = calculate_resonance_signals(
        feature_df,
        z_threshold=Z_THRESHOLD,
        y_profit_threshold=Y_PROFIT_THRESHOLD,
        use_zscore=True,
    )

    # 5. 打印汇总
    print_batch_summary(stock_results, use_zscore=True)

    # 6. 打印整体共振信号（Z-Score 模式）
    print("\n" + "-" * 70)
    print("📊 整体共振信号分析（Z-Score 模式）")
    print("-" * 70)
    print(f"有效样本数: {resonance_stats['total_samples']}")
    print(f"Z-Score 阈值: {resonance_stats['z_threshold']} 个标准差")
    print(f"超跌共振 (Z_X1 < {Z_THRESHOLD} 且 Z_X2 < {Z_THRESHOLD}):")
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
    print(f"超涨共振 (Z_X1 > {-Z_THRESHOLD} 且 Z_X2 > {-Z_THRESHOLD}):")
    print(
        f"  触发次数: {resonance_stats['reverse_count']} ({resonance_stats['reverse_ratio']*100:.1f}%)"
    )
    print(f"  胜率(Y<-0.3%): {resonance_stats['reverse_win_rate']*100:.1f}%")
    print(f"  平均收益: {resonance_stats['reverse_avg_return']*100:.4f}%")

    return stock_results, feature_df, resonance_stats


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("日内做T时序模型 - Z-Score 标准化版本")
    logger.info(f"股票列表: {STOCK_CODES}")
    logger.info(f"观察时点: {OBSERVE_TIME} | 出场时点: {EXIT_TIME}")
    logger.info(f"组合因子: Z_final = Z_X1 + {X2_WEIGHT} * Z_X2")
    logger.info(f"Z-Score 窗口: {ZSCORE_WINDOW} 天 | 共振阈值: {Z_THRESHOLD} 个标准差")
    logger.info("=" * 70)

    # 批量分析
    stock_results, feature_df, resonance_stats = run_batch_analysis(
        STOCK_CODES,
        year=YEAR,
        observe_time=OBSERVE_TIME,
        exit_time=EXIT_TIME,
        x2_weight=X2_WEIGHT,
        zscore_window=ZSCORE_WINDOW,
    )

    # 保存结果
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 保存特征数据
    feature_df.to_csv(OUTPUT_DIR / "intraday_features.csv", index=False)

    # 保存汇总结果
    stock_results.to_csv(OUTPUT_DIR / "stock_correlation_summary.csv", index=False)

    logger.success(f"结果已保存至: {OUTPUT_DIR}")

    return stock_results, feature_df, resonance_stats


if __name__ == "__main__":
    stock_results, feature_df, resonance_stats = main()
