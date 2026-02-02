"""
特征计算模块

负责计算日内做T相关的特征和标签
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


def calculate_intraday_features(
    df: pd.DataFrame,
    observe_time: str = "10:30",
    exit_time: str = "14:50",
    x2_weight: float = 2.0,
) -> pd.DataFrame:
    """
    计算日内特征和标签

    Args:
        df: 分钟数据 DataFrame，需包含 date, time, SecuCode, close, volume, amount 列
        observe_time: 观察时间点（如 "10:30"）
        exit_time: 出场时间点（如 "14:50"）
        x2_weight: 组合因子中X2的权重（相对于X1）

    Returns:
        DataFrame: 每日每股票的特征和标签
    """
    logger.info(f"开始计算特征，观察时点: {observe_time}，出场时点: {exit_time}")

    results = []

    # 按股票和日期分组
    for (stock_code, date), day_df in df.groupby(["SecuCode", "date"]):
        day_df = day_df.sort_values("TradingDay")

        # 获取观察时点之前的数据（开盘 ~ observe_time）
        morning_df = day_df[day_df["time"] <= observe_time]
        if len(morning_df) == 0:
            continue

        # 获取观察时点的数据
        obs_df = day_df[day_df["time"] == observe_time]
        if len(obs_df) == 0:
            continue

        # 获取出场时点的数据
        exit_df = day_df[day_df["time"] == exit_time]
        if len(exit_df) == 0:
            continue

        # 计算 VWAP (成交量加权平均价)
        total_amount = morning_df["amount"].sum()
        total_volume = morning_df["volume"].sum()
        if total_volume == 0:
            continue
        vwap = total_amount / total_volume

        # 计算 TWAP (时间加权平均价) - 每分钟close的简单平均
        twap = morning_df["close"].mean()

        # 获取观察时点的价格
        price_obs = obs_df["close"].values[0]

        # 获取出场时点的价格
        price_exit = exit_df["close"].values[0]

        # 计算原始特征
        x1 = price_obs / vwap - 1  # 价格偏离
        x2 = vwap / twap - 1  # 能量偏离

        # 计算原始组合因子 Z = X1 + w2 * X2
        z = x1 + x2_weight * x2

        # 计算标签
        y = price_exit / price_obs - 1  # 下午收益

        results.append(
            {
                "date": date,
                "stock_code": stock_code,
                "price_obs": price_obs,
                "price_exit": price_exit,
                "vwap": vwap,
                "twap": twap,
                "X1": x1,  # 价格偏离
                "X2": x2,  # 能量偏离
                "Z": z,  # 组合因子
                "Y": y,  # 下午收益
                "morning_volume": total_volume,
                "morning_amount": total_amount,
            }
        )

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        n_stocks = result_df["stock_code"].nunique()
        n_days = result_df["date"].nunique()
        logger.success(
            f"特征计算完成: {len(result_df)} 条记录, {n_stocks} 只股票, {n_days} 个交易日"
        )
    else:
        logger.warning("未计算出任何特征")

    return result_df


def calculate_resonance_signals(
    df: pd.DataFrame,
    x1_threshold: float = -0.01,  # X1 < -1%
    x2_threshold: float = -0.005,  # X2 < -0.5%
    y_profit_threshold: float = 0.003,  # Y > 0.3%
) -> Dict:
    """
    计算共振信号统计

    共振条件：X1 和 X2 同时处于极端值

    Args:
        df: 特征数据 DataFrame
        x1_threshold: X1阈值（低于此值视为超跌）
        x2_threshold: X2阈值（低于此值视为低位换手）
        y_profit_threshold: Y盈利阈值

    Returns:
        Dict: 共振信号统计结果
    """
    logger.info(
        f"开始计算共振信号，X1阈值: {x1_threshold*100:.1f}%, X2阈值: {x2_threshold*100:.2f}%"
    )

    # 共振条件：X1 < 阈值 且 X2 < 阈值（双重超跌）
    resonance_mask = (df["X1"] < x1_threshold) & (df["X2"] < x2_threshold)
    resonance_df = df[resonance_mask]

    # 反向共振：X1 > 阈值 且 X2 > 阈值（双重超涨）
    reverse_mask = (df["X1"] > -x1_threshold) & (df["X2"] > -x2_threshold)
    reverse_df = df[reverse_mask]

    # 统计胜率
    if len(resonance_df) > 0:
        win_count = (resonance_df["Y"] > y_profit_threshold).sum()
        win_rate = win_count / len(resonance_df)
        avg_return = resonance_df["Y"].mean()
    else:
        win_count = 0
        win_rate = 0
        avg_return = 0

    if len(reverse_df) > 0:
        reverse_win_count = (reverse_df["Y"] < -y_profit_threshold).sum()
        reverse_win_rate = reverse_win_count / len(reverse_df)
        reverse_avg_return = reverse_df["Y"].mean()
    else:
        reverse_win_count = 0
        reverse_win_rate = 0
        reverse_avg_return = 0

    result = {
        "total_samples": len(df),
        # 超跌共振（买入信号）
        "resonance_count": len(resonance_df),
        "resonance_ratio": len(resonance_df) / len(df) if len(df) > 0 else 0,
        "resonance_win_rate": win_rate,
        "resonance_avg_return": avg_return,
        # 超涨共振（卖出信号）
        "reverse_count": len(reverse_df),
        "reverse_ratio": len(reverse_df) / len(df) if len(df) > 0 else 0,
        "reverse_win_rate": reverse_win_rate,
        "reverse_avg_return": reverse_avg_return,
    }

    logger.success(
        f"共振信号统计完成: 超跌共振 {len(resonance_df)} 次, 胜率 {win_rate*100:.1f}%"
    )

    return result


def calculate_quantile_analysis(
    df: pd.DataFrame,
    feature_col: str = "X1",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    分位数分析

    Args:
        df: 特征数据 DataFrame
        feature_col: 特征列名
        n_quantiles: 分位数数量

    Returns:
        DataFrame: 分位数分析结果
    """
    labels = [f"Q{i+1}" for i in range(n_quantiles)]
    df = df.copy()
    df["quantile"] = pd.qcut(
        df[feature_col], q=n_quantiles, labels=labels, duplicates="drop"
    )

    analysis = (
        df.groupby("quantile", observed=True)
        .agg({"Y": ["mean", "std", "count"], feature_col: "mean"})
        .round(6)
    )

    analysis.columns = ["Y_mean", "Y_std", "count", f"{feature_col}_mean"]
    analysis = analysis.reset_index()

    return analysis
