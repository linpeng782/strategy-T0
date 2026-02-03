"""特征计算核心逻辑"""

import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


def calculate_intraday_features(
    df: pd.DataFrame,
    observe_time: str = "10:30",
    exit_time: str = "14:50",
    x1_weight: float = 0.5,
    x2_weight: float = 2.5,
    zscore_window: int = 20,
) -> pd.DataFrame:
    """
    计算日内特征和标签（含 Z-Score 标准化）- 向量化版本

    Args:
        df: 分钟数据 DataFrame，需包含 date, time, SecuCode, close, volume, amount 列
        observe_time: 观察时间点（如 "10:30"）
        exit_time: 出场时间点（如 "14:50"）
        x1_weight: X1 权重
        x2_weight: X2 权重
        zscore_window: Z-Score 滚动窗口大小（天数）

    Returns:
        DataFrame: 每日每股票的特征和标签（含原始特征和 Z-Score 标准化特征）
    """
    logger.info(
        f"开始计算特征，观察时点: {observe_time}，出场时点: {exit_time}，Z-Score窗口: {zscore_window}天"
    )

    # 向量化计算：按股票和日期分组聚合
    # 1. 划分上午数据（开盘 ~ observe_time）
    morning_mask = df["time"] <= observe_time
    morning_df = df[morning_mask].copy()

    # 2. 划分下午数据（observe_time ~ exit_time）
    afternoon_mask = (df["time"] > observe_time) & (df["time"] <= exit_time)
    afternoon_df = df[afternoon_mask].copy()

    # 3. 上午聚合：VWAP, TWAP, 观察价格, 成交量, 成交额
    morning_agg = (
        morning_df.groupby(["SecuCode", "date"])
        .agg(
            morning_amount=("amount", "sum"),
            morning_volume=("volume", "sum"),
            twap=("close", "mean"),  # TWAP = 简单平均
            price_obs=("close", "last"),  # 观察时点价格 = 最后一根K线收盘价
        )
        .reset_index()
    )

    # 4. 下午聚合：出场价格
    afternoon_agg = (
        afternoon_df.groupby(["SecuCode", "date"])
        .agg(
            price_exit=("close", "last"),  # 出场价格 = 最后一根K线收盘价
        )
        .reset_index()
    )

    # 5. 合并上午和下午数据
    result_df = morning_agg.merge(afternoon_agg, on=["SecuCode", "date"], how="inner")

    # 6. 过滤无效数据（成交量为0）
    result_df = result_df[result_df["morning_volume"] > 0].copy()

    if len(result_df) == 0:
        logger.warning("未计算出任何特征")
        return pd.DataFrame()

    # 7. 向量化计算特征
    result_df["vwap"] = result_df["morning_amount"] / result_df["morning_volume"]
    result_df["X1"] = result_df["price_obs"] / result_df["vwap"] - 1  # 价格偏离
    result_df["X2"] = result_df["vwap"] / result_df["twap"] - 1  # 能量偏离
    result_df["Z"] = result_df["X1"] + x2_weight * result_df["X2"]  # 组合因子
    result_df["Y"] = result_df["price_exit"] / result_df["price_obs"] - 1  # 下午收益

    # 8. 重命名列
    result_df = result_df.rename(columns={"SecuCode": "stock_code"})

    if len(result_df) == 0:
        logger.warning("未计算出任何特征")
        return result_df

    # 按日期排序（用于滚动计算）
    result_df = result_df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 计算 Z-Score 标准化特征（按股票分组，使用滚动窗口）
    # 注意：使用 shift(1) 避免用到当天信息导致未来函数
    for col in ["X1", "X2"]:
        mu_col = f"{col}_mu"
        std_col = f"{col}_std"
        z_col = f"Z_{col}"

        # 滚动均值和标准差（shift(1) 避免未来函数）
        result_df[mu_col] = result_df.groupby("stock_code")[col].transform(
            lambda x: x.rolling(zscore_window, min_periods=5).mean().shift(1)
        )
        result_df[std_col] = result_df.groupby("stock_code")[col].transform(
            lambda x: x.rolling(zscore_window, min_periods=5).std().shift(1)
        )

        # 计算 Z-Score
        result_df[z_col] = (result_df[col] - result_df[mu_col]) / result_df[std_col]

    # 计算标准化后的组合因子 Z_final = Z_X1 + x2_weight * Z_X2
    result_df["Z_final"] = result_df["Z_X1"] + x2_weight * result_df["Z_X2"]

    # 统计有效样本（排除 NaN）
    valid_mask = result_df["Z_X1"].notna() & result_df["Z_X2"].notna()
    n_valid = valid_mask.sum()
    n_total = len(result_df)
    n_stocks = result_df["stock_code"].nunique()
    n_days = result_df["date"].nunique()

    logger.success(
        f"特征计算完成: {n_total} 条记录, {n_stocks} 只股票, {n_days} 个交易日"
    )
    logger.info(
        f"Z-Score 有效样本: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%), "
        f"前 {zscore_window-1} 天为预热期"
    )

    return result_df


def calculate_resonance_signals(
    df: pd.DataFrame,
    z_threshold: float,
    y_profit_threshold: float,
    use_zscore: bool = True,
) -> Dict:
    """
    计算共振信号统计（基于 Z-Score 标准化）

    共振条件：Z_X1 和 Z_X2 同时处于极端值（超过 z_threshold 个标准差）

    Args:
        df: 特征数据 DataFrame（需包含 Z_X1, Z_X2 列）
        z_threshold: Z-Score 阈值（默认 -2.0，即 2 个标准差以外）
        y_profit_threshold: Y盈利阈值
        use_zscore: 是否使用 Z-Score 特征（False 则使用原始特征）

    Returns:
        Dict: 共振信号统计结果
    """
    # 过滤有效样本（Z-Score 非空）
    if use_zscore:
        valid_df = df[df["Z_X1"].notna() & df["Z_X2"].notna()].copy()
        x1_col, x2_col = "Z_X1", "Z_X2"
        logger.info(f"开始计算共振信号（Z-Score模式），阈值: {z_threshold} 个标准差")
    else:
        valid_df = df.copy()
        x1_col, x2_col = "X1", "X2"
        logger.info("开始计算共振信号（原始特征模式）")

    if len(valid_df) == 0:
        logger.warning("无有效样本")
        return {"total_samples": 0}

    # 超跌共振：Z_X1 < 阈值 且 Z_X2 < 阈值（双重超跌）
    resonance_mask = (valid_df[x1_col] < z_threshold) & (valid_df[x2_col] < z_threshold)
    resonance_df = valid_df[resonance_mask]

    # 超涨共振：Z_X1 > -阈值 且 Z_X2 > -阈值（双重超涨）
    reverse_mask = (valid_df[x1_col] > -z_threshold) & (valid_df[x2_col] > -z_threshold)
    reverse_df = valid_df[reverse_mask]

    # 统计超跌共振胜率
    if len(resonance_df) > 0:
        win_count = (resonance_df["Y"] > y_profit_threshold).sum()
        win_rate = win_count / len(resonance_df)
        avg_return = resonance_df["Y"].mean()
        std_return = resonance_df["Y"].std()
    else:
        win_count = 0
        win_rate = 0
        avg_return = 0
        std_return = 0

    # 统计超涨共振胜率
    if len(reverse_df) > 0:
        reverse_win_count = (reverse_df["Y"] < -y_profit_threshold).sum()
        reverse_win_rate = reverse_win_count / len(reverse_df)
        reverse_avg_return = reverse_df["Y"].mean()
        reverse_std_return = reverse_df["Y"].std()
    else:
        reverse_win_count = 0
        reverse_win_rate = 0
        reverse_avg_return = 0
        reverse_std_return = 0

    result = {
        "total_samples": len(valid_df),
        "z_threshold": z_threshold,
        # 超跌共振（买入信号）
        "resonance_count": len(resonance_df),
        "resonance_ratio": (
            len(resonance_df) / len(valid_df) if len(valid_df) > 0 else 0
        ),
        "resonance_win_rate": win_rate,
        "resonance_avg_return": avg_return,
        "resonance_std_return": std_return,
        "resonance_details": (
            resonance_df[["date", "stock_code", x1_col, x2_col, "Y"]].to_dict("records")
            if len(resonance_df) > 0
            else []
        ),
        # 超涨共振（卖出信号）
        "reverse_count": len(reverse_df),
        "reverse_ratio": len(reverse_df) / len(valid_df) if len(valid_df) > 0 else 0,
        "reverse_win_rate": reverse_win_rate,
        "reverse_avg_return": reverse_avg_return,
        "reverse_std_return": reverse_std_return,
        "reverse_details": (
            reverse_df[["date", "stock_code", x1_col, x2_col, "Y"]].to_dict("records")
            if len(reverse_df) > 0
            else []
        ),
    }

    logger.success(
        f"共振信号统计完成: 超跌共振 {len(resonance_df)} 次, 胜率 {win_rate*100:.1f}%, 平均收益 {avg_return*100:.2f}%"
    )

    return result


def calculate_quantile_analysis(
    df: pd.DataFrame,
    feature_col: str = "Z_final",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    分位数分析

    Args:
        df: 特征数据 DataFrame
        feature_col: 用于分组的特征列
        n_quantiles: 分位数数量

    Returns:
        DataFrame: 分位数分析结果
    """
    if feature_col not in df.columns:
        logger.warning(f"特征列 {feature_col} 不存在")
        return pd.DataFrame()

    # 过滤有效样本
    valid_df = df[df[feature_col].notna()].copy()
    if len(valid_df) == 0:
        return pd.DataFrame()

    # 计算分位数
    valid_df["quantile"] = pd.qcut(
        valid_df[feature_col], q=n_quantiles, labels=False, duplicates="drop"
    )

    # 按分位数分组统计
    result = valid_df.groupby("quantile").agg(
        {
            feature_col: "mean",
            "Y": ["mean", "std", "count"],
        }
    )

    result.columns = [f"{feature_col}_mean", "Y_mean", "Y_std", "count"]
    result = result.reset_index()

    return result
