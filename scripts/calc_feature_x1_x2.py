"""
特征生产脚本：计算X1/X2特征、Z-Score、组合因子和多时间维度标签

职责：读取预聚合5分钟Bar数据，计算所有特征和标签，保存到缓存目录
输出：包含特征+标签的完整DataFrame (pkl)

bar_time 使用 ceil（Bar的结束时间），与米筐get_price()一致
语义：bar_time=10:00 表示 09:56~10:00 的数据，在10:00时刻全部可用

特征（无未来函数）：
  - X1 = close / cum_vwap - 1
  - X2 = cum_vwap / cum_twap - 1
  - X1_zscore, X2_zscore: 历史同时间点的Z-Score
  - Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore

标签（上帝视角，买入成本 = 下一个Bar的VWAP）：
  - Y: next_vwap_5m / V_rest
  - Y_15m ~ Y_120m: 不同时间维度的标签

依赖：aggregate_5m_bars.py 预聚合的5分钟Bar数据

作者：量化研究
日期：2025-02
"""

import json

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024

# 成分股配置文件（按年份区分）
COMPONENTS_PATH = Path(
    f"/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config/csi1000_components_{YEAR}.json"
)
ZSCORE_WINDOW = 20  # Z-Score 滚动窗口


def load_5m_data(year: int) -> pd.DataFrame:
    """加载预聚合的5分钟Bar数据，并筛选成分股"""
    pkl_path = CACHE_DIR / f"df_5m_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"5分钟Bar数据不存在: {pkl_path}\n请先运行 aggregate_5m_bars.py"
        )

    logger.info(f"加载5分钟Bar数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    logger.info(f"全量数据: {len(df):,} 个Bar, {df['SecuCode'].nunique()} 只股票")

    # 筛选成分股
    if COMPONENTS_PATH.exists():
        with open(COMPONENTS_PATH, "r") as f:
            components = json.load(f)
        df = df[df["SecuCode"].isin(components)].reset_index(drop=True)
        logger.info(
            f"筛选成分股后: {len(df):,} 个Bar, {df['SecuCode'].nunique()} 只股票"
        )
    else:
        logger.warning(f"成分股文件不存在: {COMPONENTS_PATH}，使用全量数据")

    n_stocks = df["SecuCode"].nunique()
    n_days = df["date"].nunique()
    logger.success(f"加载完成: {len(df):,} 个Bar, {n_stocks} 只股票, {n_days} 个交易日")
    return df


def calculate_features_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征和多时间维度标签（全向量化实现）

    特征（无未来函数，ceil语义下当前Bar数据在bar_time时刻已全部可用）：
      - X1 = close / cum_vwap - 1
      - X2 = cum_vwap / cum_twap - 1
    标签（上帝视角，买入成本 = 下一个Bar的VWAP）：
      - Y: next_vwap_5m / V_rest（买入成本 vs 之后全天均价）
      - Y_15m ~ Y_120m: 不同时间维度的标签
    """
    logger.info("计算特征和标签...")

    # 基础派生列（聚合脚本只输出原始OHLCV）
    df["vwap_5m"] = (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
    # entry_time: 入场决策时间（即bar_time的时分字符串）
    df["entry_time"] = df["bar_time"].dt.strftime("%H:%M")

    grp_keys = ["SecuCode", "date"]
    grp = df.groupby(grp_keys)

    # ========== 特征部分（当前Bar数据在bar_time时刻已全部可用）==========
    df["cum_volume"] = grp["volume"].cumsum()
    df["cum_amount"] = grp["amount"].cumsum()
    df["cum_vwap"] = (df["cum_amount"] / df["cum_volume"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["cum_twap"] = (
        grp["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )

    df["X1"] = df["close"] / df["cum_vwap"] - 1
    df["X2"] = df["cum_vwap"] / df["cum_twap"] - 1

    # ========== 标签部分（上帝视角）==========
    # 下一个Bar的VWAP（买入成本）
    df["next_vwap_5m"] = grp["vwap_5m"].shift(-1)

    # 截至下一个Bar（含）的累积量
    cum_amt_s1 = grp["cum_amount"].shift(-1)
    cum_vol_s1 = grp["cum_volume"].shift(-1)
    total_amount = grp["amount"].transform("sum")
    total_volume = grp["volume"].transform("sum")

    # 统一计算所有时间维度的标签
    # "rest" 表示到收盘（全天剩余），其余为固定分钟数
    horizons = [
        ("rest", "rest"),
        (15, "15m"),
        (30, "30m"),
        (60, "60m"),
        (90, "90m"),
        (120, "120m"),
    ]
    for minutes, label_suffix in horizons:
        if minutes == "rest":
            # 到收盘：用全天总量 - 下一个Bar的累积量
            cum_amt_sN = total_amount
            cum_vol_sN = total_volume
        else:
            # 固定分钟数：shift(-offset) 取未来第offset个Bar的累积量
            offset = minutes // 5 + 1
            cum_amt_sN = grp["cum_amount"].shift(-offset)
            cum_vol_sN = grp["cum_volume"].shift(-offset)

        next_amount = cum_amt_sN - cum_amt_s1
        next_volume = cum_vol_sN - cum_vol_s1
        df[f"V_next_{label_suffix}"] = (next_amount / next_volume).replace(
            [np.inf, -np.inf], np.nan
        )
        df[f"Y_{label_suffix}"] = df["next_vwap_5m"] / df[f"V_next_{label_suffix}"]

    # 保留兼容列名: Y = Y_rest
    df["V_rest"] = df["V_next_rest"]
    df["Y"] = df["Y_rest"]

    logger.success(f"特征和标签计算完成: {len(df):,} 行")
    return df


def calculate_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    计算 Z-Score 和组合因子 Z_final

    按 (股票, entry_time) 分组，使用历史同时间点的数据
    shift(1) 避免使用当天数据计算均值/标准差
    Z_final = 2.5 * X2_zscore - 0.5 * X1_zscore
    """
    logger.info(f"计算 Z-Score 和 Z_final (窗口={window}天)...")

    result_list = []

    for (stock, entry_time), group_df in df.groupby(["SecuCode", "entry_time"]):
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

        # 组合因子
        group_df["Z_final"] = 2.5 * group_df["X2_zscore"] - 0.5 * group_df["X1_zscore"]

        result_list.append(group_df)

    result = pd.concat(result_list, ignore_index=True)

    # 过滤无效样本（Z-Score窗口期不足）
    valid_mask = result["X2_zscore"].notna() & result["X1_zscore"].notna()
    result = result[valid_mask]

    logger.success(f"Z-Score 和 Z_final 计算完成, 有效样本: {len(result):,}")
    return result


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("特征生产：X1/X2 + Z-Score + 多时间维度标签")
    logger.info("=" * 60)

    # 1. 加载预聚合的5分钟Bar数据
    df = load_5m_data(YEAR)

    # 2. 计算特征和标签
    df = calculate_features_and_labels(df)

    # 3. 计算 Z-Score 和 Z_final
    df = calculate_zscore(df, window=ZSCORE_WINDOW)

    # 4. 保存到缓存
    output_path = CACHE_DIR / f"df_features_{YEAR}.pkl"
    df.to_pickle(output_path)
    logger.success(f"特征数据已保存: {output_path}")
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"列: {list(df.columns)}")
    logger.info(f"股票数: {df['SecuCode'].nunique()}")
    logger.info(f"交易日数: {df['date'].nunique()}")

    return df


if __name__ == "__main__":
    df = main()
