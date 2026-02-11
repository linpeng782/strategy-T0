"""
时序标签构建脚本

职责：读取时序特征数据，构建模型标签并清洗数据
输入：df_ts_features_{year}.pkl
输出：df_ts_ready_{year}.pkl（含特征 + 标签，可直接用于训练）

标签逻辑：
  1. raw_ret = 1/Y_120m - 1（做多原始收益）
  2. market_ret = 同日所有股票 raw_ret 的均值（去市场Beta）
  3. excess_ret = raw_ret - market_ret（超额收益）
  4. vol_adj_ret = excess_ret / hist_vol_20d（波动率标准化超额收益）
  5. label = (vol_adj_ret > VOL_ADJ_THRESHOLD)（二分类标签）

依赖：build_ts_features.py 生产的时序特征数据

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
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

# 处理年份
YEARS = [2024, 2025]

# 持有期标签列
HOLD_COL = "Y_120m"

# 波动率标准化阈值: vol_adj_ret > 0.0 为正类（超额收益为正）
VOL_ADJ_THRESHOLD = 0.0

# 特征列表（与 build_ts_features.py 一致）
FEATURES = [
    # 第1类：当前bar的X1/X2
    "X1",
    "X2",
    "X1_zscore",
    "X2_zscore",
    # 第2类：日内价格轨迹
    "X1_lag1",
    "X1_lag3",
    "X1_lag6",
    "X1_lag10",
    "X2_lag1",
    "X2_lag3",
    "X2_lag6",
    "X1_diff1",
    "X1_diff3",
    "X1_slope_6bar",
    "X1_morning_std",
    # 第3类：量能演变
    "rel_vol",
    "vol_accel",
    "vol_concentration",
    # 第4类：日内价格形态
    "morning_ret",
    "price_position",
    "morning_range",
    "intraday_vol",
    # 第5类：跨日特征
    "overnight_gap",
    "prev_day_ret",
    "momentum_5d",
    "hist_vol_20d",
]


# ==================== 数据加载 ====================
def load_ts_features(year: int) -> pd.DataFrame:
    """加载时序特征数据"""
    pkl_path = OUTPUT_DIR / f"df_ts_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"时序特征数据不存在: {pkl_path}\n请先运行 build_ts_features.py"
        )
    df = pd.read_pickle(pkl_path)
    logger.success(f"加载 {year} 年时序特征: {len(df):,} 行")
    return df


# ==================== 标签构建 ====================
def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建时序模型标签

    步骤:
      1. 计算做多原始收益 raw_ret
      2. 计算截面市场均值收益 market_ret（去Beta）
      3. 计算超额收益 excess_ret
      4. 波动率标准化 vol_adj_ret
      5. 二分类标签 label
    """
    logger.info("构建标签...")

    # 做多收益 = 1/Y_120m - 1
    df["raw_ret"] = 1.0 / df[HOLD_COL] - 1.0

    # 市场均值收益（同日所有股票的 raw_ret 平均值）
    df["market_ret"] = df.groupby("date")["raw_ret"].transform("mean")

    # 超额收益
    df["excess_ret"] = df["raw_ret"] - df["market_ret"]

    # 波动率标准化超额收益
    df["vol_adj_ret"] = df["excess_ret"] / df["hist_vol_20d"]

    # 二分类标签
    df["label"] = (df["vol_adj_ret"] > VOL_ADJ_THRESHOLD).astype(int)

    pos_ratio = df["label"].mean()
    logger.info(f"标签阈值: vol_adj_ret > {VOL_ADJ_THRESHOLD}")
    logger.info(f"正样本比例: {pos_ratio:.2%}")
    logger.info(
        f"收益统计: raw_ret 均值={df['raw_ret'].mean()*10000:.1f}bps, "
        f"excess_ret 均值={df['excess_ret'].mean()*10000:.1f}bps"
    )

    return df


# ==================== 数据清洗 ====================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤缺失值和异常值

    清洗规则:
      - 所有特征列非空且有限
      - Y_120m 在合理范围 (0.9, 1.1)
      - hist_vol_20d > 0（波动率标准化需要）
      - vol_adj_ret 有限（排除 inf）
    """
    n_before = len(df)

    # 特征非空且有限
    feat_cols = [c for c in FEATURES if c in df.columns]
    mask = df[feat_cols].notna().all(axis=1)
    mask = mask & df[feat_cols].apply(np.isfinite).all(axis=1)

    # 标签列非空且合理
    mask = mask & df[HOLD_COL].notna() & np.isfinite(df[HOLD_COL])
    mask = mask & (df[HOLD_COL] > 0.9) & (df[HOLD_COL] < 1.1)

    # 波动率有效
    mask = mask & (df["hist_vol_20d"] > 0) & np.isfinite(df["hist_vol_20d"])

    # vol_adj_ret 有限
    mask = mask & df["vol_adj_ret"].notna() & np.isfinite(df["vol_adj_ret"])

    df = df[mask].copy()
    logger.info(f"数据清洗: {n_before:,} -> {len(df):,} ({len(df)/n_before:.1%} 保留)")

    return df


# ==================== 主函数 ====================
def process_year(year: int):
    """处理一个年份"""
    logger.info(f"{'='*60}")
    logger.info(f"构建 {year} 年时序标签")
    logger.info(f"{'='*60}")

    # 1. 加载特征数据
    df = load_ts_features(year)

    # 2. 构建标签
    df = build_label(df)

    # 3. 数据清洗
    df = clean_data(df)

    # 4. 保存
    output_path = OUTPUT_DIR / f"df_ts_ready_{year}.pkl"
    df.to_pickle(output_path)
    logger.success(
        f"{year} 年训练数据已保存: {output_path}\n"
        f"  行数: {len(df):,}, 正类: {df['label'].sum():,} ({df['label'].mean():.2%})"
    )

    return df


def main():
    """主函数"""
    for year in YEARS:
        process_year(year)
    logger.success("全部年份标签构建完成!")


if __name__ == "__main__":
    main()
