"""
数据聚合脚本：将1分钟K线数据聚合为5分钟Bar

职责：仅做数据聚合，不计算任何特征或标签
输入：1分钟K线数据 (pkl)
输出：5分钟Bar数据 (pkl)，保存到 backtest_cache 目录

bar_time 使用 ceil（Bar的结束时间），与米筐get_price()一致
语义：bar_time=10:00 表示 09:56~10:00 的数据，在10:00时刻全部可用

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
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024


def load_minute_data(year: int) -> pd.DataFrame:
    """加载全量1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")

    logger.info(f"加载 {year} 年全量1分钟数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    df = df.sort_values(["SecuCode", "TradingDay"]).reset_index(drop=True)
    df["date"] = df["TradingDay"].dt.date

    n_stocks = df["SecuCode"].nunique()
    n_days = df["date"].nunique()
    logger.success(f"加载完成: {len(df):,} 条, {n_stocks} 只股票, {n_days} 个交易日")
    return df


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    将1分钟数据聚合为5分钟Bar

    bar_time 使用 ceil（Bar的结束时间），与米筐get_price()一致
    语义：bar_time=10:00 表示 09:56~10:00 的数据，在10:00时刻全部可用

    输出列：SecuCode, date, bar_time, open, high, low, close, volume, amount
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

    # 按股票+日期+时间排序
    agg_df = agg_df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("开始数据聚合：1分钟 → 5分钟Bar")
    logger.info("=" * 50)

    # 1. 加载全量1分钟数据
    df_1m = load_minute_data(YEAR)

    # 2. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 3. 保存到缓存目录
    output_path = CACHE_DIR / f"df_5m_{YEAR}.pkl"
    df_5m.to_pickle(output_path)
    logger.success(f"5分钟Bar数据已保存: {output_path}")
    logger.info(f"数据形状: {df_5m.shape}")
    logger.info(f"列: {list(df_5m.columns)}")
    logger.info(f"股票数: {df_5m['SecuCode'].nunique()}")
    logger.info(f"交易日数: {df_5m['date'].nunique()}")

    return df_5m


if __name__ == "__main__":
    df = main()
