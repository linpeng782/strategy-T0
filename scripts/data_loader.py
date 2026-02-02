"""
数据加载模块

负责从pkl文件加载分钟级别数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from loguru import logger


# 默认数据目录（按年份拆分的 pkl 文件）
DEFAULT_DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)


def load_minute_data(
    stock_codes: Union[str, List[str]],
    year: int = 2025,
    months: Optional[List[int]] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    加载指定股票的分钟数据

    Args:
        stock_codes: 股票代码或股票代码列表（如 "000001" 或 ["000001", "002591"]）
        year: 年份
        months: 月份列表，默认1-12月
        data_dir: 数据目录

    Returns:
        DataFrame: 分钟数据，包含 TradingDay, SecuCode, open, high, low, close, volume, amount 等列
    """
    # 统一转换为列表
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]

    if months is None:
        months = list(range(1, 13))

    logger.info(f"开始加载 {len(stock_codes)} 只股票 {year}年的分钟数据...")

    # 新数据格式：按年份拆分的 pkl 文件
    pkl_path = data_dir / f"{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")

    logger.debug(f"读取: {pkl_path.name}")
    df = pd.read_pickle(pkl_path)

    # 筛选目标股票
    result = df[df["SecuCode"].isin(stock_codes)].copy()

    if len(result) == 0:
        raise ValueError(f"未找到 {stock_codes} 在 {year}年 的数据")

    # 如果指定了月份，进一步筛选
    if months is not None and len(months) < 12:
        result["_month"] = result["TradingDay"].dt.month
        result = result[result["_month"].isin(months)].drop(columns=["_month"])

    result = result.sort_values(["SecuCode", "TradingDay"]).reset_index(drop=True)

    # 添加辅助列
    result["date"] = result["TradingDay"].dt.date
    result["time"] = result["TradingDay"].dt.strftime("%H:%M")

    n_stocks = result["SecuCode"].nunique()
    n_days = result["date"].nunique()
    logger.success(
        f"加载完成: {len(result):,} 条分钟数据, {n_stocks} 只股票, {n_days} 个交易日"
    )

    return result


def get_available_stocks(
    year: int = 2025, month: int = 1, data_dir: Path = DEFAULT_DATA_DIR
) -> List[str]:
    """
    获取指定年份可用的股票代码列表

    Args:
        year: 年份
        month: 月份（保留参数兼容性，但新格式按年份存储）
        data_dir: 数据目录

    Returns:
        List[str]: 股票代码列表
    """
    pkl_path = data_dir / f"{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"文件不存在: {pkl_path}")

    df = pd.read_pickle(pkl_path)
    stocks = sorted(df["SecuCode"].unique().tolist())
    logger.info(f"{year}年 共有 {len(stocks)} 只股票")
    return stocks
