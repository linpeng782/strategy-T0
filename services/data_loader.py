"""数据加载服务"""

import pandas as pd
from pathlib import Path
from typing import List, Optional
from loguru import logger
import rqdatac

# 初始化 rqdatac
rqdatac.init()


class DataLoader:
    """数据加载服务类"""

    def __init__(self, config):
        """
        初始化数据加载器

        Args:
            config: 配置对象
        """
        self.config = config
        self.data_dir = Path(config.data_dir)

    def load_minute_data(
        self,
        stock_codes: list = None,
        year: int = None,
    ) -> pd.DataFrame:
        """
        加载分钟级别数据

        Args:
            stock_codes: 股票代码列表，默认使用配置中的股票列表
            year: 年份，默认使用配置中的年份

        Returns:
            DataFrame: 分钟数据
        """
        if stock_codes is None:
            stock_codes = self.config.stock_codes
        if year is None:
            year = self.config.year

        # 支持单个股票代码
        if isinstance(stock_codes, str):
            stock_codes = [stock_codes]

        logger.info(f"开始加载 {len(stock_codes)} 只股票 {year}年的分钟数据...")

        # 读取年度数据文件
        file_path = self.data_dir / f"{year}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.debug(f"读取: {file_path.name}")
        df = pd.read_pickle(file_path)

        # 筛选指定股票
        df = df[df["SecuCode"].isin(stock_codes)]

        if len(df) == 0:
            logger.warning(f"未找到指定股票的数据")
            return pd.DataFrame()

        # 处理日期和时间
        df["TradingDay"] = pd.to_datetime(df["TradingDay"])
        df["date"] = df["TradingDay"].dt.strftime("%Y-%m-%d")
        df["time"] = df["TradingDay"].dt.strftime("%H:%M")

        n_stocks = df["SecuCode"].nunique()
        n_days = df["date"].nunique()
        logger.success(
            f"加载完成: {len(df):,} 条分钟数据, {n_stocks} 只股票, {n_days} 个交易日"
        )

        return df

    def get_index_components(self, index_code: str, date: str = None) -> List[str]:
        """
        获取指数成分股

        Args:
            index_code: 指数代码，如 "932000.INDX" (中证2000)
            date: 日期，默认为最近交易日

        Returns:
            List[str]: 成分股代码列表（6位数字格式）
        """
        logger.info(f"从指数 {index_code} 获取成分股...")

        # 获取指数成分股
        components = rqdatac.index_components(index_code, date=date)

        if components is None or len(components) == 0:
            logger.warning(f"未获取到指数 {index_code} 的成分股")
            return []

        # 转换为6位数字格式（去掉后缀）
        stock_codes = [c.split(".")[0] for c in components]

        logger.success(f"获取到 {len(stock_codes)} 只成分股")
        return stock_codes

    def load_minute_data_with_index(
        self,
        index_code: str = None,
        year: int = None,
    ) -> pd.DataFrame:
        """
        从指数获取成分股，并与分钟数据做交集

        Args:
            index_code: 指数代码，默认使用配置中的指数
            year: 年份，默认使用配置中的年份

        Returns:
            DataFrame: 分钟数据（仅包含指数成分股与数据的交集）
        """
        import time as time_module

        if index_code is None:
            index_code = self.config.index_code
        if year is None:
            year = self.config.year

        # 1. 获取指数成分股
        index_stocks = self.get_index_components(index_code)
        if len(index_stocks) == 0:
            raise ValueError(f"指数 {index_code} 成分股为空")

        logger.info(f"开始加载 {year}年 分钟数据...")

        # 2. 读取年度数据文件
        file_path = self.data_dir / f"{year}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        t0 = time_module.time()
        logger.debug(f"读取: {file_path.name}")
        df = pd.read_pickle(file_path)
        logger.info(
            f"读取完成，耗时 {time_module.time() - t0:.1f}s，共 {len(df):,} 条记录"
        )

        # 3. 获取数据中可用的股票
        available_stocks = set(df["SecuCode"].unique())
        logger.info(f"数据中共有 {len(available_stocks)} 只股票")

        # 4. 计算交集
        index_stocks_set = set(index_stocks)
        intersection = available_stocks & index_stocks_set
        logger.info(f"指数成分股与数据交集: {len(intersection)} 只股票")

        # 5. 筛选交集股票（使用 isin，比 merge 更快且内存更少）
        t0 = time_module.time()
        logger.info("开始筛选股票数据...")
        mask = df["SecuCode"].isin(intersection)
        df = df.loc[mask].copy()  # 使用 copy() 避免 SettingWithCopyWarning
        logger.info(
            f"筛选完成: {len(df):,} 条记录，耗时 {time_module.time() - t0:.1f}s"
        )

        if len(df) == 0:
            logger.warning(f"交集为空，无有效数据")
            return pd.DataFrame()

        # 处理日期和时间（使用向量化方法，比 strftime 快很多）
        t0 = time_module.time()
        logger.info("开始处理日期时间...")
        df["TradingDay"] = pd.to_datetime(df["TradingDay"])
        # 使用 dt.date 和 astype(str) 替代 strftime，性能提升 10x+
        df["date"] = df["TradingDay"].dt.date.astype(str)
        # 使用 dt.hour 和 dt.minute 组合替代 strftime，性能提升 50x+
        df["time"] = (
            df["TradingDay"].dt.hour.astype(str).str.zfill(2)
            + ":"
            + df["TradingDay"].dt.minute.astype(str).str.zfill(2)
        )
        logger.info(f"日期时间处理完成，耗时 {time_module.time() - t0:.1f}s")

        n_stocks = df["SecuCode"].nunique()
        n_days = df["date"].nunique()
        logger.success(
            f"加载完成: {len(df):,} 条分钟数据, {n_stocks} 只股票, {n_days} 个交易日"
        )

        return df
