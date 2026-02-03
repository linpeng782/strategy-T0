"""数据加载服务"""

import pandas as pd
from pathlib import Path
from loguru import logger


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
