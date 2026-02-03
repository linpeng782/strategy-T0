"""组件：加载分钟数据"""

from loguru import logger
from pipeline.base_component import BaseComponent


class LoadMinuteData(BaseComponent):
    """加载分钟级别数据"""

    def run(self, context: dict) -> dict:
        """
        加载分钟数据并存入上下文

        Args:
            context: 上下文字典

        Returns:
            dict: 更新后的上下文，包含 'minute_data'
        """
        logger.info("开始加载分钟数据...")

        # 判断使用指数模式还是股票列表模式
        index_code = self.config.index_code
        if index_code:
            # 指数模式：从指数获取成分股并与数据做交集
            logger.info(f"使用指数模式: {index_code}")
            df = self.data_loader.load_minute_data_with_index(index_code)
            stock_codes = df["SecuCode"].unique().tolist() if len(df) > 0 else []
        else:
            # 股票列表模式
            stock_codes = self.config.stock_codes
            year = self.config.year
            logger.info(f"使用股票列表模式: {len(stock_codes)} 只股票")
            df = self.data_loader.load_minute_data(stock_codes, year)

        # 存入上下文
        context["minute_data"] = df
        context["stock_codes"] = stock_codes

        logger.success(
            f"分钟数据加载完成: {len(df):,} 条记录, {len(stock_codes)} 只股票"
        )
        return context
