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

        # 从配置获取参数
        stock_codes = self.config.stock_codes
        year = self.config.year

        # 加载数据
        df = self.data_loader.load_minute_data(stock_codes, year)

        # 存入上下文
        context["minute_data"] = df
        context["stock_codes"] = stock_codes

        logger.success(f"分钟数据加载完成: {len(df):,} 条记录")
        return context
