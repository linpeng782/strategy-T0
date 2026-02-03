"""组件：保存结果"""

from loguru import logger
from pipeline.base_component import BaseComponent


class SaveResults(BaseComponent):
    """保存分析结果"""

    def run(self, context: dict) -> dict:
        """
        保存结果到文件

        Args:
            context: 上下文字典，需包含 'feature_df' 和 'stock_results'

        Returns:
            dict: 更新后的上下文
        """
        logger.info("开始保存结果...")

        feature_df = context.get("feature_df")
        stock_results = context.get("stock_results")
        feature_std_df = context.get("feature_std_df")

        # 保存特征数据
        if feature_df is not None and len(feature_df) > 0:
            self.data_manager.save_csv(feature_df, "intraday_features.csv")

        # 保存汇总结果
        if stock_results is not None and len(stock_results) > 0:
            self.data_manager.save_csv(stock_results, "stock_correlation_summary.csv")

        # 保存 X1, X2 标准差统计
        if feature_std_df is not None and len(feature_std_df) > 0:
            self.data_manager.save_csv(feature_std_df, "feature_std_by_stock.csv")

        logger.success(f"结果已保存至: {self.config.output_dir}")

        return context
