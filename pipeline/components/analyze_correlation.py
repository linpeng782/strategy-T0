"""组件：相关性分析"""

from loguru import logger
from pipeline.base_component import BaseComponent
from core.correlation_helpers import analyze_by_stock, print_batch_summary


class AnalyzeCorrelation(BaseComponent):
    """按股票分组分析相关性"""

    def run(self, context: dict) -> dict:
        """
        分析相关性并存入上下文

        Args:
            context: 上下文字典，需包含 'feature_df'

        Returns:
            dict: 更新后的上下文，包含 'stock_results'
        """
        logger.info("开始分析相关性...")

        feature_df = context.get("feature_df")
        if feature_df is None or len(feature_df) == 0:
            logger.warning("无特征数据，跳过相关性分析")
            context["stock_results"] = None
            return context

        # 按股票分组分析（使用 Z-Score 特征）
        stock_results = analyze_by_stock(feature_df, use_zscore=True)

        # 打印汇总
        z_threshold = self.config.resonance_params.z_threshold
        print_batch_summary(stock_results, use_zscore=True, z_threshold=z_threshold)

        # 存入上下文
        context["stock_results"] = stock_results

        return context
