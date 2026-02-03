"""组件：计算日内特征"""

from loguru import logger
from pipeline.base_component import BaseComponent
from core.feature_helpers import calculate_intraday_features


class ComputeIntradayFeatures(BaseComponent):
    """计算日内特征（含 Z-Score 标准化）"""

    def run(self, context: dict) -> dict:
        """
        计算日内特征并存入上下文

        Args:
            context: 上下文字典，需包含 'minute_data'

        Returns:
            dict: 更新后的上下文，包含 'feature_df'
        """
        logger.info("开始计算日内特征...")

        minute_data = context.get("minute_data")
        if minute_data is None or len(minute_data) == 0:
            logger.warning("无分钟数据，跳过特征计算")
            context["feature_df"] = None
            return context

        # 从配置获取参数
        feature_df = calculate_intraday_features(
            minute_data,
            observe_time=self.config.observe_time,
            exit_time=self.config.exit_time,
            x1_weight=self.config.feature_params.x1_weight,
            x2_weight=self.config.feature_params.x2_weight,
            zscore_window=self.config.feature_params.zscore_window,
        )

        # 存入上下文
        context["feature_df"] = feature_df

        return context
