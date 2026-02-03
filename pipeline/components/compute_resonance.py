"""组件：计算共振信号"""

from loguru import logger
from pipeline.base_component import BaseComponent
from core.feature_helpers import calculate_resonance_signals
from core.correlation_helpers import print_resonance_summary


class ComputeResonanceSignals(BaseComponent):
    """计算共振信号统计"""

    def run(self, context: dict) -> dict:
        """
        计算共振信号并存入上下文

        Args:
            context: 上下文字典，需包含 'feature_df'

        Returns:
            dict: 更新后的上下文，包含 'resonance_stats'
        """
        logger.info("开始计算共振信号...")

        feature_df = context.get("feature_df")
        if feature_df is None or len(feature_df) == 0:
            logger.warning("无特征数据，跳过共振信号计算")
            context["resonance_stats"] = None
            return context

        # 从配置获取参数
        z_threshold = self.config.resonance_params.z_threshold
        y_profit_threshold = self.config.resonance_params.y_profit_threshold

        # 计算共振信号
        resonance_stats = calculate_resonance_signals(
            feature_df,
            z_threshold=z_threshold,
            y_profit_threshold=y_profit_threshold,
            use_zscore=True,
        )

        # 打印汇总
        print_resonance_summary(resonance_stats, z_threshold)

        # 存入上下文
        context["resonance_stats"] = resonance_stats

        return context
