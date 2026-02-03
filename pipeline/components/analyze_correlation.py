"""组件：相关性分析"""

from loguru import logger
from pipeline.base_component import BaseComponent
from core.correlation_helpers import (
    analyze_by_stock,
    print_batch_summary,
    calculate_feature_std_by_stock,
)


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

        # 计算 X1, X2 标准差
        feature_std_df = calculate_feature_std_by_stock(feature_df)

        # 打印 X1, X2 标准差
        print("\n" + "=" * 70)
        print("📊 原始特征 X1, X2 标准差统计")
        print("=" * 70)
        print(
            f"{'股票代码':>10} {'X1_mean':>12} {'X1_std':>12} {'X2_mean':>12} {'X2_std':>12} {'样本数':>8}"
        )
        print("-" * 70)
        for _, row in feature_std_df.iterrows():
            print(
                f"{row['stock_code']:>10} {row['X1_mean']*100:>11.4f}% {row['X1_std']*100:>11.4f}% "
                f"{row['X2_mean']*100:>11.4f}% {row['X2_std']*100:>11.4f}% {int(row['n_samples']):>8}"
            )
        # 打印平均值
        print("-" * 70)
        avg_x1_mean = feature_std_df["X1_mean"].mean()
        avg_x1_std = feature_std_df["X1_std"].mean()
        avg_x2_mean = feature_std_df["X2_mean"].mean()
        avg_x2_std = feature_std_df["X2_std"].mean()
        total_samples = feature_std_df["n_samples"].sum()
        print(
            f"{'平均值':>10} {avg_x1_mean*100:>11.4f}% {avg_x1_std*100:>11.4f}% "
            f"{avg_x2_mean*100:>11.4f}% {avg_x2_std*100:>11.4f}% {int(total_samples):>8}"
        )
        print("=" * 70)

        # 按股票分组分析（使用 Z-Score 特征）
        stock_results = analyze_by_stock(feature_df, use_zscore=True)

        # 打印汇总
        z_threshold = self.config.resonance_params.z_threshold
        print_batch_summary(stock_results, use_zscore=True, z_threshold=z_threshold)

        # 存入上下文
        context["stock_results"] = stock_results
        context["feature_std_df"] = feature_std_df

        return context
