"""
日内做T时序模型 - 管道式架构主入口

使用方法:
    python run.py
"""

import os
import sys
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import load_config
from services.data_loader import DataLoader
from services.data_manager import DataManager
from pipeline.executor import PipelineExecutor, Pipeline

# 导入流水线组件
from pipeline.components.load_data import LoadMinuteData
from pipeline.components.compute_features import ComputeIntradayFeatures
from pipeline.components.analyze_correlation import AnalyzeCorrelation
from pipeline.components.compute_resonance import ComputeResonanceSignals
from pipeline.components.save_results import SaveResults


def main():
    """主函数"""
    # 1. 加载配置
    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    # 2. 打印启动信息
    logger.info("=" * 70)
    logger.info("日内做T时序模型 - Z-Score 标准化版本（管道式架构）")
    logger.info(f"股票列表: {config.stock_codes}")
    logger.info(f"观察时点: {config.observe_time} | 出场时点: {config.exit_time}")
    logger.info(
        f"组合因子: Z_final = {config.feature_params.x1_weight} * Z_X1 + {config.feature_params.x2_weight} * Z_X2"
    )
    logger.info(
        f"Z-Score 窗口: {config.feature_params.zscore_window} 天 | 共振阈值: {config.resonance_params.z_threshold} 个标准差"
    )
    logger.info("=" * 70)

    # 3. 初始化服务
    data_loader = DataLoader(config)
    data_manager = DataManager(config)

    # 4. 初始化执行器
    executor = PipelineExecutor(config, data_loader, data_manager)

    # 5. 定义流水线
    # 阶段1: 加载数据
    stage_load = [LoadMinuteData]
    # 阶段2: 计算特征
    stage_compute = [ComputeIntradayFeatures]
    # 阶段3: 分析
    stage_analyze = [AnalyzeCorrelation, ComputeResonanceSignals]
    # 阶段4: 保存结果
    stage_save = [SaveResults]

    pipeline = Pipeline(
        stages=[
            stage_load,
            stage_compute,
            stage_analyze,
            stage_save,
        ]
    )

    # 6. 执行流水线
    try:
        result = executor.execute(pipeline)
        logger.success("✅ 分析完成")
        return result
    except Exception as e:
        logger.critical(f"流水线执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
