"""流水线执行器"""

from loguru import logger


class Pipeline:
    """流水线定义容器，用于存储业务流程的各个阶段"""

    def __init__(self, stages: list):
        """
        初始化流水线

        Args:
            stages: 阶段列表，每个阶段是一个组件类列表
        """
        if not stages or not all(isinstance(s, list) for s in stages):
            raise ValueError("stages 必须是一个包含列表的非空列表")
        self.stages = stages


class PipelineExecutor:
    """负责按顺序执行一系列组件"""

    def __init__(self, config, data_loader, data_manager):
        """
        初始化执行器

        Args:
            config: 配置对象
            data_loader: 数据加载服务
            data_manager: 数据管理服务
        """
        self.config = config
        self.data_loader = data_loader
        self.data_manager = data_manager
        self.context = {}

    def execute(self, pipeline: Pipeline) -> dict:
        """
        执行完整的流水线

        Args:
            pipeline: 流水线定义

        Returns:
            dict: 最终的上下文数据
        """
        logger.info("开始执行流水线...")

        for i, stage_components in enumerate(pipeline.stages, 1):
            logger.info(f"=== 执行阶段 {i}/{len(pipeline.stages)} ===")
            self.context = self._run_stage(stage_components, self.context)

        logger.success("流水线执行完毕！")
        return self.context

    def _run_stage(self, stage_components: list, context: dict) -> dict:
        """
        执行一个阶段的所有组件

        Args:
            stage_components: 组件类列表
            context: 当前上下文

        Returns:
            dict: 更新后的上下文
        """
        for component_class in stage_components:
            component_name = component_class.__name__
            logger.info(f"--- 开始执行组件: {component_name} ---")

            try:
                # 依赖注入
                component_instance = component_class(
                    config=self.config,
                    data_loader=self.data_loader,
                    data_manager=self.data_manager,
                )

                context = component_instance.run(context)
                logger.success(f"--- 组件 {component_name} 执行成功 ---")

            except Exception as e:
                logger.critical(f"组件 {component_name} 执行失败: {e}")
                raise

        return context
