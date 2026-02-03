"""定义流水线的基础构建块：BaseComponent"""

from abc import ABC, abstractmethod
from loguru import logger


class BaseComponent(ABC):
    """所有组件的基类"""

    def __init__(self, config, data_loader, data_manager):
        """
        初始化组件

        Args:
            config: 配置对象
            data_loader: 数据加载服务
            data_manager: 数据管理服务
        """
        self.config = config
        self.data_loader = data_loader
        self.data_manager = data_manager
        logger.debug(f"组件 '{self.__class__.__name__}' 初始化完成")

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        每个组件必须实现的核心方法

        Args:
            context: 上下文字典，包含从上一个组件传递过来的数据

        Returns:
            dict: 处理后传给下一个组件的上下文数据
        """
        raise NotImplementedError("每个组件都必须实现 run 方法")
