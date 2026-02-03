"""数据管理服务"""

import pandas as pd
from pathlib import Path
from loguru import logger


class DataManager:
    """数据管理服务类，负责结果保存"""

    def __init__(self, config):
        """
        初始化数据管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        保存 DataFrame 为 CSV 文件

        Args:
            df: 要保存的 DataFrame
            filename: 文件名（不含路径）

        Returns:
            Path: 保存的文件路径
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"已保存: {filepath}")
        return filepath

    def save_parquet(self, df: pd.DataFrame, filename: str) -> Path:
        """
        保存 DataFrame 为 Parquet 文件

        Args:
            df: 要保存的 DataFrame
            filename: 文件名（不含路径）

        Returns:
            Path: 保存的文件路径
        """
        filepath = self.output_dir / filename
        df.to_parquet(filepath, index=False)
        logger.info(f"已保存: {filepath}")
        return filepath

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        加载 CSV 文件

        Args:
            filename: 文件名（不含路径）

        Returns:
            DataFrame: 加载的数据
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        return pd.read_csv(filepath)
