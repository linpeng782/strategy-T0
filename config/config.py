"""配置加载器"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class FeatureParams:
    """特征计算参数"""

    x1_weight: float = 0.5
    x2_weight: float = 2.5
    zscore_window: int = 20


@dataclass
class ResonanceParams:
    """共振信号参数"""

    z_threshold: float = -1.0
    y_profit_threshold: float = 0.003


@dataclass
class Config:
    """配置类"""

    # 股票列表
    stock_codes: List[str] = field(default_factory=list)

    # 时间参数
    year: int = 2025
    observe_time: str = "10:30"
    exit_time: str = "14:50"

    # 特征计算参数
    feature_params: FeatureParams = field(default_factory=FeatureParams)

    # 共振信号参数
    resonance_params: ResonanceParams = field(default_factory=ResonanceParams)

    # 数据路径
    data_dir: str = ""
    output_dir: str = ""
    project_root: str = ""


def load_config(config_path: str = None) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为 config/config.yaml

    Returns:
        Config: 配置对象
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # 解析配置
    config = Config(
        stock_codes=raw_config.get("stock_codes", []),
        year=raw_config.get("year", 2025),
        observe_time=raw_config.get("observe_time", "10:30"),
        exit_time=raw_config.get("exit_time", "14:50"),
        feature_params=FeatureParams(
            x1_weight=raw_config.get("feature_params", {}).get("x1_weight", 0.5),
            x2_weight=raw_config.get("feature_params", {}).get("x2_weight", 2.5),
            zscore_window=raw_config.get("feature_params", {}).get("zscore_window", 20),
        ),
        resonance_params=ResonanceParams(
            z_threshold=raw_config.get("resonance_params", {}).get("z_threshold", -1.0),
            y_profit_threshold=raw_config.get("resonance_params", {}).get(
                "y_profit_threshold", 0.003
            ),
        ),
        data_dir=raw_config.get("data_dir", ""),
        output_dir=raw_config.get("output_dir", ""),
        project_root=str(config_path.parent.parent),
    )

    return config
