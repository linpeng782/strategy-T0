"""
获取指数成分股列表并保存到 config 目录

使用米筐 index_components 函数获取中证1000成分股
"""

import json
from pathlib import Path
from loguru import logger
from rqdatac import init, index_components

# ==================== 配置参数 ====================
CONFIG_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CODE = "000852.XSHG"  # 中证1000
YEAR = 2024
INDEX_DATE = f"{YEAR}-12-31"


def main():
    logger.info(f"获取 {INDEX_CODE} 在 {INDEX_DATE} 的成分股...")

    init()
    components = index_components(INDEX_CODE, date=INDEX_DATE)
    logger.info(f"米筐返回 {len(components)} 只成分股")

    # 转换为纯股票代码（去掉 .XSHE/.XSHG 后缀）
    codes = sorted([c.split(".")[0] for c in components])
    logger.info(f"前10只: {codes[:10]}")

    # 保存
    output_path = CONFIG_DIR / f"csi1000_components_{YEAR}.json"
    with open(output_path, "w") as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)

    logger.success(f"已保存 {len(codes)} 只成分股到: {output_path}")


if __name__ == "__main__":
    main()
