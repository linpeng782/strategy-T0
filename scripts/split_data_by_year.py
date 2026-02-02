"""
按年份拆分分钟数据

将大型 pkl 文件按年份拆分为多个小文件，加速后续加载
源文件: /nfs/ofs-prediction/changxi/data/m1_data_20090101_20260116.pkl
目标目录: /nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import time

# 源文件路径
SOURCE_PATH = Path("/nfs/ofs-prediction/changxi/data/m1_data_20090101_20260116.pkl")

# 目标目录
TARGET_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)


def split_by_year():
    """按年份拆分数据"""

    # 创建目标目录
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"目标目录: {TARGET_DIR}")

    # 加载源数据
    logger.info(f"正在加载源文件: {SOURCE_PATH}")
    logger.info("文件约 106GB，预计需要 3-5 分钟...")

    start_time = time.time()
    df = pd.read_pickle(SOURCE_PATH)
    load_time = time.time() - start_time
    logger.success(f"加载完成，耗时 {load_time:.1f} 秒，共 {len(df):,} 条记录")

    # 获取年份列表
    # 数据格式: MultiIndex (datetime, instrument)
    datetime_idx = df.index.get_level_values("datetime")
    years = sorted(datetime_idx.year.unique())
    logger.info(f"数据包含年份: {years}")

    # 按年份拆分并保存
    for year in years:
        logger.info(f"正在处理 {year} 年...")

        # 筛选该年份的数据
        year_mask = datetime_idx.year == year
        year_df = df[year_mask].copy()

        if len(year_df) == 0:
            logger.warning(f"{year} 年无数据，跳过")
            continue

        # 重置索引，转换为普通 DataFrame
        year_df = year_df.reset_index()

        # 重命名列以兼容旧代码
        year_df = year_df.rename(
            columns={
                "datetime": "TradingDay",
                "instrument": "SecuCode",
            }
        )

        # 保存为 pkl 文件
        output_path = TARGET_DIR / f"{year}.pkl"
        year_df.to_pickle(output_path)

        # 获取文件大小
        file_size = output_path.stat().st_size / 1024 / 1024 / 1024
        logger.success(
            f"  保存 {output_path.name}: {len(year_df):,} 条记录, {file_size:.2f} GB"
        )

    logger.success("=" * 60)
    logger.success("拆分完成！")
    logger.success(f"目标目录: {TARGET_DIR}")
    logger.success("=" * 60)


if __name__ == "__main__":
    split_by_year()
