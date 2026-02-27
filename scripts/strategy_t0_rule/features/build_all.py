"""
全量特征构造入口
合并classic/volume/alpha158三组因子，做跨日同时刻标准化，生成最终缓存。

使用方式：
  # 构造CSI1000因子（默认，读取对应年份的json股票池）
  python build_all.py --year 2025 --index csi1000

  # 构造全市场因子
  python build_all.py --year 2025 --index all

  # 只重新构造某一组（其他组从缓存加载）
  python build_all.py --year 2025 --index csi1000 --rebuild classic

  # 强制重建所有缓存
  python build_all.py --year 2025 --index csi1000 --rebuild all

输出缓存：output/features_{index}_{year}.pkl
"""

import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd
from loguru import logger

# 将父目录加入路径，支持直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.base import (
    FEAT_CACHE_DIR,
    OUTPUT_DIR,
    RAW_FEATURE_COLS,
    FEATURE_GROUPS,
    load_raw_data,
)
from features.classic_indicators import build_classic_features
from features.price_volume_features import build_price_volume_features
from features.alpha158_features import build_alpha158_features
from features.normalizer import normalize_features

# 默认年份
YEAR = 2024

# CSI1000成分股json目录
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"
if not CONFIG_DIR.exists():
    CONFIG_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config")


def build_group(
    group: str,
    year: int,
    stock_list: list = None,
    use_cache: bool = True,
    df_all: pd.DataFrame = None,
) -> pd.DataFrame:
    """构造单组因子（从缓存或重新计算）

    df_all : 已加载的原始K线数据，传入则跳过内部load_raw_data，避免重复加载
    """
    if group == "classic":
        return build_classic_features(year, stock_list, use_cache, df_all=df_all)
    elif group == "price_volume":
        return build_price_volume_features(year, stock_list, use_cache, df_all=df_all)
    elif group == "alpha158":
        return build_alpha158_features(year, stock_list, use_cache, df_all=df_all)
    else:
        raise ValueError(f"未知因子组: {group}，支持 classic/price_volume/alpha158")


def merge_groups(
    year: int, stock_list: list = None, rebuild: str = None
) -> pd.DataFrame:
    """
    合并三组因子到一个DataFrame。

    参数：
        rebuild : None=全部从缓存加载；'classic'/'price_volume'/'alpha158'=重建指定组；'all'=重建全部
    """
    # 确定各组是否使用缓存
    use_cache = {
        "classic": rebuild not in ("classic", "all"),
        "price_volume": rebuild not in ("price_volume", "all"),
        "alpha158": rebuild not in ("alpha158", "all"),
    }

    logger.info(f"开始合并三组因子（rebuild={rebuild}）...")

    # 只加载一次原始 K线数据，传给各组构造函数避免重复 IO
    df_base = load_raw_data(year, stock_list)

    # 逐组构造并合并
    for group in ["classic", "price_volume", "alpha158"]:
        logger.info(f"处理 {group} 组因子（use_cache={use_cache[group]}）...")
        # 缓存失效时传入df_all，干塩时内部从缓存文件直接加载
        df_group = build_group(
            group,
            year,
            stock_list,
            use_cache[group],
            df_all=None if use_cache[group] else df_base,
        )

        # 只取新增的因子列（避免重复列）
        new_cols = [c for c in FEATURE_GROUPS[group] if c in df_group.columns]
        # 用SecuCode+date+bar_time作为连接键
        merge_keys = ["SecuCode", "date", "bar_time"]
        df_base = df_base.merge(
            df_group[merge_keys + new_cols],
            on=merge_keys,
            how="left",
        )

    logger.success(f"三组因子合并完成，shape={df_base.shape}")
    return df_base


def load_stock_list(index: str, year: int) -> list:
    """
    根据指数名称和年份加载股票池列表。
    index='csi1000' 时从json加载，index='all' 时返回None（全市场）。
    """
    if index == "all":
        return None
    json_path = CONFIG_DIR / f"{index}_components_{year}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"股票池json不存在: {json_path}")
    with open(json_path) as f:
        stock_list = json.load(f)
    logger.info(f"股票池：{index.upper()}，共 {len(stock_list)} 只（{year}年）")
    return stock_list


def build_all_features(
    year: int, index: str = "csi1000", rebuild: str = None
) -> pd.DataFrame:
    """
    构造全量特征并标准化，生成最终缓存。

    参数：
        year    : 数据年份
        index   : 股票池，'csi1000' 或 'all'（全市场）
        rebuild : None/classic/price_volume/alpha158/all，控制哪组重新构造

    返回：含原始因子列和_norm标准化列的完整DataFrame
    """
    stock_list = load_stock_list(index, year)
    final_cache = OUTPUT_DIR / f"features_{index}_{year}.pkl"

    if final_cache.exists() and rebuild is None:
        logger.info(f"加载已缓存特征文件: {final_cache}")
        return pd.read_pickle(final_cache)

    t_start = time.time()
    # 合并三组因子
    df = merge_groups(year, stock_list, rebuild)

    # 跨日同时刻标准化
    feat_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]
    df = normalize_features(df, feat_cols=feat_cols)

    df.to_pickle(final_cache)
    logger.success(
        f"全量特征已保存: {final_cache}，shape={df.shape}，总耗时={time.time()-t_start:.1f}s"
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="构造全量特征缓存")
    parser.add_argument("--year", type=int, default=YEAR, help="数据年份")
    parser.add_argument(
        "--index",
        type=str,
        default="csi1000",
        help="股票池：csi1000（默认）或 all（全市场）",
    )
    parser.add_argument(
        "--rebuild",
        type=str,
        default=None,
        choices=["classic", "price_volume", "alpha158", "all"],
        help="重新构造指定组因子（不指定则从缓存加载）",
    )
    args = parser.parse_args()

    logger.info(
        f"开始构造全量特征，year={args.year}，index={args.index}，rebuild={args.rebuild}"
    )
    df = build_all_features(year=args.year, index=args.index, rebuild=args.rebuild)
    logger.info(f"完成，shape={df.shape}")


if __name__ == "__main__":
    main()
