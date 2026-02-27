"""
量价类因子构造
包含：vol_ratio, buy_pressure, amplitude, ret_1/3/5bar, range_pos
日内计算为主，避免跨日隔夜收益污染

注：vol_price_corr（收益率与log成交量相关）已删除，与alpha158中的cord20高度重叠，
    cord20（收益率与log成交量变化率相关，跨日20bar）信息量更丰富，保留cord20即可。

缓存文件：feat_cache/price_volume_{year}.pkl
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from loguru import logger

from .base import FEAT_CACHE_DIR, load_raw_data

# 并行进程数（最多使用100核）
N_WORKERS = min(100, cpu_count())


def compute_pv_single(df: pd.DataFrame) -> pd.DataFrame:
    """对单只股票构造量价类因子"""
    grp = df.groupby("date")

    # vol_ratio：当前bar成交量 / 日内前5根bar均量（日内rolling5，消除绝对量差异）
    vol_ma = grp["volume"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    df["vol_ratio"] = np.where(vol_ma > 0, df["volume"] / vol_ma, 1.0)

    # buy_pressure：(close - low) / (high - low)，bar内买卖力量对比
    hl_range = df["high"] - df["low"]
    df["buy_pressure"] = np.where(
        hl_range > 0, (df["close"] - df["low"]) / hl_range, 0.5
    )

    # amplitude：(high - low) / open，bar内波动强度
    df["amplitude"] = np.where(
        df["open"] > 0, (df["high"] - df["low"]) / df["open"], 0.0
    )

    # 短期价格动量（日内分组，避免跨日隔夜收益污染）
    df["ret_1bar"] = grp["close"].pct_change(1).fillna(0.0)
    df["ret_3bar"] = grp["close"].pct_change(3).fillna(0.0)
    df["ret_5bar"] = grp["close"].pct_change(5).fillna(0.0)

    # range_pos：(close - 日内最低) / (日内最高 - 日内最低)，日内价格位置
    cum_high = grp["high"].transform("cummax")
    cum_low = grp["low"].transform("cummin")
    intraday_range = cum_high - cum_low
    df["range_pos"] = np.where(
        intraday_range > 0, (df["close"] - cum_low) / intraday_range, 0.5
    )

    return df


def _pv_worker(args):
    """多进程worker：处理单只股票"""
    code, df_s = args
    return compute_pv_single(df_s)


def build_price_volume_features(
    year: int,
    stock_list: list = None,
    use_cache: bool = True,
    df_all: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    构造量价类因子，支持独立缓存，多核并行加速。

    参数：
        year       : 数据年份
        stock_list : 指定股票列表，None表示全部
        use_cache  : 是否使用缓存
        df_all     : 已加载的原始K线数据，传入则跳过内部load_raw_data

    返回：含量价原始因子列的DataFrame
    """
    cache_path = FEAT_CACHE_DIR / f"price_volume_{year}.pkl"

    if use_cache and cache_path.exists():
        logger.info(f"加载量价因子缓存: {cache_path}")
        df = pd.read_pickle(cache_path)
        if stock_list is not None:
            df = df[df["SecuCode"].isin(stock_list)].copy()
        return df

    if df_all is None:
        df_all = load_raw_data(year, stock_list)
    elif stock_list is not None:
        df_all = df_all[df_all["SecuCode"].isin(stock_list)].copy()
    stocks = df_all["SecuCode"].unique()
    logger.info(f"构造量价因子，共 {len(stocks)} 只股票，使用 {N_WORKERS} 核并行...")

    # groupby预分组：O(n)一次分组，替代O(n*k)的逐股全表扫描
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    tasks = [(code, grouped[code]) for code in stocks]

    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_pv_worker, tasks)

    df_feat = pd.concat(parts, ignore_index=True)
    logger.success(f"量价因子构造完成，shape={df_feat.shape}")

    if stock_list is None:
        df_feat.to_pickle(cache_path)
        logger.success(f"量价因子已缓存: {cache_path}")

    return df_feat
