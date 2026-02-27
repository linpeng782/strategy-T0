"""
alpha158精选因子构造
包含：vwap_bias, kmid, kup2, klow2, cord20, wvma10, vsumd10
跨日连续 rolling（提供warmup效果，与MACD设计一致）

vwap_bias归属说明：价格位置类因子，与rsv/kdj同类，放在此模块更清晰
缓存文件：feat_cache/alpha158_{year}.pkl
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from loguru import logger

from .base import FEAT_CACHE_DIR, load_raw_data

# 并行进程数（最多使用100核）
N_WORKERS = min(100, cpu_count())


def compute_alpha158_single(df: pd.DataFrame) -> pd.DataFrame:
    """对单只股票构造alpha158精选因子"""

    # vwap_bias：close / 日内累计VWAP - 1（价格相对位置类）
    # X2：cum_vwap / cum_twap - 1（量加权均价 vs 时间加权均价的偏离，与vwap_bias方向相反）
    grp = df.groupby("date")
    cum_amount = grp["amount"].transform("cumsum")
    cum_volume = grp["volume"].transform("cumsum")
    cum_vwap = (cum_amount / cum_volume).replace([np.inf, -np.inf], np.nan)
    cum_twap = grp["close"].expanding().mean().reset_index(level=0, drop=True)
    df["vwap_bias"] = df["close"] / cum_vwap - 1
    df["x2"] = (cum_vwap / cum_twap - 1).replace([np.inf, -np.inf], np.nan)

    hl = df["high"] - df["low"]
    hl_safe = hl.clip(lower=1e-8)

    # K线形态因子（单bar，无需rolling）
    # kmid：(close-open)/(high-low)，归一化涨跌幅
    df["kmid"] = (df["close"] - df["open"]) / hl_safe
    # kup2：上影线比例，反映上方抛压
    df["kup2"] = (df["high"] - np.maximum(df["open"], df["close"])) / hl_safe
    # klow2：下影线比例，反映下方支撑
    df["klow2"] = (np.minimum(df["open"], df["close"]) - df["low"]) / hl_safe

    # cord20：close收益率与log(成交量变化率)的跨日滚动20bar相关
    close_ret = df["close"].pct_change(1).fillna(0.0)
    vol_prev = df["volume"].shift(1).clip(lower=1)
    log_vol_ret = np.log((df["volume"].clip(lower=1) / vol_prev).clip(lower=1e-8))
    df["cord20"] = close_ret.rolling(20, min_periods=10).corr(log_vol_ret).fillna(0.0)

    # wvma10：成交量加权价格变化波动率 = Std(|ret|*vol, 10) / Mean(|ret|*vol, 10)
    abs_ret = close_ret.abs()
    weighted = abs_ret * df["volume"]
    wvma_std = weighted.rolling(10, min_periods=5).std()
    wvma_mean = weighted.rolling(10, min_periods=5).mean()
    df["wvma10"] = np.where(wvma_mean > 0, wvma_std / wvma_mean, 0.0)

    # vsumd10：成交量方向性指标（放量占比 - 缩量占比），跨日rolling10
    vol_chg = df["volume"] - df["volume"].shift(1)
    sum_pos = vol_chg.clip(lower=0).rolling(10, min_periods=5).sum()
    sum_neg = (-vol_chg).clip(lower=0).rolling(10, min_periods=5).sum()
    sum_abs = vol_chg.abs().rolling(10, min_periods=5).sum()
    df["vsumd10"] = np.where(sum_abs > 0, (sum_pos - sum_neg) / sum_abs, 0.0)

    return df


def _alpha158_worker(args):
    """多进程worker：处理单只股票"""
    code, df_s = args
    return compute_alpha158_single(df_s)


def build_alpha158_features(
    year: int,
    stock_list: list = None,
    use_cache: bool = True,
    df_all: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    构造alpha158精选因子，支持独立缓存，多核并行加速。

    参数：
        year       : 数据年份
        stock_list : 指定股票列表，None表示全部
        use_cache  : 是否使用缓存
        df_all     : 已加载的原始K线数据，传入则跳过内部load_raw_data

    返回：含alpha158原始因子列的DataFrame
    """
    cache_path = FEAT_CACHE_DIR / f"alpha158_{year}.pkl"

    if use_cache and cache_path.exists():
        logger.info(f"加载alpha158因子缓存: {cache_path}")
        df = pd.read_pickle(cache_path)
        if stock_list is not None:
            df = df[df["SecuCode"].isin(stock_list)].copy()
        return df

    if df_all is None:
        df_all = load_raw_data(year, stock_list)
    elif stock_list is not None:
        df_all = df_all[df_all["SecuCode"].isin(stock_list)].copy()
    stocks = df_all["SecuCode"].unique()
    logger.info(
        f"构造alpha158因子，共 {len(stocks)} 只股票，使用 {N_WORKERS} 核并行..."
    )

    # groupby预分组：O(n)一次分组，替代O(n*k)的逐股全表扫描
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    tasks = [(code, grouped[code]) for code in stocks]

    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_alpha158_worker, tasks)

    df_feat = pd.concat(parts, ignore_index=True)
    logger.success(f"alpha158因子构造完成，shape={df_feat.shape}")

    if stock_list is None:
        df_feat.to_pickle(cache_path)
        logger.success(f"alpha158因子已缓存: {cache_path}")

    return df_feat
