"""
经典技术指标因子构造
包含：MACD（双参数）、KDJ、布林带、RSV
全部跨日连续计算，EMA不在每天开盘时重置（提供warmup效果）

因子列表：
  MACD快线组 (9,12,7)：dif, dea, macd, macd_area, dif_slope3
  MACD标准组 (12,26,9)：dif_std, dea_std, macd_std
  KDJ (9,3,3)：kdj_k, kdj_d, kdj_j, kdj_kd（KD差值）
  布林带 (20,2)：boll_pct_b（%B价格位置）, boll_bw（带宽）
  RSV：rsv10, rsv20（近N根bar内价格位置，KDJ的K值前身）

缓存文件：feat_cache/classic_{year}.pkl
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from loguru import logger
from scipy import signal

from .base import FEAT_CACHE_DIR, load_raw_data

# 并行进程数（最多使用100核）
N_WORKERS = min(100, cpu_count())

# ==================== 参数配置 ====================
# MACD快线组（超快线，捕捉即时动能）
MACD_FAST1, MACD_SLOW1, MACD_SIG1 = 9, 12, 7
# MACD标准组（标准参数，趋势确认）
MACD_FAST2, MACD_SLOW2, MACD_SIG2 = 12, 26, 9
# KDJ参数
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3
# 布林带参数
BOLL_N, BOLL_K = 20, 2.0


# ==================== EMA跨日连续计算（scipy加速，~28x） ====================
def _ema_continuous(arr: np.ndarray, alpha: float) -> np.ndarray:
    """跨日连续EMA，用scipy.signal.lfilter实现，比Python for loop快约28倍"""
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    zi = signal.lfilter_zi(b, a) * arr[0]
    out, _ = signal.lfilter(b, a, arr, zi=zi)
    return out


def _compute_macd(close: np.ndarray, fast: int, slow: int, sig: int) -> tuple:
    """跨日连续计算MACD，返回 (dif, dea, macd_hist)"""
    ema_f = _ema_continuous(close, 2.0 / (fast + 1))
    ema_s = _ema_continuous(close, 2.0 / (slow + 1))
    dif = ema_f - ema_s
    dea = _ema_continuous(dif, 2.0 / (sig + 1))
    return dif, dea, 2.0 * (dif - dea)


def _compute_kdj(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    n: int,
    k_smooth: int,
    d_smooth: int,
) -> tuple:
    """
    跨日连续计算KDJ。
    RSV = (close - lowest_low_N) / (highest_high_N - lowest_low_N)
    K = EMA(RSV, k_smooth)，初始值50
    D = EMA(K, d_smooth)，初始值50
    J = 3K - 2D
    """
    size = len(close)
    rsv = np.empty(size)

    # 用rolling min/max计算RSV（向量化）
    low_s = pd.Series(low)
    high_s = pd.Series(high)
    low_n = low_s.rolling(n, min_periods=1).min().values
    high_n = high_s.rolling(n, min_periods=1).max().values
    hl_range = high_n - low_n
    rsv = np.where(hl_range > 0, (close - low_n) / hl_range, 0.5)

    # K、D用EMA平滑，初始值50（标准KDJ约定）
    alpha_k = 1.0 / k_smooth
    alpha_d = 1.0 / d_smooth
    k_arr = np.empty(size)
    d_arr = np.empty(size)
    k_arr[0] = 50.0 + alpha_k * (rsv[0] * 100 - 50.0)
    d_arr[0] = 50.0 + alpha_d * (k_arr[0] - 50.0)
    for i in range(1, size):
        k_arr[i] = (1.0 - alpha_k) * k_arr[i - 1] + alpha_k * rsv[i] * 100
        d_arr[i] = (1.0 - alpha_d) * d_arr[i - 1] + alpha_d * k_arr[i]

    j_arr = 3.0 * k_arr - 2.0 * d_arr
    return k_arr, d_arr, j_arr


# ==================== 单股因子构造 ====================
def compute_classic_single(df: pd.DataFrame) -> pd.DataFrame:
    """对单只股票构造所有经典技术指标因子"""
    close = df["close"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)

    # ---- MACD快线组 (9,12,7) ----
    dif1, dea1, macd1 = _compute_macd(close, MACD_FAST1, MACD_SLOW1, MACD_SIG1)
    df["dif"] = dif1
    df["dea"] = dea1
    df["macd"] = macd1

    # macd_area：当前红/绿柱段累计面积（每次交叉后重置，向量化实现）
    macd_sign = np.sign(macd1)
    cross = np.zeros(len(macd1), dtype=bool)
    cross[1:] = macd_sign[1:] != macd_sign[:-1]
    group_id = np.cumsum(cross)
    df["macd_area"] = pd.Series(macd1, index=df.index).groupby(group_id).cumsum().values

    # dif_slope3：DIF最近3根bar的变化速度
    df["dif_slope3"] = df["dif"].diff(3).fillna(0.0)

    # ---- MACD标准组 (12,26,9) ----
    dif2, dea2, macd2 = _compute_macd(close, MACD_FAST2, MACD_SLOW2, MACD_SIG2)
    df["dif_std"] = dif2
    df["dea_std"] = dea2
    df["macd_std"] = macd2

    # ---- KDJ (9,3,3) ----
    k_arr, d_arr, j_arr = _compute_kdj(
        low, high, close, KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH
    )
    df["kdj_k"] = k_arr
    df["kdj_d"] = d_arr
    df["kdj_j"] = j_arr
    df["kdj_kd"] = k_arr - d_arr  # KD差值，类似MACD的DIF-DEA，反映K/D背离程度

    # ---- 布林带 (20,2) ----
    close_s = pd.Series(close)
    boll_mid = close_s.rolling(BOLL_N, min_periods=5).mean().values
    boll_std = close_s.rolling(BOLL_N, min_periods=5).std().values
    boll_upper = boll_mid + BOLL_K * boll_std
    boll_lower = boll_mid - BOLL_K * boll_std
    boll_range = boll_upper - boll_lower

    # %B：价格在布林带内的位置，0=下轨，1=上轨，0.5=中轨
    df["boll_pct_b"] = np.where(boll_range > 0, (close - boll_lower) / boll_range, 0.5)
    # 带宽：(上轨-下轨)/中轨，反映波动率收缩/扩张
    df["boll_bw"] = np.where(boll_mid > 0, boll_range / boll_mid, 0.0)

    # ---- RSV（跨日连续rolling，与KDJ的RSV计算一致但窗口不同） ----
    low_s = pd.Series(low)
    high_s = pd.Series(high)

    low10 = low_s.rolling(10, min_periods=3).min().values
    high10 = high_s.rolling(10, min_periods=3).max().values
    df["rsv10"] = np.where(
        (high10 - low10) > 0, (close - low10) / (high10 - low10), 0.5
    )

    low20 = low_s.rolling(20, min_periods=5).min().values
    high20 = high_s.rolling(20, min_periods=5).max().values
    df["rsv20"] = np.where(
        (high20 - low20) > 0, (close - low20) / (high20 - low20), 0.5
    )

    return df


def _classic_worker(args):
    """多进程worker：处理单只股票"""
    code, df_s = args
    return compute_classic_single(df_s)


# ==================== 批量构造 + 缓存 ====================
def build_classic_features(
    year: int,
    stock_list: list = None,
    use_cache: bool = True,
    df_all: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    构造经典技术指标因子，支持独立缓存，多核并行加速。

    参数：
        year       : 数据年份
        stock_list : 指定股票列表，None表示全部
        use_cache  : 是否使用缓存
        df_all     : 已加载的原始K线数据，传入则跳过内部load_raw_data

    返回：含经典技术指标原始因子列的DataFrame
    """
    cache_path = FEAT_CACHE_DIR / f"classic_{year}.pkl"

    if use_cache and cache_path.exists():
        logger.info(f"加载经典指标因子缓存: {cache_path}")
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
        f"构造经典技术指标因子，共 {len(stocks)} 只股票，使用 {N_WORKERS} 核并行..."
    )

    # groupby预分组：O(n)一次分组，替代O(n*k)的逐股全表扫描
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    tasks = [(code, grouped[code]) for code in stocks]

    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_classic_worker, tasks)

    df_feat = pd.concat(parts, ignore_index=True)
    logger.success(f"经典技术指标因子构造完成，shape={df_feat.shape}")

    if stock_list is None:
        df_feat.to_pickle(cache_path)
        logger.success(f"经典技术指标因子已缓存: {cache_path}")

    return df_feat
