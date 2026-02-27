"""
速度测试脚本：对比不同加速方案的耗时
目标：加速 classic_indicators.py 的因子构造

方案对比：
  A) 当前方案：全表扫描分组 + multiprocessing + Python for loop
  B) 方案1：groupby预分组 + multiprocessing + scipy lfilter EMA（减少IO开销）
  C) 方案2：全向量化（groupby transform，无multiprocessing）+ scipy lfilter EMA
  D) 方案3：groupby预分组 + multiprocessing + numba EMA（评估numba收益）

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python test_speed_classic.py
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy import signal
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.base import load_raw_data, FEAT_CACHE_DIR
from features.classic_indicators import (
    _ema_continuous,
    _compute_macd,
    _compute_kdj,
    MACD_FAST1, MACD_SLOW1, MACD_SIG1,
    MACD_FAST2, MACD_SLOW2, MACD_SIG2,
    KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH,
    BOLL_N, BOLL_K,
    compute_classic_single,
    N_WORKERS,
)

YEAR = 2024
N_TEST_STOCKS = 200  # 测试用股票数量，减少等待时间

# ============================================================
# 工具：scipy lfilter 实现向量化 EMA
# ============================================================
def _ema_scipy(arr: np.ndarray, alpha: float) -> np.ndarray:
    """用 scipy.signal.lfilter 实现 EMA，比 Python for loop 快 ~10-20x"""
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    zi = signal.lfilter_zi(b, a) * arr[0]
    out, _ = signal.lfilter(b, a, arr, zi=zi)
    return out


def _compute_macd_fast(close: np.ndarray, fast: int, slow: int, sig: int) -> tuple:
    """用 scipy EMA 计算 MACD"""
    ema_f = _ema_scipy(close, 2.0 / (fast + 1))
    ema_s = _ema_scipy(close, 2.0 / (slow + 1))
    dif = ema_f - ema_s
    dea = _ema_scipy(dif, 2.0 / (sig + 1))
    return dif, dea, 2.0 * (dif - dea)


def _compute_kdj_fast(low, high, close, n, k_smooth, d_smooth):
    """用 scipy EMA 计算 KDJ"""
    size = len(close)
    low_s = pd.Series(low)
    high_s = pd.Series(high)
    low_n = low_s.rolling(n, min_periods=1).min().values
    high_n = high_s.rolling(n, min_periods=1).max().values
    hl_range = high_n - low_n
    rsv = np.where(hl_range > 0, (close - low_n) / hl_range, 0.5)

    # K/D 用 EMA 平滑，但初始值是 50，这里用手动初始化
    alpha_k = 1.0 / k_smooth
    alpha_d = 1.0 / d_smooth
    
    # scipy lfilter 无法直接设置初始值为50（不是arr[0]），
    # 这里用 numpy cumsum 方式（IIR滤波等价）
    k_arr = np.empty(size)
    d_arr = np.empty(size)
    k_arr[0] = 50.0 + alpha_k * (rsv[0] * 100 - 50.0)
    d_arr[0] = 50.0 + alpha_d * (k_arr[0] - 50.0)
    
    # 使用 numpy 向量化（不是 Python loop，但需要累加）
    # 利用 scipy lfilter 对 rsv*100 做 EMA，然后调整初始值
    # K_t = (1-alpha_k) * K_{t-1} + alpha_k * RSV_t * 100
    # 等价于 lfilter，初始条件 zi = k_arr[0] * zi_coeff
    b_k = [alpha_k]
    a_k = [1.0, -(1.0 - alpha_k)]
    zi_k = signal.lfilter_zi(b_k, a_k) * k_arr[0]
    k_arr_full, _ = signal.lfilter(b_k, a_k, rsv * 100, zi=zi_k)
    # 注意：lfilter从索引0开始，初始状态已设为k_arr[0]，所以直接用
    k_arr = k_arr_full
    
    b_d = [alpha_d]
    a_d = [1.0, -(1.0 - alpha_d)]
    zi_d = signal.lfilter_zi(b_d, a_d) * k_arr[0]
    d_arr, _ = signal.lfilter(b_d, a_d, k_arr, zi=zi_d)
    
    j_arr = 3.0 * k_arr - 2.0 * d_arr
    return k_arr, d_arr, j_arr


def _macd_area_fast(macd1: np.ndarray) -> np.ndarray:
    """用向量化替代 Python for loop 计算 macd_area"""
    macd_sign = np.sign(macd1)
    # 找到符号变化点（交叉点）
    cross = np.zeros(len(macd1), dtype=bool)
    cross[1:] = macd_sign[1:] != macd_sign[:-1]
    # 生成每段的 group id
    group_id = np.cumsum(cross)
    # 对每个group内做累加，利用groupby（或 np.frompyfunc，但这里用pandas）
    s = pd.Series(macd1)
    area = s.groupby(group_id).cumsum().values
    return area


# ============================================================
# 方案A：原始方案（全表扫描 + multiprocessing + Python loop）
# ============================================================
def _worker_original(args):
    code, df_s = args
    return compute_classic_single(df_s)


def method_A(df_all: pd.DataFrame, stocks: list, n_workers: int) -> float:
    """原始方案：全表扫描过滤 + multiprocessing"""
    t0 = time.time()
    tasks = [(code, df_all[df_all["SecuCode"] == code].copy()) for code in stocks]
    t1 = time.time()
    logger.info(f"  [A] 构建tasks耗时: {t1-t0:.1f}s")
    
    with Pool(processes=n_workers) as pool:
        parts = pool.map(_worker_original, tasks)
    t2 = time.time()
    logger.info(f"  [A] Pool.map耗时: {t2-t1:.1f}s")
    
    df_feat = pd.concat(parts, ignore_index=True)
    t3 = time.time()
    logger.info(f"  [A] concat耗时: {t3-t2:.1f}s")
    return t3 - t0


# ============================================================
# 方案B：groupby预分组 + multiprocessing + scipy EMA
# ============================================================
def compute_classic_fast(df: pd.DataFrame) -> pd.DataFrame:
    """单只股票，用 scipy EMA 加速"""
    close = df["close"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)

    # MACD快线组
    dif1, dea1, macd1 = _compute_macd_fast(close, MACD_FAST1, MACD_SLOW1, MACD_SIG1)
    df = df.copy()
    df["dif"] = dif1
    df["dea"] = dea1
    df["macd"] = macd1
    df["macd_area"] = _macd_area_fast(macd1)
    
    close_s = pd.Series(close, index=df.index)
    df["dif_slope3"] = close_s.copy()
    df["dif_slope3"] = dif1
    df["dif_slope3"] = pd.Series(dif1, index=df.index).diff(3).fillna(0.0).values

    # MACD标准组
    dif2, dea2, macd2 = _compute_macd_fast(close, MACD_FAST2, MACD_SLOW2, MACD_SIG2)
    df["dif_std"] = dif2
    df["dea_std"] = dea2
    df["macd_std"] = macd2

    # KDJ
    k_arr, d_arr, j_arr = _compute_kdj_fast(low, high, close, KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH)
    df["kdj_k"] = k_arr
    df["kdj_d"] = d_arr
    df["kdj_j"] = j_arr
    df["kdj_kd"] = k_arr - d_arr

    # 布林带
    close_s = pd.Series(close)
    boll_mid = close_s.rolling(BOLL_N, min_periods=5).mean().values
    boll_std = close_s.rolling(BOLL_N, min_periods=5).std().values
    boll_upper = boll_mid + BOLL_K * boll_std
    boll_lower = boll_mid - BOLL_K * boll_std
    boll_range = boll_upper - boll_lower
    df["boll_pct_b"] = np.where(boll_range > 0, (close - boll_lower) / boll_range, 0.5)
    df["boll_bw"] = np.where(boll_mid > 0, boll_range / boll_mid, 0.0)

    # RSV
    low_s = pd.Series(low)
    high_s = pd.Series(high)
    low10 = low_s.rolling(10, min_periods=3).min().values
    high10 = high_s.rolling(10, min_periods=3).max().values
    df["rsv10"] = np.where((high10 - low10) > 0, (close - low10) / (high10 - low10), 0.5)
    low20 = low_s.rolling(20, min_periods=5).min().values
    high20 = high_s.rolling(20, min_periods=5).max().values
    df["rsv20"] = np.where((high20 - low20) > 0, (close - low20) / (high20 - low20), 0.5)

    return df


def _worker_fast(args):
    code, df_s = args
    return compute_classic_fast(df_s)


def method_B(df_all: pd.DataFrame, stocks: list, n_workers: int) -> float:
    """groupby预分组 + multiprocessing + scipy EMA"""
    t0 = time.time()
    # 用 groupby 预分组，避免 1000 次全表扫描
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    tasks = [(code, grouped[code]) for code in stocks if code in grouped]
    t1 = time.time()
    logger.info(f"  [B] groupby预分组耗时: {t1-t0:.1f}s")

    with Pool(processes=n_workers) as pool:
        parts = pool.map(_worker_fast, tasks)
    t2 = time.time()
    logger.info(f"  [B] Pool.map耗时: {t2-t1:.1f}s")

    df_feat = pd.concat(parts, ignore_index=True)
    t3 = time.time()
    logger.info(f"  [B] concat耗时: {t3-t2:.1f}s")
    return t3 - t0


# ============================================================
# 方案C：全向量化（无multiprocessing，groupby transform）
# ============================================================
def _ema_groupby_vectorized(series: pd.Series, alpha: float, groupby_col: pd.Series) -> np.ndarray:
    """
    对每个 group 独立做 EMA（跨日连续，不重置）。
    用 groupby + apply，但 apply 内部用 scipy lfilter。
    """
    def _ema_group(x):
        arr = x.values.astype(np.float64)
        b = [alpha]
        a = [1.0, -(1.0 - alpha)]
        zi = signal.lfilter_zi(b, a) * arr[0]
        out, _ = signal.lfilter(b, a, arr, zi=zi)
        return pd.Series(out, index=x.index)

    return series.groupby(groupby_col).transform(_ema_group)


def method_C(df_all: pd.DataFrame, stocks: list) -> float:
    """全向量化（groupby transform），无multiprocessing"""
    t0 = time.time()
    df = df_all.copy()
    code_col = df["SecuCode"]

    # MACD快线组：对所有股票的close列同时做EMA
    close = df["close"].astype(np.float64)
    ema_f1 = _ema_groupby_vectorized(close, 2.0 / (MACD_FAST1 + 1), code_col)
    ema_s1 = _ema_groupby_vectorized(close, 2.0 / (MACD_SLOW1 + 1), code_col)
    dif1 = ema_f1 - ema_s1
    dea1 = _ema_groupby_vectorized(dif1, 2.0 / (MACD_SIG1 + 1), code_col)
    df["dif"] = dif1
    df["dea"] = dea1
    df["macd"] = 2.0 * (dif1 - dea1)
    
    # macd_area 向量化
    df["macd_area"] = df.groupby("SecuCode")["macd"].transform(
        lambda x: _macd_area_fast(x.values)
    )
    df["dif_slope3"] = df.groupby("SecuCode")["dif"].transform(lambda x: x.diff(3).fillna(0.0))

    t1 = time.time()
    logger.info(f"  [C] MACD快线组耗时: {t1-t0:.1f}s")

    # MACD标准组
    ema_f2 = _ema_groupby_vectorized(close, 2.0 / (MACD_FAST2 + 1), code_col)
    ema_s2 = _ema_groupby_vectorized(close, 2.0 / (MACD_SLOW2 + 1), code_col)
    dif2 = ema_f2 - ema_s2
    dea2 = _ema_groupby_vectorized(dif2, 2.0 / (MACD_SIG2 + 1), code_col)
    df["dif_std"] = dif2
    df["dea_std"] = dea2
    df["macd_std"] = 2.0 * (dif2 - dea2)
    t2 = time.time()
    logger.info(f"  [C] MACD标准组耗时: {t2-t1:.1f}s")

    # KDJ - rolling min/max 向量化
    low = df["low"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low_n = low.groupby(code_col).transform(lambda x: x.rolling(KDJ_N, min_periods=1).min())
    high_n = high.groupby(code_col).transform(lambda x: x.rolling(KDJ_N, min_periods=1).max())
    hl_range = high_n - low_n
    rsv_kdj = np.where(hl_range > 0, (close - low_n) / hl_range, 0.5)
    rsv_series = pd.Series(rsv_kdj * 100, index=df.index)
    
    alpha_k = 1.0 / KDJ_K_SMOOTH
    alpha_d = 1.0 / KDJ_D_SMOOTH
    # K/D 的初始值是50，用 lfilter 但需要从50开始
    def _kdj_k_group(x):
        arr = x.values.astype(np.float64)
        b = [alpha_k]
        a = [1.0, -(1.0 - alpha_k)]
        init_k = 50.0 + alpha_k * (arr[0] - 50.0)
        zi = signal.lfilter_zi(b, a) * init_k
        out, _ = signal.lfilter(b, a, arr, zi=zi)
        return pd.Series(out, index=x.index)
    
    k_series = rsv_series.groupby(code_col).transform(_kdj_k_group)
    
    def _kdj_d_group(x):
        arr = x.values.astype(np.float64)
        b = [alpha_d]
        a = [1.0, -(1.0 - alpha_d)]
        zi = signal.lfilter_zi(b, a) * arr[0]
        out, _ = signal.lfilter(b, a, arr, zi=zi)
        return pd.Series(out, index=x.index)
    
    d_series = k_series.groupby(code_col).transform(_kdj_d_group)
    df["kdj_k"] = k_series
    df["kdj_d"] = d_series
    df["kdj_j"] = 3.0 * k_series - 2.0 * d_series
    df["kdj_kd"] = k_series - d_series
    t3 = time.time()
    logger.info(f"  [C] KDJ耗时: {t3-t2:.1f}s")

    # 布林带
    boll_mid = close.groupby(code_col).transform(lambda x: x.rolling(BOLL_N, min_periods=5).mean())
    boll_std_v = close.groupby(code_col).transform(lambda x: x.rolling(BOLL_N, min_periods=5).std())
    boll_upper = boll_mid + BOLL_K * boll_std_v
    boll_lower = boll_mid - BOLL_K * boll_std_v
    boll_range = boll_upper - boll_lower
    df["boll_pct_b"] = np.where(boll_range > 0, (close - boll_lower) / boll_range, 0.5)
    df["boll_bw"] = np.where(boll_mid > 0, boll_range / boll_mid, 0.0)
    t4 = time.time()
    logger.info(f"  [C] 布林带耗时: {t4-t3:.1f}s")

    # RSV
    low10 = low.groupby(code_col).transform(lambda x: x.rolling(10, min_periods=3).min())
    high10 = high.groupby(code_col).transform(lambda x: x.rolling(10, min_periods=3).max())
    df["rsv10"] = np.where((high10 - low10) > 0, (close - low10) / (high10 - low10), 0.5)
    low20 = low.groupby(code_col).transform(lambda x: x.rolling(20, min_periods=5).min())
    high20 = high.groupby(code_col).transform(lambda x: x.rolling(20, min_periods=5).max())
    df["rsv20"] = np.where((high20 - low20) > 0, (close - low20) / (high20 - low20), 0.5)
    t5 = time.time()
    logger.info(f"  [C] RSV耗时: {t5-t4:.1f}s")

    return t5 - t0


# ============================================================
# 方案D：groupby预分组 + multiprocessing + scipy EMA（最大化并行）
# ============================================================
# 这与方案B相同，但使用不同 n_workers 来评估并行扩展性
def method_D(df_all: pd.DataFrame, stocks: list, n_workers: int) -> float:
    """groupby预分组 + 更大并行数 multiprocessing + scipy EMA"""
    return method_B(df_all, stocks, n_workers)


# ============================================================
# 单个 EMA 函数速度对比
# ============================================================
def benchmark_ema_single(arr_len: int = 11596):
    """对比单次 EMA 调用：Python loop vs scipy lfilter"""
    arr = np.random.randn(arr_len).cumsum()
    alpha = 2.0 / (12 + 1)

    # Python loop
    t0 = time.time()
    for _ in range(1000):
        _ema_continuous(arr, alpha)
    t_loop = (time.time() - t0) / 1000 * 1000  # ms per call

    # scipy lfilter
    t0 = time.time()
    for _ in range(1000):
        _ema_scipy(arr, alpha)
    t_scipy = (time.time() - t0) / 1000 * 1000  # ms per call

    logger.info(f"单次EMA对比（arr_len={arr_len}）:")
    logger.info(f"  Python loop: {t_loop:.3f} ms/call")
    logger.info(f"  scipy lfilter: {t_scipy:.3f} ms/call")
    logger.info(f"  加速比: {t_loop/t_scipy:.1f}x")


# ============================================================
# 主函数
# ============================================================
def main():
    logger.info(f"加载 {YEAR} 年 CSI1000 数据（前{N_TEST_STOCKS}只股票）...")
    
    # 加载数据
    cache_path = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache/df_5m_2024.pkl")
    t0 = time.time()
    
    # 读取CSI1000成分股列表
    import json
    csi1000_path = "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config/csi1000_components_2024.json"
    with open(csi1000_path) as f:
        csi1000 = json.load(f)
    # 取前N_TEST_STOCKS只
    test_stocks = csi1000[:N_TEST_STOCKS]
    
    df_all = load_raw_data(YEAR, test_stocks)
    logger.info(f"数据加载完成，shape={df_all.shape}，耗时={time.time()-t0:.1f}s")
    
    stocks = df_all["SecuCode"].unique().tolist()
    logger.info(f"实际股票数：{len(stocks)}")
    
    # === EMA 单函数对比 ===
    logger.info("\n=== EMA单函数速度对比 ===")
    benchmark_ema_single(arr_len=len(df_all) // len(stocks))

    # === 方案A：原始方案 ===
    logger.info(f"\n=== 方案A：原始方案（全表扫描 + multiprocessing，{N_WORKERS}核）===")
    t_A = method_A(df_all, stocks, N_WORKERS)
    logger.info(f"方案A 总耗时: {t_A:.1f}s")

    # === 方案B：groupby预分组 + scipy EMA ===
    logger.info(f"\n=== 方案B：groupby预分组 + scipy EMA（{N_WORKERS}核）===")
    t_B = method_B(df_all, stocks, N_WORKERS)
    logger.info(f"方案B 总耗时: {t_B:.1f}s，相比A加速 {t_A/t_B:.1f}x")

    # === 方案C：全向量化，无multiprocessing ===
    logger.info(f"\n=== 方案C：全向量化 groupby transform（无multiprocessing）===")
    t_C = method_C(df_all, stocks)
    logger.info(f"方案C 总耗时: {t_C:.1f}s，相比A加速 {t_A/t_C:.1f}x")

    # === 汇总 ===
    logger.info(f"\n{'='*50}")
    logger.info(f"{'方案':<10} {'耗时(s)':<12} {'加速比':<10}")
    logger.info(f"{'-'*32}")
    logger.info(f"{'A(原始)':<10} {t_A:<12.1f} {'1.0x':<10}")
    logger.info(f"{'B(scipy)':<10} {t_B:<12.1f} {f'{t_A/t_B:.1f}x':<10}")
    logger.info(f"{'C(全向量)':<10} {t_C:<12.1f} {f'{t_A/t_C:.1f}x':<10}")
    logger.info(f"{'='*50}")
    logger.info(f"注：以上为 {N_TEST_STOCKS} 只股票的测试结果，全量1000只可参考比例推算")


if __name__ == "__main__":
    main()
