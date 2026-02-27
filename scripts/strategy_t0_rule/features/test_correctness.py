"""
正确性验证脚本：对比方案B（groupby预分组 + scipy EMA）与原始方案A的数值结果是否完全一致。
验证三组因子：classic_indicators / price_volume_features / alpha158_features

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python test_correctness.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy import signal
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.base import load_raw_data, FEAT_CACHE_DIR
from features.classic_indicators import (
    compute_classic_single,
    _ema_continuous,
    MACD_FAST1,
    MACD_SLOW1,
    MACD_SIG1,
    MACD_FAST2,
    MACD_SLOW2,
    MACD_SIG2,
    KDJ_N,
    KDJ_K_SMOOTH,
    KDJ_D_SMOOTH,
    BOLL_N,
    BOLL_K,
    N_WORKERS,
)
from features.price_volume_features import compute_pv_single
from features.alpha158_features import compute_alpha158_single, _alpha158_worker

YEAR = 2024
N_TEST_STOCKS = 50  # 50只股票足够验证正确性


# ============================================================
# scipy EMA 实现（方案B核心）
# ============================================================
def _ema_scipy(arr: np.ndarray, alpha: float) -> np.ndarray:
    """用 scipy.signal.lfilter 实现 EMA，初始条件与 Python loop 一致"""
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    # zi 初始条件 = arr[0]，与 Python loop 的 out[0] = arr[0] 一致
    zi = signal.lfilter_zi(b, a) * arr[0]
    out, _ = signal.lfilter(b, a, arr, zi=zi)
    return out


def _compute_macd_scipy(close: np.ndarray, fast: int, slow: int, sig: int) -> tuple:
    ema_f = _ema_scipy(close, 2.0 / (fast + 1))
    ema_s = _ema_scipy(close, 2.0 / (slow + 1))
    dif = ema_f - ema_s
    dea = _ema_scipy(dif, 2.0 / (sig + 1))
    return dif, dea, 2.0 * (dif - dea)


def _macd_area_vectorized(macd1: np.ndarray) -> np.ndarray:
    """向量化 macd_area，替代 Python for loop"""
    macd_sign = np.sign(macd1)
    cross = np.zeros(len(macd1), dtype=bool)
    cross[1:] = macd_sign[1:] != macd_sign[:-1]
    group_id = np.cumsum(cross)
    area = pd.Series(macd1).groupby(group_id).cumsum().values
    return area


def _compute_kdj_scipy(low, high, close, n, k_smooth, d_smooth):
    """
    KDJ：rolling min/max 向量化，K/D 的 EMA 保留 Python loop。
    原因：K/D 初始值是50（非 rsv[0]），lfilter 的 zi 机制无法精确等价，
    保留 Python loop 确保与原始代码数值完全一致。
    KDJ 本身计算量不大（主要耗时在 rolling min/max，已向量化），
    Python loop 部分不是瓶颈。
    """
    low_s = pd.Series(low)
    high_s = pd.Series(high)
    low_n = low_s.rolling(n, min_periods=1).min().values
    high_n = high_s.rolling(n, min_periods=1).max().values
    hl_range = high_n - low_n
    rsv = np.where(hl_range > 0, (close - low_n) / hl_range, 0.5)

    # 保持与原始代码完全一致的初始值和递推公式
    alpha_k = 1.0 / k_smooth
    alpha_d = 1.0 / d_smooth
    size = len(close)
    k_arr = np.empty(size)
    d_arr = np.empty(size)
    k_arr[0] = 50.0 + alpha_k * (rsv[0] * 100 - 50.0)
    d_arr[0] = 50.0 + alpha_d * (k_arr[0] - 50.0)
    for i in range(1, size):
        k_arr[i] = (1.0 - alpha_k) * k_arr[i - 1] + alpha_k * rsv[i] * 100
        d_arr[i] = (1.0 - alpha_d) * d_arr[i - 1] + alpha_d * k_arr[i]

    j_arr = 3.0 * k_arr - 2.0 * d_arr
    return k_arr, d_arr, j_arr


# ============================================================
# 方案B：classic 单股函数（scipy版本）
# ============================================================
def compute_classic_single_fast(df: pd.DataFrame) -> pd.DataFrame:
    """方案B的单股计算，用 scipy EMA 替代 Python for loop"""
    close = df["close"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    df = df.copy()

    # MACD快线组
    dif1, dea1, macd1 = _compute_macd_scipy(close, MACD_FAST1, MACD_SLOW1, MACD_SIG1)
    df["dif"] = dif1
    df["dea"] = dea1
    df["macd"] = macd1
    df["macd_area"] = _macd_area_vectorized(macd1)
    df["dif_slope3"] = pd.Series(dif1, index=df.index).diff(3).fillna(0.0).values

    # MACD标准组
    dif2, dea2, macd2 = _compute_macd_scipy(close, MACD_FAST2, MACD_SLOW2, MACD_SIG2)
    df["dif_std"] = dif2
    df["dea_std"] = dea2
    df["macd_std"] = macd2

    # KDJ
    k_arr, d_arr, j_arr = _compute_kdj_scipy(
        low, high, close, KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH
    )
    df["kdj_k"] = k_arr
    df["kdj_d"] = d_arr
    df["kdj_j"] = j_arr
    df["kdj_kd"] = k_arr - d_arr

    # 布林带（本身已是向量化，无需改动）
    close_s = pd.Series(close)
    boll_mid = close_s.rolling(BOLL_N, min_periods=5).mean().values
    boll_std = close_s.rolling(BOLL_N, min_periods=5).std().values
    boll_upper = boll_mid + BOLL_K * boll_std
    boll_lower = boll_mid - BOLL_K * boll_std
    boll_range = boll_upper - boll_lower
    df["boll_pct_b"] = np.where(boll_range > 0, (close - boll_lower) / boll_range, 0.5)
    df["boll_bw"] = np.where(boll_mid > 0, boll_range / boll_mid, 0.0)

    # RSV（本身已是向量化，无需改动）
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


# ============================================================
# 方案B：build 函数（groupby预分组）
# ============================================================
def _worker_fast(args):
    code, df_s = args
    return compute_classic_single_fast(df_s)


def _worker_original(args):
    code, df_s = args
    return compute_classic_single(df_s)


def _worker_pv(args):
    code, df_s = args
    return compute_pv_single(df_s)


def build_with_groupby(df_all, worker_fn, n_workers):
    """通用：用groupby预分组代替全表扫描，然后Pool.map"""
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    stocks = list(grouped.keys())
    tasks = [(code, grouped[code]) for code in stocks]
    with Pool(processes=n_workers) as pool:
        parts = pool.map(worker_fn, tasks)
    return pd.concat(parts, ignore_index=True)


def build_with_scan(df_all, worker_fn, n_workers):
    """原始：全表扫描分组，然后Pool.map"""
    stocks = df_all["SecuCode"].unique()
    tasks = [(code, df_all[df_all["SecuCode"] == code].copy()) for code in stocks]
    with Pool(processes=n_workers) as pool:
        parts = pool.map(worker_fn, tasks)
    return pd.concat(parts, ignore_index=True)


# ============================================================
# 验证函数
# ============================================================
def compare_dataframes(
    df_orig: pd.DataFrame, df_fast: pd.DataFrame, cols_to_check: list, name: str
) -> bool:
    """
    对比两个 DataFrame 的指定列数值是否一致。
    返回 True 表示完全一致（或误差在浮点精度内）。
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"验证 [{name}]")

    # 按 SecuCode + date + bar_time 排序对齐
    sort_keys = ["SecuCode", "date", "bar_time"]
    df_orig = df_orig.sort_values(sort_keys).reset_index(drop=True)
    df_fast = df_fast.sort_values(sort_keys).reset_index(drop=True)

    if len(df_orig) != len(df_fast):
        logger.error(f"行数不一致：原始={len(df_orig)}, 方案B={len(df_fast)}")
        return False

    all_pass = True
    for col in cols_to_check:
        if col not in df_orig.columns:
            logger.warning(f"  列 {col} 不在原始结果中，跳过")
            continue
        if col not in df_fast.columns:
            logger.warning(f"  列 {col} 不在方案B结果中，跳过")
            continue

        try:
            orig_vals = df_orig[col].values.astype(np.float64)
            fast_vals = df_fast[col].values.astype(np.float64)
        except (ValueError, TypeError):
            logger.info(f"  [{col}] 非数值列，跳过")
            continue

        # NaN 位置必须一致
        nan_orig = np.isnan(orig_vals)
        nan_fast = np.isnan(fast_vals)
        if not np.array_equal(nan_orig, nan_fast):
            logger.error(f"  [{col}] NaN位置不一致！")
            all_pass = False
            continue

        # 非NaN值的最大绝对误差
        mask = ~nan_orig
        if mask.sum() == 0:
            logger.warning(f"  [{col}] 全是NaN，跳过")
            continue

        max_err = np.abs(orig_vals[mask] - fast_vals[mask]).max()
        rel_err = max_err / (np.abs(orig_vals[mask]).max() + 1e-10)

        # 浮点数容忍：绝对误差 < 1e-8 或相对误差 < 1e-6
        tol_abs = 1e-8
        tol_rel = 1e-6
        passed = (max_err < tol_abs) or (rel_err < tol_rel)
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  [{col}] {status}: max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}"
        )

        if not passed:
            all_pass = False
            # 打印几个不一致的行
            bad_idx = np.where(np.abs(orig_vals[mask] - fast_vals[mask]) > tol_abs)[0][
                :3
            ]
            for i in bad_idx:
                logger.error(
                    f"    行{i}: 原始={orig_vals[mask][i]:.10f}, 方案B={fast_vals[mask][i]:.10f}"
                )

    if all_pass:
        logger.success(f"[{name}] 全部列数值一致 ✓")
    else:
        logger.error(f"[{name}] 存在数值差异 ✗")

    return all_pass


# ============================================================
# 主流程
# ============================================================
def main():
    # 加载测试数据
    logger.info(f"加载 {N_TEST_STOCKS} 只CSI1000股票数据...")
    with open(
        "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config/csi1000_components_2024.json"
    ) as f:
        csi1000 = json.load(f)
    test_stocks = csi1000[:N_TEST_STOCKS]

    df_all = load_raw_data(YEAR, test_stocks)
    logger.info(
        f"数据加载完成，shape={df_all.shape}，股票数={df_all['SecuCode'].nunique()}"
    )

    results = {}

    # ============================================================
    # 验证1：classic_indicators — EMA scipy vs Python loop
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("验证1：classic_indicators 单股EMA对比（scipy vs Python loop）")

    # 取第一只股票单独验证 EMA 数值
    code0 = df_all["SecuCode"].unique()[0]
    df_one = df_all[df_all["SecuCode"] == code0].copy()
    close = df_one["close"].values.astype(np.float64)

    for fast in [MACD_FAST1, MACD_FAST2]:
        alpha = 2.0 / (fast + 1)
        ema_loop = _ema_continuous(close, alpha)
        ema_sci = _ema_scipy(close, alpha)
        max_err = np.abs(ema_loop - ema_sci).max()
        logger.info(
            f"  EMA(alpha={alpha:.4f}): max_abs_err={max_err:.2e} {'PASS' if max_err < 1e-8 else 'FAIL'}"
        )

    # 单股完整因子对比
    df_orig_single = compute_classic_single(df_one.copy())
    df_fast_single = compute_classic_single_fast(df_one.copy())
    classic_cols = [
        "dif",
        "dea",
        "macd",
        "macd_area",
        "dif_slope3",
        "dif_std",
        "dea_std",
        "macd_std",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "kdj_kd",
        "boll_pct_b",
        "boll_bw",
        "rsv10",
        "rsv20",
    ]
    ok_single = compare_dataframes(
        df_orig_single, df_fast_single, classic_cols, f"classic 单股({code0})"
    )

    # 全部50只股票对比（原始方案 vs 方案B）
    logger.info("\n对比50只股票的 classic 因子（原始全表扫描 vs groupby预分组）...")
    t0 = time.time()
    df_classic_orig = build_with_scan(df_all, _worker_original, N_WORKERS)
    t1 = time.time()
    df_classic_fast = build_with_groupby(df_all, _worker_fast, N_WORKERS)
    t2 = time.time()
    logger.info(f"  原始方案耗时: {t1-t0:.1f}s，方案B耗时: {t2-t1:.1f}s")
    ok_classic = compare_dataframes(
        df_classic_orig, df_classic_fast, classic_cols, "classic 50只"
    )
    results["classic"] = ok_classic

    # ============================================================
    # 验证2：price_volume_features — groupby预分组 vs 全表扫描
    # ============================================================
    # price_volume 内部没有 Python for loop EMA，全是 pandas 向量化
    # 唯一区别是 tasks 构建方式，计算逻辑完全相同，理论上100%一致
    logger.info("\n" + "=" * 60)
    logger.info("验证2：price_volume_features（groupby预分组 vs 全表扫描）")
    t0 = time.time()
    df_pv_orig = build_with_scan(df_all, _worker_pv, N_WORKERS)
    t1 = time.time()
    df_pv_fast = build_with_groupby(df_all, _worker_pv, N_WORKERS)
    t2 = time.time()
    logger.info(f"  原始方案耗时: {t1-t0:.1f}s，方案B耗时: {t2-t1:.1f}s")
    pv_cols = [
        "vol_ratio",
        "buy_pressure",
        "amplitude",
        "ret_1bar",
        "ret_3bar",
        "ret_5bar",
        "range_pos",
    ]
    ok_pv = compare_dataframes(df_pv_orig, df_pv_fast, pv_cols, "price_volume 50只")
    results["price_volume"] = ok_pv

    # ============================================================
    # 验证3：alpha158_features — groupby预分组 vs 全表扫描
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("验证3：alpha158_features（groupby预分组 vs 全表扫描）")

    t0 = time.time()
    df_a158_orig = build_with_scan(df_all, _alpha158_worker, N_WORKERS)
    t1 = time.time()
    df_a158_fast = build_with_groupby(df_all, _alpha158_worker, N_WORKERS)
    t2 = time.time()
    logger.info(f"  原始方案耗时: {t1-t0:.1f}s，方案B耗时: {t2-t1:.1f}s")

    # alpha158 列：取公共的新增列
    base_cols = [
        "SecuCode",
        "date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    a158_cols = [c for c in df_a158_orig.columns if c not in base_cols]
    logger.info(f"  alpha158 共 {len(a158_cols)} 个因子列")
    ok_a158 = compare_dataframes(df_a158_orig, df_a158_fast, a158_cols, "alpha158 50只")
    results["alpha158"] = ok_a158

    # ============================================================
    # 汇总
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("验证汇总：")
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        logger.info(f"  {name:<20} {status}")
        if not ok:
            all_pass = False

    if all_pass:
        logger.success(
            "\n所有因子组验证通过，方案B与原始方案A数值完全一致，可以安全替换。"
        )
    else:
        logger.error("\n存在数值差异，请检查上方详细日志。")


if __name__ == "__main__":
    main()
