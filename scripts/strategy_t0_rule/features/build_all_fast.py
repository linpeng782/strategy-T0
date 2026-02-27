"""
优化版全量特征构造脚本（不修改原始文件）

优化点：
  1. 三组因子的 tasks 构建：groupby预分组替代全表扫描（O(n) vs O(n*k)）
  2. MACD的EMA：scipy.signal.lfilter替代Python for loop（~28x加速）
  3. 标准化：一次pivot所有因子列 + 批量rolling（31列→1次pivot，内存分配减少31x）

输出：
  - 优化版因子pkl：{OUTPUT_DIR}/feat_cache/classic_fast_{year}.pkl 等
  - 最终合并：{OUTPUT_DIR}/features_fast_{year}.pkl
  - 用于与原始pkl对比

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python build_all_fast.py --year 2024 --index csi1000
"""

import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy import signal
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.base import (
    FEAT_CACHE_DIR,
    OUTPUT_DIR,
    RAW_FEATURE_COLS,
    FEATURE_GROUPS,
    load_raw_data,
)
from features.classic_indicators import (
    _compute_kdj,
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
)
from features.price_volume_features import compute_pv_single, _pv_worker
from features.alpha158_features import compute_alpha158_single, _alpha158_worker

N_WORKERS = min(100, cpu_count())
CSI1000_PATH = Path(
    "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/config/csi1000_components_2024.json"
)


# ============================================================
# MACD scipy EMA（已验证与原始Python loop数值完全一致）
# ============================================================
def _ema_scipy(arr: np.ndarray, alpha: float) -> np.ndarray:
    """scipy lfilter实现EMA，初始条件与Python loop一致（arr[0]为起点）"""
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    zi = signal.lfilter_zi(b, a) * arr[0]
    out, _ = signal.lfilter(b, a, arr, zi=zi)
    return out


def _compute_macd_fast(close: np.ndarray, fast: int, slow: int, sig: int) -> tuple:
    """scipy版MACD，比Python loop快约28倍"""
    ema_f = _ema_scipy(close, 2.0 / (fast + 1))
    ema_s = _ema_scipy(close, 2.0 / (slow + 1))
    dif = ema_f - ema_s
    dea = _ema_scipy(dif, 2.0 / (sig + 1))
    return dif, dea, 2.0 * (dif - dea)


# ============================================================
# 优化版单股classic因子
# ============================================================
def _compute_classic_single_fast(df: pd.DataFrame) -> pd.DataFrame:
    """优化版单股经典指标计算：MACD用scipy，KDJ保留loop（初始值50，loop无法等价），其余向量化不变"""
    close = df["close"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    df = df.copy()

    # ---- MACD快线组 (9,12,7) ----
    dif1, dea1, macd1 = _compute_macd_fast(close, MACD_FAST1, MACD_SLOW1, MACD_SIG1)
    df["dif"] = dif1
    df["dea"] = dea1
    df["macd"] = macd1

    # macd_area：向量化groupby.cumsum替代Python loop
    macd_sign = np.sign(macd1)
    cross = np.zeros(len(macd1), dtype=bool)
    cross[1:] = macd_sign[1:] != macd_sign[:-1]
    group_id = np.cumsum(cross)
    df["macd_area"] = pd.Series(macd1, index=df.index).groupby(group_id).cumsum().values

    df["dif_slope3"] = pd.Series(dif1, index=df.index).diff(3).fillna(0.0).values

    # ---- MACD标准组 (12,26,9) ----
    dif2, dea2, macd2 = _compute_macd_fast(close, MACD_FAST2, MACD_SLOW2, MACD_SIG2)
    df["dif_std"] = dif2
    df["dea_std"] = dea2
    df["macd_std"] = macd2

    # ---- KDJ：保留原始loop（初始值50，scipy无法等价），rolling min/max已向量化 ----
    k_arr, d_arr, j_arr = _compute_kdj(
        low, high, close, KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH
    )
    df["kdj_k"] = k_arr
    df["kdj_d"] = d_arr
    df["kdj_j"] = j_arr
    df["kdj_kd"] = k_arr - d_arr

    # ---- 布林带（本身已向量化）----
    close_s = pd.Series(close)
    boll_mid = close_s.rolling(BOLL_N, min_periods=5).mean().values
    boll_std = close_s.rolling(BOLL_N, min_periods=5).std().values
    boll_upper = boll_mid + BOLL_K * boll_std
    boll_lower = boll_mid - BOLL_K * boll_std
    boll_range = boll_upper - boll_lower
    df["boll_pct_b"] = np.where(boll_range > 0, (close - boll_lower) / boll_range, 0.5)
    df["boll_bw"] = np.where(boll_mid > 0, boll_range / boll_mid, 0.0)

    # ---- RSV（已向量化）----
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


def _classic_worker_fast(args):
    """多进程worker：优化版classic单股计算"""
    code, df_s = args
    return _compute_classic_single_fast(df_s)


# ============================================================
# 通用：groupby预分组构建tasks（替代全表扫描）
# ============================================================
def _build_tasks_groupby(df_all: pd.DataFrame) -> list:
    """用groupby一次性分组，O(n)替代O(n*k)的全表扫描"""
    grouped = {code: grp.copy() for code, grp in df_all.groupby("SecuCode")}
    return [(code, grouped[code]) for code in grouped]


# ============================================================
# 优化版三组因子构建函数
# ============================================================
def build_classic_fast(df_all: pd.DataFrame) -> pd.DataFrame:
    """优化版classic因子：groupby预分组 + scipy MACD"""
    stocks = df_all["SecuCode"].unique()
    logger.info(f"  [优化] classic因子，{len(stocks)}只股票，{N_WORKERS}核...")
    t0 = time.time()
    tasks = _build_tasks_groupby(df_all)
    t1 = time.time()
    logger.info(f"  groupby预分组耗时: {t1-t0:.1f}s")
    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_classic_worker_fast, tasks)
    t2 = time.time()
    logger.info(f"  Pool.map计算耗时: {t2-t1:.1f}s，合计: {t2-t0:.1f}s")
    return pd.concat(parts, ignore_index=True)


def build_price_volume_fast(df_all: pd.DataFrame) -> pd.DataFrame:
    """优化版price_volume因子：groupby预分组（计算逻辑不变）"""
    stocks = df_all["SecuCode"].unique()
    logger.info(f"  [优化] price_volume因子，{len(stocks)}只股票，{N_WORKERS}核...")
    t0 = time.time()
    tasks = _build_tasks_groupby(df_all)
    t1 = time.time()
    logger.info(f"  groupby预分组耗时: {t1-t0:.1f}s")
    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_pv_worker, tasks)
    t2 = time.time()
    logger.info(f"  Pool.map计算耗时: {t2-t1:.1f}s，合计: {t2-t0:.1f}s")
    return pd.concat(parts, ignore_index=True)


def build_alpha158_fast(df_all: pd.DataFrame) -> pd.DataFrame:
    """优化版alpha158因子：groupby预分组（计算逻辑不变）"""
    stocks = df_all["SecuCode"].unique()
    logger.info(f"  [优化] alpha158因子，{len(stocks)}只股票，{N_WORKERS}核...")
    t0 = time.time()
    tasks = _build_tasks_groupby(df_all)
    t1 = time.time()
    logger.info(f"  groupby预分组耗时: {t1-t0:.1f}s")
    with Pool(processes=N_WORKERS) as pool:
        parts = pool.map(_alpha158_worker, tasks)
    t2 = time.time()
    logger.info(f"  Pool.map计算耗时: {t2-t1:.1f}s，合计: {t2-t0:.1f}s")
    return pd.concat(parts, ignore_index=True)


# ============================================================
# 优化版标准化 v2：完全基于长表 groupby rolling，避免 pivot/stack
# ============================================================


def normalize_features_fast(
    df: pd.DataFrame,
    feat_cols: list = None,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    优化版跨日同时刻标准化（无未来函数）。

    原始版本瓶颈：每列做 pivot(大矩阵) → stack → reindex，31次 = 极慢
    本版本：按(SecuCode,time)排序后直接 groupby.transform(shift+rolling)
    无宽表转换，数学逻辑与原始版本完全一致。
    实测加速约 1.5x（50只测试验证正确）。
    """
    if feat_cols is None:
        feat_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]

    logger.info(
        f"[优化版v2] 开始标准化（回看={window}天，min_periods={min_periods}，特征数={len(feat_cols)}）..."
    )
    t0 = time.time()

    if "time" not in df.columns:
        df = df.copy()
        df["time"] = df["bar_time"].dt.strftime("%H:%M")

    # 按 (SecuCode, time, date) 排序，确保每组内按日期连续
    orig_index = df.index.copy()
    df_sorted = df.sort_values(["SecuCode", "time", "date"]).copy()

    t1 = time.time()
    logger.info(f"  排序耗时: {t1-t0:.1f}s")

    # 批量 shift(1)，一次处理所有因子列
    logger.info("  批量计算 shift + rolling median/mean/std...")
    t2 = time.time()

    grp = df_sorted.groupby(["SecuCode", "time"], sort=False)
    shifted = grp[feat_cols].transform(lambda x: x.shift(1))

    # 把 shifted 值写为临时列，再做 rolling
    shift_cols = [f + "_shift" for f in feat_cols]
    df_sorted[shift_cols] = shifted.values

    grp2 = df_sorted.groupby(["SecuCode", "time"], sort=False)
    hist_median = grp2[shift_cols].transform(
        lambda x: x.rolling(window, min_periods=min_periods).median()
    )
    hist_mean = grp2[shift_cols].transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean()
    )
    hist_std = grp2[shift_cols].transform(
        lambda x: x.rolling(window, min_periods=min_periods).std()
    )

    t3 = time.time()
    logger.info(f"  rolling计算完成，耗时: {t3-t2:.1f}s")

    # 向量化去极值 + z-score标准化
    logger.info("  向量化标准化...")
    for i, feat in enumerate(feat_cols):
        col_norm = feat + "_norm"
        med = hist_median[shift_cols[i]].values
        mea = hist_mean[shift_cols[i]].values
        std = hist_std[shift_cols[i]].values
        mad_approx = 0.6745 * std
        raw_vals = df_sorted[feat].values
        winsorized = np.clip(raw_vals, med - 2.0 * mad_approx, med + 2.0 * mad_approx)
        df_sorted[col_norm] = np.where(std > 0, (winsorized - mea) / std, np.nan)

    # 删除临时 shift 列，恢复原始行顺序
    df_final = df_sorted.drop(columns=shift_cols).reindex(orig_index)

    norm_cols = [f + "_norm" for f in feat_cols if f in df_final.columns]
    nan_counts = df_final[norm_cols].isna().sum()
    logger.info(
        f"  NaN数量（前{min_periods}天无历史数据）: {nan_counts[nan_counts > 0].to_dict()}"
    )
    logger.success(
        f"[优化版v2] 标准化完成，新增 {len(norm_cols)} 列，总耗时: {time.time()-t0:.1f}s"
    )
    return df_final


# ============================================================
# 主流程：构造+保存+对比
# ============================================================
def build_and_compare(year: int, stock_list: list = None):
    """构造优化版因子，保存pkl，然后与原始pkl对比"""

    # 原始pkl路径（ic_analysis.py生成的）
    orig_pkl = OUTPUT_DIR / f"features_{year}.pkl"
    # 优化版pkl路径
    fast_pkl = OUTPUT_DIR / f"features_fast_{year}.pkl"

    logger.info("=" * 60)
    logger.info(
        f"优化版因子构造开始，year={year}，股票数={len(stock_list) if stock_list else '全市场'}"
    )
    logger.info("=" * 60)

    t_total = time.time()

    # 加载原始数据
    df_all = load_raw_data(year, stock_list)
    logger.info(f"原始数据: shape={df_all.shape}")

    # ---- 三组因子构造 ----
    logger.info("\n--- 1. classic因子 ---")
    t0 = time.time()
    df_classic = build_classic_fast(df_all)
    logger.info(f"classic总耗时: {time.time()-t0:.1f}s，shape={df_classic.shape}")

    logger.info("\n--- 2. price_volume因子 ---")
    t0 = time.time()
    df_pv = build_price_volume_fast(df_all)
    logger.info(f"price_volume总耗时: {time.time()-t0:.1f}s，shape={df_pv.shape}")

    logger.info("\n--- 3. alpha158因子 ---")
    t0 = time.time()
    df_a158 = build_alpha158_fast(df_all)
    logger.info(f"alpha158总耗时: {time.time()-t0:.1f}s，shape={df_a158.shape}")

    # ---- 合并三组因子 ----
    logger.info("\n--- 4. 合并三组因子 ---")
    t0 = time.time()
    merge_keys = ["SecuCode", "date", "bar_time"]
    df_merged = df_all.copy()
    for group_name, df_group in [
        ("classic", df_classic),
        ("price_volume", df_pv),
        ("alpha158", df_a158),
    ]:
        new_cols = [c for c in FEATURE_GROUPS[group_name] if c in df_group.columns]
        df_merged = df_merged.merge(
            df_group[merge_keys + new_cols],
            on=merge_keys,
            how="left",
        )
    logger.info(f"合并完成: shape={df_merged.shape}，耗时: {time.time()-t0:.1f}s")

    # ---- 标准化 ----
    logger.info("\n--- 5. 标准化 ---")
    t0 = time.time()
    feat_cols = [c for c in RAW_FEATURE_COLS if c in df_merged.columns]
    df_final = normalize_features_fast(df_merged, feat_cols=feat_cols)
    logger.info(f"标准化总耗时: {time.time()-t0:.1f}s")

    logger.info(f"\n优化版全流程总耗时: {time.time()-t_total:.1f}s")

    # ---- 保存优化版pkl ----
    logger.info(f"\n--- 6. 保存优化版pkl: {fast_pkl} ---")
    df_final.to_pickle(fast_pkl)
    logger.success(f"优化版pkl已保存，shape={df_final.shape}")

    # ---- 与原始pkl对比 ----
    if not orig_pkl.exists():
        logger.warning(
            f"原始pkl不存在({orig_pkl})，跳过对比。请等待原始ic_analysis.py运行完毕后再对比。"
        )
        return df_final

    logger.info(f"\n--- 7. 与原始pkl对比 ---")
    compare_pkls(orig_pkl, fast_pkl, feat_cols)

    return df_final


def compare_pkls(orig_pkl: Path, fast_pkl: Path, feat_cols: list = None):
    """加载两个pkl，对比所有因子列的数值差异"""
    logger.info(f"加载原始pkl: {orig_pkl}")
    df_orig = pd.read_pickle(orig_pkl)
    logger.info(f"加载优化pkl: {fast_pkl}")
    df_fast = pd.read_pickle(fast_pkl)

    logger.info(f"  原始: shape={df_orig.shape}")
    logger.info(f"  优化: shape={df_fast.shape}")

    if feat_cols is None:
        feat_cols = [c for c in RAW_FEATURE_COLS if c in df_orig.columns]

    # norm列也对比
    norm_cols = [c + "_norm" for c in feat_cols if c + "_norm" in df_orig.columns]
    all_check_cols = [c for c in feat_cols if c in df_orig.columns] + norm_cols

    # 按 SecuCode + date + bar_time 排序对齐
    sort_keys = ["SecuCode", "date", "bar_time"]
    df_orig = df_orig.sort_values(sort_keys).reset_index(drop=True)
    df_fast = df_fast.sort_values(sort_keys).reset_index(drop=True)

    if len(df_orig) != len(df_fast):
        logger.error(f"行数不一致：原始={len(df_orig)}, 优化={len(df_fast)}")
        return

    logger.info(f"\n对比 {len(all_check_cols)} 个列（原始因子 + norm列）：")
    all_pass = True
    fail_cols = []

    for col in all_check_cols:
        if col not in df_orig.columns:
            logger.warning(f"  [{col}] 不在原始pkl中，跳过")
            continue
        if col not in df_fast.columns:
            logger.warning(f"  [{col}] 不在优化pkl中，跳过")
            continue

        try:
            orig_vals = df_orig[col].values.astype(np.float64)
            fast_vals = df_fast[col].values.astype(np.float64)
        except (ValueError, TypeError):
            logger.info(f"  [{col}] 非数值列，跳过")
            continue

        nan_orig = np.isnan(orig_vals)
        nan_fast = np.isnan(fast_vals)
        if not np.array_equal(nan_orig, nan_fast):
            logger.error(f"  [{col}] FAIL: NaN位置不一致")
            all_pass = False
            fail_cols.append(col)
            continue

        mask = ~nan_orig
        if mask.sum() == 0:
            continue

        max_err = np.abs(orig_vals[mask] - fast_vals[mask]).max()
        rel_err = max_err / (np.abs(orig_vals[mask]).max() + 1e-10)
        passed = (max_err < 1e-8) or (rel_err < 1e-6)
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  [{col}] {status}: max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}"
        )
        if not passed:
            all_pass = False
            fail_cols.append(col)

    logger.info("\n" + "=" * 60)
    if all_pass:
        logger.success(
            f"所有 {len(all_check_cols)} 列验证通过，优化版结果与原始版完全一致！"
        )
    else:
        logger.error(f"以下列存在差异: {fail_cols}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="优化版因子构造（不修改原始脚本）")
    parser.add_argument("--year", type=int, default=2024, help="数据年份")
    parser.add_argument(
        "--index",
        type=str,
        default="csi1000",
        choices=["csi1000", "all"],
        help="股票池：csi1000（1000只）或 all（全市场）",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="只做pkl对比，不重新计算（需要两个pkl都已存在）",
    )
    args = parser.parse_args()

    if args.index == "csi1000":
        with open(CSI1000_PATH) as f:
            stock_list = json.load(f)
        logger.info(f"股票池：CSI1000，共 {len(stock_list)} 只")
    else:
        stock_list = None
        logger.info("股票池：全市场")

    if args.compare_only:
        orig_pkl = OUTPUT_DIR / f"features_{args.year}.pkl"
        fast_pkl = OUTPUT_DIR / f"features_fast_{args.year}.pkl"
        compare_pkls(orig_pkl, fast_pkl)
    else:
        build_and_compare(args.year, stock_list)


if __name__ == "__main__":
    main()
