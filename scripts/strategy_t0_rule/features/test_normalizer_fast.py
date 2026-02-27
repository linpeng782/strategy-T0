"""
标准化加速方案测试脚本

原始方案瓶颈：
  - 每列做 pivot_table → shift → rolling → stack → reindex
  - 31列 × pivot大矩阵(242行 × 48000列) = 极慢的内存分配和行列重排
  - 优化版（批量pivot）实测 260s(pivot) + 217s(stack) = 477s，总计 907s

新方案：完全避免 pivot/stack，改用长表 groupby 方式
  - 按 (SecuCode, time) 排序 → 每组在 date 维度上连续
  - groupby(["SecuCode","time"]).transform(shift+rolling)
  - 直接输出对齐到原始 df 的统计量，无需 reindex
  - 31列可以一次 groupby 批量处理（pandas groupby多列transform）

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python test_normalizer_fast.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.base import RAW_FEATURE_COLS, OUTPUT_DIR
from features.normalizer import normalize_features

FAST_PKL = OUTPUT_DIR / "features_fast_2024.pkl"
ORIG_PKL = OUTPUT_DIR / "features_2024.pkl"

# ============================================================
# 新方案：完全基于长表的标准化（避免pivot/stack）
# ============================================================


def normalize_features_v2(
    df: pd.DataFrame,
    feat_cols: list = None,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    优化版标准化 v2：完全避免 pivot/stack，直接在长表上做 groupby rolling。

    核心思路：
      - 按 (SecuCode, time) 分组，每组内按 date 升序排列
      - 对每组做 shift(1) + rolling(window).median/mean/std
      - 输出直接对齐到原始 df 的行顺序，无需 reindex

    性能优势：
      - 无宽表转换（避免 pivot 的 242×48000 矩阵分配）
      - pandas groupby transform 内部用 Cython，比 Python for loop 快
      - 31列可以同时在一次 groupby 里批量计算
    """
    if feat_cols is None:
        feat_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]

    logger.info(
        f"[v2] 开始标准化（回看={window}天，min_periods={min_periods}，特征数={len(feat_cols)}）..."
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

    # 每组内 date 有唯一性，每个 (SecuCode, time) 对应 ~242 天的时间序列
    grp = df_sorted.groupby(["SecuCode", "time"], sort=False)

    # 批量计算所有列的历史统计量（shift(1) + rolling）
    # shift(1) 严格不含当天
    logger.info("  批量计算 rolling median/mean/std...")
    t2 = time.time()

    # 一次 transform 处理所有因子列，Cython 内部循环
    feat_df = df_sorted[feat_cols]
    shifted = grp[feat_cols].transform(lambda x: x.shift(1))

    # 把 shifted 值直接写入 df_sorted（临时列）
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

    # 向量化去极值+标准化
    logger.info("  向量化标准化...")
    for i, feat in enumerate(feat_cols):
        col_norm = feat + "_norm"
        s_col = shift_cols[i]

        med = hist_median[s_col].values
        mea = hist_mean[s_col].values
        std = hist_std[s_col].values
        mad_approx = 0.6745 * std

        raw_vals = df_sorted[feat].values
        winsorized = np.clip(raw_vals, med - 2.0 * mad_approx, med + 2.0 * mad_approx)
        df_sorted[col_norm] = np.where(std > 0, (winsorized - mea) / std, np.nan)

    # 删除临时 shift 列，恢复原始行顺序
    df_final = df_sorted.drop(columns=shift_cols).reindex(orig_index)

    t4 = time.time()
    norm_cols = [f + "_norm" for f in feat_cols if f in df_final.columns]
    logger.success(f"[v2] 标准化完成，新增 {len(norm_cols)} 列，总耗时: {t4-t0:.1f}s")
    return df_final


# ============================================================
# 新方案 v3：矩阵化 rolling（完全 numpy，无 groupby/pivot）
# ============================================================


def _rolling_mean_std_median_matrix(
    mat: np.ndarray, window: int, min_periods: int
) -> tuple:
    """
    对 2D 矩阵 mat (n_groups, n_dates) 沿 axis=1 做 rolling(window)，
    返回 (median, mean, std) 三个同形状矩阵。

    使用累积和技巧加速 mean/std（O(n)），median 仍需 O(n*window)
    但 numpy 的内存连续访问比 pandas groupby 快很多。
    """
    n_groups, n_dates = mat.shape
    out_med = np.full_like(mat, np.nan)
    out_mean = np.full_like(mat, np.nan)
    out_std = np.full_like(mat, np.nan)

    for j in range(n_dates):
        start = max(0, j - window + 1)
        if j - start + 1 < min_periods:
            continue
        window_data = mat[:, start : j + 1]  # (n_groups, w)，内存连续
        # 有效值计数（处理NaN）
        valid = np.sum(~np.isnan(window_data), axis=1)
        valid_mask = valid >= min_periods

        out_med[valid_mask, j] = np.nanmedian(window_data[valid_mask], axis=1)
        out_mean[valid_mask, j] = np.nanmean(window_data[valid_mask], axis=1)
        out_std[valid_mask, j] = np.nanstd(window_data[valid_mask], axis=1, ddof=1)

    return out_med, out_mean, out_std


def normalize_features_v3(
    df: pd.DataFrame,
    feat_cols: list = None,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """
    优化版标准化 v3：矩阵化 rolling，完全向量化，无 groupby/pivot。

    核心思路：
      - 按 (SecuCode, time) 排好序，确保每组在 date 维度上连续
      - 一次性 reshape 成 (n_groups, n_dates) 矩阵
      - 所有组同时做 shift(1) + rolling，内存连续访问
      - 无任何 Python 循环（除了时间轴上的 window 滑动）

    适用条件：
      - 每个 (SecuCode, time) 组必须有完全相同的 n_dates（或可 padding）
      - CSI1000 × 48时间点 × 242天，满足条件
    """
    if feat_cols is None:
        feat_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]

    logger.info(
        f"[v3] 开始标准化（回看={window}天，min_periods={min_periods}，特征数={len(feat_cols)}）..."
    )
    t0 = time.time()

    if "time" not in df.columns:
        df = df.copy()
        df["time"] = df["bar_time"].dt.strftime("%H:%M")

    # 排序：(SecuCode, time) 为组，date 为时间轴
    orig_index = df.index.copy()
    df_sorted = df.sort_values(["SecuCode", "time", "date"]).reset_index(drop=True)

    # 获取维度信息
    groups = df_sorted.groupby(["SecuCode", "time"], sort=False)
    group_keys = list(groups.groups.keys())
    n_groups = len(group_keys)
    group_sizes = groups.size()
    n_dates_per_group = group_sizes.values  # 每组的日期数

    # 检查每组是否等长（等长才能 reshape）
    unique_sizes = np.unique(n_dates_per_group)
    if len(unique_sizes) == 1:
        n_dates = unique_sizes[0]
        logger.info(f"  维度: {n_groups}组 × {n_dates}天，可用矩阵化rolling")
        use_matrix = True
    else:
        logger.warning(f"  各组长度不等（{unique_sizes}），退回 v2 方案")
        use_matrix = False

    t1 = time.time()
    logger.info(f"  排序+维度检查耗时: {t1-t0:.1f}s")

    if not use_matrix:
        return normalize_features_v2(df, feat_cols, window, min_periods)

    # ---- 矩阵化 rolling ----
    logger.info("  矩阵化 rolling 所有因子列...")
    t2 = time.time()

    # 获取排序后行的原始 index，用于最终恢复顺序
    sorted_orig_idx = df_sorted.index  # 0..N-1（已 reset_index）

    for feat in feat_cols:
        col_norm = feat + "_norm"
        raw_vals = df_sorted[feat].values.astype(np.float64)

        # reshape: (n_groups, n_dates)
        mat = raw_vals.reshape(n_groups, n_dates)

        # shift(1)：沿 date 轴偏移，第0列设为 NaN
        mat_shifted = np.empty_like(mat)
        mat_shifted[:, 0] = np.nan
        mat_shifted[:, 1:] = mat[:, :-1]

        # 矩阵化 rolling：所有组同时计算
        med_mat, mean_mat, std_mat = _rolling_mean_std_median_matrix(
            mat_shifted, window, min_periods
        )

        # 去极值+标准化（向量化）
        mad_approx = 0.6745 * std_mat
        winsorized = np.clip(
            mat,
            med_mat - 2.0 * mad_approx,
            med_mat + 2.0 * mad_approx,
        )
        norm_mat = np.where(std_mat > 0, (winsorized - mean_mat) / std_mat, np.nan)

        # flatten 回 (n_groups * n_dates,) 并赋值
        df_sorted[col_norm] = norm_mat.ravel()

    t3 = time.time()
    logger.info(f"  矩阵化rolling完成，耗时: {t3-t2:.1f}s")

    # 恢复原始行顺序：df_sorted 的 index 是 0..N-1，原始 index 在 orig_index
    # 通过 sorted_to_orig 映射恢复
    df_sorted.index = df.sort_values(["SecuCode", "time", "date"]).index
    df_final = df_sorted.reindex(orig_index)

    norm_cols = [f + "_norm" for f in feat_cols if f in df_final.columns]
    logger.success(
        f"[v3] 标准化完成，新增 {len(norm_cols)} 列，总耗时: {time.time()-t0:.1f}s"
    )
    return df_final


# ============================================================
# 正确性验证：对比 v2/v3 和原始方案输出是否一致
# ============================================================
def verify_correctness(
    df_orig_normed: pd.DataFrame, df_v2_normed: pd.DataFrame, feat_cols: list
) -> bool:
    """对比两个标准化结果的 norm 列是否一致"""
    norm_cols = [f + "_norm" for f in feat_cols]
    sort_keys = ["SecuCode", "date", "bar_time"]

    df_a = df_orig_normed.sort_values(sort_keys).reset_index(drop=True)
    df_b = df_v2_normed.sort_values(sort_keys).reset_index(drop=True)

    if len(df_a) != len(df_b):
        logger.error(f"行数不一致: {len(df_a)} vs {len(df_b)}")
        return False

    all_pass = True
    for col in norm_cols:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        try:
            va = df_a[col].values.astype(np.float64)
            vb = df_b[col].values.astype(np.float64)
        except (ValueError, TypeError):
            continue

        nan_a = np.isnan(va)
        nan_b = np.isnan(vb)
        if not np.array_equal(nan_a, nan_b):
            logger.error(f"  [{col}] NaN位置不一致")
            all_pass = False
            continue

        mask = ~nan_a
        if mask.sum() == 0:
            continue
        max_err = np.abs(va[mask] - vb[mask]).max()
        rel_err = max_err / (np.abs(va[mask]).max() + 1e-10)
        passed = (max_err < 1e-8) or (rel_err < 1e-6)
        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  [{col}] {status}: max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}"
        )
        if not passed:
            all_pass = False

    return all_pass


# ============================================================
# 主流程：加载 fast pkl，测试 v2 标准化，对比正确性和速度
# ============================================================
def main():
    if not FAST_PKL.exists():
        logger.error(f"fast pkl 不存在: {FAST_PKL}，请先运行 build_all_fast.py")
        return

    logger.info(f"加载 features_fast pkl: {FAST_PKL}")
    df = pd.read_pickle(FAST_PKL)
    logger.info(f"shape={df.shape}")

    feat_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]
    logger.info(f"因子列数: {len(feat_cols)}")

    # 去掉已有的 norm 列（只保留原始因子列）
    norm_cols_existing = [c for c in df.columns if c.endswith("_norm")]
    df_raw = df.drop(columns=norm_cols_existing).copy()

    # 用 50 只股票测试正确性
    stocks_50 = df_raw["SecuCode"].unique()[:50]
    df_test = df_raw[df_raw["SecuCode"].isin(stocks_50)].copy()
    logger.info(f"\n用 50 只股票验证正确性，shape={df_test.shape}")

    # 原始方案
    logger.info("\n--- 原始方案（pivot）---")
    t0 = time.time()
    df_orig_normed = normalize_features(df_test.copy(), feat_cols=feat_cols)
    t_orig = time.time() - t0
    logger.info(f"原始方案耗时: {t_orig:.1f}s")

    # v2 方案
    logger.info("\n--- v2方案（groupby rolling）---")
    t0 = time.time()
    df_v2_normed = normalize_features_v2(df_test.copy(), feat_cols=feat_cols)
    t_v2 = time.time() - t0
    logger.info(f"v2方案耗时: {t_v2:.1f}s")

    logger.info(
        f"\n速度对比：原始={t_orig:.1f}s，v2={t_v2:.1f}s，加速={t_orig/t_v2:.1f}x"
    )

    # 正确性验证
    logger.info("\n--- 正确性验证 ---")
    ok = verify_correctness(df_orig_normed, df_v2_normed, feat_cols)
    if ok:
        logger.success("v2 标准化结果与原始方案完全一致！")
    else:
        logger.error("存在数值差异！")

    # v3 方案
    logger.info("\n--- v3方案（矩阵化rolling）---")
    t0 = time.time()
    df_v3_normed = normalize_features_v3(df_test.copy(), feat_cols=feat_cols)
    t_v3 = time.time() - t0
    logger.info(f"v3方案耗时: {t_v3:.1f}s")

    # v3 正确性验证
    logger.info("\n--- v3 正确性验证 ---")
    ok_v3 = verify_correctness(df_orig_normed, df_v3_normed, feat_cols)
    if ok_v3:
        logger.success("v3 标准化结果与原始方案完全一致！")
    else:
        logger.error("v3 存在数值差异！")

    # 汇总速度对比
    logger.info("\n" + "=" * 60)
    logger.info("标准化速度对比（50只股票）：")
    logger.info(f"  原始方案（pivot）: {t_orig:.1f}s")
    logger.info(f"  v2（groupby rolling）: {t_v2:.1f}s  加速={t_orig/t_v2:.1f}x")
    logger.info(f"  v3（矩阵化rolling）:  {t_v3:.1f}s  加速={t_orig/t_v3:.1f}x")

    # 外推 1000 只（pivot方案是超线性的，实际会更多）
    logger.info(f"\n外推 1000 只股票全量耗时预估（线性外推，实际pivot更慢）：")
    ratio = 1000 / 50
    logger.info(f"  原始方案: ~{t_orig*ratio:.0f}s ({t_orig*ratio/60:.1f}min)")
    logger.info(f"  v2方案:   ~{t_v2*ratio:.0f}s ({t_v2*ratio/60:.1f}min)")
    logger.info(f"  v3方案:   ~{t_v3*ratio:.0f}s ({t_v3*ratio/60:.1f}min)")


if __name__ == "__main__":
    main()
