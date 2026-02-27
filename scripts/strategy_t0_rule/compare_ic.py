"""
IC对比脚本：对原始pkl和优化版pkl分别计算IC，对比ICIR结果是否完全对齐。

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python compare_ic.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from ic_analysis import (
    compute_forward_returns,
    compute_ic_daily_random,
    summarize_ic,
    NORM_FEATURE_COLS,
    HOLD_PERIODS,
)

ORIG_PKL = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/features_csi1000_2024.pkl")
FAST_PKL = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/features_fast_2024.pkl")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")


def run_ic(pkl_path: Path, tag: str) -> pd.DataFrame:
    """加载pkl，计算未来收益率，跑IC，返回summary"""
    logger.info(f"\n{'='*60}")
    logger.info(f"[{tag}] 加载: {pkl_path}")
    t0 = time.time()
    df = pd.read_pickle(pkl_path)
    logger.info(f"[{tag}] shape={df.shape}，加载耗时: {time.time()-t0:.1f}s")

    df = compute_forward_returns(df)
    valid_mask = df["buy_open"].notna()
    df_valid = df[valid_mask].copy()

    feat_cols = [c for c in NORM_FEATURE_COLS if c in df_valid.columns]
    logger.info(f"[{tag}] 有效行数={len(df_valid):,}，因子列数={len(feat_cols)}")

    t1 = time.time()
    ic_df = compute_ic_daily_random(df_valid, feat_cols, seed=42)
    logger.info(f"[{tag}] IC计算耗时: {time.time()-t1:.1f}s")

    summary = summarize_ic(ic_df)
    return summary


def compare_summaries(s_orig: pd.DataFrame, s_fast: pd.DataFrame):
    """并排对比两个summary的ICIR，输出差异表"""
    logger.info(f"\n{'='*80}")
    logger.info("ICIR 对比：原始版 vs 优化版（相同结果则说明因子完全一致）")
    logger.info(f"{'='*80}")

    period_order = list(HOLD_PERIODS.keys())

    def pivot_icir(s):
        p = s.pivot(index="feature", columns="period", values="ICIR")
        return p.reindex(columns=[c for c in period_order if c in p.columns])

    def pivot_meanic(s):
        p = s.pivot(index="feature", columns="period", values="MeanIC")
        return p.reindex(columns=[c for c in period_order if c in p.columns])

    icir_orig = pivot_icir(s_orig)
    icir_fast = pivot_icir(s_fast)
    meanic_orig = pivot_meanic(s_orig)
    meanic_fast = pivot_meanic(s_fast)

    # 按原始15min ICIR绝对值排序
    if "15min" in icir_orig.columns:
        order = icir_orig["15min"].abs().sort_values(ascending=False).index
        icir_orig = icir_orig.reindex(order)
        icir_fast = icir_fast.reindex(order)
        meanic_orig = meanic_orig.reindex(order)
        meanic_fast = meanic_fast.reindex(order)

    # --- 打印 ICIR 并排对比 ---
    logger.info("\nICIR 对比（orig vs fast，差值=fast-orig）：")
    periods = [p for p in period_order if p in icir_orig.columns]
    header = f"{'因子':<25}" + "".join([f"  {p}(orig) {p}(fast)   diff" for p in periods])
    print(header)
    print("-" * len(header))

    max_diff = 0.0
    for feat in icir_orig.index:
        row = f"{feat:<25}"
        for p in periods:
            v_orig = icir_orig.loc[feat, p] if feat in icir_orig.index else np.nan
            v_fast = icir_fast.loc[feat, p] if feat in icir_fast.index else np.nan
            diff = v_fast - v_orig if not (np.isnan(v_orig) or np.isnan(v_fast)) else np.nan
            row += f"  {v_orig:7.3f}  {v_fast:7.3f}  {diff:+6.3f}"
            if not np.isnan(diff):
                max_diff = max(max_diff, abs(diff))
        print(row)

    logger.info(f"\nICIR最大差值（abs）: {max_diff:.4f}")

    # --- 打印 MeanIC 并排对比 ---
    logger.info("\nMeanIC 对比（orig vs fast，差值=fast-orig）：")
    header2 = f"{'因子':<25}" + "".join([f"  {p}(orig) {p}(fast)   diff" for p in periods])
    print(header2)
    print("-" * len(header2))

    max_diff_ic = 0.0
    for feat in meanic_orig.index:
        row = f"{feat:<25}"
        for p in periods:
            v_orig = meanic_orig.loc[feat, p] if feat in meanic_orig.index else np.nan
            v_fast = meanic_fast.loc[feat, p] if feat in meanic_fast.index else np.nan
            diff = v_fast - v_orig if not (np.isnan(v_orig) or np.isnan(v_fast)) else np.nan
            row += f"  {v_orig:7.4f}  {v_fast:7.4f}  {diff:+7.4f}"
            if not np.isnan(diff):
                max_diff_ic = max(max_diff_ic, abs(diff))
        print(row)

    logger.info(f"\nMeanIC最大差值（abs）: {max_diff_ic:.6f}")

    # --- 结论 ---
    logger.info(f"\n{'='*60}")
    if max_diff < 0.001 and max_diff_ic < 0.0001:
        logger.success("IC结果完全对齐！优化版因子与原始版等价。")
    else:
        logger.warning(f"存在差异：ICIR最大差={max_diff:.4f}，MeanIC最大差={max_diff_ic:.6f}")
        logger.info("（差异来源可能是：标准化方案不同导致的数值微差，需进一步分析）")


def main():
    logger.info("开始 IC 对比分析...")
    logger.info(f"原始pkl: {ORIG_PKL}")
    logger.info(f"优化pkl: {FAST_PKL}")

    s_orig = run_ic(ORIG_PKL, "原始版")
    s_fast = run_ic(FAST_PKL, "优化版")

    # 保存两份summary
    s_orig.to_csv(OUTPUT_DIR / "ic_summary_orig.csv", index=False)
    s_fast.to_csv(OUTPUT_DIR / "ic_summary_fast.csv", index=False)
    logger.info("两份IC summary已保存")

    compare_summaries(s_orig, s_fast)


if __name__ == "__main__":
    main()
