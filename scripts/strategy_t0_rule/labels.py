"""
标签构造模块（单一职责：只负责从 fwd_ret 生成训练标签）

支持4种标签，通过 build_label(df, label_type) 统一入口调用：

  fwdret    : 原始 fwd_ret_30min，不做任何处理
  cs_demean : 仅截面去均值，ret_ex = fwd_ret - 同时刻截面均值
  wf_rank   : 过去20日滚动rank（0~1），无前视（shift(1)），基于原始 fwd_ret
  refined   : 4步精炼标签（截面去均值→时序Z-Score→二次截面中性化→clip）
"""

import time

import numpy as np
import pandas as pd
from loguru import logger

# ==================== 标签共享参数 ====================
BARs_PER_DAY = 48
LABEL_LOOKBACK_DAYS = 20
LABEL_MIN_PERIODS = 5
WF_RANK_CLIP = 5.0

LABEL_TYPES = ["fwdret", "cs_demean", "wf_rank", "refined"]


def build_label(
    df: pd.DataFrame,
    label_type: str,
    ret_col: str = "fwd_ret_30min",
) -> pd.DataFrame:
    """
    统一标签构造入口。
    在 df 上新增 'label' 列，返回新 DataFrame（不修改原始数据）。
    label_type 取值见模块文档。
    """
    if label_type not in LABEL_TYPES:
        raise ValueError(f"label_type 须为 {LABEL_TYPES}，收到: {label_type}")
    builders = {
        "fwdret": _build_fwdret,
        "cs_demean": _build_cs_demean,
        "wf_rank": _build_wf_rank,
        "refined": _build_refined,
    }
    return builders[label_type](df, ret_col)


# ------------------------------------------------------------------ #
# 标签1：原始收益率（baseline）
# ------------------------------------------------------------------ #
def _build_fwdret(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    """直接用 fwd_ret 作为标签，不做任何处理"""
    t0 = time.time()
    df = df.copy()
    df["label"] = df[ret_col]
    logger.info(
        f"[fwdret] 标签构建完成，range=[{df['label'].min():.6f}, {df['label'].max():.6f}]，"
        f"耗时={time.time()-t0:.1f}s"
    )
    return df


# ------------------------------------------------------------------ #
# 标签2：仅截面去均值
# ------------------------------------------------------------------ #
def _build_cs_demean(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    """
    截面去均值：ret_ex = fwd_ret - 同时刻（date+bar_time）全市场均值
    剔除大盘 Beta，保留个股超额收益，无前视偏差
    """
    t0 = time.time()
    df = df.copy()
    df["label"] = df[ret_col] - df.groupby(["date", "bar_time"])[ret_col].transform(
        "mean"
    )
    n_valid = df["label"].notna().sum()
    logger.info(
        f"[cs_demean] 标签构建完成，有效={n_valid:,}，"
        f"range=[{df['label'].min():.6f}, {df['label'].max():.6f}]，"
        f"耗时={time.time()-t0:.1f}s"
    )
    return df


# ------------------------------------------------------------------ #
# 标签3：WF滚动rank（基于原始 fwd_ret）
# ------------------------------------------------------------------ #
def _build_wf_rank(
    df: pd.DataFrame,
    ret_col: str,
    window_days: int = LABEL_LOOKBACK_DAYS,
    min_days: int = LABEL_MIN_PERIODS,
) -> pd.DataFrame:
    """
    Walk-Forward 滚动百分位rank（向量化实现）：
      - 对每只股票每个 bar_time，用过去 window_days 天同时刻的 fwd_ret 排名
      - shift(1) 确保无前视偏差
      - 输出 0~1 之间的浮点数（历史分位数）
      - 实现：pivot成 (date × SecuCode*bar_time) 宽表，利用 pandas rolling.rank 向量化计算
    """
    t0 = time.time()
    df = df.copy()
    df = df.sort_values(["SecuCode", "bar_time", "date"])

    # pivot：行=date，列=(SecuCode, bar_time)，值=ret_col
    # 每列是一个"个股×时刻"序列，长度=交易日数，天然满足"同时刻同股票"的分组需求
    wide = df.pivot_table(
        index="date", columns=["SecuCode", "bar_time"], values=ret_col, aggfunc="first"
    )

    # 向量化 rolling rank：一次对所有列同时计算，利用 pandas 底层 C 实现
    ranked = (
        wide.rolling(window=window_days, min_periods=min_days).rank(pct=True).shift(1)
    )

    # 将宽表 melt 回长表，与原始 df merge
    ranked_long = ranked.stack(
        ["SecuCode", "bar_time"], future_stack=True
    ).reset_index()
    ranked_long.columns = ["date", "SecuCode", "bar_time", "label"]
    ranked_long["date"] = ranked_long["date"].astype(df["date"].dtype)

    df = df.merge(ranked_long, on=["date", "SecuCode", "bar_time"], how="left")

    n_valid = df["label"].notna().sum()
    n_total = len(df)
    logger.info(
        f"[wf_rank] 标签构建完成（window={window_days}天，min_periods={min_days}），"
        f"有效={n_valid:,}/{n_total:,}，预热期={n_total-n_valid:,}，"
        f"耗时={time.time()-t0:.1f}s"
    )
    return df


# ------------------------------------------------------------------ #
# 标签4：4步精炼标签（当前最优）
# ------------------------------------------------------------------ #
def _build_refined(
    df: pd.DataFrame,
    ret_col: str,
    window_days: int = LABEL_LOOKBACK_DAYS,
    min_days: int = LABEL_MIN_PERIODS,
    clip_bound: float = WF_RANK_CLIP,
) -> pd.DataFrame:
    """
    精炼版T0标签（4步流程）：
      Step1: 截面去均值 → ret_ex（剔除Beta）
      Step2: 个股时序滚动Z-Score（shift(1)无前视）
      Step3: 二次截面中性化（消除日内波动率漂移）
      Step4: Winsorize缩尾 clip(-5, 5)
    """
    t0 = time.time()
    df = df.copy()
    df = df.sort_values(["SecuCode", "date", "bar_time"])

    window_size = window_days * BARs_PER_DAY
    min_periods = min_days * BARs_PER_DAY

    # Step1: 截面去均值
    df["ret_ex"] = df[ret_col] - df.groupby(["date", "bar_time"])[ret_col].transform(
        "mean"
    )

    # Step2: 个股时序滚动Z-Score（shift(1)无前视）
    grp = df.groupby("SecuCode", sort=False)["ret_ex"]
    roll_mean_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).mean().shift(1)
    )
    roll_std_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).std().shift(1)
    )
    df["zscore_raw"] = (df["ret_ex"] - roll_mean_lag) / (roll_std_lag + 1e-8)

    # Step3: 二次截面中性化
    df["label"] = df["zscore_raw"] - df.groupby(["date", "bar_time"])[
        "zscore_raw"
    ].transform("mean")

    # Step4: Winsorize缩尾
    df["label"] = df["label"].clip(-clip_bound, clip_bound)

    df.drop(columns=["ret_ex", "zscore_raw"], inplace=True)

    n_valid = df["label"].notna().sum()
    n_total = len(df)
    logger.info(
        f"[refined] 标签构建完成（window={window_days}天/{window_size}bar，clip=±{clip_bound}），"
        f"有效={n_valid:,}/{n_total:,}，预热期={n_total-n_valid:,}bar，"
        f"耗时={time.time()-t0:.1f}s"
    )
    return df
