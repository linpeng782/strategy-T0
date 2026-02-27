"""
模型分析模块（单一职责：可视化 + 分层收益分析）

提供两个公共函数：
  visualize_model_logic(model, label_tag, feat_cols, lgbm_params, output_dir)
  quantile_return_analysis(result_csv, ret_col, n_bins, output_dir)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from loguru import logger
from pathlib import Path


def visualize_model_logic(
    model,
    label_tag: str,
    feat_cols: list,
    lgbm_params: dict,
    output_dir: Path,
) -> None:
    """
    可视化模型决策逻辑：
      1) 前3棵树的结构图（分裂特征、阈值、叶子节点样本数）
      2) 特征重要性柱状图（split次数 + gain两种口径）
    """
    n_trees_to_plot = min(3, model.n_estimators_)
    for tree_idx in range(n_trees_to_plot):
        try:
            fig, ax = plt.subplots(figsize=(24, 12))
            lgb.plot_tree(
                model,
                tree_index=tree_idx,
                ax=ax,
                show_info=["split_gain", "internal_count", "leaf_count"],
                precision=4,
            )
            ax.set_title(
                f"LGBM Tree #{tree_idx}  [{label_tag}]  "
                f"(num_leaves={model.num_leaves}  "
                f"feature_fraction={lgbm_params['feature_fraction']})",
                fontsize=13,
            )
            save_path = output_dir / f"lgbm_tree_{label_tag}_tree{tree_idx}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.success(f"树结构图已保存: {save_path}")
        except Exception as e:
            logger.error(f"树结构图 tree#{tree_idx} 生成失败: {e}")

    try:
        imp_split = pd.Series(
            model.feature_importances_, index=feat_cols, name="split"
        ).sort_values(ascending=False)
        imp_gain = pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=feat_cols,
            name="gain",
        ).sort_values(ascending=False)

        top_n = 20
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        imp_split.head(top_n).sort_values().plot(
            kind="barh", ax=axes[0], color="steelblue"
        )
        axes[0].set_title(f"Feature Importance - Split Count (Top{top_n})", fontsize=12)
        axes[0].set_xlabel("Split Count")

        imp_gain.head(top_n).sort_values().plot(
            kind="barh", ax=axes[1], color="darkorange"
        )
        axes[1].set_title(f"Feature Importance - Gain (Top{top_n})", fontsize=12)
        axes[1].set_xlabel("Total Gain")

        fig.suptitle(f"Feature Importance [{label_tag}]", fontsize=14)
        plt.tight_layout()
        save_path = output_dir / f"lgbm_feat_importance_{label_tag}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"特征重要性图已保存: {save_path}")
    except Exception as e:
        logger.error(f"特征重要性图生成失败: {e}")


def quantile_return_analysis(
    result_csv: Path,
    ret_col: str = "fwd_ret_30min",
    n_bins: int = 20,
    output_dir: Path = None,
) -> tuple:
    """
    按预测值分层，分析各层平均收益、胜率；输出日度spread稳定性。
    返回 (layer_stats, daily_spread)。
    """
    if output_dir is None:
        output_dir = result_csv.parent

    label_tag = result_csv.stem.replace("lgbm_result_", "")
    logger.info(f"\n{'='*60}")
    logger.info(f"分层收益分析: {label_tag}")
    logger.info(f"{'='*60}")

    pred_col = "pred"
    df = pd.read_csv(result_csv, parse_dates=["date", "bar_time"])
    if pred_col not in df.columns and "prob" in df.columns:
        pred_col = "prob"
    df = df.dropna(subset=[pred_col, ret_col])
    logger.info(f"加载完成，shape={df.shape}，预测列={pred_col}")

    # 日内分层（每天内按pred排名切分，消除跨天截面bias）
    df["quintile"] = (
        df.groupby("date")[pred_col]
        .transform(lambda x: pd.qcut(x, n_bins, labels=False, duplicates="drop"))
        .add(1)
        .astype("Int64")
    )
    df_valid_layer = df.dropna(subset=["quintile"])
    grp = df_valid_layer.groupby("quintile")[ret_col]
    layer_stats = pd.DataFrame(
        {
            "样本数": grp.count(),
            "平均收益": grp.mean(),
            "胜率(>0)": grp.apply(lambda x: (x > 0).mean()),
            "胜率(>1%)": grp.apply(lambda x: (x > 0.01).mean()),
            "中位收益": grp.median(),
            "收益std": grp.std(),
        }
    )
    layer_stats["平均收益(bps)"] = layer_stats["平均收益"] * 10000
    logger.info(
        f"\n日内分层统计（Q1=低{pred_col} ~ Q{n_bins}=高{pred_col}，每天内按pred排名切分）:"
    )
    logger.info(
        "\n"
        + layer_stats[
            ["样本数", "平均收益(bps)", "胜率(>0)", "胜率(>1%)", "收益std"]
        ].to_string()
    )

    # 多空收益差（Q_max - Q1）
    q1_ret = layer_stats.loc[1, "平均收益"]
    qn_ret = layer_stats.loc[n_bins, "平均收益"]
    spread = (qn_ret - q1_ret) * 10000
    logger.info(f"\nQ{n_bins} - Q1 收益差 = {spread:.2f} bps")

    # 日度spread稳定性
    daily_layer = (
        df_valid_layer.groupby(["date", "quintile"])[ret_col].mean().unstack("quintile")
    )
    daily_spread = daily_layer[n_bins] - daily_layer[1]
    n_days = daily_spread.dropna().__len__()
    mean_spread = daily_spread.mean() * 10000
    std_spread = daily_spread.std() * 10000
    daily_ir = mean_spread / std_spread if std_spread > 0 else np.nan
    annual_ir = daily_ir * np.sqrt(252) if not np.isnan(daily_ir) else np.nan
    t_stat = mean_spread / (std_spread / np.sqrt(n_days)) if std_spread > 0 else np.nan
    win_rate = (daily_spread > 0).mean()
    logger.info(f"\n日度多空收益差（Q{n_bins}-Q1，日内分层，{n_days}天）:")
    logger.success(f"  日均spread = {mean_spread:.2f} bps")
    logger.info(f"  spread_std = {std_spread:.2f} bps")
    logger.info(f"  日度IR     = {daily_ir:.3f}")
    logger.info(f"  年化IR     = {annual_ir:.2f}  (≈年化Sharpe)")
    logger.info(f"  t-stat     = {t_stat:.2f}  (>2视为显著)")
    logger.info(f"  胜率(>0)   = {win_rate:.1%}")

    # 月度spread分解
    df_valid_layer = df_valid_layer.copy()
    df_valid_layer["month"] = pd.to_datetime(df_valid_layer["date"]).dt.to_period("M")
    monthly_spread = (
        df_valid_layer.groupby(["month", "quintile"])[ret_col]
        .mean()
        .unstack("quintile")
        .pipe(lambda x: (x[n_bins] - x[1]) * 10000)
        .rename("spread_bps")
    )
    logger.info(f"\n月度多空spread（Q{n_bins}-Q1，bps）:")
    logger.info("\n" + monthly_spread.to_string())

    out_path = output_dir / f"layer_analysis_{label_tag}.csv"
    layer_stats.to_csv(out_path)
    logger.success(f"分层结果已保存: {out_path}")

    return layer_stats, daily_spread
