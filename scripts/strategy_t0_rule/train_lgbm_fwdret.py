"""
LGBM 回归模型训练脚本（传统 forward_ret 标签版）

与 train_lgbm.py（Walk-Forward标签版）的核心区别：
  - 标签：直接用 fwd_ret_30min 的日内截面rank（组内rank by SecuCode+date）
    - 含义：这个bar在当天所有时刻中属于哪个分位（截面问题）
    - 无前视偏差：rank是事后计算，用于监督学习的目标变量，不属于"特征"中的前视
  - 无需Walk-Forward预热期，所有训练样本均可用
  - 输出文件：lgbm_reg_fwdret.pkl / lgbm_result_reg_fwdret.csv
  - 配套回测：t0_backtest.py 中使用截面rank信号（每天每时刻全市场pred排名），
    而非WF滚动阈值（两种框架的信号消费方式完全不同）
"""

import sys
import time
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

# ==================== 路径配置 ====================
BASE_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research")
FEAT_PKLS = [
    BASE_DIR / "output/features_csi1000_2024.pkl",
    BASE_DIR / "output/features_csi1000_2025.pkl",
]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 特征列（与 train_lgbm.py 完全一致） ====================
FEAT_COLS = [
    # classic组
    "dif_norm",
    "dea_norm",
    "macd_norm",
    "macd_area_norm",
    "dif_slope3_norm",
    "dif_std_norm",
    "dea_std_norm",
    "macd_std_norm",
    "kdj_k_norm",
    "kdj_d_norm",
    "kdj_j_norm",
    "kdj_kd_norm",
    "boll_pct_b_norm",
    "boll_bw_norm",
    "rsv10_norm",
    "rsv20_norm",
    # price_volume组
    "vol_ratio_norm",
    "buy_pressure_norm",
    "amplitude_norm",
    "ret_1bar_norm",
    "ret_3bar_norm",
    "ret_5bar_norm",
    "range_pos_norm",
    # alpha158组
    "vwap_bias_norm",
    "x2_norm",
    "kmid_norm",
    "kup2_norm",
    "klow2_norm",
    "cord20_norm",
    "wvma10_norm",
    "vsumd10_norm",
]

# ==================== 训练参数 ====================
HOLD_PERIOD = "30min"  # 持有期
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")  # 2024年训练，2025年全年测试

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 10,
    "min_child_samples": 1000,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,
    "n_jobs": 64,
    "verbose": -1,
    "random_state": 42,
}


def load_features(paths: list) -> pd.DataFrame:
    """加载多个特征文件并合并"""
    parts = []
    for path in paths:
        if not Path(path).exists():
            logger.warning(f"特征文件不存在，跳过: {path}")
            continue
        logger.info(f"加载特征文件: {path}")
        t0 = time.time()
        df = pd.read_pickle(path)
        logger.info(f"加载完成，shape={df.shape}，耗时={time.time()-t0:.1f}s")
        parts.append(df)
    result = pd.concat(parts, ignore_index=True)
    logger.info(f"合并完成，总shape={result.shape}")
    return result


def split_by_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp):
    """按截止日期时序切分"""
    cutoff_date = cutoff.date()
    train_mask = df["date"] < cutoff_date
    test_mask = df["date"] >= cutoff_date
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    train_dates = sorted(df_train["date"].unique())
    test_dates = sorted(df_test["date"].unique())
    logger.info(
        f"训练集日期: {train_dates[0]} ~ {train_dates[-1]}（{len(train_dates)}天）"
    )
    logger.info(
        f"测试集日期: {test_dates[0]} ~ {test_dates[-1]}（{len(test_dates)}天）"
    )
    return df_train, df_test


def daily_nonoverlap_sample(
    df: pd.DataFrame, hold_period: str, seed: int = 42
) -> pd.DataFrame:
    """
    每日随机非重叠采样（向量化实现）：与 ic_analysis.py 保持一致。
    按持有期计算采样间隔（bars），每只股票每天随机起始后等间隔取样，
    保证相邻样本的未来收益窗口不重叠，避免标签自相关。
    """
    period_to_bars = {"15min": 3, "30min": 6, "45min": 9, "60min": 12}
    step = period_to_bars.get(hold_period, 6)
    rng = np.random.default_rng(seed)

    grp_keys = ["SecuCode", "date"]
    df = df.copy()
    df["_row_in_grp"] = df.groupby(grp_keys).cumcount()

    groups = df[grp_keys].drop_duplicates().reset_index(drop=True)
    groups["_offset"] = rng.integers(0, step, size=len(groups))
    df = df.merge(groups, on=grp_keys, how="left")

    mask = ((df["_row_in_grp"] - df["_offset"]) % step == 0) & (
        df["_row_in_grp"] >= df["_offset"]
    )
    result = df[mask].drop(columns=["_row_in_grp", "_offset"])
    logger.info(
        f"非重叠采样完成（step={step}bar），原始={len(df):,} -> 采样后={len(result):,}"
    )
    return result


def train_and_eval(df: pd.DataFrame):
    """
    传统 forward_ret 标签训练：
      - target = fwd_ret
      - 无WF预热期，训练样本更多
      - IC评估与 WF 版完全相同口径，方便直接对比
    """
    ret_col = f"fwd_ret_{HOLD_PERIOD}"

    # 过滤：特征和收益率均有效
    valid_mask = df[FEAT_COLS].notna().all(axis=1) & df[ret_col].notna()
    df_valid = df[valid_mask].copy()
    logger.info(f"有效样本数: {len(df_valid):,}")

    # 直接用原始 fwd_ret 作为训练标签
    df_valid["target"] = df_valid[ret_col]
    logger.info(
        f"原始收益率标签，target范围: [{df_valid['target'].min():.6f}, {df_valid['target'].max():.6f}]"
    )

    # 按截止日期切分
    df_train, df_test = split_by_cutoff(df_valid, TRAIN_CUTOFF)

    # 训练集非重叠降采样
    logger.info("对训练集做每日随机非重叠采样...")
    df_train_sampled = daily_nonoverlap_sample(df_train, HOLD_PERIOD, seed=42)
    logger.info(f"采样后训练集: {len(df_train_sampled):,} 行")

    X_train = df_train_sampled[FEAT_COLS].values
    y_train = df_train_sampled["target"].values

    # 测试集非重叠采样（IC评估用）
    logger.info("对测试集做每日随机非重叠采样...")
    df_test_sampled = daily_nonoverlap_sample(df_test, HOLD_PERIOD, seed=99)
    logger.info(f"采样后测试集（IC评估）: {len(df_test_sampled):,} 行")
    X_test_sampled = df_test_sampled[FEAT_COLS].values

    # 全量测试集用于最终分层收益分析
    X_test_all = df_test[FEAT_COLS].values

    # 训练
    logger.info(f"开始训练 LGBM 回归（固定 {LGBM_PARAMS['n_estimators']} 轮）...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        callbacks=[lgb.log_evaluation(100)],
    )
    logger.info(f"训练完成，耗时={time.time()-t0:.1f}s")

    # 预测
    pred_train = model.predict(X_train)
    pred_sampled = model.predict(X_test_sampled)
    pred_all = model.predict(X_test_all)

    # 时序IC计算（与 WF 版完全相同口径：按 SecuCode+date 分组的 Spearman）
    df_train_sampled = df_train_sampled.copy()
    df_train_sampled["pred"] = pred_train
    df_test_sampled = df_test_sampled.copy()
    df_test_sampled["pred"] = pred_sampled

    def _ts_ic_fast(df_sub: pd.DataFrame) -> pd.Series:
        """全向量化时序IC：组内rank后Pearson = Spearman"""
        grp_keys = ["SecuCode", "date"]
        tmp = df_sub[["SecuCode", "date"]].copy()
        tmp["pr"] = df_sub.groupby(grp_keys)["pred"].rank(pct=True)
        tmp["rr"] = df_sub.groupby(grp_keys)[ret_col].rank(pct=True)

        grp = tmp.groupby(grp_keys)
        tmp["pr_c"] = tmp["pr"] - grp["pr"].transform("mean")
        tmp["rr_c"] = tmp["rr"] - grp["rr"].transform("mean")

        tmp["dxdy"] = tmp["pr_c"] * tmp["rr_c"]
        tmp["dx2"] = tmp["pr_c"] ** 2
        tmp["dy2"] = tmp["rr_c"] ** 2
        sum_dxdy = grp["dxdy"].transform("sum")
        sum_dx2 = grp["dx2"].transform("sum")
        sum_dy2 = grp["dy2"].transform("sum")
        denom = np.sqrt(sum_dx2 * sum_dy2)
        tmp["ic"] = np.where(denom > 1e-12, sum_dxdy / denom, np.nan)

        ic_per_stock = tmp.groupby(grp_keys)["ic"].first().dropna().reset_index()
        return ic_per_stock.groupby("date")["ic"].mean().rename("ic")

    t_ic0 = time.time()
    ts_ic_train = _ts_ic_fast(df_train_sampled)
    ts_daily_ic = _ts_ic_fast(df_test_sampled)
    logger.info(f"时序IC计算耗时={time.time()-t_ic0:.1f}s")

    ts_ic_train_mean = ts_ic_train.mean()
    ts_mean_ic = ts_daily_ic.mean()
    ts_std_ic = ts_daily_ic.std()
    ts_icir = ts_mean_ic / ts_std_ic if ts_std_ic > 0 else np.nan
    ts_ic_pos = (ts_daily_ic > 0).mean()

    logger.success(
        f"[时序IC] 训练集日均IC={ts_ic_train_mean:.4f}  测试集日均IC={ts_mean_ic:.4f}"
    )
    logger.success(
        f"[时序IC] 测试集 ICIR={ts_icir:.3f}  IC_std={ts_std_ic:.4f}  IC>0={ts_ic_pos:.1%}"
    )

    # 特征重要性
    importance = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(
        ascending=False
    )
    logger.info("特征重要性 Top15:")
    for feat, imp in importance.head(15).items():
        logger.info(f"    {feat:<25} {imp:>6}")

    # 保存模型和结果
    model_path = OUTPUT_DIR / "lgbm_reg_fwdret.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"模型已保存: {model_path}")

    result_df = df_test[["SecuCode", "date", "bar_time", ret_col]].copy()
    result_df["pred"] = pred_all
    result_path = OUTPUT_DIR / "lgbm_result_reg_fwdret.csv"
    result_df.to_csv(result_path, index=False)
    logger.success(f"结果已保存: {result_path}")

    ic_gap = ts_ic_train_mean - ts_mean_ic
    if ic_gap > 0.1:
        logger.warning(f"过拟合预警：IC Gap={ic_gap:.4f} > 0.1")
    else:
        logger.success(f"IC Gap={ic_gap:.4f}，训练测试一致性良好")

    return {
        "n_train": len(y_train),
        "n_test_sampled": len(X_test_sampled),
        "n_test_all": len(df_test),
        "ts_ic_train": ts_ic_train_mean,
        "ts_daily_mean_ic": ts_mean_ic,
        "ts_icir": ts_icir,
        "ic_gap": ic_gap,
        "model": model,
    }


def visualize_model_logic(model, label_tag: str = "reg_fwdret"):
    """
    可视化模型决策逻辑：
      1) 前3棵树的结构图（分裂特征、阈值、叶子节点样本数）
      2) 特征重要性柱状图（split次数 + gain两种口径）
    """
    # ---- 图1/2/3：前3棵树结构 ----
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
                f"feature_fraction={LGBM_PARAMS['feature_fraction']})",
                fontsize=13,
            )
            save_path = OUTPUT_DIR / f"lgbm_tree_{label_tag}_tree{tree_idx}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.success(f"树结构图已保存: {save_path}")
        except Exception as e:
            logger.error(f"树结构图 tree#{tree_idx} 生成失败: {e}")

    # ---- 图4：特征重要性（split + gain双口径对比）----
    try:
        imp_split = pd.Series(
            model.feature_importances_, index=FEAT_COLS, name="split"
        ).sort_values(ascending=False)
        imp_gain = pd.Series(
            model.booster_.feature_importance(importance_type="gain"),
            index=FEAT_COLS,
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
        save_path = OUTPUT_DIR / f"lgbm_feat_importance_{label_tag}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"特征重要性图已保存: {save_path}")
    except Exception as e:
        logger.error(f"特征重要性图生成失败: {e}")


def quantile_return_analysis(
    result_csv: Path, ret_col: str = f"fwd_ret_{HOLD_PERIOD}", n_bins: int = 20
):
    """按预测值分层，分析各层平均收益、胜率；输出日度spread稳定性"""
    label_tag = result_csv.stem.replace("lgbm_result_", "")
    logger.info(f"\n{'='*60}")
    logger.info(f"分层收益分析: {label_tag}")
    logger.info(f"{'='*60}")

    pred_col = "pred"
    df = pd.read_csv(result_csv, parse_dates=["date", "bar_time"])
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
            "收益std": grp.std(),
        }
    )
    layer_stats["平均收益(bps)"] = layer_stats["平均收益"] * 10000
    logger.info(
        f"\n日内分层统计（Q1=低pred ~ Q{n_bins}=高pred，每天内按pred排名切分）:"
    )
    logger.info(
        "\n"
        + layer_stats[
            ["样本数", "平均收益(bps)", "胜率(>0)", "胜率(>1%)", "收益std"]
        ].to_string()
    )

    # 多空收益差
    q1_ret = layer_stats.loc[1, "平均收益"]
    q5_ret = layer_stats.loc[n_bins, "平均收益"]
    spread = (q5_ret - q1_ret) * 10000
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

    out_path = result_csv.parent / f"layer_analysis_{label_tag}.csv"
    layer_stats.to_csv(out_path)
    logger.success(f"分层结果已保存: {out_path}")

    return layer_stats, daily_spread


def main():
    from ic_analysis import compute_forward_returns

    df = load_features(FEAT_PKLS)

    logger.info("计算未来收益率...")
    df = compute_forward_returns(df)

    missing = [c for c in FEAT_COLS if c not in df.columns]
    if missing:
        logger.error(f"缺少特征列: {missing}")
        return

    logger.info(f"\n{'='*60}")
    logger.info("LGBM 传统forward_ret标签版：预测 fwd_ret_30min 无rank")
    logger.info(f"{'='*60}")
    logger.success(
        f"[模型参数] n_estimators={LGBM_PARAMS['n_estimators']}  "
        f"learning_rate={LGBM_PARAMS['learning_rate']}  "
        f"num_leaves={LGBM_PARAMS['num_leaves']}  "
        f"min_child_samples={LGBM_PARAMS['min_child_samples']}  "
        f"feature_fraction={LGBM_PARAMS['feature_fraction']}"
    )
    res = train_and_eval(df)

    logger.info(f"\n{'='*60}")
    logger.info("训练结果汇总")
    logger.info(f"{'='*60}")
    for k, v in res.items():
        if k != "model":
            logger.info(f"  {k:<20} = {v}")

    # 可视化模型决策逻辑
    visualize_model_logic(res["model"], label_tag="reg_fwdret")


if __name__ == "__main__":
    main()
    # 分层收益分析
    csv_path = OUTPUT_DIR / "lgbm_result_reg_fwdret.csv"
    if csv_path.exists():
        quantile_return_analysis(csv_path, ret_col=f"fwd_ret_{HOLD_PERIOD}")
    else:
        logger.warning(f"结果文件不存在，跳过: {csv_path}")
