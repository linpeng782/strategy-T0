"""
LGBM 回归模型训练脚本（T0策略）

输入：features_csi1000_2024.pkl（含31个_norm特征 + 未来收益率）
标签：fwd_ret_30min（直接回归收益率，涨0.5%和2%的区别被充分利用）
切分：按日期时序切分，前70%训练，后30%测试（无shuffle）
非重叠采样：每日按持有期间隔降采样，避免标签自相关导致过拟合
输出：
  - 模型文件：output/lgbm_reg.pkl
  - 结果文件：output/lgbm_result_reg.csv（每bar的预测收益 + 实际收益）
  - 汇总打印：IC(Spearman) / ICIR / 分层收益
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
# 实验配置：2024年训练，2025年全年测试（加载两年数据，TRAIN_CUTOFF=2025-01-01）
# 恢复原来实验：将下面两行注释掉，取消注释上面FEAT_PKLS和TRAIN_CUTOFF=2025-07-01
# FEAT_PKLS = [BASE_DIR / "output/features_csi1000_2024.pkl", BASE_DIR / "output/features_csi1000_2025.pkl"]
# TRAIN_CUTOFF = pd.Timestamp("2025-07-01")
FEAT_PKLS = [
    BASE_DIR / "output/features_csi1000_2024.pkl",
    BASE_DIR / "output/features_csi1000_2025.pkl",
]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 特征列（保留全部31个） ====================
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
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")  # 仅2024年训练，2025全年测试

# 精炼标签参数
LABEL_LOOKBACK_DAYS = 20  # 时序Z-Score滚动窗口（交易日）
LABEL_MIN_PERIODS = 5  # 最少所需天数
BARs_PER_DAY = 48  # 每天5分钟bar数

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "num_leaves": 10,  # 31→12：降低树复杂度，减少过拟合
    "min_child_samples": 1000,  # 200→1000：提高叶节点最小样本，增强泛化
    "feature_fraction": 0.4,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 5.0,  # 1.0→5.0：加强L2正则
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
    """按截止日期时序切分，cutoff之前为训练集，之后为测试集"""
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


def build_refined_labels(
    df: pd.DataFrame,
    ret_col: str = "fwd_ret_30min",
    window_days: int = LABEL_LOOKBACK_DAYS,
    min_days: int = LABEL_MIN_PERIODS,
    clip_bound: float = 5.0,
) -> pd.DataFrame:
    """
    精炼版T0标签构造（4步流程，解决诊断发现的两个核心问题）：

    Step1: 截面去均值 (ret_ex)
      - 剔除大盘共同成分（Beta），IC提升约6倍
      - ret_ex = fwd_ret - 同时刻截面均值

    Step2: 个股时序滚动Z-Score（无前视偏差）
      - 用历史window_days*48根bar的均值和标准差标准化ret_ex
      - shift(1)确保当前bar只使用历史数据

    Step3: 二次截面中性化（消除日内漂移）
      - 诊断发现target_zscore在下午均值漂移至0.025
      - 原因：下午波动率收缩→分母缩小→Z值虚高
      - 再次扣除同时刻截面均值，强制各时刻期望值=0

    Step4: Winsorize缩尾 clip(-clip_bound, clip_bound)
      - 诊断发现Kurt≈45，极端厚尾会严重干扰RMSE优化方向
      - clip(-5,5)去除0.1%极端样本，保护损失函数
    """
    t0 = time.time()
    df = df.copy()
    df = df.sort_values(["SecuCode", "date", "bar_time"])

    window_size = window_days * BARs_PER_DAY
    min_periods = min_days * BARs_PER_DAY

    # --- Step1: 截面去均值 ---
    df["ret_ex"] = df[ret_col] - df.groupby(["date", "bar_time"])[ret_col].transform(
        "mean"
    )

    # --- Step2: 个股时序滚动Z-Score（shift(1)无前视）---
    grp = df.groupby("SecuCode", sort=False)["ret_ex"]
    roll_mean_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).mean().shift(1)
    )
    roll_std_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).std().shift(1)
    )
    df["zscore_raw"] = (df["ret_ex"] - roll_mean_lag) / (roll_std_lag + 1e-8)

    # --- Step3: 二次截面中性化（消除日内漂移）---
    df["label"] = df["zscore_raw"] - df.groupby(["date", "bar_time"])[
        "zscore_raw"
    ].transform("mean")

    # --- Step4: Winsorize缩尾 ---
    df["label"] = df["label"].clip(-clip_bound, clip_bound)

    df.drop(columns=["ret_ex", "zscore_raw"], inplace=True)

    n_valid = df["label"].notna().sum()
    n_total = len(df)
    logger.info(
        f"精炼标签构建完成（window={window_days}天/{window_size}bar，clip=±{clip_bound}），"
        f"有效={n_valid:,}/{n_total:,}，预热期={n_total-n_valid:,}bar，"
        f"耗时={time.time()-t0:.1f}s"
    )
    return df


def daily_nonoverlap_sample(
    df: pd.DataFrame, hold_period: str, seed: int = 42
) -> pd.DataFrame:
    """
    每日随机非重叠采样（向量化实现）：与ic_analysis.py保持一致。
    按持有期计算采样间隔（bars），每只股票每天随机起始后等间隔取样，
    保证相邻样本的未来收益窗口不重叠，避免标签自相关导致过拟合。
    """
    period_to_bars = {"15min": 3, "30min": 6, "45min": 9, "60min": 12}
    step = period_to_bars.get(hold_period, 6)
    rng = np.random.default_rng(seed)

    # 组内行号（每只股票每天从0开始计数）
    grp_keys = ["SecuCode", "date"]
    df = df.copy()
    df["_row_in_grp"] = df.groupby(grp_keys).cumcount()

    # 每组随机起始偏移（向量化：先得到唯一组 -> 随机offset -> merge回来）
    groups = df[grp_keys].drop_duplicates().reset_index(drop=True)
    groups["_offset"] = rng.integers(0, step, size=len(groups))
    df = df.merge(groups, on=grp_keys, how="left")

    # 保留满足 (row_in_grp - offset) % step == 0 且 row_in_grp >= offset 的行
    mask = ((df["_row_in_grp"] - df["_offset"]) % step == 0) & (
        df["_row_in_grp"] >= df["_offset"]
    )
    result = df[mask].drop(columns=["_row_in_grp", "_offset"])
    logger.info(
        f"非重叠采样完成（step={step}bar），原始={len(df):,} -> 采样后={len(result):,}"
    )
    return result


def train_and_eval(df: pd.DataFrame):
    """LGBM回归模式：Walk-Forward标签训练，与回测阈值逻辑严格对齐"""
    ret_col = f"fwd_ret_{HOLD_PERIOD}"

    # 过滤：特征和标签均有效
    valid_mask = df[FEAT_COLS].notna().all(axis=1) & df[ret_col].notna()
    df_valid = df[valid_mask].copy()
    logger.info(f"有效样本数: {len(df_valid):,}")

    # 精炼标签：截面去均值→时序Z-Score→二次截面中性化→clip(-5,5)
    logger.info("构建精炼标签（4步流程：截面去均值→Z-Score→二次中性化→缩尾）...")
    df_valid = build_refined_labels(df_valid, ret_col=ret_col)
    target_col = "label"

    # 按截止日期切分
    df_train, df_test = split_by_cutoff(df_valid, TRAIN_CUTOFF)

    # ---- 训练集非重叠降采样 ----
    logger.info("对训练集做每日随机非重叠采样...")
    df_train_sampled = daily_nonoverlap_sample(df_train, HOLD_PERIOD, seed=42)
    # 过滤掉预热期无label的样本
    df_train_sampled = df_train_sampled[df_train_sampled[target_col].notna()].copy()
    logger.info(f"采样后训练集（过滤预热期后）: {len(df_train_sampled):,} 行")

    X_train = df_train_sampled[FEAT_COLS].values
    y_train = df_train_sampled[target_col].values  # Walk-Forward历史分位数label

    # ---- 测试集也做非重叠采样（IC评估用，避免相邻bar自相关） ----
    logger.info("对测试集做每日随机非重叠采样...")
    df_test_sampled = daily_nonoverlap_sample(df_test, HOLD_PERIOD, seed=99)
    logger.info(f"采样后测试集（IC评估）: {len(df_test_sampled):,} 行")
    X_test_sampled = df_test_sampled[FEAT_COLS].values
    y_test_sampled = df_test_sampled[ret_col].values

    # 全量测试集用于最终分层收益分析
    X_test_all = df_test[FEAT_COLS].values

    # 训练（收益率噪声极大，RMSE无法有效early stop，固定轮数训练）
    logger.info(f"开始训练 LGBM 回归（固定 {LGBM_PARAMS['n_estimators']} 轮）...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        callbacks=[lgb.log_evaluation(100)],
    )
    logger.info(f"训练完成，耗时={time.time()-t0:.1f}s")

    # 预测（采样集用于IC，全量用于分层保存）
    pred_train = model.predict(X_train)
    pred_sampled = model.predict(X_test_sampled)
    pred_all = model.predict(X_test_all)

    # ---- 时序IC：同一股票同一天内，pred高的时刻fwd_ret是否也高（按SecuCode+date分组）----
    # 这是T+0择时策略的核心指标，训练集和测试集同口径对比才能判断时序方向是否过拟合
    # 向量化实现：先组内rank，再用corrcoef（rank后的Pearson = Spearman），避免逐组apply
    df_train_sampled["pred"] = pred_train
    df_test_sampled = df_test_sampled.copy()
    df_test_sampled["pred"] = pred_sampled

    def _ts_ic_fast(df: pd.DataFrame) -> pd.Series:
        """
        全向量化时序IC：组内rank后，用transform(sum)计算每个(SecuCode,date)组的
        Pearson相关系数，再按date取均值，无任何Python层apply回调。
        rank后Pearson = Spearman。
        """
        grp_keys = ["SecuCode", "date"]
        tmp = df[["SecuCode", "date"]].copy()
        tmp["pr"] = df.groupby(grp_keys)["pred"].rank(pct=True)
        tmp["rr"] = df.groupby(grp_keys)[ret_col].rank(pct=True)

        # 组内去均值
        grp = tmp.groupby(grp_keys)
        tmp["pr_c"] = tmp["pr"] - grp["pr"].transform("mean")
        tmp["rr_c"] = tmp["rr"] - grp["rr"].transform("mean")

        # Pearson = sum(dx*dy) / sqrt(sum(dx²)*sum(dy²))  全向量化
        tmp["dxdy"] = tmp["pr_c"] * tmp["rr_c"]
        tmp["dx2"] = tmp["pr_c"] ** 2
        tmp["dy2"] = tmp["rr_c"] ** 2
        sum_dxdy = grp["dxdy"].transform("sum")
        sum_dx2 = grp["dx2"].transform("sum")
        sum_dy2 = grp["dy2"].transform("sum")
        denom = np.sqrt(sum_dx2 * sum_dy2)
        tmp["ic"] = np.where(denom > 1e-12, sum_dxdy / denom, np.nan)

        # 每组取第一行（组内ic值相同），再按date取均值
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

    # ---- 评估3：特征重要性 Top15 ----
    importance = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(
        ascending=False
    )
    logger.info("特征重要性 Top15:")
    for feat, imp in importance.head(15).items():
        logger.info(f"    {feat:<25} {imp:>6}")

    # ---- 保存 ----
    model_path = OUTPUT_DIR / "lgbm_reg_refined_label.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"模型已保存: {model_path}")

    result_df = df_test[["SecuCode", "date", "bar_time", ret_col]].copy()
    result_df["pred"] = pred_all
    result_path = OUTPUT_DIR / "lgbm_result_refined_label.csv"
    result_df.to_csv(result_path, index=False)
    logger.success(f"结果已保存: {result_path}")

    ic_gap = ts_ic_train_mean - ts_mean_ic
    if ic_gap > 0.1:
        logger.warning(f"过拟合预警：IC Gap={ic_gap:.4f} > 0.1，训练集IC明显高于测试集")
    else:
        logger.success(f"IC Gap={ic_gap:.4f}，训练测试一致性良好")

    return {
        "n_train": len(y_train),
        "n_test_sampled": len(y_test_sampled),
        "n_test_all": len(df_test),
        "ts_ic_train": ts_ic_train_mean,
        "ts_daily_mean_ic": ts_mean_ic,
        "ts_icir": ts_icir,
        "ic_gap": ic_gap,
        "model": model,
    }


def main():
    from ic_analysis import compute_forward_returns

    df = load_features(FEAT_PKLS)

    # 补充未来收益率（特征pkl中不含该列）
    logger.info("计算未来收益率...")
    df = compute_forward_returns(df)

    # 检查特征列是否齐全
    missing = [c for c in FEAT_COLS if c not in df.columns]
    if missing:
        logger.error(f"缺少特征列: {missing}")
        return

    logger.info(f"\n{'='*60}")
    logger.info("LGBM 回归模式：预测 fwd_ret_30min")
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
    visualize_model_logic(res["model"], label_tag="refined_label")


def visualize_model_logic(model, label_tag: str = "refined_label"):
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
    result_csv: Path, ret_col: str = "fwd_ret_30min", n_bins: int = 20
):
    """
    按预测概率分层，分析各层平均收益、胜率、样本数。
    同时输出日度分层（按天 × 层 计算平均收益）验证稳定性。
    """
    label_tag = result_csv.stem.replace("lgbm_result_", "")
    logger.info(f"\n{'='*60}")
    logger.info(f"分层收益分析: {label_tag}")
    logger.info(f"{'='*60}")

    # 自动识别预测列名（回归用pred，分类用prob）
    pred_col = "pred"
    df = pd.read_csv(result_csv, parse_dates=["date", "bar_time"])
    if pred_col not in df.columns and "prob" in df.columns:
        pred_col = "prob"
    df = df.dropna(subset=[pred_col, ret_col])
    logger.info(f"加载完成，shape={df.shape}，预测列={pred_col}")

    # ---- 日内分层（每天内按pred分位切分，消除跨天截面bias）----
    # pred是组内rank(0~1)，全局切分会把同一tick的相邻bar集中到同一层
    # 改为每天内分层：每天所有股票按pred排名后切分到Q1~Q5
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

    # ---- 多空收益差（Q5 - Q1）----
    q1_ret = layer_stats.loc[1, "平均收益"]
    q5_ret = layer_stats.loc[n_bins, "平均收益"]
    spread = (q5_ret - q1_ret) * 10000
    logger.info(f"\nQ{n_bins} - Q1 收益差 = {spread:.2f} bps")

    # ---- 日度分层稳定性 ----
    daily_layer = (
        df_valid_layer.groupby(["date", "quintile"])[ret_col].mean().unstack("quintile")
    )
    daily_spread = daily_layer[n_bins] - daily_layer[1]
    n_days = daily_spread.dropna().__len__()
    mean_spread = daily_spread.mean() * 10000
    std_spread = daily_spread.std() * 10000
    daily_ir = mean_spread / std_spread if std_spread > 0 else np.nan
    annual_ir = daily_ir * np.sqrt(252) if daily_ir is not np.nan else np.nan
    t_stat = mean_spread / (std_spread / np.sqrt(n_days)) if std_spread > 0 else np.nan
    win_rate = (daily_spread > 0).mean()
    logger.info(f"\n日度多空收益差（Q{n_bins}-Q1，日内分层，{n_days}天）:")
    logger.success(f"  日均spread = {mean_spread:.2f} bps")
    logger.info(f"  spread_std = {std_spread:.2f} bps")
    logger.info(f"  日度IR     = {daily_ir:.3f}")
    logger.info(f"  年化IR     = {annual_ir:.2f}  (≈年化Sharpe)")
    logger.info(f"  t-stat     = {t_stat:.2f}  (>2视为显著)")
    logger.info(f"  胜率(>0)   = {win_rate:.1%}")

    # ---- 月度spread分解（验证稳定性）----
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

    # 保存分层结果
    out_path = result_csv.parent / f"layer_analysis_{label_tag}.csv"
    layer_stats.to_csv(out_path)
    logger.success(f"分层结果已保存: {out_path}")

    return layer_stats, daily_spread


if __name__ == "__main__":
    main()
    # ---- 精炼标签版分层收益分析 ----
    csv_path = OUTPUT_DIR / "lgbm_result_refined_label.csv"
    if csv_path.exists():
        quantile_return_analysis(csv_path)
    else:
        logger.warning(f"结果文件不存在，跳过: {csv_path}")
