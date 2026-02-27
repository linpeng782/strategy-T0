"""
LGBM 统一训练入口（SRP重构版）

用法：
  python train_lgbm_unified.py --label refined
  python train_lgbm_unified.py --label cs_demean
  python train_lgbm_unified.py --label wf_rank
  python train_lgbm_unified.py --label fwdret

支持的标签类型（详见 labels.py）：
  refined   : 4步精炼标签（当前最优，截面去均值→Z-Score→二次中性化→clip）
  cs_demean : 仅截面去均值（ret_ex = fwd_ret - 截面均值）
  wf_rank   : 过去20日滚动rank（0~1，无前视）
  fwdret    : 原始 fwd_ret_30min（baseline）
"""

import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from labels import build_label, LABEL_TYPES
from analysis import visualize_model_logic, quantile_return_analysis

import lightgbm as lgb

# ==================== 路径配置 ====================
BASE_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research")
FEAT_PKLS = [
    BASE_DIR / "output/pkl/features_csi1000_2024.pkl",
    BASE_DIR / "output/pkl/features_csi1000_2025.pkl",
]
OUTPUT_DIR = BASE_DIR / "output"
CSV_DIR = OUTPUT_DIR / "csv"
PNG_DIR = OUTPUT_DIR / "png"
PKL_DIR = OUTPUT_DIR / "pkl"
MODELS_DIR = OUTPUT_DIR / "models"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)
PKL_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 特征列 ====================
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
HOLD_PERIOD = "30min"
TRAIN_CUTOFF = pd.Timestamp("2025-01-01")

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
    df_train = df[df["date"] < cutoff_date].copy()
    df_test = df[df["date"] >= cutoff_date].copy()
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
    每日随机非重叠采样（向量化实现）。
    按持有期计算采样间隔，保证相邻样本的未来收益窗口不重叠。
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


def _ts_ic_fast(df: pd.DataFrame, ret_col: str) -> pd.Series:
    """
    全向量化时序IC（组内rank后Pearson = Spearman）。
    按 SecuCode+date 分组，计算 pred 与 ret_col 的 Spearman IC，再按 date 取均值。
    """
    grp_keys = ["SecuCode", "date"]
    tmp = df[["SecuCode", "date"]].copy()
    tmp["pr"] = df.groupby(grp_keys)["pred"].rank(pct=True)
    tmp["rr"] = df.groupby(grp_keys)[ret_col].rank(pct=True)

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


def train_and_eval(df: pd.DataFrame, label_type: str) -> dict:
    """
    核心训练函数：
      1. 调用 labels.build_label 生成指定类型的标签
      2. 时序切分 + 非重叠采样
      3. LGBM 训练
      4. 时序IC评估
      5. 保存模型和预测结果
    """
    ret_col = f"fwd_ret_{HOLD_PERIOD}"

    # 过滤：特征和收益率均有效
    valid_mask = df[FEAT_COLS].notna().all(axis=1) & df[ret_col].notna()
    df_valid = df[valid_mask].copy()
    logger.info(f"有效样本数: {len(df_valid):,}")

    # 构建标签
    logger.info(f"构建标签: {label_type} ...")
    df_valid = build_label(df_valid, label_type, ret_col=ret_col)

    # 按截止日期切分（标签构建在全量数据上，切分在后）
    df_train, df_test = split_by_cutoff(df_valid, TRAIN_CUTOFF)

    # 训练集非重叠采样
    logger.info("对训练集做每日随机非重叠采样...")
    df_train_sampled = daily_nonoverlap_sample(df_train, HOLD_PERIOD, seed=42)
    df_train_sampled = df_train_sampled[df_train_sampled["label"].notna()].copy()
    logger.info(f"采样后训练集（过滤无label后）: {len(df_train_sampled):,} 行")

    X_train = df_train_sampled[FEAT_COLS].values
    y_train = df_train_sampled["label"].values

    # 测试集非重叠采样（IC评估用）
    logger.info("对测试集做每日随机非重叠采样...")
    df_test_sampled = daily_nonoverlap_sample(df_test, HOLD_PERIOD, seed=99)
    logger.info(f"采样后测试集（IC评估）: {len(df_test_sampled):,} 行")
    X_test_all = df_test[FEAT_COLS].values

    # 训练
    logger.info(f"开始训练 LGBM 回归（固定 {LGBM_PARAMS['n_estimators']} 轮）...")
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(100)])
    logger.info(f"训练完成，耗时={time.time()-t0:.1f}s")

    # 预测
    pred_train = model.predict(X_train)
    pred_sampled = model.predict(df_test_sampled[FEAT_COLS].values)
    pred_all = model.predict(X_test_all)

    # 时序IC
    df_train_sampled = df_train_sampled.copy()
    df_train_sampled["pred"] = pred_train
    df_test_sampled = df_test_sampled.copy()
    df_test_sampled["pred"] = pred_sampled

    t_ic0 = time.time()
    ts_ic_train = _ts_ic_fast(df_train_sampled, ret_col)
    ts_daily_ic = _ts_ic_fast(df_test_sampled, ret_col)
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

    # 特征重要性 Top15
    importance = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values(
        ascending=False
    )
    logger.info("特征重要性 Top15:")
    for feat, imp in importance.head(15).items():
        logger.info(f"    {feat:<25} {imp:>6}")

    # 保存模型和结果
    model_path = MODELS_DIR / f"lgbm_reg_{label_type}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"模型已保存: {model_path}")

    result_df = df_test[["SecuCode", "date", "bar_time", ret_col]].copy()
    result_df["pred"] = pred_all
    result_path = CSV_DIR / f"lgbm_result_{label_type}.csv"
    result_df.to_csv(result_path, index=False)
    logger.success(f"结果已保存: {result_path}")

    ic_gap = ts_ic_train_mean - ts_mean_ic
    if ic_gap > 0.1:
        logger.warning(f"过拟合预警：IC Gap={ic_gap:.4f} > 0.1，训练集IC明显高于测试集")
    else:
        logger.success(f"IC Gap={ic_gap:.4f}，训练测试一致性良好")

    # 特征重要性Top5
    top5_features = importance.head(5).to_dict()

    return {
        "label_type": label_type,
        "n_train": len(y_train),
        "n_test_sampled": len(df_test_sampled),
        "n_test_all": len(df_test),
        "ts_ic_train": ts_ic_train_mean,
        "ts_daily_mean_ic": ts_mean_ic,
        "ts_icir": ts_icir,
        "ic_gap": ic_gap,
        "ts_ic_std": ts_std_ic,
        "ts_ic_pos": ts_ic_pos,
        "top5_features": top5_features,
        "model": model,
    }


def main(label_type: str) -> None:
    from ic_analysis import compute_forward_returns

    logger.success(
        f"[模型参数] n_estimators={LGBM_PARAMS['n_estimators']}  "
        f"learning_rate={LGBM_PARAMS['learning_rate']}  "
        f"num_leaves={LGBM_PARAMS['num_leaves']}  "
        f"min_child_samples={LGBM_PARAMS['min_child_samples']}  "
        f"feature_fraction={LGBM_PARAMS['feature_fraction']}"
    )
    logger.info(f"标签类型: {label_type}")

    df = load_features(FEAT_PKLS)
    logger.info("计算未来收益率...")
    df = compute_forward_returns(df)

    missing = [c for c in FEAT_COLS if c not in df.columns]
    if missing:
        logger.error(f"缺少特征列: {missing}")
        return

    res = train_and_eval(df, label_type)

    logger.info(f"\n{'='*60}")
    logger.info("训练结果汇总")
    logger.info(f"{'='*60}")
    for k, v in res.items():
        if k != "model" and k != "top5_features":
            logger.info(f"  {k:<20} = {v}")

    # 保存指标到 csv
    metrics_dict = {
        "label_type": res["label_type"],
        "训练集IC": f"{res['ts_ic_train']:.4f}",
        "测试集IC": f"{res['ts_daily_mean_ic']:.4f}",
        "测试集ICIR": f"{res['ts_icir']:.3f}",
        "IC_std": f"{res['ts_ic_std']:.4f}",
        "IC>0比例": f"{res['ts_ic_pos']:.1%}",
        "IC_Gap": f"{res['ic_gap']:.4f}",
        "训练样本数": res["n_train"],
        "测试样本数": res["n_test_all"],
    }
    # 添加Top5特征
    for i, (feat, imp) in enumerate(res["top5_features"].items(), 1):
        metrics_dict[f"Top{i}_特征"] = feat
        metrics_dict[f"Top{i}_重要性"] = f"{imp:.0f}"

    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = CSV_DIR / f"metrics_{label_type}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.success(f"指标已保存: {metrics_path}")

    # 可视化
    visualize_model_logic(res["model"], label_type, FEAT_COLS, LGBM_PARAMS, PNG_DIR)

    # 分层收益分析
    csv_path = CSV_DIR / f"lgbm_result_{label_type}.csv"
    if csv_path.exists():
        quantile_return_analysis(
            csv_path, ret_col=f"fwd_ret_{HOLD_PERIOD}", output_dir=CSV_DIR
        )
    else:
        logger.warning(f"结果文件不存在，跳过分层分析: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LGBM 统一训练入口")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=LABEL_TYPES,
        help=f"标签类型，可选: {LABEL_TYPES}",
    )
    args = parser.parse_args()
    main(args.label)
