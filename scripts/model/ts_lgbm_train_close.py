"""
时序 LightGBM 回归模型 — Close 离场标签版 (F2)

与原版 ts_lgbm_train_reg.py 的区别：
  - 标签: close_excess_ret（收盘价离场的超额收益），而非 excess_ret（VWAP离场）
  - 目的: 让模型学习"持续上涨到收盘"的股票，而非"日内冲高"的股票

逻辑:
  close_ret = 1/Y_rest - 1  (入场VWAP → 收盘价)
  close_market_ret = 日截面均值
  close_excess_ret = close_ret - close_market_ret

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

YEAR_TRAIN = 2024
YEAR_TEST = 2025

FEATURES = [
    "X1", "X2", "X1_zscore", "X2_zscore",
    "X1_lag1", "X1_lag3", "X1_lag6", "X1_lag10",
    "X2_lag1", "X2_lag3", "X2_lag6",
    "X1_diff1", "X1_diff3", "X1_slope_6bar", "X1_morning_std",
    "rel_vol", "vol_accel", "vol_concentration",
    "morning_ret", "price_position", "morning_range", "intraday_vol",
    "overnight_gap", "prev_day_ret", "momentum_5d", "hist_vol_20d",
]

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50

# 标签列（close版超额收益）
LABEL_COL = "close_excess_ret"


# ==================== 数据加载与标签构建 ====================
def load_and_build(year: int) -> pd.DataFrame:
    """
    加载特征数据，构建 close_ret 标签

    从 df_ts_features 中获取 Y_rest（close离场收益比），
    与 df_ts_ready 合并以复用已有的清洗逻辑
    """
    # 加载已清洗的 ready 数据（含所有特征和 VWAP 标签）
    ready = pd.read_pickle(OUTPUT_DIR / f"df_ts_ready_{year}.pkl")
    # 加载原始特征数据（含 Y_rest）
    feats = pd.read_pickle(OUTPUT_DIR / f"df_ts_features_{year}.pkl")

    # 提取 Y_rest 并计算 close_ret
    feats_sub = feats[["SecuCode", "date", "entry_time", "Y_rest"]].copy()
    feats_sub["close_ret"] = 1.0 / feats_sub["Y_rest"] - 1.0

    # 合并到 ready 数据
    df = ready.merge(
        feats_sub[["SecuCode", "date", "entry_time", "close_ret"]],
        on=["SecuCode", "date", "entry_time"],
        how="left",
    )

    # 计算 close 版市场收益和超额收益
    df["close_market_ret"] = df.groupby("date")["close_ret"].transform("mean")
    df["close_excess_ret"] = df["close_ret"] - df["close_market_ret"]

    # 清除 NaN
    mask = df["close_ret"].notna() & np.isfinite(df["close_ret"])
    df = df[mask].copy()

    logger.success(
        f"加载 {year} 年数据: {len(df):,} 行, "
        f"close_excess_ret 均值={df['close_excess_ret'].mean()*10000:.2f}bps, "
        f"std={df['close_excess_ret'].std()*10000:.2f}bps"
    )

    # VWAP vs Close 标签对比
    corr, _ = spearmanr(df["excess_ret"], df["close_excess_ret"])
    logger.info(
        f"VWAP超额 vs Close超额 Spearman相关: {corr:.4f} "
        f"(相关性越低，两个标签学到的东西差异越大)"
    )

    return df


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> lgb.Booster:
    """训练 LightGBM 回归模型（close标签版）"""
    X_train = train_df[FEATURES]
    y_train = train_df[LABEL_COL]
    X_test = test_df[FEATURES]
    y_test = test_df[LABEL_COL]

    logger.info(f"特征数: {len(FEATURES)}")
    logger.info(
        f"训练集: {len(X_train):,} 行, "
        f"标签均值={y_train.mean()*10000:.2f}bps, std={y_train.std()*10000:.2f}bps"
    )
    logger.info(
        f"测试集: {len(X_test):,} 行, "
        f"标签均值={y_test.mean()*10000:.2f}bps, std={y_test.std()*10000:.2f}bps"
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
        ],
    )

    logger.success(f"训练完成! 最佳迭代: {model.best_iteration}")

    # 特征重要性
    importance = model.feature_importance(importance_type="gain")
    logger.info("特征重要性 (gain):")
    for feat, imp in sorted(zip(FEATURES, importance), key=lambda x: -x[1]):
        logger.info(f"  {feat:>20s}: {imp:.4f}")

    return model


# ==================== 预测与评估 ====================
def predict_and_evaluate(model: lgb.Booster, test_df: pd.DataFrame) -> pd.DataFrame:
    """预测并评估（同时对比 VWAP 和 Close 两种离场）"""
    X_test = test_df[FEATURES]

    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    test_df["pred_prob"] = y_pred

    # 对 close_excess_ret 的 Rank IC（模型直接优化的目标）
    global_ic_close, _ = spearmanr(y_pred, test_df["close_excess_ret"])
    # 对 excess_ret (VWAP) 的 Rank IC（实际回测用的）
    global_ic_vwap, _ = spearmanr(y_pred, test_df["excess_ret"])

    logger.success(f"全局 Rank IC vs close_excess_ret: {global_ic_close:.4f}")
    logger.success(f"全局 Rank IC vs excess_ret(VWAP):  {global_ic_vwap:.4f}")

    # 日度 Rank IC（两种标签）
    daily_ic_close = test_df.groupby("date").apply(
        lambda g: spearmanr(g["pred_prob"], g["close_excess_ret"])[0]
        if len(g) > 10 else np.nan,
        include_groups=False,
    ).dropna()

    daily_ic_vwap = test_df.groupby("date").apply(
        lambda g: spearmanr(g["pred_prob"], g["excess_ret"])[0]
        if len(g) > 10 else np.nan,
        include_groups=False,
    ).dropna()

    logger.success(
        f"日度IC vs Close: 均值={daily_ic_close.mean():.4f}, "
        f"std={daily_ic_close.std():.4f}, IC>0={((daily_ic_close>0).mean()):.1%}"
    )
    logger.success(
        f"日度IC vs VWAP:  均值={daily_ic_vwap.mean():.4f}, "
        f"std={daily_ic_vwap.std():.4f}, IC>0={((daily_ic_vwap>0).mean()):.1%}"
    )

    # 分位分析
    test_df["pred_decile"] = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
    logger.info("预测分位 vs 实际收益 (VWAP / Close):")
    for d in sorted(test_df["pred_decile"].unique()):
        g = test_df[test_df["pred_decile"] == d]
        logger.info(
            f"  D{d}: VWAP_exc={g['excess_ret'].mean()*10000:>6.2f}bps, "
            f"Close_exc={g['close_excess_ret'].mean()*10000:>6.2f}bps, "
            f"raw={g['raw_ret'].mean()*10000:>6.2f}bps, "
            f"close_ret={g['close_ret'].mean()*10000:>6.2f}bps"
        )

    return test_df


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("时序 LightGBM 回归模型 — Close 离场标签版 (F2)")
    logger.info(f"标签: {LABEL_COL} (收盘价超额收益)")
    logger.info("=" * 60)

    # 1. 加载数据
    train_df = load_and_build(YEAR_TRAIN)
    test_df = load_and_build(YEAR_TEST)

    # 2. 训练模型
    model = train_model(train_df, test_df)

    # 3. 预测与评估
    test_result = predict_and_evaluate(model, test_df)

    # 4. 保存预测结果
    save_cols = (
        ["SecuCode", "date", "entry_time"]
        + FEATURES
        + ["raw_ret", "market_ret", "excess_ret", "close_ret",
           "close_market_ret", "close_excess_ret", "vol_adj_ret", "label", "pred_prob"]
    )
    save_df = test_result[save_cols].copy()
    pred_path = OUTPUT_DIR / "ts_lgbm_predictions_close_label.pkl"
    save_df.to_pickle(pred_path)
    logger.success(f"预测结果已保存: {pred_path} ({len(save_df):,} 行)")

    # 5. 保存模型
    model_path = OUTPUT_DIR / "ts_lgbm_model_close.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    logger.info("=" * 60)
    logger.info("Close标签回归模型训练预测完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
