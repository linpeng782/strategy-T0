"""
时序 LightGBM 回归模型训练与预测脚本

与 ts_lgbm_train.py 的区别：
  - objective: regression（而非 binary）
  - 标签: excess_ret（连续值，而非 0/1）
  - 评估: Rank IC（Spearman相关系数）
  - 输出: pred_prob 列实际为预测的 excess_ret，用于排名选股

目的：让模型学习收益的"量级"，而非"正负"

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

# 特征列表（与 build_ts_label.py 一致）
FEATURES = [
    # 第1类：当前bar的X1/X2
    "X1",
    "X2",
    "X1_zscore",
    "X2_zscore",
    # 第2类：日内价格轨迹
    "X1_lag1",
    "X1_lag3",
    "X1_lag6",
    "X1_lag10",
    "X2_lag1",
    "X2_lag3",
    "X2_lag6",
    "X1_diff1",
    "X1_diff3",
    "X1_slope_6bar",
    "X1_morning_std",
    # 第3类：量能演变
    "rel_vol",
    "vol_accel",
    "vol_concentration",
    # 第4类：日内价格形态
    "morning_ret",
    "price_position",
    "morning_range",
    "intraday_vol",
    # 第5类：跨日特征
    "overnight_gap",
    "prev_day_ret",
    "momentum_5d",
    "hist_vol_20d",
]

# LightGBM 回归参数
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

# 回归标签列
LABEL_COL = "excess_ret"


# ==================== 数据加载 ====================
def load_ready_data(year: int) -> pd.DataFrame:
    """加载训练就绪数据"""
    pkl_path = OUTPUT_DIR / f"df_ts_ready_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"训练数据不存在: {pkl_path}\n请先运行 build_ts_label.py"
        )
    df = pd.read_pickle(pkl_path)
    logger.success(
        f"加载 {year} 年数据: {len(df):,} 行, "
        f"{LABEL_COL} 均值={df[LABEL_COL].mean()*10000:.2f}bps"
    )
    return df


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> lgb.Booster:
    """训练 LightGBM 回归模型"""
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
    """预测并用 Rank IC 评估"""
    X_test = test_df[FEATURES]
    y_true = test_df[LABEL_COL]

    # 预测
    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    # 保存为 pred_prob 列以兼容回测脚本（实际是预测的 excess_ret）
    test_df["pred_prob"] = y_pred

    # 全局 Rank IC
    global_ic, _ = spearmanr(y_pred, y_true)
    logger.success(f"全局 Rank IC (Spearman): {global_ic:.4f}")

    # 日度 Rank IC
    daily_ic = test_df.groupby("date").apply(
        lambda g: spearmanr(g["pred_prob"], g[LABEL_COL])[0] if len(g) > 10 else np.nan,
        include_groups=False,
    )
    daily_ic = daily_ic.dropna()
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    logger.success(
        f"日度 Rank IC: 均值={ic_mean:.4f}, std={ic_std:.4f}, ICIR={icir:.4f}"
    )
    logger.info(f"IC > 0 的天数占比: {(daily_ic > 0).mean():.1%}")

    # 预测值分布
    logger.info(
        f"预测值分布: min={y_pred.min()*10000:.2f}bps, "
        f"median={np.median(y_pred)*10000:.2f}bps, max={y_pred.max()*10000:.2f}bps"
    )

    # 按预测值分10组看实际收益
    test_df["pred_decile"] = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
    logger.info("预测分位 vs 实际收益:")
    for d in sorted(test_df["pred_decile"].unique()):
        g = test_df[test_df["pred_decile"] == d]
        logger.info(
            f"  D{d}: pred=[{g['pred_prob'].min()*10000:>6.1f}, {g['pred_prob'].max()*10000:>6.1f}]bps, "
            f"实际raw={g['raw_ret'].mean()*10000:.2f}bps, excess={g['excess_ret'].mean()*10000:.2f}bps"
        )

    return test_df


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("时序 LightGBM 回归模型训练与预测")
    logger.info(f"标签: {LABEL_COL} (连续值)")
    logger.info("=" * 60)

    # 1. 加载数据
    train_df = load_ready_data(YEAR_TRAIN)
    test_df = load_ready_data(YEAR_TEST)

    # 2. 训练模型
    model = train_model(train_df, test_df)

    # 3. 预测与评估
    test_result = predict_and_evaluate(model, test_df)

    # 4. 保存预测结果（覆盖，兼容回测脚本）
    save_cols = (
        ["SecuCode", "date", "entry_time"]
        + FEATURES
        + ["raw_ret", "market_ret", "excess_ret", "vol_adj_ret", "label", "pred_prob"]
    )
    save_df = test_result[save_cols].copy()
    pred_path = OUTPUT_DIR / "ts_lgbm_predictions.pkl"
    save_df.to_pickle(pred_path)
    logger.success(f"预测结果已保存: {pred_path} ({len(save_df):,} 行)")

    # 5. 保存模型
    model_path = OUTPUT_DIR / "ts_lgbm_model_reg.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    logger.info("=" * 60)
    logger.info("回归模型训练预测完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
