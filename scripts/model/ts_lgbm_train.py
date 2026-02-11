"""
时序 LightGBM 模型训练与预测脚本

职责：加载训练/测试数据，训练 LightGBM 二分类模型，输出预测结果
输入：df_ts_ready_2024.pkl (训练), df_ts_ready_2025.pkl (测试)
输出：ts_lgbm_predictions.pkl, ts_lgbm_model.txt

模型设定：
  - 入场时间: 10:30（由 build_ts_features.py 确定）
  - 持有期: 120分钟
  - 标签: vol_adj_ret > 0.5（由 build_ts_label.py 构建）
  - 特征: 25个时序特征（5大类）

依赖：build_ts_label.py 生产的训练就绪数据

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
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

# LightGBM 参数
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
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
        f"正类: {df['label'].sum():,} ({df['label'].mean():.2%})"
    )
    return df


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> lgb.Booster:
    """
    训练 LightGBM 二分类模型

    使用 early_stopping 防止过拟合
    """
    X_train = train_df[FEATURES]
    y_train = train_df["label"]
    X_test = test_df[FEATURES]
    y_test = test_df["label"]

    logger.info(f"特征数: {len(FEATURES)}")
    logger.info(
        f"训练集: {len(X_train):,} 行, 正类: {y_train.sum():,} ({y_train.mean():.2%})"
    )
    logger.info(
        f"测试集: {len(X_test):,} 行, 正类: {y_test.sum():,} ({y_test.mean():.2%})"
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
        logger.info(f"  {feat:>20s}: {imp:,.0f}")

    return model


# ==================== 预测与评估 ====================
def predict_and_evaluate(model: lgb.Booster, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    在测试集上预测，评估性能，保存结果
    """
    X_test = test_df[FEATURES]
    y_test = test_df["label"]

    # 预测概率
    y_prob = model.predict(X_test)
    test_df = test_df.copy()
    test_df["pred_prob"] = y_prob

    # AUC
    test_auc = roc_auc_score(y_test, y_prob)
    logger.success(f"样本外 AUC: {test_auc:.4f}")

    # 概率分布
    logger.info(
        f"预测概率分布: min={y_prob.min():.4f}, "
        f"median={np.median(y_prob):.4f}, max={y_prob.max():.4f}"
    )

    # 不同阈值下的精确率/召回率
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (y_prob > thr).astype(int)
        n_pred = pred.sum()
        if n_pred > 0:
            precision = y_test[pred == 1].mean()
            recall = y_test[pred == 1].sum() / y_test.sum() if y_test.sum() > 0 else 0
            logger.info(
                f"  阈值={thr:.2f}: 预测正类={n_pred:,}, "
                f"精确率={precision:.2%}, 召回率={recall:.2%}"
            )

    return test_df


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("时序 LightGBM 模型训练与预测")
    logger.info(f"入场时间: 10:30, 持有期: 120分钟")
    logger.info("=" * 60)

    # 1. 加载数据
    train_df = load_ready_data(YEAR_TRAIN)
    test_df = load_ready_data(YEAR_TEST)

    # 2. 训练模型
    model = train_model(train_df, test_df)

    # 3. 预测与评估
    test_result = predict_and_evaluate(model, test_df)

    # 4. 保存预测结果
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
    model_path = OUTPUT_DIR / "ts_lgbm_model.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    logger.info("=" * 60)
    logger.info("时序模型训练预测完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
