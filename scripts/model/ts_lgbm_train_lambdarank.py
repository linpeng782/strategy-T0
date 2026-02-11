"""
时序 LightGBM LambdaRank 排序模型训练与预测脚本 (S4)

与回归版的区别：
  - objective: lambdarank（排序学习）
  - 标签: excess_ret 按日内分位转为 0-4 的相关性等级
  - 分组: 每天的股票为一个 query group
  - 评估: NDCG@20,50,100
  - 直接优化 Top-N 排序质量，而非绝对值预测

目的：让模型学习"在每天的~1000只股票中，哪些排在前面"

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

# 相关性等级数（将 excess_ret 分为 N_GRADES 个等级）
N_GRADES = 5  # 0, 1, 2, 3, 4

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

# LightGBM LambdaRank 参数
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [20, 50, 100],
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
        f"{df['date'].nunique()} 天, {df['SecuCode'].nunique()} 只股票"
    )
    return df


# ==================== 标签转换 ====================
def create_relevance_labels(df: pd.DataFrame, n_grades: int = 5) -> np.ndarray:
    """
    将 excess_ret 转换为日内分位相关性等级

    每天内部：
      - excess_ret 最低的 20% → 等级 0
      - excess_ret 次低的 20% → 等级 1
      - ...
      - excess_ret 最高的 20% → 等级 4

    返回: 与 df 等长的整数数组
    """
    labels = np.zeros(len(df), dtype=np.int32)

    for date, group in df.groupby("date"):
        idx = group.index
        # 日内分位
        ranks = group[LABEL_COL].rank(pct=True)
        # 转为 0 ~ (n_grades-1) 的等级
        grades = np.clip((ranks * n_grades).astype(int), 0, n_grades - 1)
        labels[df.index.get_indexer(idx)] = grades.values

    logger.info(f"相关性等级分布: {np.bincount(labels, minlength=n_grades)}")
    return labels


def create_group_data(df: pd.DataFrame) -> np.ndarray:
    """
    创建 query group 数据

    每天的股票为一个 group，返回每个 group 的样本数数组。
    要求: df 已按 date 排序
    """
    group_sizes = df.groupby("date").size().values
    logger.info(
        f"Query groups: {len(group_sizes)} 天, "
        f"每组: {group_sizes.min()}~{group_sizes.max()} 只, "
        f"均值={group_sizes.mean():.0f}"
    )
    return group_sizes


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> lgb.Booster:
    """训练 LightGBM LambdaRank 模型"""
    # 确保按日期排序（group 要求）
    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    # 创建相关性等级标签
    logger.info(f"创建 {N_GRADES} 级相关性标签...")
    y_train = create_relevance_labels(train_df, N_GRADES)
    y_test = create_relevance_labels(test_df, N_GRADES)

    # 创建 group 数据
    logger.info("创建 query group 数据...")
    group_train = create_group_data(train_df)
    group_test = create_group_data(test_df)

    logger.info(f"特征数: {len(FEATURES)}")
    logger.info(f"训练集: {len(X_train):,} 行, {len(group_train)} 个 query group")
    logger.info(f"测试集: {len(X_test):,} 行, {len(group_test)} 个 query group")

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_data = lgb.Dataset(X_test, label=y_test, group=group_test, reference=train_data)

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

    return model, test_df


# ==================== 预测与评估 ====================
def predict_and_evaluate(
    model: lgb.Booster, test_df: pd.DataFrame
) -> pd.DataFrame:
    """预测并用 Rank IC 评估"""
    X_test = test_df[FEATURES]
    y_true = test_df[LABEL_COL]

    y_pred = model.predict(X_test)
    test_df = test_df.copy()
    # 保存为 pred_prob 列以兼容回测脚本
    test_df["pred_prob"] = y_pred

    # 全局 Rank IC（pred_prob vs 真实 excess_ret）
    global_ic, _ = spearmanr(y_pred, y_true)
    logger.success(f"全局 Rank IC (Spearman): {global_ic:.4f}")

    # 日度 Rank IC
    daily_ic = test_df.groupby("date").apply(
        lambda g: spearmanr(g["pred_prob"], g[LABEL_COL])[0]
        if len(g) > 10
        else np.nan,
        include_groups=False,
    )
    daily_ic = daily_ic.dropna()
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0

    logger.success(f"日度 Rank IC: 均值={ic_mean:.4f}, std={ic_std:.4f}, ICIR={icir:.4f}")
    logger.info(f"IC > 0 的天数占比: {(daily_ic > 0).mean():.1%}")

    # 预测值分布
    logger.info(
        f"预测分数分布: min={y_pred.min():.4f}, "
        f"median={np.median(y_pred):.4f}, max={y_pred.max():.4f}"
    )

    # 按预测值分10组看实际收益
    test_df["pred_decile"] = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
    logger.info("预测分位 vs 实际收益:")
    for d in sorted(test_df["pred_decile"].unique()):
        g = test_df[test_df["pred_decile"] == d]
        logger.info(
            f"  D{d}: 实际raw={g['raw_ret'].mean()*10000:.2f}bps, "
            f"excess={g['excess_ret'].mean()*10000:.2f}bps, "
            f"n={len(g):,}"
        )

    return test_df


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("时序 LightGBM LambdaRank 排序模型 (S4)")
    logger.info(f"标签: {LABEL_COL} → {N_GRADES} 级相关性等级")
    logger.info(f"NDCG 评估位置: {LGB_PARAMS['ndcg_eval_at']}")
    logger.info("=" * 60)

    # 1. 加载数据
    train_df = load_ready_data(YEAR_TRAIN)
    test_df = load_ready_data(YEAR_TEST)

    # 2. 训练模型
    model, test_df_sorted = train_model(train_df, test_df)

    # 3. 预测与评估
    test_result = predict_and_evaluate(model, test_df_sorted)

    # 4. 保存预测结果（独立文件）
    save_cols = (
        ["SecuCode", "date", "entry_time"]
        + FEATURES
        + ["raw_ret", "market_ret", "excess_ret", "vol_adj_ret", "label", "pred_prob"]
    )
    save_df = test_result[save_cols].copy()
    pred_path = OUTPUT_DIR / "ts_lgbm_predictions_lambdarank.pkl"
    save_df.to_pickle(pred_path)
    logger.success(f"预测结果已保存: {pred_path} ({len(save_df):,} 行)")

    # 5. 保存模型
    model_path = OUTPUT_DIR / "ts_lgbm_model_lambdarank.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    logger.info("=" * 60)
    logger.info("LambdaRank 排序模型训练预测完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
