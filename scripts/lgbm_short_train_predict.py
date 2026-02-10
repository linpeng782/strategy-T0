"""
LightGBM 二分类模型 — 做空方向 (训练与预测)

与做多模型 (lgbm_train_predict.py) 完全独立：
  - 标签：is_short_profitable = (excess_ret < -10bps)，即超额收益显著为负
  - 模型预测高概率 → 该股票大概率跑输市场 → 做空信号
  - 输出文件均带 short_ 前缀，不会覆盖做多模型结果

收益语义：
  做空收益 = -(raw_ret) = 1 - 1/Y = (buy_price - sell_price) / sell_price
  做空净收益 = 做空收益 - 成本

输入：df_features_2024.pkl, df_features_2025.pkl
输出：lgbm_short_test_predictions.pkl, lgbm_short_classifier.txt

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR_TRAIN = 2024
YEAR_TEST = 2025

# 标签配置
HOLD_COL = "Y_120m"
EXCESS_THRESHOLD = 0.0010  # 超额收益阈值 10bps（做空方向取反：excess_ret < -10bps）

# 特征列表（与做多模型一致）
FEATURES = [
    "X1_zscore",
    "X2_zscore",
    "X1_zscore_rank",  # 截面排名
    "X2_zscore_rank",  # 截面排名
    "rel_vol",
    "X1_delta_15m",  # 动量斜率
    "time_index",  # 入场时间编码
]

# 相对成交量窗口
VOL_WINDOW = 10

# 动量斜率窗口（3个5分钟Bar = 15分钟）
DELTA_BARS = 3

# LightGBM 参数（与做多模型一致）
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": 1,
    "seed": 42,
}
NUM_BOOST_ROUND = 100


# ==================== 数据准备 ====================
def load_and_prepare(year: int) -> pd.DataFrame:
    """加载特征数据并构建增强特征（与做多模型相同，仅标签不同）"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    logger.info(f"加载 {year} 年特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    logger.success(
        f"加载完成: {len(df):,} 行, {df['SecuCode'].nunique()} 只股票, "
        f"{df['date'].nunique()} 个交易日"
    )

    # ---------- 1. 相对成交量 ----------
    logger.info(f"计算相对成交量 (窗口={VOL_WINDOW}天)...")
    df = df.sort_values(["SecuCode", "entry_time", "date"])
    grp_vol = df.groupby(["SecuCode", "entry_time"])["cum_volume"]
    vol_mean = grp_vol.rolling(VOL_WINDOW, min_periods=2).mean().shift(1)
    vol_mean = vol_mean.reset_index(level=[0, 1], drop=True)
    df["rel_vol"] = df["cum_volume"] / vol_mean
    logger.success(f"相对成交量计算完成, 有效: {df['rel_vol'].notna().sum():,}")

    # ---------- 2. 截面排名 ----------
    logger.info("计算截面排名...")
    for feat in ["X1_zscore", "X2_zscore"]:
        df[f"{feat}_rank"] = df.groupby(["date", "entry_time"])[feat].rank(pct=True)

    # ---------- 3. 动量斜率 ----------
    logger.info(f"计算动量斜率 (delta={DELTA_BARS}bars)...")
    df = df.sort_values(["SecuCode", "date", "entry_time"])
    df["X1_delta_15m"] = df.groupby(["SecuCode", "date"])["X1_zscore"].diff(DELTA_BARS)

    # ---------- 4. 时间编码 ----------
    def time_to_minutes(t_str):
        h, m = map(int, t_str.split(":"))
        return h * 60 + m - 9 * 60 - 30

    df["time_index"] = df["entry_time"].apply(time_to_minutes)

    # ---------- 5. 做多收益与标签 ----------
    df["raw_ret"] = 1.0 / df[HOLD_COL] - 1.0

    # 市场平均收益
    df["market_ret"] = df.groupby(["date", "entry_time"])["raw_ret"].transform("mean")
    df["excess_ret"] = df["raw_ret"] - df["market_ret"]

    # ★ 做空标签：超额收益 < -10bps 为正类（跑输市场）
    df["is_short_profitable"] = (df["excess_ret"] < -EXCESS_THRESHOLD).astype(int)

    # ---------- 6. 过滤异常值和缺失值 ----------
    feat_cols = [c for c in FEATURES if c in df.columns]
    mask = df[feat_cols].notna().all(axis=1)
    mask = mask & df[feat_cols].apply(np.isfinite).all(axis=1)
    mask = mask & df[HOLD_COL].notna() & np.isfinite(df[HOLD_COL])
    mask = mask & (df[HOLD_COL] > 0.9) & (df[HOLD_COL] < 1.1)
    df = df[mask].copy()
    logger.info(f"清洗后样本: {len(df):,} 行")

    # 正类比例（跑输市场的比例）
    pos_ratio = df["is_short_profitable"].mean()
    logger.info(f"做空正类比例 (超额收益<-{EXCESS_THRESHOLD*10000:.0f}bps): {pos_ratio:.2%}")

    return df


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame) -> lgb.Booster:
    """训练做空方向 LightGBM 分类器"""
    X_train = train_df[FEATURES]
    y_train = train_df["is_short_profitable"]

    logger.info(f"开始训练做空模型... 特征: {FEATURES}")
    logger.info(
        f"训练集: {len(X_train):,} 行, 正类(做空): {y_train.sum():,} ({y_train.mean():.2%})"
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(LGB_PARAMS, train_data, num_boost_round=NUM_BOOST_ROUND)

    # 训练集 AUC
    y_train_prob = model.predict(X_train)
    train_auc = roc_auc_score(y_train, y_train_prob)
    logger.success(f"训练完成! 训练集 AUC: {train_auc:.4f}")

    # 特征重要性
    importance = model.feature_importance(importance_type="gain")
    for feat, imp in sorted(zip(FEATURES, importance), key=lambda x: -x[1]):
        logger.info(f"  特征重要性: {feat} = {imp:.0f}")

    return model


# ==================== 预测与保存 ====================
def predict_and_save(model: lgb.Booster, test_df: pd.DataFrame) -> float:
    """在测试集上预测并保存结果"""
    X_test = test_df[FEATURES]
    y_test = test_df["is_short_profitable"]

    # 预测概率（做空概率：概率越高，越可能跑输市场）
    y_prob = model.predict(X_test)
    test_df["short_pred_prob"] = y_prob

    # 样本外 AUC
    test_auc = roc_auc_score(y_test, y_prob)
    logger.success(f"样本外 AUC: {test_auc:.4f}")

    # 概率分布
    logger.info(
        f"做空概率分布: min={y_prob.min():.4f}, "
        f"median={np.median(y_prob):.4f}, max={y_prob.max():.4f}"
    )

    # 保存测试集预测结果（使用 short_ 前缀，与做多模型隔离）
    save_cols = (
        ["SecuCode", "date", "entry_time"]
        + FEATURES
        + [
            "raw_ret",
            "market_ret",
            "excess_ret",
            "is_short_profitable",
            "short_pred_prob",
        ]
    )
    save_df = test_df[save_cols].copy()
    pred_path = OUTPUT_DIR / "lgbm_short_test_predictions.pkl"
    save_df.to_pickle(pred_path)
    logger.success(f"测试集预测结果已保存: {pred_path} ({len(save_df):,} 行)")

    return test_auc


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("LightGBM 做空模型 — 训练与预测")
    logger.info("=" * 60)

    # 1. 准备数据
    train_df = load_and_prepare(YEAR_TRAIN)
    test_df = load_and_prepare(YEAR_TEST)

    # 2. 训练模型
    model = train_model(train_df)

    # 3. 预测并保存
    test_auc = predict_and_save(model, test_df)

    # 4. 保存模型
    model_path = OUTPUT_DIR / "lgbm_short_classifier.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    logger.info("=" * 60)
    logger.info(
        f"做空模型训练预测完成! OOS AUC={test_auc:.4f}, "
        f"请运行 lgbm_short_backtest.py 进行回测分析"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
