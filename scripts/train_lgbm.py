#!/usr/bin/env python3
"""
LGBM 信号分类模型训练脚本
==========================
从"规则执行者"进化为"概率博弈者"

设计目标：
1. 直接从缓存读取数据（由 prepare_backtest_data.py 生成）
2. 路径敏感型标签：在不触碰止损的前提下摸到止盈才算赢
3. 特征：X1_zscore, X2_zscore, Z_final, time_val, yesterday_range

使用方法：
1. 先运行 prepare_backtest_data.py 生成缓存
2. 再运行本脚本训练模型
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
import sys
import time as time_module
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== 配置参数 ====================
# 缓存目录（与 prepare_backtest_data.py 保持一致）
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")

# 模型输出目录
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 建模参数
Z_THRESHOLD = -1.5  # 潜在买入信号阈值
TARGET_PROFIT_BP = 30  # 止盈目标（60bp）
STOP_LOSS_BP = 40  # 止损目标（40bp）
TRAIN_RATIO = 0.8  # 训练集比例

# 特征列表（V3增强版：新增大盘环境特征 mkt_oversold_ratio, mkt_avg_z）
ML_FEATURES = [
    "X1_zscore",
    "X2_zscore",
    "Z_final",
    "time_val",
    "yesterday_range",
    "vol_burst",
    "z_final_slope",
    "x1_slope",
    "mkt_oversold_ratio",
    "mkt_avg_z",
]

# 设置字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_cached_data() -> pd.DataFrame:
    """从缓存加载数据"""
    logger.info(f"从缓存加载数据: {CACHE_DIR}")

    df_5m_path = CACHE_DIR / "df_5m.pkl"
    if not df_5m_path.exists():
        raise FileNotFoundError(
            f"缓存文件不存在: {df_5m_path}\n请先运行 prepare_backtest_data.py 生成缓存"
        )

    df = pd.read_pickle(df_5m_path)

    # 加载元数据
    metadata_path = CACHE_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(
            f"缓存元数据: {metadata['n_stocks']} 只股票, {metadata['df_5m_rows']:,} 行"
        )

    logger.success(f"数据加载完成: {len(df):,} 行, {df['SecuCode'].nunique()} 只股票")
    return df


def prepare_modeling_data(df: pd.DataFrame) -> tuple:
    """
    准备建模数据 (V2 路径敏感型标签)

    核心逻辑：
    1. 只在昨日高波动股中，挑选有买入潜力的信号点
    2. 路径敏感型标签：在不触碰止损的前提下摸到止盈才算赢
       - 修正"乐观偏见"：不再只看是否摸到止盈线
       - 必须满足：future_high >= 止盈价 且 future_low > 止损价
    """
    logger.info("准备建模数据 (V2 路径敏感型)...")

    # 1. 昨日高波动滤网（全局90%分位数）
    vol_threshold = df["yesterday_range"].quantile(0.9)
    logger.info(f"昨日波动率阈值（90%分位数）: {vol_threshold*100:.2f}%")

    # 2. 核心过滤：只训练潜在买入信号点
    df_model = df[
        (df["yesterday_range"] > vol_threshold)  # 昨日高波动
        & (df["Z_final"] < Z_THRESHOLD)  # 潜在买入信号
        & (df["close"] < df["day_high"])  # 非新高
        & (df["time_str"] < "14:15")  # 剔除尾盘
        & np.isfinite(df["Z_final"])
        & np.isfinite(df["X1_zscore"])
        & np.isfinite(df["X2_zscore"])
        & np.isfinite(df["future_high"])
        & np.isfinite(df["future_low"])
        & np.isfinite(df["vol_burst"])  # 新增特征
        & np.isfinite(df["z_final_slope"])  # 新增特征
        & np.isfinite(df["x1_slope"])  # 新增特征
        & np.isfinite(df["mkt_oversold_ratio"])  # 大盘环境特征
        & np.isfinite(df["mkt_avg_z"])  # 大盘环境特征
    ].copy()

    logger.info(f"筛选后样本数: {len(df_model):,}")

    # 3. 路径敏感型打标签（Triple Barrier Method 简化版）
    # 止盈止损阈值（与回测参数保持一致）
    tp_ratio = TARGET_PROFIT_BP / 10000  # +60bp
    sl_ratio = STOP_LOSS_BP / 10000  # -40bp

    # 计算止盈价和止损价
    df_model["tp_price"] = df_model["close"] * (1 + tp_ratio)
    df_model["sl_price"] = df_model["close"] * (1 - sl_ratio)

    # 判定为 1 的严格条件：
    # 必须摸到止盈价，且在此期间绝对不能跌破止损价
    # 注意：这是保守估计，因为5分钟Bar内无法判断先后顺序
    # 我们假设如果这段时间内跌破过止损，这单就算输
    win_condition = (df_model["future_high"] >= df_model["tp_price"]) & (
        df_model["future_low"] > df_model["sl_price"]
    )
    df_model["label"] = win_condition.astype(int)

    # 4. 过滤 NaN
    df_model = df_model.dropna(subset=ML_FEATURES + ["label"])

    pos_ratio = df_model["label"].mean()
    logger.info(f"修正后正样本比例 (真实胜率): {pos_ratio:.2%}")

    return df_model, ML_FEATURES


def train_lgbm(df: pd.DataFrame, features: list) -> tuple:
    """
    训练 LightGBM 模型

    按时间划分训练集和测试集（极其重要：不能随机乱分！）
    """
    logger.info("开始训练 LightGBM 模型...")

    # 1. 按时间划分
    df = df.sort_values("date")
    unique_dates = sorted(df["date"].unique())
    split_idx = int(len(unique_dates) * TRAIN_RATIO)
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_train, y_train = train_df[features], train_df["label"]
    X_test, y_test = test_df[features], test_df["label"]

    logger.info(f"训练集: {len(X_train):,} 样本, 正样本比例: {y_train.mean():.2%}")
    logger.info(f"测试集: {len(X_test):,} 样本, 正样本比例: {y_test.mean():.2%}")
    logger.info(f"划分日期: {split_date}")

    # 2. 设置 LGBM 参数（优化版：降低学习率、增加正则化、处理类别不平衡）
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.01,  # 降慢学习速度，学得更细
        "num_leaves": 31,  # 增加叶子数，捕捉更复杂的组合
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "is_unbalance": True,  # 自动处理正负样本不平衡
        "lambda_l1": 0.5,  # L1 正则，防止特征权重过大
        "lambda_l2": 0.5,  # L2 正则，防止过拟合
        "min_data_in_leaf": 150,  # 每个叶子最少150个样本，防止过拟合
        "seed": 42,
        "verbose": -1,
    }

    # 3. 训练
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=1000,  # 学习率降低后需要更多轮次
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    logger.success(f"模型训练完成, 最佳迭代次数: {model.best_iteration}")

    return model, X_train, y_train, X_test, y_test, test_df


def evaluate_model(model, X_test, y_test, features: list, save_dir: Path = None):
    """
    评估模型性能
    """
    logger.info("评估模型...")

    # 1. 预测概率
    y_prob = model.predict(X_test)

    # 2. 计算 AUC
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 80)
    print("LGBM 模型评估报告")
    print("=" * 80)

    print(f"\n📊 模型性能:")
    print(f"   测试集 AUC: {auc:.4f}")

    # 3. 不同阈值下的表现
    print(f"\n📈 不同概率阈值下的表现:")
    print(
        f"   {'阈值':>8} | {'精确率':>8} | {'召回率':>8} | {'样本数':>8} | {'正样本率':>10}"
    )
    print(f"   {'-' * 55}")

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
        y_pred = (y_prob >= threshold).astype(int)
        n_pred = y_pred.sum()
        if n_pred > 0:
            precision = y_test[y_pred == 1].mean()
            recall = y_test[y_pred == 1].sum() / y_test.sum() if y_test.sum() > 0 else 0
            print(
                f"   {threshold:>8.2f} | {precision:>7.2%} | {recall:>7.2%} | {n_pred:>8,} | {precision:>10.2%}"
            )

    # 4. 特征重要性
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values(by="importance", ascending=False)

    print(f"\n🎯 特征重要性排名 (Gain):")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] / importance["importance"].max() * 20)
        print(f"   {row['feature']:>15}: {bar} {row['importance']:.1f}")

    # 5. 绘制图表
    if save_dir:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 特征重要性
        ax1 = axes[0, 0]
        importance_sorted = importance.sort_values("importance", ascending=True)
        ax1.barh(
            importance_sorted["feature"],
            importance_sorted["importance"],
            color="steelblue",
        )
        ax1.set_title("Feature Importance (Gain)")
        ax1.set_xlabel("Importance")

        # 图2: 概率分布
        ax2 = axes[0, 1]
        ax2.hist(
            y_prob[y_test == 0], bins=50, alpha=0.5, label="Negative (0)", color="red"
        )
        ax2.hist(
            y_prob[y_test == 1], bins=50, alpha=0.5, label="Positive (1)", color="green"
        )
        ax2.set_title("Predicted Probability Distribution")
        ax2.set_xlabel("Probability")
        ax2.legend()

        # 图3: 按概率分组的实际正样本率
        ax3 = axes[1, 0]
        df_eval = pd.DataFrame({"prob": y_prob, "label": y_test.values})
        df_eval["prob_bin"] = pd.qcut(df_eval["prob"], q=10, duplicates="drop")
        bin_stats = df_eval.groupby("prob_bin")["label"].agg(["mean", "count"])
        bin_stats["mean"].plot(kind="bar", ax=ax3, color="teal", alpha=0.7)
        ax3.set_title("Actual Win Rate by Probability Bin")
        ax3.set_xlabel("Probability Bin")
        ax3.set_ylabel("Win Rate")
        ax3.tick_params(axis="x", rotation=45)

        # 图4: ROC 曲线
        ax4 = axes[1, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax4.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC={auc:.4f})")
        ax4.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax4.set_title("ROC Curve")
        ax4.set_xlabel("False Positive Rate")
        ax4.set_ylabel("True Positive Rate")
        ax4.legend()

        plt.tight_layout()
        save_path = save_dir / "lgbm_evaluation.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"评估图表已保存: {save_path}")
        plt.close()

    return y_prob, auc, importance


def analyze_by_score(test_df: pd.DataFrame, y_prob: np.ndarray):
    """
    按模型得分分组分析实际收益
    """
    print(f"\n📉 按模型得分分组的实际收益分析:")

    df_analysis = test_df.copy()
    df_analysis["score"] = y_prob
    df_analysis["mfe_bp"] = (
        df_analysis["future_high"] / df_analysis["close"] - 1
    ) * 10000

    # 按得分分组
    df_analysis["score_bin"] = pd.qcut(
        df_analysis["score"], q=5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"]
    )

    group_stats = df_analysis.groupby("score_bin").agg(
        {"mfe_bp": ["mean", "median", "count"], "label": "mean"}
    )
    group_stats.columns = ["MFE均值(bp)", "MFE中位数(bp)", "样本数", "实际胜率"]

    print(
        f"\n   {'分组':>10} | {'MFE均值(bp)':>12} | {'MFE中位数(bp)':>14} | {'样本数':>8} | {'实际胜率':>10}"
    )
    print(f"   {'-' * 70}")
    for idx, row in group_stats.iterrows():
        print(
            f"   {idx:>10} | {row['MFE均值(bp)']:>12.1f} | {row['MFE中位数(bp)']:>14.1f} | {int(row['样本数']):>8} | {row['实际胜率']:>10.2%}"
        )

    # 结论
    q5_win_rate = group_stats.loc["Q5(高)", "实际胜率"]
    q1_win_rate = group_stats.loc["Q1(低)", "实际胜率"]
    lift = q5_win_rate / q1_win_rate if q1_win_rate > 0 else float("inf")

    print(f"\n🎯 结论:")
    print(f"   高分组(Q5)胜率: {q5_win_rate:.2%}")
    print(f"   低分组(Q1)胜率: {q1_win_rate:.2%}")
    print(f"   Lift (提升倍数): {lift:.2f}x")

    if lift > 1.2:
        print(f"   ✅ 模型具有显著的信号筛选能力!")
    else:
        print(f"   ⚠️ 模型筛选能力有限，建议增加特征或调整参数")


def main():
    """主函数"""
    total_start = time_module.time()

    logger.info("=" * 80)
    logger.info("LGBM 信号分类模型训练 (从缓存加载数据)")
    logger.info("=" * 80)

    # 1. 从缓存加载数据
    df_5m = load_cached_data()

    # 2. 准备建模数据
    df_model, features = prepare_modeling_data(df_5m)
    del df_5m  # 释放内存

    # 3. 训练模型
    model, X_train, y_train, X_test, y_test, test_df = train_lgbm(df_model, features)

    # 4. 评估模型
    y_prob, auc, importance = evaluate_model(
        model, X_test, y_test, features, save_dir=OUTPUT_DIR
    )

    # 5. 按得分分组分析
    analyze_by_score(test_df, y_prob)

    # 6. 保存模型
    model_path = OUTPUT_DIR / "lgbm_signal_classifier.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")

    total_time = time_module.time() - total_start
    logger.success(f"\n总耗时: {total_time:.1f}s")

    return model, df_model, features


if __name__ == "__main__":
    model, df_model, features = main()
