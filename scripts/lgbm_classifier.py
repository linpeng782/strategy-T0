"""
LightGBM 二分类策略 (增强版)

改进点：
  1. 截面排名：X1/X2 转换为同时刻全市场百分位排名，消除日间波动率差异
  2. 动量斜率：X1_delta_15m = X1(t) - X1(t-15m)，捕捉加速/减速
  3. 标签中性化：剔除市场Beta，预测超额收益是否 > 10bps
  4. 多时间点：加入 time_index 特征，让模型自动学习不同时段规律
  5. 防过拟合：减小树复杂度

收益语义：
  Y_120m = buy_price / sell_price
  做多收益 = 1/Y - 1
  超额收益 = raw_ret - market_ret

输入：df_features_2024.pkl, df_features_2025.pkl
依赖：calc_feature_x1_x2.py 生产的特征数据

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from pathlib import Path
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COST_BPS = 15
COST_RATE = COST_BPS / 10000
EXCESS_THRESHOLD = 0.0010  # 超额收益阈值 10bps

# 特征列表（增强版）
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

# 概率阈值扫描
PROB_THRESHOLDS = [0.45]

# LightGBM 参数（防过拟合）
LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 15,  # 减小，防过拟合
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,  # 增大，防过拟合
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": 1,
    "seed": 42,
}
NUM_BOOST_ROUND = 100  # 减少轮数


# ==================== 数据准备 ====================
def load_and_prepare(year: int) -> pd.DataFrame:
    """加载特征数据并构建增强特征"""
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
    # 同一 (date, entry_time) 下的百分位排名，消除日间波动率差异
    logger.info("计算截面排名...")
    for feat in ["X1_zscore", "X2_zscore"]:
        df[f"{feat}_rank"] = df.groupby(["date", "entry_time"])[feat].rank(pct=True)

    # ---------- 3. 动量斜率 ----------
    # X1(t) - X1(t - 15m)，捕捉加速/减速
    logger.info(f"计算动量斜率 (delta={DELTA_BARS}bars)...")
    df = df.sort_values(["SecuCode", "date", "entry_time"])
    df["X1_delta_15m"] = df.groupby(["SecuCode", "date"])["X1_zscore"].diff(DELTA_BARS)

    # ---------- 4. 时间编码 ----------
    # 将 entry_time 转换为距离开盘的分钟数
    def time_to_minutes(t_str):
        h, m = map(int, t_str.split(":"))
        return h * 60 + m - 9 * 60 - 30  # 距离 09:30 的分钟数

    df["time_index"] = df["entry_time"].apply(time_to_minutes)

    # ---------- 5. 做多收益与标签中性化 ----------
    df["raw_ret"] = 1.0 / df[HOLD_COL] - 1.0

    # 市场平均收益（同一 date+entry_time 下全部股票的均值）
    df["market_ret"] = df.groupby(["date", "entry_time"])["raw_ret"].transform("mean")
    df["excess_ret"] = df["raw_ret"] - df["market_ret"]

    # 标签：超额收益 > 10bps 为正类
    df["is_profitable"] = (df["excess_ret"] > EXCESS_THRESHOLD).astype(int)

    # ---------- 6. 过滤异常值和缺失值 ----------
    feat_cols = [c for c in FEATURES if c in df.columns]
    mask = df[feat_cols].notna().all(axis=1)
    mask = mask & df[feat_cols].apply(np.isfinite).all(axis=1)
    mask = mask & df[HOLD_COL].notna() & np.isfinite(df[HOLD_COL])
    mask = mask & (df[HOLD_COL] > 0.9) & (df[HOLD_COL] < 1.1)
    df = df[mask].copy()
    logger.info(f"清洗后样本: {len(df):,} 行")

    # 正类比例
    pos_ratio = df["is_profitable"].mean()
    logger.info(f"正类比例 (超额收益>{EXCESS_THRESHOLD*10000:.0f}bps): {pos_ratio:.2%}")

    return df


# ==================== 模型训练 ====================
def train_model(train_df: pd.DataFrame) -> lgb.Booster:
    """训练 LightGBM 分类器"""
    X_train = train_df[FEATURES]
    y_train = train_df["is_profitable"]

    logger.info(f"开始训练 LightGBM... 特征: {FEATURES}")
    logger.info(
        f"训练集: {len(X_train):,} 行, 正类: {y_train.sum():,} ({y_train.mean():.2%})"
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


# ==================== 预测与回测 ====================
def predict_and_backtest(model: lgb.Booster, test_df: pd.DataFrame) -> list:
    """在测试集上预测并回测"""
    X_test = test_df[FEATURES]
    y_test = test_df["is_profitable"]

    # 预测概率
    y_prob = model.predict(X_test)
    test_df["pred_prob"] = y_prob

    # 样本外 AUC
    test_auc = roc_auc_score(y_test, y_prob)
    logger.success(f"样本外 AUC: {test_auc:.4f}")

    # 概率分布
    logger.info(
        f"预测概率分布: min={y_prob.min():.4f}, "
        f"median={np.median(y_prob):.4f}, max={y_prob.max():.4f}"
    )

    # 保存测试集预测结果（供信号分析脚本使用）
    save_cols = (
        ["SecuCode", "date", "entry_time"]
        + FEATURES
        + [
            "raw_ret",
            "market_ret",
            "excess_ret",
            "is_profitable",
            "pred_prob",
        ]
    )
    save_df = test_df[save_cols].copy()
    pred_path = OUTPUT_DIR / "lgbm_test_predictions.pkl"
    save_df.to_pickle(pred_path)
    logger.success(f"测试集预测结果已保存: {pred_path} ({len(save_df):,} 行)")

    # 对每个概率阈值做回测
    all_results = []
    for prob_th in PROB_THRESHOLDS:
        result = backtest_with_threshold(test_df, prob_th)
        result["auc"] = test_auc
        all_results.append(result)

    return all_results


def backtest_with_threshold(test_df: pd.DataFrame, prob_threshold: float) -> dict:
    """在指定概率阈值下回测"""
    trades = test_df[test_df["pred_prob"] > prob_threshold].copy()

    if len(trades) == 0:
        logger.info(f"  Threshold={prob_threshold}: no signal")
        return {
            "prob_threshold": prob_threshold,
            "n_trades": 0,
        }

    # 净收益
    trades["net_ret"] = trades["raw_ret"] - COST_RATE
    trades["net_ret_bps"] = trades["net_ret"] * 10000

    n_trades = len(trades)
    win_rate = (trades["net_ret"] > 0).mean()
    avg_raw_bps = trades["raw_ret"].mean() * 10000
    avg_net_bps = trades["net_ret_bps"].mean()

    # 精确率：模型预测为正类中，真正赚钱的比例
    precision = trades["is_profitable"].mean()

    # 日汇总
    daily = trades.groupby("date")["net_ret"].mean()
    n_days = len(daily)
    day_win_rate = (daily > 0).mean()
    avg_daily_count = n_trades / n_days if n_days > 0 else 0

    sharpe = 0.0
    if n_days > 1 and daily.std() > 0:
        sharpe = daily.mean() / daily.std() * np.sqrt(252)

    cum = daily.cumsum()
    max_dd = (cum - cum.cummax()).min() if len(cum) > 0 else 0

    # 获利因子
    gross_profit = trades.loc[trades["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(trades.loc[trades["net_ret"] < 0, "net_ret"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    logger.info(
        f"  Threshold={prob_threshold:.2f}: "
        f"{n_trades} trades/{n_days} days, "
        f"Prec={precision:.1%}, WinRate={win_rate:.1%}, "
        f"NetBps={avg_net_bps:.1f}, Sharpe={sharpe:.2f}"
    )

    return {
        "prob_threshold": prob_threshold,
        "n_trades": n_trades,
        "n_days": n_days,
        "avg_daily_count": avg_daily_count,
        "precision": precision,
        "win_rate": win_rate,
        "day_win_rate": day_win_rate,
        "avg_raw_bps": avg_raw_bps,
        "avg_net_bps": avg_net_bps,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "daily_ret": daily,
    }


# ==================== 报告 ====================
def print_report(all_results: list):
    """打印回测报告"""
    print("\n" + "=" * 140)
    print("LGBM Binary Classification Backtest Report")
    print(
        f"  Train: {YEAR_TRAIN} | Test: {YEAR_TEST} | "
        f"Hold: 120m | Cost: {COST_BPS}bps | ExcessThresh: {EXCESS_THRESHOLD*10000:.0f}bps"
    )
    print(f"  Features: {FEATURES}")
    auc = all_results[0].get("auc", 0)
    print(f"  OOS AUC: {auc:.4f}")
    print("=" * 140)

    header = (
        f"{'ProbTh':>8}  {'Trades':>8}  {'Days':>8}  {'AvgDay':>8}  "
        f"{'Prec':>8}  {'WinRate':>8}  {'DayWR':>8}  "
        f"{'GrossBps':>10}  {'NetBps':>10}  {'Sharpe':>8}  "
        f"{'MaxDD':>10}  {'PF':>8}"
    )
    print(f"\n{header}")
    print("-" * 140)

    for r in all_results:
        if r["n_trades"] == 0:
            print(f"{r['prob_threshold']:>8.2f}  {'N/A':>8}")
            continue

        print(
            f"{r['prob_threshold']:>8.2f}  "
            f"{r['n_trades']:>8,}  {r['n_days']:>8}  "
            f"{r['avg_daily_count']:>8.1f}  "
            f"{r['precision']:>7.1%}  {r['win_rate']:>7.1%}  "
            f"{r['day_win_rate']:>7.1%}  "
            f"{r['avg_raw_bps']:>10.2f}  {r['avg_net_bps']:>10.2f}  "
            f"{r['sharpe']:>8.2f}  "
            f"{r['max_drawdown']:>9.2%}  "
            f"{r['profit_factor']:>8.2f}"
        )

    print("-" * 140)

    valid = [r for r in all_results if r["n_trades"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\nBest Threshold (by Sharpe): {best['prob_threshold']:.2f}")
        print(
            f"  Sharpe={best['sharpe']:.2f}, Precision={best['precision']:.1%}, "
            f"DayWinRate={best['day_win_rate']:.1%}, NetBps={best['avg_net_bps']:.2f}bps"
        )

    print("=" * 140)


def plot_results(all_results: list, model: lgb.Booster):
    """绘制回测图表"""
    valid = [r for r in all_results if r["n_trades"] > 0 and "daily_ret" in r]
    if not valid:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))

    # 1. 资金曲线（x轴显示日期标签）
    for r in valid:
        daily_ret = r["daily_ret"]
        cum_bps = daily_ret.cumsum() * 10000
        label = f"P>{r['prob_threshold']:.2f} (S={r['sharpe']:.2f}, N={r['n_trades']})"
        axes[0].plot(range(len(cum_bps)), cum_bps.values, label=label, alpha=0.8)

        # 添加日期刻度（每月首个交易日标注）
        dates = daily_ret.index
        date_strs = [str(d) for d in dates]
        tick_pos, tick_labels = [], []
        seen_months = set()
        for i, d in enumerate(date_strs):
            month_key = d[:7]  # "YYYY-MM"
            if month_key not in seen_months:
                seen_months.add(month_key)
                tick_pos.append(i)
                tick_labels.append(d[5:10])  # "MM-DD"
        axes[0].set_xticks(tick_pos)
        axes[0].set_xticklabels(tick_labels, rotation=45, fontsize=8)

    axes[0].set_title(
        f"LGBM Equity Curve (Train={YEAR_TRAIN}, Test={YEAR_TEST}, Cost={COST_BPS}bps)",
        fontsize=12,
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Net Return (bps)")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)

    # 2. 阈值 vs 指标
    thresholds = [r["prob_threshold"] for r in valid]
    sharpes = [r["sharpe"] for r in valid]
    net_bps = [r["avg_net_bps"] for r in valid]
    n_trades = [r["n_trades"] for r in valid]

    ax2_twin = axes[1].twinx()
    axes[1].bar(
        thresholds, n_trades, width=0.015, alpha=0.3, color="gray", label="Trades"
    )
    axes[1].set_ylabel("Num Trades", color="gray")
    ax2_twin.plot(thresholds, sharpes, "bo-", label="Sharpe", markersize=6)
    ax2_twin.plot(thresholds, net_bps, "rs-", label="Net bps", markersize=6)
    ax2_twin.set_ylabel("Sharpe / Net bps")
    ax2_twin.legend(fontsize=8, loc="best")
    axes[1].set_xlabel("Prob Threshold")
    axes[1].set_title("Threshold vs Backtest Metrics", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 3. 特征重要性
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)
    axes[2].barh(
        [FEATURES[i] for i in sorted_idx],
        importance[sorted_idx],
        color="steelblue",
    )
    axes[2].set_title("Feature Importance (Gain)", fontsize=12)
    axes[2].set_xlabel("Importance")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "lgbm_classifier_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("LightGBM 二分类策略")
    logger.info("=" * 60)

    # 1. 准备数据
    train_df = load_and_prepare(YEAR_TRAIN)
    test_df = load_and_prepare(YEAR_TEST)

    # 2. 训练模型
    model = train_model(train_df)

    # 3. 预测与回测
    all_results = predict_and_backtest(model, test_df)

    # 4. 打印报告
    print_report(all_results)

    # 5. 绘图
    plot_results(all_results, model)

    # 6. 保存模型
    model_path = OUTPUT_DIR / "lgbm_classifier.txt"
    model.save_model(str(model_path))
    logger.success(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()
