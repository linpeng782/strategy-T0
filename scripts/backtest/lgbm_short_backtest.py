"""
LightGBM 做空模型 — 回测脚本 (SRP: 仅负责回测、报告和绘图)

加载 lgbm_short_train_predict.py 生成的做空预测结果，按概率阈值做回测。
所有输出文件带 short_ 前缀，与做多模型完全隔离。

做空收益语义：
  short_ret = -(raw_ret) = 买入价/卖出价 - 1 = Y - 1
  做空净收益 = short_ret - 成本

输入：output/lgbm_short_test_predictions.pkl
输出：终端报告 + lgbm_short_classifier_backtest.png

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_short_test_predictions.pkl"
MODEL_PATH = OUTPUT_DIR / "lgbm_short_classifier.txt"

YEAR_TRAIN = 2024
YEAR_TEST = 2025

COST_BPS = 15
COST_RATE = COST_BPS / 10000
EXCESS_THRESHOLD = 0.0010

# 概率阈值扫描（做空模型：概率越高越看空）
PROB_THRESHOLDS = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]

# 特征列表
FEATURES = [
    "X1_zscore",
    "X2_zscore",
    "X1_zscore_rank",
    "X2_zscore_rank",
    "rel_vol",
    "X1_delta_15m",
    "time_index",
]


# ==================== 数据加载 ====================
def load_predictions() -> pd.DataFrame:
    """加载做空模型的测试集预测结果"""
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"预测结果文件不存在: {PRED_PATH}\n"
            f"请先运行 lgbm_short_train_predict.py 生成预测结果"
        )
    df = pd.read_pickle(PRED_PATH)
    logger.success(f"加载做空预测结果: {len(df):,} 行, {df['date'].nunique()} 个交易日")
    return df


def load_model() -> lgb.Booster:
    """加载做空模型（用于绘制特征重要性）"""
    if not MODEL_PATH.exists():
        logger.warning(f"模型文件不存在: {MODEL_PATH}，跳过特征重要性绘制")
        return None
    model = lgb.Booster(model_file=str(MODEL_PATH))
    logger.success(f"做空模型已加载: {MODEL_PATH}")
    return model


# ==================== 回测 ====================
def backtest_with_threshold(test_df: pd.DataFrame, prob_threshold: float) -> dict:
    """在指定概率阈值下做空回测

    做空逻辑：short_pred_prob > 阈值 → 做空（卖出，120分钟后买回）
    """
    trades = test_df[test_df["short_pred_prob"] > prob_threshold].copy()

    if len(trades) == 0:
        logger.info(f"  Threshold={prob_threshold}: 无信号")
        return {"prob_threshold": prob_threshold, "n_trades": 0}

    # ★ 做空收益：与做多方向相反
    trades["short_ret"] = -trades["raw_ret"]
    trades["net_ret"] = trades["short_ret"] - COST_RATE
    trades["net_ret_bps"] = trades["net_ret"] * 10000

    n_trades = len(trades)
    win_rate = (trades["net_ret"] > 0).mean()
    avg_short_bps = trades["short_ret"].mean() * 10000
    avg_net_bps = trades["net_ret_bps"].mean()

    # 精确率：模型预测为做空正类中，实际跑输市场的比例
    precision = trades["is_short_profitable"].mean()

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
        "avg_short_bps": avg_short_bps,
        "avg_net_bps": avg_net_bps,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "daily_ret": daily,
    }


# ==================== 报告 ====================
def print_report(all_results: list, test_auc: float):
    """打印做空回测报告"""
    print("\n" + "=" * 140)
    print("LGBM Short Model Backtest Report (做空模型)")
    print(
        f"  Train: {YEAR_TRAIN} | Test: {YEAR_TEST} | "
        f"Hold: 120m | Cost: {COST_BPS}bps | "
        f"ShortLabel: excess_ret < -{EXCESS_THRESHOLD*10000:.0f}bps"
    )
    print(f"  Features: {FEATURES}")
    print(f"  OOS AUC: {test_auc:.4f}")
    print("=" * 140)

    header = (
        f"{'ProbTh':>8}  {'Trades':>8}  {'Days':>8}  {'AvgDay':>8}  "
        f"{'Prec':>8}  {'WinRate':>8}  {'DayWR':>8}  "
        f"{'ShortBps':>10}  {'NetBps':>10}  {'Sharpe':>8}  "
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
            f"{r['avg_short_bps']:>10.2f}  {r['avg_net_bps']:>10.2f}  "
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


# ==================== 信号分布诊断 ====================
def diagnose_signal(test_df: pd.DataFrame):
    """诊断做空模型的信号分布"""
    print("\n" + "=" * 80)
    print("做空模型信号分布诊断")
    print("=" * 80)

    prob = test_df["short_pred_prob"]
    print(f"\n概率分布: min={prob.min():.4f}, Q25={prob.quantile(0.25):.4f}, "
          f"median={prob.median():.4f}, Q75={prob.quantile(0.75):.4f}, max={prob.max():.4f}")

    # 按概率分箱看实际做空收益
    bins = [0.0, 0.30, 0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 1.0]
    test_df["prob_bin"] = pd.cut(prob, bins=bins)

    stats = test_df.groupby("prob_bin", observed=True).agg(
        count=("raw_ret", "size"),
        raw_ret_bps=("raw_ret", lambda x: x.mean() * 10000),
        excess_ret_bps=("excess_ret", lambda x: x.mean() * 10000),
        short_profit_ratio=("is_short_profitable", "mean"),
    )
    # 做空净收益
    stats["short_net_bps"] = -stats["raw_ret_bps"] - COST_BPS

    print(f"\n{'prob_bin':>16} {'count':>10} {'raw_bps':>10} {'excess_bps':>12} "
          f"{'short_net':>11} {'short_prec':>12}")
    print("-" * 80)
    for idx, row in stats.iterrows():
        print(f"{str(idx):>16} {row['count']:>10.0f} {row['raw_ret_bps']:>10.1f} "
              f"{row['excess_ret_bps']:>12.1f} {row['short_net_bps']:>11.1f} "
              f"{row['short_profit_ratio']:>11.1%}")

    # 与做多模型对比
    print(f"\n--- 做空模型高概率端 vs 做多模型 ---")
    for th in [0.45, 0.48, 0.50]:
        subset = test_df[prob > th]
        if len(subset) == 0:
            continue
        short_net = (-subset["raw_ret"] - COST_RATE).mean() * 10000
        excess = subset["excess_ret"].mean() * 10000
        print(f"  short_pred_prob > {th}: {len(subset):>8,} 笔, "
              f"做空NetBps={short_net:>7.1f}, excess_ret={excess:>7.1f} bps")


# ==================== 绘图 ====================
def plot_results(all_results: list, model: lgb.Booster = None):
    """绘制做空回测图表"""
    valid = [r for r in all_results if r["n_trades"] > 0 and "daily_ret" in r]
    if not valid:
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 16))

    # 1. 资金曲线
    for r in valid:
        daily_ret = r["daily_ret"]
        cum_bps = daily_ret.cumsum() * 10000
        label = f"P>{r['prob_threshold']:.2f} (S={r['sharpe']:.2f}, N={r['n_trades']})"
        axes[0].plot(range(len(cum_bps)), cum_bps.values, label=label, alpha=0.8)

        # 日期刻度
        dates = daily_ret.index
        date_strs = [str(d) for d in dates]
        tick_pos, tick_labels = [], []
        seen_months = set()
        for i, d in enumerate(date_strs):
            month_key = d[:7]
            if month_key not in seen_months:
                seen_months.add(month_key)
                tick_pos.append(i)
                tick_labels.append(d[5:10])
        axes[0].set_xticks(tick_pos)
        axes[0].set_xticklabels(tick_labels, rotation=45, fontsize=8)

    axes[0].set_title(
        f"LGBM SHORT Equity Curve (Train={YEAR_TRAIN}, Test={YEAR_TEST}, Cost={COST_BPS}bps)",
        fontsize=12,
    )
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Short Net Return (bps)")
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
    axes[1].set_title("Short Model: Threshold vs Backtest Metrics", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 3. 特征重要性
    if model is not None:
        importance = model.feature_importance(importance_type="gain")
        sorted_idx = np.argsort(importance)
        axes[2].barh(
            [FEATURES[i] for i in sorted_idx],
            importance[sorted_idx],
            color="indianred",
        )
        axes[2].set_title("Short Model: Feature Importance (Gain)", fontsize=12)
        axes[2].set_xlabel("Importance")
    else:
        axes[2].text(
            0.5, 0.5, "Model file not found\nSkipping feature importance",
            ha="center", va="center", fontsize=14, color="gray",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Short Model: Feature Importance (Gain)", fontsize=12)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "lgbm_short_classifier_backtest.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("LightGBM 做空模型回测")
    logger.info("=" * 60)

    # 1. 加载预测结果
    test_df = load_predictions()

    # 计算 OOS AUC
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(test_df["is_short_profitable"], test_df["short_pred_prob"])
    logger.info(f"样本外 AUC: {test_auc:.4f}")

    # 2. 信号分布诊断
    diagnose_signal(test_df)

    # 3. 对每个概率阈值做回测
    all_results = []
    for prob_th in PROB_THRESHOLDS:
        result = backtest_with_threshold(test_df, prob_th)
        result["auc"] = test_auc
        all_results.append(result)

    # 4. 打印报告
    print_report(all_results, test_auc)

    # 5. 加载模型并绘图
    model = load_model()
    plot_results(all_results, model)

    logger.success("做空模型回测完成!")


if __name__ == "__main__":
    main()
