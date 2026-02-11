"""
P0: 鲁棒标签实验 — 三种标签变体对比

变体:
  A) min(exc_vwap, exc_close)  — 保守标签，只奖励VWAP和Close都涨的
  B) (exc_vwap + exc_close) / 2 — 中间标签，平均收益
  C) 0.7*exc_vwap + 0.3*exc_close — 偏VWAP加权标签

基线:
  原版 VWAP 标签 (excess_ret)
  Close 标签 (close_excess_ret)

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
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "backtest"))
from ts_portfolio_backtest import run_topn_backtest, run_topn_backtest_turnover

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


# ==================== 数据加载 ====================
def load_data(year: int) -> pd.DataFrame:
    """加载 ready 数据并添加 close_ret 和各种混合标签"""
    ready = pd.read_pickle(OUTPUT_DIR / f"df_ts_ready_{year}.pkl")
    feats = pd.read_pickle(OUTPUT_DIR / f"df_ts_features_{year}.pkl")

    # 合并 Y_rest → 计算 close_ret
    feats_sub = feats[["SecuCode", "date", "entry_time", "Y_rest"]].copy()
    feats_sub["close_ret"] = 1.0 / feats_sub["Y_rest"] - 1.0

    df = ready.merge(
        feats_sub[["SecuCode", "date", "entry_time", "close_ret"]],
        on=["SecuCode", "date", "entry_time"],
        how="left",
    )

    # close 超额
    df["close_market_ret"] = df.groupby("date")["close_ret"].transform("mean")
    df["close_excess_ret"] = df["close_ret"] - df["close_market_ret"]

    # 三种鲁棒标签
    df["label_min"] = np.minimum(df["excess_ret"], df["close_excess_ret"])
    df["label_avg"] = (df["excess_ret"] + df["close_excess_ret"]) / 2
    df["label_w73"] = 0.7 * df["excess_ret"] + 0.3 * df["close_excess_ret"]

    # 清除 NaN
    mask = df["close_ret"].notna() & np.isfinite(df["close_ret"])
    df = df[mask].copy()

    logger.success(f"加载 {year} 年: {len(df):,} 行")
    return df


# ==================== 单次训练+评估 ====================
def train_and_evaluate(train_df, test_df, label_col, label_name):
    """训练一个模型并返回评估结果"""
    logger.info(f"\n{'='*80}")
    logger.info(f"训练标签: {label_name} ({label_col})")
    logger.info(f"{'='*80}")

    X_train = train_df[FEATURES]
    y_train = train_df[label_col]
    X_test = test_df[FEATURES]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(
        X_test, label=test_df[label_col], reference=train_data
    )

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=100),
        ],
    )

    logger.info(f"最佳迭代: {model.best_iteration}")

    # 特征重要性 Top-5
    importance = model.feature_importance(importance_type="gain")
    sorted_fi = sorted(zip(FEATURES, importance), key=lambda x: -x[1])
    logger.info("特征重要性 Top-5:")
    for feat, imp in sorted_fi[:5]:
        logger.info(f"  {feat:>20s}: {imp:.4f}")

    # 预测
    y_pred = model.predict(X_test)
    test_result = test_df.copy()
    test_result["pred_prob"] = y_pred

    # Rank IC (对多种目标)
    results = {"name": label_name, "best_iter": model.best_iteration}

    for target_name, target_col in [
        ("VWAP超额", "excess_ret"),
        ("Close超额", "close_excess_ret"),
        ("标签自身", label_col),
    ]:
        daily_ic = test_result.groupby("date").apply(
            lambda g: spearmanr(g["pred_prob"], g[target_col])[0]
            if len(g) > 10 else np.nan,
            include_groups=False,
        ).dropna()
        results[f"IC_{target_name}"] = daily_ic.mean()
        results[f"IC_{target_name}_std"] = daily_ic.std()

    # 分位分析
    test_result["pred_decile"] = pd.qcut(
        y_pred, 10, labels=False, duplicates="drop"
    )
    d_max = test_result["pred_decile"].max()
    top_decile = test_result[test_result["pred_decile"] == d_max]
    results["top_decile_vwap_exc"] = top_decile["excess_ret"].mean() * 10000
    results["top_decile_close_exc"] = top_decile["close_excess_ret"].mean() * 10000
    results["top_decile_close_ret"] = top_decile["close_ret"].mean() * 10000

    # 分位单调性
    logger.info("分位 vs 收益:")
    for d in sorted(test_result["pred_decile"].unique()):
        g = test_result[test_result["pred_decile"] == d]
        logger.info(
            f"  D{d}: VWAP_exc={g['excess_ret'].mean()*10000:>+6.2f}, "
            f"Close_exc={g['close_excess_ret'].mean()*10000:>+6.2f}, "
            f"raw={g['raw_ret'].mean()*10000:>+6.2f}, "
            f"close_ret={g['close_ret'].mean()*10000:>+6.2f}"
        )

    return test_result, results, model


# ==================== 回测对比 ====================
def run_all_backtests(test_result, label_name):
    """对一个模型的预测结果，分别用 VWAP 和 Close 离场回测"""
    bt_results = {}

    # VWAP 离场（原版 raw_ret / excess_ret）
    for top_n in [20, 50, 100]:
        r = run_topn_backtest_turnover(test_result, top_n=top_n, buffer_ratio=2.0)
        bt_results[f"VWAP_Top{top_n}_net"] = r["avg_net_bps"]
        bt_results[f"VWAP_Top{top_n}_exc"] = r["avg_excess_bps"]
        bt_results[f"VWAP_Top{top_n}_shp_net"] = r["sharpe_net"]
        bt_results[f"VWAP_Top{top_n}_shp_exc"] = r["sharpe_excess"]
        bt_results[f"VWAP_Top{top_n}_turnover"] = r["avg_turnover"]

    # Close 离场
    df_c = test_result.copy()
    close_mkt = df_c.groupby("date")["close_ret"].transform("mean")
    df_c["raw_ret"] = df_c["close_ret"]
    df_c["excess_ret"] = df_c["close_ret"] - close_mkt
    df_c["market_ret"] = close_mkt

    for top_n in [20, 50, 100]:
        r = run_topn_backtest_turnover(df_c, top_n=top_n, buffer_ratio=2.0)
        bt_results[f"Close_Top{top_n}_net"] = r["avg_net_bps"]
        bt_results[f"Close_Top{top_n}_exc"] = r["avg_excess_bps"]
        bt_results[f"Close_Top{top_n}_shp_net"] = r["sharpe_net"]
        bt_results[f"Close_Top{top_n}_shp_exc"] = r["sharpe_excess"]

    return bt_results


# ==================== 主函数 ====================
def main():
    logger.info("=" * 80)
    logger.info("P0: 鲁棒标签实验 — 三种标签变体 + 两个基线")
    logger.info("=" * 80)

    # 1. 加载数据
    train_df = load_data(YEAR_TRAIN)
    test_df = load_data(YEAR_TEST)

    # 2. 定义所有标签变体
    variants = [
        ("excess_ret", "基线A: VWAP标签"),
        ("close_excess_ret", "基线B: Close标签"),
        ("label_min", "S11a: min(VWAP,Close)"),
        ("label_avg", "S11b: avg(VWAP,Close)"),
        ("label_w73", "S11c: 0.7V+0.3C"),
    ]

    all_results = []
    all_bt = []

    for label_col, label_name in variants:
        # 训练+评估
        test_result, eval_results, model = train_and_evaluate(
            train_df, test_df, label_col, label_name
        )

        # 回测
        bt = run_all_backtests(test_result, label_name)
        bt["name"] = label_name

        all_results.append(eval_results)
        all_bt.append(bt)

        # 保存预测结果
        safe_name = label_col.replace("excess_ret", "vwap").replace("close_excess_ret", "close")
        pred_path = OUTPUT_DIR / f"ts_lgbm_predictions_{safe_name}.pkl"
        save_cols = (
            ["SecuCode", "date", "entry_time"]
            + FEATURES
            + ["raw_ret", "market_ret", "excess_ret",
               "close_ret", "close_excess_ret", "pred_prob"]
        )
        test_result[save_cols].to_pickle(pred_path)

    # 3. 汇总对比表
    print("\n\n" + "=" * 140)
    print("P0 综合对比: 信号质量")
    print("=" * 140)

    header = (
        f"{'标签':>25s} | {'迭代':>4s} | "
        f"{'IC(VWAP)':>8s} {'IC(Close)':>9s} {'IC(自身)':>8s} | "
        f"{'TopD VWAP':>9s} {'TopD Close':>10s}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['name']:>25s} | {r['best_iter']:>4d} | "
            f"{r['IC_VWAP超额']:>+8.4f} {r['IC_Close超额']:>+9.4f} {r['IC_标签自身']:>+8.4f} | "
            f"{r['top_decile_vwap_exc']:>+9.2f} {r['top_decile_close_exc']:>+10.2f}"
        )

    print("\n\n" + "=" * 140)
    print("P0 综合对比: VWAP离场回测 (2x缓冲)")
    print("=" * 140)

    header2 = (
        f"{'标签':>25s} | "
        f"{'T20 Net':>7s} {'T20 Exc':>7s} {'T20 ShpE':>8s} | "
        f"{'T50 Net':>7s} {'T50 Exc':>7s} {'T50 ShpE':>8s} | "
        f"{'T100 Net':>8s} {'T100 Exc':>8s} {'T100 ShpE':>9s} | "
        f"{'T50换手':>7s}"
    )
    print(header2)
    print("-" * 140)
    for bt in all_bt:
        print(
            f"{bt['name']:>25s} | "
            f"{bt['VWAP_Top20_net']:>+7.2f} {bt['VWAP_Top20_exc']:>+7.2f} {bt['VWAP_Top20_shp_exc']:>8.2f} | "
            f"{bt['VWAP_Top50_net']:>+7.2f} {bt['VWAP_Top50_exc']:>+7.2f} {bt['VWAP_Top50_shp_exc']:>8.2f} | "
            f"{bt['VWAP_Top100_net']:>+8.2f} {bt['VWAP_Top100_exc']:>+8.2f} {bt['VWAP_Top100_shp_exc']:>9.2f} | "
            f"{bt['VWAP_Top50_turnover']:>6.1%}"
        )

    print("\n\n" + "=" * 140)
    print("P0 综合对比: Close离场回测 (2x缓冲)")
    print("=" * 140)

    header3 = (
        f"{'标签':>25s} | "
        f"{'T20 Net':>7s} {'T20 Exc':>7s} {'T20 ShpE':>8s} | "
        f"{'T50 Net':>7s} {'T50 Exc':>7s} {'T50 ShpE':>8s} | "
        f"{'T100 Net':>8s} {'T100 Exc':>8s} {'T100 ShpE':>9s}"
    )
    print(header3)
    print("-" * 120)
    for bt in all_bt:
        print(
            f"{bt['name']:>25s} | "
            f"{bt['Close_Top20_net']:>+7.2f} {bt['Close_Top20_exc']:>+7.2f} {bt['Close_Top20_shp_exc']:>8.2f} | "
            f"{bt['Close_Top50_net']:>+7.2f} {bt['Close_Top50_exc']:>+7.2f} {bt['Close_Top50_shp_exc']:>8.2f} | "
            f"{bt['Close_Top100_net']:>+8.2f} {bt['Close_Top100_exc']:>+8.2f} {bt['Close_Top100_shp_exc']:>9.2f}"
        )

    logger.success("\nP0 鲁棒标签实验完成!")


if __name__ == "__main__":
    main()
