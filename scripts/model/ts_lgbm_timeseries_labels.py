"""
Step 1: 纯时序标签 vs 截面标签 对比实验

标签变体:
  TS-A: raw_ret (绝对收益，不减beta)
  TS-B: raw_ret / hist_vol_20d (波动率标准化绝对收益)
  TS-C: raw_ret - self_mean_ret_20d (减去自身均值)
  CS-Base: excess_ret (减beta，截面基线)
  CS-Best: 0.7*vwap_exc + 0.3*close_exc (P0冠军)

同一套特征、同一套模型参数，只换标签，统一回测对比

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
from ts_portfolio_backtest import run_topn_backtest_turnover

# ==================== 配置 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

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


def load_data(year):
    """加载数据"""
    ready = pd.read_pickle(OUTPUT_DIR / f"df_ts_ready_{year}.pkl")
    feats = pd.read_pickle(OUTPUT_DIR / f"df_ts_features_{year}.pkl")

    # 合并 close_ret (来自 feats 的 Y_rest)
    feats_sub = feats[["SecuCode", "date", "entry_time", "Y_rest"]].copy()
    feats_sub["close_ret"] = 1.0 / feats_sub["Y_rest"] - 1.0

    df = ready.merge(
        feats_sub[["SecuCode", "date", "entry_time", "close_ret"]],
        on=["SecuCode", "date", "entry_time"],
        how="left",
    )
    df["close_market_ret"] = df.groupby("date")["close_ret"].transform("mean")
    df["close_excess_ret"] = df["close_ret"] - df["close_market_ret"]

    # 计算自身历史均值 (用于TS-C标签)
    df = df.sort_values(["SecuCode", "date"])
    df["self_mean_ret_20d"] = (
        df.groupby("SecuCode")["raw_ret"]
        .transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))
    )

    mask = (
        df["raw_ret"].notna() & np.isfinite(df["raw_ret"])
        & df["close_ret"].notna() & np.isfinite(df["close_ret"])
        & df["hist_vol_20d"].notna() & (df["hist_vol_20d"] > 0)
    )
    df = df[mask].copy()
    logger.info(f"  {year}: {len(df):,} 行")
    return df


def build_labels(df):
    """构建所有标签变体"""
    labels = {}

    # TS-A: 绝对收益
    labels["TS-A: raw_ret"] = df["raw_ret"].copy()

    # TS-B: 波动率标准化绝对收益
    labels["TS-B: raw/vol"] = (df["raw_ret"] / df["hist_vol_20d"]).copy()

    # TS-C: 减去自身均值
    ts_c = (df["raw_ret"] - df["self_mean_ret_20d"]).copy()
    labels["TS-C: raw-self_mean"] = ts_c

    # CS-Base: 超额收益 (减beta)
    labels["CS-Base: excess"] = df["excess_ret"].copy()

    # CS-Best: 0.7*vwap_exc + 0.3*close_exc
    labels["CS-Best: 0.7V+0.3C"] = (
        0.7 * df["excess_ret"] + 0.3 * df["close_excess_ret"]
    ).copy()

    return labels


def train_and_evaluate(train_df, test_df, label_name, train_labels, test_labels):
    """训练模型并评估"""
    # 过滤有效标签
    train_mask = train_labels.notna() & np.isfinite(train_labels)
    test_mask = test_labels.notna() & np.isfinite(test_labels)

    X_train = train_df.loc[train_mask, FEATURES]
    y_train = train_labels[train_mask]
    X_test = test_df.loc[test_mask, FEATURES]
    y_test = test_labels[test_mask]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        LGB_PARAMS, train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),
        ],
    )

    # 预测
    y_pred = model.predict(X_test)

    result = test_df.loc[test_mask].copy()
    result["pred_prob"] = y_pred

    # Rank IC (对VWAP excess, Close excess, 自身标签)
    dates = result["date"].unique()
    ic_vwap, ic_close, ic_self = [], [], []
    for d in dates:
        day = result[result["date"] == d]
        if len(day) < 30:
            continue
        pred = day["pred_prob"].values
        ic_v, _ = spearmanr(pred, day["excess_ret"].values)
        if not np.isnan(ic_v):
            ic_vwap.append(ic_v)

        if "close_excess_ret" in day.columns:
            ic_c, _ = spearmanr(pred, day["close_excess_ret"].values)
            if not np.isnan(ic_c):
                ic_close.append(ic_c)

        ic_s, _ = spearmanr(pred, y_test.loc[day.index].values)
        if not np.isnan(ic_s):
            ic_self.append(ic_s)

    mean_ic_vwap = np.mean(ic_vwap) if ic_vwap else 0
    mean_ic_close = np.mean(ic_close) if ic_close else 0
    mean_ic_self = np.mean(ic_self) if ic_self else 0

    # 分位分析
    result["decile"] = result.groupby("date")["pred_prob"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    decile_stats = result.groupby("decile").agg(
        vwap_exc=("excess_ret", "mean"),
        close_exc=("close_excess_ret", "mean"),
        raw=("raw_ret", "mean"),
    )
    top_decile_vwap = decile_stats["vwap_exc"].iloc[-1] * 10000
    top_decile_close = decile_stats["close_exc"].iloc[-1] * 10000
    d0_vwap = decile_stats["vwap_exc"].iloc[0] * 10000
    spread = top_decile_vwap - d0_vwap

    # 特征重要性Top3
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
    imp = imp / imp.sum()
    top3 = imp.nlargest(3)
    top3_str = ", ".join([f"{k}:{v:.2f}" for k, v in top3.items()])

    return {
        "label": label_name,
        "best_iter": model.best_iteration,
        "ic_vwap": mean_ic_vwap,
        "ic_close": mean_ic_close,
        "ic_self": mean_ic_self,
        "top_d_vwap": top_decile_vwap,
        "top_d_close": top_decile_close,
        "d0_vwap": d0_vwap,
        "spread": spread,
        "top3_feat": top3_str,
        "predictions": result,
    }


def backtest_vwap(pred_df, top_n, buffer_ratio):
    """VWAP离场回测"""
    return run_topn_backtest_turnover(pred_df, top_n=top_n, buffer_ratio=buffer_ratio)


def backtest_close(pred_df, top_n, buffer_ratio):
    """Close离场回测"""
    df_c = pred_df.copy()
    close_mkt = df_c.groupby("date")["close_ret"].transform("mean")
    df_c["raw_ret"] = df_c["close_ret"]
    df_c["excess_ret"] = df_c["close_ret"] - close_mkt
    df_c["market_ret"] = close_mkt
    return run_topn_backtest_turnover(df_c, top_n=top_n, buffer_ratio=buffer_ratio)


def main():
    logger.info("加载数据...")
    train_df = load_data(2024)
    test_df = load_data(2025)

    train_labels = build_labels(train_df)
    test_labels = build_labels(test_df)

    # ===== 训练并评估所有标签 =====
    results = {}
    for label_name in train_labels:
        logger.info(f"\n训练: {label_name}")
        r = train_and_evaluate(
            train_df, test_df,
            label_name,
            train_labels[label_name],
            test_labels[label_name],
        )
        results[label_name] = r

    # ===== 表1: 信号质量对比 =====
    print("\n" + "=" * 140)
    print("表1: 信号质量 — 时序标签 vs 截面标签")
    print("=" * 140)
    header = (
        f"{'标签':>25s} | {'迭代':>4s} | {'IC(VWAP)':>8s} {'IC(Close)':>9s} "
        f"{'IC(自身)':>8s} | {'TopD_VWAP':>9s} {'TopD_Close':>10s} "
        f"{'D0_VWAP':>7s} {'Spread':>6s} | {'Top3特征'}"
    )
    print(header)
    print("-" * 140)
    for name, r in results.items():
        print(
            f"{name:>25s} | {r['best_iter']:>4d} | "
            f"{r['ic_vwap']:>+8.4f} {r['ic_close']:>+9.4f} "
            f"{r['ic_self']:>+8.4f} | "
            f"{r['top_d_vwap']:>+9.2f} {r['top_d_close']:>+10.2f} "
            f"{r['d0_vwap']:>+7.2f} {r['spread']:>+6.1f} | "
            f"{r['top3_feat']}"
        )

    # ===== 表2: VWAP离场回测 (Top-50, 多buffer) =====
    print("\n" + "=" * 140)
    print("表2: VWAP离场回测 — 年化收益 (Top-50)")
    print("=" * 140)

    buffers = [1.5, 2.0, 2.5, 3.0]
    sub_h = f"{'标签':>25s}"
    for br in buffers:
        sub_h += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpN':>5s} {'ShpE':>5s} {'Turn':>5s}"
    print(f"{'':>25s}", end="")
    for br in buffers:
        print(f" |       buf={br:.1f}x" + " " * 18, end="")
    print()
    print(sub_h)
    print("-" * (25 + len(buffers) * 42))

    for name, r in results.items():
        row = f"{name:>25s}"
        for br in buffers:
            bt = backtest_vwap(r["predictions"], top_n=50, buffer_ratio=br)
            row += (
                f" | {bt['annual_net']:>+7.1f} {bt['annual_excess']:>+7.1f}"
                f" {bt['sharpe_net']:>5.2f} {bt['sharpe_excess']:>5.2f}"
                f" {bt['avg_turnover']:>4.0%}"
            )
        print(row)

    # ===== 表3: Close离场回测 (Top-50, Buffer=2.0) =====
    print("\n" + "=" * 140)
    print("表3: Close离场回测 — 年化收益 (Top-50, Buffer=2.0x)")
    print("=" * 140)

    top_ns = [20, 50, 100]
    sub_h3 = f"{'标签':>25s}"
    for tn in top_ns:
        sub_h3 += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpN':>5s} {'ShpE':>5s}"
    print(f"{'':>25s}", end="")
    for tn in top_ns:
        print(f" |       Top-{tn:<5d}" + " " * 16, end="")
    print()
    print(sub_h3)
    print("-" * (25 + len(top_ns) * 38))

    for name, r in results.items():
        row = f"{name:>25s}"
        for tn in top_ns:
            bt = backtest_close(r["predictions"], top_n=tn, buffer_ratio=2.0)
            row += (
                f" | {bt['annual_net']:>+7.1f} {bt['annual_excess']:>+7.1f}"
                f" {bt['sharpe_net']:>5.2f} {bt['sharpe_excess']:>5.2f}"
            )
        print(row)

    # ===== 表4: VWAP离场 各TopN (Buffer=2.0) =====
    print("\n" + "=" * 140)
    print("表4: VWAP离场回测 — 各TopN (Buffer=2.0x)")
    print("=" * 140)

    print(f"{'':>25s}", end="")
    for tn in top_ns:
        print(f" |       Top-{tn:<5d}" + " " * 20, end="")
    print()

    sub_h4 = f"{'标签':>25s}"
    for tn in top_ns:
        sub_h4 += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpN':>5s} {'ShpE':>5s} {'Turn':>5s}"
    print(sub_h4)
    print("-" * (25 + len(top_ns) * 42))

    for name, r in results.items():
        row = f"{name:>25s}"
        for tn in top_ns:
            bt = backtest_vwap(r["predictions"], top_n=tn, buffer_ratio=2.0)
            row += (
                f" | {bt['annual_net']:>+7.1f} {bt['annual_excess']:>+7.1f}"
                f" {bt['sharpe_net']:>5.2f} {bt['sharpe_excess']:>5.2f}"
                f" {bt['avg_turnover']:>4.0%}"
            )
        print(row)

    # ===== 表5: 预测值分布分析 — 时序标签的防御特性 =====
    print("\n" + "=" * 140)
    print("表5: 预测值分布 — 时序标签的防御特性")
    print("=" * 140)
    print(f"{'标签':>25s} | {'pred_mean':>9s} {'pred_std':>8s} | {'日均pred>0比例':>14s} | {'最差5天pred均值':>14s} {'最好5天pred均值':>14s}")
    print("-" * 110)

    for name, r in results.items():
        pred = r["predictions"]
        # 每日平均预测值
        daily_pred_mean = pred.groupby("date")["pred_prob"].mean()
        # pred > 0 的比例（按天）
        daily_pos_pct = pred.groupby("date")["pred_prob"].apply(lambda x: (x > 0).mean())

        # 市场最差5天
        daily_mkt = pred.groupby("date")["market_ret"].first().sort_values()
        worst_5 = daily_mkt.head(5).index
        best_5 = daily_mkt.tail(5).index

        worst_pred_mean = daily_pred_mean.loc[worst_5].mean()
        best_pred_mean = daily_pred_mean.loc[best_5].mean()

        print(
            f"{name:>25s} | "
            f"{pred['pred_prob'].mean():>+9.6f} {pred['pred_prob'].std():>8.6f} | "
            f"{daily_pos_pct.mean():>13.1%} | "
            f"{worst_pred_mean:>+14.6f} {best_pred_mean:>+14.6f}"
        )

    logger.success("\nStep 1 纯时序标签实验完成!")


if __name__ == "__main__":
    main()
