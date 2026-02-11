"""
P0+: 权重微调 + 年化收益 + 多Buffer对比

测试权重: 0.6:0.4, 0.65:0.35, 0.7:0.3, 0.75:0.25, 0.8:0.2
Buffer: 1.0, 1.5, 2.0, 2.5, 3.0
TopN: 20, 50, 100

输出: 年化净收益%, 年化超额%, Sharpe(净/超额), 换手率

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

# 权重列表: (vwap_weight, close_weight, 名称)
WEIGHT_VARIANTS = [
    (1.0, 0.0, "1.0V+0.0C (VWAP基线)"),
    (0.8, 0.2, "0.8V+0.2C"),
    (0.75, 0.25, "0.75V+0.25C"),
    (0.7, 0.3, "0.7V+0.3C"),
    (0.65, 0.35, "0.65V+0.35C"),
    (0.6, 0.4, "0.6V+0.4C"),
    (0.0, 1.0, "0.0V+1.0C (Close基线)"),
]

BUFFER_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]
TOP_NS = [20, 50, 100]


def load_data(year):
    """加载数据并预计算 close_excess_ret"""
    ready = pd.read_pickle(OUTPUT_DIR / f"df_ts_ready_{year}.pkl")
    feats = pd.read_pickle(OUTPUT_DIR / f"df_ts_features_{year}.pkl")

    feats_sub = feats[["SecuCode", "date", "entry_time", "Y_rest"]].copy()
    feats_sub["close_ret"] = 1.0 / feats_sub["Y_rest"] - 1.0

    df = ready.merge(
        feats_sub[["SecuCode", "date", "entry_time", "close_ret"]],
        on=["SecuCode", "date", "entry_time"],
        how="left",
    )
    df["close_market_ret"] = df.groupby("date")["close_ret"].transform("mean")
    df["close_excess_ret"] = df["close_ret"] - df["close_market_ret"]

    mask = df["close_ret"].notna() & np.isfinite(df["close_ret"])
    df = df[mask].copy()
    return df


def train_model(train_df, test_df, vw, cw, name):
    """训练一个模型并返回测试集预测"""
    # 构造标签
    train_label = vw * train_df["excess_ret"] + cw * train_df["close_excess_ret"]

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    train_data = lgb.Dataset(X_train, label=train_label)
    valid_label = vw * test_df["excess_ret"] + cw * test_df["close_excess_ret"]
    valid_data = lgb.Dataset(X_test, label=valid_label, reference=train_data)

    model = lgb.train(
        LGB_PARAMS, train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0),  # 静默训练
        ],
    )

    y_pred = model.predict(X_test)
    result = test_df.copy()
    result["pred_prob"] = y_pred

    logger.info(f"  {name}: 迭代={model.best_iteration}")
    return result


def backtest_vwap(df, top_n, buffer_ratio):
    """VWAP 离场回测"""
    return run_topn_backtest_turnover(df, top_n=top_n, buffer_ratio=buffer_ratio)


def backtest_close(df, top_n, buffer_ratio):
    """Close 离场回测"""
    df_c = df.copy()
    close_mkt = df_c.groupby("date")["close_ret"].transform("mean")
    df_c["raw_ret"] = df_c["close_ret"]
    df_c["excess_ret"] = df_c["close_ret"] - close_mkt
    df_c["market_ret"] = close_mkt
    return run_topn_backtest_turnover(df_c, top_n=top_n, buffer_ratio=buffer_ratio)


def main():
    logger.info("加载数据...")
    train_df = load_data(2024)
    test_df = load_data(2025)

    # ===== 第一步: 训练所有权重模型 =====
    logger.info("=" * 80)
    logger.info("训练所有权重变体...")
    predictions = {}
    for vw, cw, name in WEIGHT_VARIANTS:
        pred = train_model(train_df, test_df, vw, cw, name)
        predictions[name] = pred

    # ===== 第二步: 综合回测 =====

    # ---- 表1: VWAP离场 各权重 × 各Buffer (Top-50) ----
    print("\n\n" + "=" * 140)
    print("表1: VWAP离场 — 各权重 × 各Buffer (Top-50, 年化)")
    print("=" * 140)

    header = f"{'标签':>22s}"
    for br in BUFFER_RATIOS:
        header += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpNet':>6s} {'ShpExc':>6s} {'换手':>5s}"
    header += f" | buf="
    for br in BUFFER_RATIOS:
        header += f"   {br:.1f}x" + " " * 27
    print(header[:22], end="")
    for br in BUFFER_RATIOS:
        print(f" |  buf={br:.1f}x" + " " * 26, end="")
    print()

    sub_header = f"{'':>22s}"
    for _ in BUFFER_RATIOS:
        sub_header += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpN':>5s} {'ShpE':>5s} {'Turn':>5s}"
    print(sub_header)
    print("-" * (22 + len(BUFFER_RATIOS) * 42))

    for vw, cw, name in WEIGHT_VARIANTS:
        row = f"{name:>22s}"
        for br in BUFFER_RATIOS:
            r = backtest_vwap(predictions[name], top_n=50, buffer_ratio=br)
            row += (
                f" | {r['annual_net']:>+7.1f} {r['annual_excess']:>+7.1f}"
                f" {r['sharpe_net']:>5.2f} {r['sharpe_excess']:>5.2f}"
                f" {r['avg_turnover']:>4.0%}"
            )
        print(row)

    # ---- 表2: VWAP离场 各权重 × 各TopN (Buffer=2.0) ----
    print("\n\n" + "=" * 140)
    print("表2: VWAP离场 — 各权重 × 各TopN (Buffer=2.0x, 年化)")
    print("=" * 140)

    sub_header2 = f"{'标签':>22s}"
    for tn in TOP_NS:
        sub_header2 += f" | {'AnnNet%':>7s} {'AnnExc%':>7s} {'ShpN':>5s} {'ShpE':>5s} {'Turn':>5s}"
    print(f"{'':>22s}", end="")
    for tn in TOP_NS:
        print(f" |       Top-{tn:<5d}" + " " * 20, end="")
    print()
    print(sub_header2)
    print("-" * (22 + len(TOP_NS) * 42))

    for vw, cw, name in WEIGHT_VARIANTS:
        row = f"{name:>22s}"
        for tn in TOP_NS:
            r = backtest_vwap(predictions[name], top_n=tn, buffer_ratio=2.0)
            row += (
                f" | {r['annual_net']:>+7.1f} {r['annual_excess']:>+7.1f}"
                f" {r['sharpe_net']:>5.2f} {r['sharpe_excess']:>5.2f}"
                f" {r['avg_turnover']:>4.0%}"
            )
        print(row)

    # ---- 表3: Close离场 各权重 × 各TopN (Buffer=2.0) ----
    print("\n\n" + "=" * 140)
    print("表3: Close离场 — 各权重 × 各TopN (Buffer=2.0x, 年化)")
    print("=" * 140)

    print(f"{'':>22s}", end="")
    for tn in TOP_NS:
        print(f" |       Top-{tn:<5d}" + " " * 20, end="")
    print()
    print(sub_header2)
    print("-" * (22 + len(TOP_NS) * 42))

    for vw, cw, name in WEIGHT_VARIANTS:
        row = f"{name:>22s}"
        for tn in TOP_NS:
            r = backtest_close(predictions[name], top_n=tn, buffer_ratio=2.0)
            row += (
                f" | {r['annual_net']:>+7.1f} {r['annual_excess']:>+7.1f}"
                f" {r['sharpe_net']:>5.2f} {r['sharpe_excess']:>5.2f}"
                f" {r['avg_turnover']:>4.0%}"
            )
        print(row)

    # ---- 表4: 最优权重的详细月度表现 ----
    print("\n\n" + "=" * 140)
    print("表4: Top-50 Buffer=2.0x VWAP离场 — 月度年化净收益对比")
    print("=" * 140)

    # 只挑3个关键权重
    key_names = ["1.0V+0.0C (VWAP基线)", "0.7V+0.3C", "0.75V+0.25C"]
    for name in key_names:
        df_bt = predictions[name].copy()
        df_bt["rank"] = df_bt.groupby("date")["pred_prob"].rank(ascending=False, method="first")
        dates = sorted(df_bt["date"].unique())
        prev_holdings = set()
        daily_recs = []
        buffer_n = 100

        for date in dates:
            day = df_bt[df_bt["date"] == date]
            if len(prev_holdings) > 0:
                kept_mask = day["SecuCode"].isin(prev_holdings) & (day["rank"] <= buffer_n)
                kept = set(day.loc[kept_mask, "SecuCode"])
                if len(kept) > 50:
                    kept = set(day[day["SecuCode"].isin(kept)].nsmallest(50, "rank")["SecuCode"])
                n_fill = 50 - len(kept)
                if n_fill > 0:
                    fills = set(day[(day["rank"] <= 50) & (~day["SecuCode"].isin(kept))].nsmallest(n_fill, "rank")["SecuCode"])
                else:
                    fills = set()
                holdings = kept | fills
                n_new = len(holdings - prev_holdings)
            else:
                holdings = set(day[day["rank"] <= 50]["SecuCode"])
                n_new = len(holdings)

            port = day[day["SecuCode"].isin(holdings)]
            if len(port) > 0:
                turnover = n_new / len(holdings)
                cost = turnover * 0.0015
                daily_recs.append({
                    "date": pd.to_datetime(date),
                    "net_ret": port["raw_ret"].mean() - cost,
                    "excess_ret": port["excess_ret"].mean(),
                })
            prev_holdings = holdings

        dr = pd.DataFrame(daily_recs)
        dr["month"] = dr["date"].dt.to_period("M")
        monthly = dr.groupby("month").agg(
            net_mean=("net_ret", "mean"),
            exc_mean=("excess_ret", "mean"),
            days=("net_ret", "count"),
        )
        monthly["ann_net"] = monthly["net_mean"] * 242 * 100
        monthly["ann_exc"] = monthly["exc_mean"] * 242 * 100

        print(f"\n  {name}:")
        print(f"  {'月份':>8s} {'交易日':>5s} {'月均Net bps':>11s} {'年化Net%':>8s} {'年化Exc%':>8s}")
        for m, row in monthly.iterrows():
            bar_n = "+" * max(0, int(row["ann_net"] / 5)) if row["ann_net"] > 0 else "-" * max(0, int(-row["ann_net"] / 5))
            print(f"  {str(m):>8s} {int(row['days']):>5d} {row['net_mean']*10000:>+11.2f} {row['ann_net']:>+8.1f} {row['ann_exc']:>+8.1f}  {bar_n}")
        total_ann_net = dr["net_ret"].mean() * 242 * 100
        total_ann_exc = dr["excess_ret"].mean() * 242 * 100
        total_shp = dr["net_ret"].mean() / dr["net_ret"].std() * np.sqrt(242) if dr["net_ret"].std() > 0 else 0
        print(f"  {'全年':>8s} {len(dr):>5d} {dr['net_ret'].mean()*10000:>+11.2f} {total_ann_net:>+8.1f} {total_ann_exc:>+8.1f}  Sharpe={total_shp:.2f}")

    logger.success("\nP0+ 微调权重实验完成!")


if __name__ == "__main__":
    main()
