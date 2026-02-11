"""
时序模型结果深度分析脚本

职责：对模型预测结果做多维度分析，挖掘有用信息
输入：ts_lgbm_predictions.pkl
输出：终端打印报告

分析维度：
  1. 预测概率分位数 — 按 pred_prob 分10组，看各组实际收益
  2. 每日 Top-N 选股 — 每天选 pred_prob 最高的 N 只，计算日均超额
  3. 按月收益曲线 — 月度汇总看稳定性
  4. 特征贡献 — 高得分股票 vs 低得分股票的特征差异
  5. 条件分析 — 什么市场环境下模型表现好/差

依赖：ts_lgbm_train.py 生产的预测数据

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置参数 ====================
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

# 每日选股数量列表
TOP_N_LIST = [20, 50, 100, 200]

# 特征列表
FEATURES = [
    "X1", "X2", "X1_zscore", "X2_zscore",
    "X1_lag1", "X1_lag3", "X1_lag6", "X1_lag10",
    "X2_lag1", "X2_lag3", "X2_lag6",
    "X1_diff1", "X1_diff3",
    "X1_slope_6bar", "X1_morning_std",
    "rel_vol", "vol_accel", "vol_concentration",
    "morning_ret", "price_position", "morning_range", "intraday_vol",
    "overnight_gap", "prev_day_ret", "momentum_5d", "hist_vol_20d",
]


# ==================== 数据加载 ====================
def load_predictions() -> pd.DataFrame:
    """加载预测结果"""
    pkl_path = OUTPUT_DIR / "ts_lgbm_predictions.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"预测数据不存在: {pkl_path}\n请先运行 ts_lgbm_train.py")
    df = pd.read_pickle(pkl_path)
    logger.success(f"加载预测数据: {len(df):,} 行, {df['date'].nunique()} 天")
    return df


# ==================== 分析1: 预测概率分位数 ====================
def analyze_decile(df: pd.DataFrame):
    """
    按 pred_prob 分10组，观察各组的实际收益

    如果模型有效，高概率组应该有更高的超额收益
    """
    print("\n" + "=" * 90)
    print("分析1: 预测概率分位数（全样本按 pred_prob 分10组）")
    print("=" * 90)

    df = df.copy()
    df["decile"] = pd.qcut(df["pred_prob"], 10, labels=False, duplicates="drop")

    stats = df.groupby("decile").agg(
        count=("raw_ret", "size"),
        prob_mean=("pred_prob", "mean"),
        prob_min=("pred_prob", "min"),
        prob_max=("pred_prob", "max"),
        raw_bps=("raw_ret", lambda x: x.mean() * 10000),
        excess_bps=("excess_ret", lambda x: x.mean() * 10000),
        win_rate=("raw_ret", lambda x: (x > 0).mean()),
        label_rate=("label", "mean"),
    )

    header = (
        f"  {'组':>4s}  {'样本数':>8s}  {'概率范围':>20s}  "
        f"{'raw_bps':>10s}  {'excess_bps':>12s}  {'胜率':>8s}  {'正类率':>8s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 80)

    for grp_id, row in stats.iterrows():
        prob_range = f"[{row['prob_min']:.3f}, {row['prob_max']:.3f}]"
        print(
            f"  {grp_id:>4.0f}  {row['count']:>8,.0f}  {prob_range:>20s}  "
            f"{row['raw_bps']:>10.2f}  {row['excess_bps']:>12.2f}  "
            f"{row['win_rate']:>7.1%}  {row['label_rate']:>7.1%}"
        )

    # 头尾差
    if len(stats) >= 2:
        spread = stats.iloc[-1]["excess_bps"] - stats.iloc[0]["excess_bps"]
        print(f"\n  头尾组 excess 差: {spread:+.2f} bps")
        # 多空组合
        long_short = (stats.iloc[-1]["raw_bps"] - stats.iloc[0]["raw_bps"]) / 2
        print(f"  多空组合 raw: {long_short:+.2f} bps/天")


# ==================== 分析2: 每日 Top-N 选股 ====================
def analyze_daily_topn(df: pd.DataFrame):
    """
    每天选 pred_prob 最高的 N 只股票

    计算日均超额收益、胜率、年化收益
    """
    print("\n" + "=" * 90)
    print("分析2: 每日 Top-N 选股表现")
    print("=" * 90)

    header = (
        f"  {'Top-N':>8s}  {'交易天数':>8s}  {'日均raw_bps':>12s}  "
        f"{'日均excess_bps':>14s}  {'日胜率':>8s}  {'年化excess%':>12s}  {'Sharpe':>8s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 80)

    for top_n in TOP_N_LIST:
        # 每天取 pred_prob 最高的 top_n 只
        daily_top = (
            df.sort_values(["date", "pred_prob"], ascending=[True, False])
            .groupby("date")
            .head(top_n)
        )

        # 日度汇总
        daily_ret = daily_top.groupby("date").agg(
            raw_ret=("raw_ret", "mean"),
            excess_ret=("excess_ret", "mean"),
        )

        n_days = len(daily_ret)
        avg_raw = daily_ret["raw_ret"].mean() * 10000
        avg_excess = daily_ret["excess_ret"].mean() * 10000
        win_rate = (daily_ret["excess_ret"] > 0).mean()
        annual_excess = daily_ret["excess_ret"].mean() * 242 * 100
        sharpe = (
            daily_ret["excess_ret"].mean() / daily_ret["excess_ret"].std() * np.sqrt(242)
            if daily_ret["excess_ret"].std() > 0 else 0
        )

        print(
            f"  {top_n:>8d}  {n_days:>8d}  {avg_raw:>12.2f}  "
            f"{avg_excess:>14.2f}  {win_rate:>7.1%}  "
            f"{annual_excess:>11.1f}%  {sharpe:>8.2f}"
        )


# ==================== 分析3: 月度表现 ====================
def analyze_monthly(df: pd.DataFrame):
    """
    按月汇总 Top-50 选股的超额收益

    观察模型在不同月份的稳定性
    """
    print("\n" + "=" * 90)
    print("分析3: 月度表现（每日 Top-50 选股）")
    print("=" * 90)

    top_n = 50
    daily_top = (
        df.sort_values(["date", "pred_prob"], ascending=[True, False])
        .groupby("date")
        .head(top_n)
    )

    daily_ret = daily_top.groupby("date").agg(
        excess_ret=("excess_ret", "mean"),
        raw_ret=("raw_ret", "mean"),
    )
    daily_ret["month"] = pd.to_datetime(daily_ret.index).to_period("M")

    monthly = daily_ret.groupby("month").agg(
        n_days=("excess_ret", "size"),
        excess_bps=("excess_ret", lambda x: x.mean() * 10000),
        cum_excess=("excess_ret", lambda x: x.sum() * 10000),
        raw_bps=("raw_ret", lambda x: x.mean() * 10000),
        win_rate=("excess_ret", lambda x: (x > 0).mean()),
    )

    header = (
        f"  {'月份':>10s}  {'交易日':>6s}  {'日均excess_bps':>14s}  "
        f"{'月累计bps':>10s}  {'日均raw_bps':>12s}  {'日胜率':>8s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 70)

    for month, row in monthly.iterrows():
        print(
            f"  {str(month):>10s}  {row['n_days']:>6.0f}  {row['excess_bps']:>14.2f}  "
            f"{row['cum_excess']:>10.1f}  {row['raw_bps']:>12.2f}  "
            f"{row['win_rate']:>7.1%}"
        )

    # 汇总
    total_excess = monthly["cum_excess"].sum()
    avg_excess = daily_ret["excess_ret"].mean() * 10000
    print(f"\n  全年累计 excess: {total_excess:.1f} bps, 日均: {avg_excess:.2f} bps")


# ==================== 分析4: 高得分 vs 低得分特征差异 ====================
def analyze_feature_profile(df: pd.DataFrame):
    """
    比较预测得分最高/最低 10% 股票的特征均值

    揭示模型认为什么样的股票会涨
    """
    print("\n" + "=" * 90)
    print("分析4: 高得分 vs 低得分股票特征画像")
    print("=" * 90)

    q90 = df["pred_prob"].quantile(0.9)
    q10 = df["pred_prob"].quantile(0.1)

    high = df[df["pred_prob"] >= q90]
    low = df[df["pred_prob"] <= q10]

    header = (
        f"  {'特征':>20s}  {'高得分(Top10%)':>16s}  {'低得分(Bot10%)':>16s}  "
        f"{'差异':>10s}  {'方向':>6s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 76)

    feat_cols = [c for c in FEATURES if c in df.columns]
    for feat in feat_cols:
        h_mean = high[feat].mean()
        l_mean = low[feat].mean()
        diff = h_mean - l_mean
        direction = "↑" if diff > 0 else "↓"
        print(
            f"  {feat:>20s}  {h_mean:>16.4f}  {l_mean:>16.4f}  "
            f"{diff:>+10.4f}  {direction:>6s}"
        )

    # 实际收益对比
    print(f"\n  高得分组 excess: {high['excess_ret'].mean()*10000:+.2f} bps")
    print(f"  低得分组 excess: {low['excess_ret'].mean()*10000:+.2f} bps")
    print(f"  差值: {(high['excess_ret'].mean()-low['excess_ret'].mean())*10000:+.2f} bps")


# ==================== 分析5: 市场环境条件分析 ====================
def analyze_market_condition(df: pd.DataFrame):
    """
    在不同市场环境下，模型的选股效果

    按当日市场收益分组：市场上涨日 vs 市场下跌日
    """
    print("\n" + "=" * 90)
    print("分析5: 不同市场环境下的模型表现（每日 Top-50）")
    print("=" * 90)

    top_n = 50
    daily_top = (
        df.sort_values(["date", "pred_prob"], ascending=[True, False])
        .groupby("date")
        .head(top_n)
    )

    daily_ret = daily_top.groupby("date").agg(
        excess_ret=("excess_ret", "mean"),
        raw_ret=("raw_ret", "mean"),
        market_ret=("market_ret", "first"),
    )

    # 按市场涨跌分组
    conditions = {
        "大涨(>50bps)": daily_ret["market_ret"] > 0.005,
        "小涨(0~50bps)": (daily_ret["market_ret"] >= 0) & (daily_ret["market_ret"] <= 0.005),
        "小跌(-50~0bps)": (daily_ret["market_ret"] >= -0.005) & (daily_ret["market_ret"] < 0),
        "大跌(<-50bps)": daily_ret["market_ret"] < -0.005,
    }

    header = (
        f"  {'市场环境':>16s}  {'天数':>6s}  {'日均excess_bps':>14s}  "
        f"{'日胜率':>8s}  {'日均raw_bps':>12s}"
    )
    print(f"\n{header}")
    print("  " + "-" * 64)

    for label, mask in conditions.items():
        sub = daily_ret[mask]
        if len(sub) < 5:
            continue
        print(
            f"  {label:>16s}  {len(sub):>6d}  "
            f"{sub['excess_ret'].mean()*10000:>14.2f}  "
            f"{(sub['excess_ret']>0).mean():>7.1%}  "
            f"{sub['raw_ret'].mean()*10000:>12.2f}"
        )


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("时序模型结果深度分析")
    logger.info("=" * 60)

    # 加载数据
    df = load_predictions()

    # 1. 分位数分析
    analyze_decile(df)

    # 2. 每日 Top-N 选股
    analyze_daily_topn(df)

    # 3. 月度表现
    analyze_monthly(df)

    # 4. 特征画像
    analyze_feature_profile(df)

    # 5. 市场环境分析
    analyze_market_condition(df)

    print("\n" + "=" * 90)
    print("深度分析完成!")
    print("=" * 90)


if __name__ == "__main__":
    main()
