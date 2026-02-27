"""
T+0 策略 Walk-Forward 严格回测脚本

前视偏差修复说明：
原始版本用 rank(pred) / n_bars 在当天全部bar上做事后排名触发信号，实盘不可行。
本版本用历史滚动分位数（walk-forward）触发信号：
  - 对每只股票，按交易日粒度计算每天所有bar的pred均值作为该日代表值
  - 用过去 LOOKBACK_DAYS 个交易日的日均pred，滚动估计第signal_pct和(1-signal_pct)分位数阈值
  - 用shift(1)将阈值向后移动一天，确保D日触发信号时只能看到D-1日及以前的历史
  - 每根bar触发时，只用历史数据判断，与实盘逻辑完全一致

代价：前LOOKBACK_DAYS天无阈值，无信号，属于预热期。
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from loguru import logger
from scipy import stats

# ==================== 回测参数 ====================
RESULT_CSV = Path(
    "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/lgbm_result_refined_label.csv"
)
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
RET_COL = "fwd_ret_30min"


SIGNAL_PCT = 0.001  # 历史分布的极端分位，触发做多/做空（狙击手策略：Bot 0.1%）
LOOKBACK_DAYS = 10  # 滚动历史窗口（交易日），相当于约一个月

# 双边摩擦成本（bps）
COST_BPS = 10
COST_RATE = COST_BPS / 10000.0

# 年化天数
ANNUAL_DAYS = 252


def load_data(csv_path: Path) -> pd.DataFrame:
    """加载预测结果，解析日期列"""
    logger.info(f"加载预测结果: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date", "bar_time"])
    df = df.dropna(subset=["pred", RET_COL]).copy()
    logger.info(
        f"有效样本: {len(df):,}行，日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}"
    )
    return df


def build_cross_section_signals(
    df: pd.DataFrame,
    signal_pct: float = SIGNAL_PCT,
) -> pd.DataFrame:
    """
    截面排名信号触发：每根bar按当天全市场 pred 截面百分位排名触发信号。

    逻辑：与 signal_threshold_analysis.py 一致
    - 每天内，所有股票所有bar按pred进行截面百分位排名
    - pred_pct >= (1 - signal_pct) 触发做多（Top signal_pct%）
    - pred_pct <= signal_pct 触发做空（Bottom signal_pct%）
    - 完全无前视：不看未来数据，仅依据当天截面中相对强弱
    - 实盘可行：当根bar内所有股票pred收齐后，即可实时计算截面排名触发

    返回：在原df上新增 direction 和 trade_ret 两列
    """
    t0 = time.time()
    df = df.copy()

    # 【实时截面排名】按每根bar（date+bar_time）做截面排名
    # 模拟实战：每5分钟bar结束时，在当前时刻所有股票中选出最强/最弱
    df["pred_pct"] = df.groupby(["date", "bar_time"])["pred"].rank(pct=True)

    long_mask = df["pred_pct"] >= (1 - signal_pct)
    short_mask = df["pred_pct"] <= signal_pct

    df["direction"] = 0
    df.loc[long_mask, "direction"] = 1
    df.loc[short_mask, "direction"] = -1

    # 计算净收益：fwd_ret 为纯价格收益率，双边成本 = COST_RATE（已含买卖双边）
    trade_mask = df["direction"] != 0
    df["trade_ret"] = np.nan
    long_trade = trade_mask & (df["direction"] == 1)
    short_trade = trade_mask & (df["direction"] == -1)
    df.loc[long_trade, "trade_ret"] = df.loc[long_trade, RET_COL] - COST_RATE
    df.loc[short_trade, "trade_ret"] = -df.loc[short_trade, RET_COL] - COST_RATE

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    logger.info(
        f"截面排名信号触发（Top/Bot {signal_pct*100:.1f}%）：做多 {n_long:,} 笔，做空 {n_short:,} 笔，"
        f"合计 {n_long+n_short:,} 笔，耗时={time.time()-t0:.1f}s"
    )
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    [deprecated] 个股时序阈值触发方式，已替换为build_cross_section_signals截面排名方式。
    保留此函数以兼容旧代码。
    """
    return build_cross_section_signals(df, signal_pct=SIGNAL_PCT)


def apply_cooldown(df: pd.DataFrame, hold_bars: int = 6) -> pd.DataFrame:
    """
    对每只股票每天内按时序贪心过滤信号，确保相邻两笔交易间隔 >= hold_bars 根bar。
    向量化实现：先按(SecuCode,date)排序，用numpy切片对每组做贪心过滤，避免逐行iterrows开销。
    """
    t0_cd = time.time()
    n_before = (df["direction"] != 0).sum()

    df = df.copy()
    # 按(SecuCode, date)排序，确保同组行连续
    sort_cols = ["SecuCode", "date", "bar_time"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["_bar_idx"] = df.groupby(["SecuCode", "date"]).cumcount()

    # 取出需要的numpy数组，全程在numpy层运算
    direction_arr = df["direction"].values.copy()
    bar_idx_arr = df["_bar_idx"].values
    # 构造组分界：找出每个(SecuCode,date)组的起始行强度
    group_ids = df.groupby(["SecuCode", "date"], sort=False).ngroup().values
    boundaries = np.where(np.diff(group_ids, prepend=-1) != 0)[0]  # 每组第一行的位置
    boundaries = np.append(boundaries, len(df))  # 加上哨兵尾部

    keep_arr = np.zeros(len(df), dtype=bool)
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        d_slice = direction_arr[s:e]
        idx_slice = bar_idx_arr[s:e]
        last = -hold_bars
        for j in range(e - s):
            if d_slice[j] == 0:
                continue
            if idx_slice[j] - last >= hold_bars:
                keep_arr[s + j] = True
                last = idx_slice[j]

    # 把冷却期内的信号清除
    remove_mask = (direction_arr != 0) & ~keep_arr
    df.loc[remove_mask, "direction"] = 0
    df.loc[remove_mask, "trade_ret"] = np.nan
    df.drop(columns=["_bar_idx"], inplace=True)

    n_after = (df["direction"] != 0).sum()
    logger.info(
        f"冷却期过滤（hold_bars={hold_bars}）: {n_before:,} → {n_after:,} 笔"
        f"（过滤掉 {n_before - n_after:,} 笔重叠信号，保留率={n_after/n_before:.1%}，耗时={time.time()-t0_cd:.1f}s）"
    )
    return df


def aggregate_daily(df: pd.DataFrame, tag: str = "多空合并") -> pd.DataFrame:
    """
    按日聚合：等权平均当天所有触发信号的净收益。
    同时统计每日交易笔数、单笔胜率、多空各笔数。
    """
    trades = df[df["direction"] != 0].copy()

    daily = (
        trades.groupby("date")
        .agg(
            daily_ret=("trade_ret", "mean"),
            n_trades=("trade_ret", "count"),
            win_rate=("trade_ret", lambda x: (x > 0).mean()),
            n_long=("direction", lambda x: (x == 1).sum()),
            n_short=("direction", lambda x: (x == -1).sum()),
        )
        .reset_index()
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["cum_ret"] = daily["daily_ret"].cumsum()

    logger.info(
        f"[{tag}] 有效交易日：{len(daily)} 天，日均笔数：{daily['n_trades'].mean():.1f}"
    )
    return daily


def diagnose_long_short(df: pd.DataFrame):
    """诊断多空信号各自的净收益，帮助理解哪个方向有效"""
    cost = COST_RATE
    long_df = df[df["direction"] == 1]
    short_df = df[df["direction"] == -1]

    for name, sub, sign in [
        ("做多(Top N%)", long_df, 1),
        ("做空(Bot N%)", short_df, -1),
    ]:
        if len(sub) == 0:
            continue
        gross = sub[RET_COL].mean() * sign * 10000
        net = gross - COST_BPS
        daily_net = sub.groupby("date")["trade_ret"].mean()
        sr = (daily_net.mean() * ANNUAL_DAYS) / (daily_net.std() * np.sqrt(ANNUAL_DAYS))
        logger.info(
            f"  [{name}] 毛收益={gross:.2f}bps  净收益={net:.2f}bps  "
            f"日度SR={sr:.2f}  样本={len(sub):,}"
        )


def check_monotonicity(df: pd.DataFrame, n_layers: int = 10):
    """
    pred绝对值分N层，验证fwd_ret的单调性（诊断用，不参与信号触发）。
    直接对pred绝对值做全局qcut分层，无前视偏差。
    """
    df = df.copy()
    # 直接对pred绝对值全局分层（诊断用，不影响信号触发逻辑）
    df["pred_layer"] = pd.qcut(df["pred"], n_layers, labels=False, duplicates="drop")
    # 还原原始价格收益（fwd_ret已含-15bps）
    df["price_ret"] = df[RET_COL] + COST_RATE

    tbl = (
        df.groupby("pred_layer")["price_ret"]
        .mean()
        .mul(10000)
        .rename("均值(bps)")
        .reset_index()
    )
    tbl["pred_layer"] = tbl["pred_layer"].astype(int)
    logger.info("pred绝对值分层单调性验证（原始价格收益，Layer0=最低pred）:")
    logger.info("\n" + tbl.to_string(index=False))

    vals = tbl["均值(bps)"].values
    monotone = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    logger.info(f"单调性检验: {'✓ 完全单调递增' if monotone else '✗ 存在非单调区间'}")
    return tbl


def evaluate_performance(daily: pd.DataFrame):
    """计算并打印标准量化绩效指标"""
    r = daily["daily_ret"]
    n = len(r)

    mean_r = r.mean()
    std_r = r.std()
    total_ret = r.sum()
    annual_ret = mean_r * ANNUAL_DAYS
    annual_vol = std_r * np.sqrt(ANNUAL_DAYS)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

    # t统计量（均值是否显著异于0）
    t_stat = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else np.nan
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    # 最大回撤
    cum = daily["cum_ret"]
    drawdown = cum.cummax() - cum
    max_dd = drawdown.max()
    calmar = annual_ret / max_dd if max_dd > 0 else np.nan

    # 日度胜率
    day_win = (r > 0).mean()

    # 单笔统计
    mean_ret_bps = mean_r * 10000
    daily_ir = mean_r / std_r if std_r > 0 else np.nan
    annual_ir = daily_ir * np.sqrt(ANNUAL_DAYS)

    logger.info("\n" + "=" * 60)
    logger.info(f"T+0 策略绩效报告（双边成本 {COST_BPS} bps）")
    logger.info("=" * 60)
    logger.info(f"  测试天数         : {n} 天")
    logger.info(f"  日均交易笔数     : {daily['n_trades'].mean():.1f} 笔/天")
    logger.info(f"  日均单笔净利润   : {mean_ret_bps:.2f} bps")
    logger.info(f"  累计净收益       : {total_ret * 10000:.1f} bps")
    logger.info(f"  日度胜率         : {day_win:.1%} ({(r>0).sum()}/{n})")
    logger.info(f"  年化收益率       : {annual_ret:.2%}")
    logger.info(f"  年化波动率       : {annual_vol:.2%}")
    logger.info(f"  夏普比率(SR)     : {sharpe:.2f}")
    logger.info(f"  最大回撤(MDD)    : {max_dd * 10000:.1f} bps")
    logger.info(f"  卡玛比率(Calmar) : {calmar:.2f}")
    logger.info(f"  日度IR           : {daily_ir:.3f}")
    logger.info(f"  年化IR           : {annual_ir:.2f}")
    logger.info(f"  t-stat           : {t_stat:.2f}  p-value={p_val:.4f}")
    logger.info("=" * 60)

    # 月度分解
    daily_copy = daily.copy()
    daily_copy["month"] = daily_copy["date"].dt.to_period("M")
    monthly = daily_copy.groupby("month")["daily_ret"].agg(
        月均收益bps=lambda x: x.mean() * 10000,
        月累计收益bps=lambda x: x.sum() * 10000,
        胜率=lambda x: (x > 0).mean(),
        交易日数="count",
    )
    logger.info("\n月度绩效分解:")
    logger.info("\n" + monthly.to_string())

    return {
        "sharpe": sharpe,
        "annual_ret": annual_ret,
        "max_dd": max_dd,
        "calmar": calmar,
        "t_stat": t_stat,
        "day_win": day_win,
    }


def plot_equity_curve(daily: pd.DataFrame, metrics: dict):
    """绘制累计收益曲线 + 回撤图"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    dates = daily["date"]
    cum_ret_bps = daily["cum_ret"] * 10000
    drawdown_bps = (daily["cum_ret"].cummax() - daily["cum_ret"]) * 10000

    # 主图：累计收益
    ax1.plot(
        dates,
        cum_ret_bps,
        color="firebrick",
        linewidth=1.5,
        label="Cumulative PnL (bps)",
    )
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative PnL (bps)", fontsize=11)
    ax1.set_title(
        f"T+0 Strategy Equity Curve (Cost={COST_BPS}bps, Top/Bot {SIGNAL_PCT*100:.0f}%)\n"
        f"Sharpe={metrics['sharpe']:.2f}  AnnRet={metrics['annual_ret']:.2%}  "
        f"MDD={metrics['max_dd']*10000:.1f}bps  t-stat={metrics['t_stat']:.1f}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # 副图：回撤
    ax2.fill_between(
        dates, -drawdown_bps, 0, color="gray", alpha=0.5, label="Drawdown (bps)"
    )
    ax2.set_ylabel("Drawdown (bps)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    plt.tight_layout()
    tag = RESULT_CSV.stem.replace("lgbm_result_reg", "").strip("_") or "default"
    out_path = OUTPUT_DIR / f"t0_equity_curve_{tag}_cost{COST_BPS}bps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.success(f"资金曲线已保存: {out_path}")
    plt.close()


def _calc_perf(daily_r: pd.Series) -> dict:
    """计算一组日度收益序列的绩效指标"""
    n = len(daily_r)
    if n == 0:
        return {}
    mean_r = daily_r.mean()
    std_r = daily_r.std()
    sr = (
        (mean_r * ANNUAL_DAYS) / (std_r * np.sqrt(ANNUAL_DAYS)) if std_r > 0 else np.nan
    )
    t_s = mean_r / (std_r / np.sqrt(n)) if std_r > 0 else np.nan
    return {
        "夏普": round(sr, 2),
        "t-stat": round(t_s, 2),
        "日均净利(bps)": round(mean_r * 10000, 2),
        "日度胜率": f"{(daily_r > 0).mean():.1%}",
    }


def sensitivity_analysis(df: pd.DataFrame):
    """
    Walk-Forward敏感性分析：在不同signal_pct/lookback/cost组合下对比绩效。
    每种参数组合重新计算rolling阈值，严格无前视偏差。
    """
    signal_pcts = [0.03, 0.05, 0.08, 0.10]
    lookbacks = [10, 20, 40]
    costs = [10, 15, 20]

    rows_long, rows_short, rows_both = [], [], []

    for pct in signal_pcts:
        for lb in lookbacks:
            # 重新计算该参数组合的rolling阈值
            df_thr = build_rolling_thresholds(df, signal_pct=pct, lookback=lb)
            valid = df_thr["threshold_long"].notna()
            long_m = valid & (df_thr["pred"] > df_thr["threshold_long"])
            short_m = valid & (df_thr["pred"] < df_thr["threshold_short"])
            label = f"pct={pct*100:.0f}%/lb={lb}d"

            for cost in costs:
                rate = cost / 10000.0

                # fwd_ret 为纯价格收益率，双边成本 = 2 * rate
                # 纯做多净利 = fwd_ret - 2 * rate
                tmp_l = df_thr[long_m].copy()
                tmp_l["tr"] = tmp_l[RET_COL] - 2 * rate
                perf_l = _calc_perf(tmp_l.groupby("date")["tr"].mean())
                if perf_l:
                    rows_long.append({"参数": label, "成本(bps)": cost, **perf_l})

                # 纯做空净利 = -fwd_ret - 2 * rate
                tmp_s = df_thr[short_m].copy()
                tmp_s["tr"] = -tmp_s[RET_COL] - 2 * rate
                perf_s = _calc_perf(tmp_s.groupby("date")["tr"].mean())
                if perf_s:
                    rows_short.append({"参数": label, "成本(bps)": cost, **perf_s})

                # 多空合并
                trade_m = long_m | short_m
                tmp_b = df_thr[trade_m].copy()
                tmp_b["tr"] = np.where(
                    tmp_b["pred"] > tmp_b["threshold_long"],
                    tmp_b[RET_COL] - 2 * rate,  # 做多
                    -tmp_b[RET_COL] - 2 * rate,  # 做空
                )
                perf_b = _calc_perf(tmp_b.groupby("date")["tr"].mean())
                if perf_b:
                    rows_both.append({"参数": label, "成本(bps)": cost, **perf_b})

    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward敏感性分析 — 纯做多")
    logger.info("=" * 60)
    logger.info("\n" + pd.DataFrame(rows_long).to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward敏感性分析 — 纯做空")
    logger.info("=" * 60)
    logger.info("\n" + pd.DataFrame(rows_short).to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward敏感性分析 — 多空合并")
    logger.info("=" * 60)
    logger.info("\n" + pd.DataFrame(rows_both).to_string(index=False))


if __name__ == "__main__":
    df = load_data(RESULT_CSV)

    # Walk-Forward阈值计算（核心步骤，消除前视偏差）
    # 截面排名触发信号（与signal_threshold_analysis逻辑一致）
    df = build_cross_section_signals(df, signal_pct=SIGNAL_PCT)

    # 冷却期过滤：每只股票每天内，相邻两笔交易间隔 >= 持有期（6根5min bar = 30min）
    df = apply_cooldown(df, hold_bars=6)

    # pred绝对值分层单调性验证（诊断用）
    check_monotonicity(df)

    # 诊断多空各自的方向性
    logger.info("\n多空信号诊断（未合并）:")
    diagnose_long_short(df)

    # 分别跑三种组合
    for mode, mask_fn, tag in [
        ("多空合并", lambda d: d, "多空合并"),
        (
            "纯做多",
            lambda d: d.assign(
                direction=lambda x: x["direction"].clip(0, 1),
                trade_ret=lambda x: x["trade_ret"].where(x["direction"] == 1, np.nan),
            ),
            "纯做多",
        ),
        (
            "纯做空",
            lambda d: d.assign(
                direction=lambda x: x["direction"].clip(-1, 0),
                trade_ret=lambda x: x["trade_ret"].where(x["direction"] == -1, np.nan),
            ),
            "纯做空",
        ),
    ]:
        logger.info(f"\n{'='*60}\n回测模式：{tag}\n{'='*60}")
        d = mask_fn(df.copy())
        daily = aggregate_daily(d, tag=tag)
        if len(daily) == 0:
            logger.warning(f"[{tag}] 无有效交易日，跳过")
            continue
        metrics = evaluate_performance(daily)
        plot_equity_curve(daily, metrics)

    # sensitivity_analysis(df)
