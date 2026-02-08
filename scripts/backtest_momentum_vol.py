"""
高X1动量 + 成交量确认 策略回测

策略逻辑：
  1. 在 10:30 时刻，筛选 X1_zscore > X1_THRESH 且 rel_vol > VOL_THRESH 的股票
  2. 在下一个Bar（10:35）以VWAP买入
  3. 持仓120分钟后以区间VWAP卖出（与Y_120m标签语义一致）
  4. 扣除交易成本后计算净收益

收益语义：
  Y_120m = buy_price / sell_price
  做多收益 = sell_price / buy_price - 1 = 1/Y - 1
  Y < 1 表示做多盈利

输入：df_features_{year}.pkl
依赖：calc_feature_x1_x2.py 生产的特征数据

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
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

YEAR = 2025

# 入场时间
ENTRY_TIME = "10:30"

# 持仓配置
HOLD_BARS = 24  # 120分钟 = 24个5分钟Bar
HOLD_COL = "Y_120m"  # 对应的标签列

# 交易成本
COST_BPS = 15  # 双边成本 15 bps

# 相对成交量配置
VOL_WINDOW = 10  # 计算相对成交量的滚动窗口（天数）

# 参数扫描：X1阈值 x 成交量阈值
X1_THRESH_LIST = [3.0, 3.5, 4.0, 4.5]
VOL_THRESH_LIST = [0.0, 1.0, 1.2, 1.5]  # 0.0 表示不过滤


# ==================== 数据加载 ====================
def load_features(year: int) -> pd.DataFrame:
    """加载特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 calc_feature_x1_x2.py"
        )
    logger.info(f"加载特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    df = df.sort_values(["SecuCode", "date", "entry_time"]).reset_index(drop=True)
    logger.success(
        f"加载完成: {len(df):,} 行, {df['SecuCode'].nunique()} 只股票, "
        f"{df['date'].nunique()} 个交易日"
    )
    return df


def calc_relative_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算相对成交量：当前Bar的cum_volume / 过去N天同一entry_time的cum_volume均值
    shift(1) 避免使用当天数据
    """
    logger.info(f"计算相对成交量 (窗口={VOL_WINDOW}天)...")
    df = df.sort_values(["SecuCode", "entry_time", "date"])
    grp = df.groupby(["SecuCode", "entry_time"])["cum_volume"]
    vol_mean = grp.rolling(VOL_WINDOW, min_periods=2).mean().shift(1)
    vol_mean = vol_mean.reset_index(level=[0, 1], drop=True)
    df["rel_vol"] = df["cum_volume"] / vol_mean
    # 恢复排序
    df = df.sort_values(["SecuCode", "date", "entry_time"]).reset_index(drop=True)
    n_valid = df["rel_vol"].notna().sum()
    logger.success(f"相对成交量计算完成, 有效样本: {n_valid:,}")
    return df


# ==================== 回测核心 ====================
def run_backtest(
    df: pd.DataFrame,
    x1_thresh: float,
    vol_thresh: float,
) -> pd.DataFrame:
    """
    单次回测

    做多逻辑：X1_zscore > x1_thresh 且 rel_vol > vol_thresh
    买入价：下一个Bar的VWAP
    卖出价：持仓期间的加权均价（与Y_120m一致）
    """
    # 1. 筛选入场时间
    df_entry = df[df["entry_time"] == ENTRY_TIME].copy()

    # 2. 生成买入信号
    mask = df_entry["X1_zscore"] > x1_thresh
    if vol_thresh > 0:
        mask = mask & (df_entry["rel_vol"] > vol_thresh)

    # 过滤无效标签
    mask = mask & df_entry[HOLD_COL].notna() & np.isfinite(df_entry[HOLD_COL])
    mask = mask & (df_entry[HOLD_COL] > 0.9) & (df_entry[HOLD_COL] < 1.1)

    trades = df_entry[mask].copy()

    if trades.empty:
        return pd.DataFrame()

    # 3. 计算做多收益
    # Y_120m = buy_price / sell_price
    # 做多毛收益 = 1/Y - 1
    trades["raw_ret"] = 1.0 / trades[HOLD_COL] - 1.0
    trades["raw_ret_bps"] = trades["raw_ret"] * 10000
    trades["net_ret_bps"] = trades["raw_ret_bps"] - COST_BPS
    trades["net_ret"] = trades["net_ret_bps"] / 10000

    return trades[
        ["SecuCode", "date", "entry_time", "X1_zscore", "X2_zscore",
         "Z_final", "rel_vol", HOLD_COL, "raw_ret", "raw_ret_bps",
         "net_ret_bps", "net_ret"]
    ].copy()


def analyze_trades(trades: pd.DataFrame, label: str) -> dict:
    """分析交易结果"""
    if trades.empty:
        return {"label": label, "n_trades": 0}

    n_trades = len(trades)
    win_trades = (trades["net_ret"] > 0).sum()
    win_rate = win_trades / n_trades

    avg_raw_bps = trades["raw_ret_bps"].mean()
    avg_net_bps = trades["net_ret_bps"].mean()

    # 按日汇总（每日等权平均）
    daily = trades.groupby("date").agg(
        daily_ret=("net_ret", "mean"),
        daily_count=("net_ret", "count"),
    )
    n_days = len(daily)
    avg_daily_count = daily["daily_count"].mean()

    sharpe = 0.0
    if n_days > 1 and daily["daily_ret"].std() > 0:
        sharpe = daily["daily_ret"].mean() / daily["daily_ret"].std() * np.sqrt(252)

    # 最大回撤
    cum = daily["daily_ret"].cumsum()
    max_dd = (cum - cum.cummax()).min()

    # 获利因子
    gross_profit = trades.loc[trades["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(trades.loc[trades["net_ret"] < 0, "net_ret"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # 日胜率
    day_win_rate = (daily["daily_ret"] > 0).mean()

    return {
        "label": label,
        "n_trades": n_trades,
        "n_days": n_days,
        "avg_daily_count": avg_daily_count,
        "win_rate": win_rate,
        "day_win_rate": day_win_rate,
        "avg_raw_bps": avg_raw_bps,
        "avg_net_bps": avg_net_bps,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "daily_ret": daily["daily_ret"],
    }


# ==================== 参数扫描 ====================
def parameter_scan(df: pd.DataFrame) -> list:
    """多参数组合扫描"""
    logger.info("开始参数扫描...")
    all_results = []

    for x1_th in X1_THRESH_LIST:
        for vol_th in VOL_THRESH_LIST:
            vol_desc = f"vol>{vol_th}" if vol_th > 0 else "无过滤"
            label = f"X1>{x1_th}, {vol_desc}"

            trades = run_backtest(df, x1_thresh=x1_th, vol_thresh=vol_th)
            stats = analyze_trades(trades, label)
            stats["x1_thresh"] = x1_th
            stats["vol_thresh"] = vol_th
            stats["trades_df"] = trades
            all_results.append(stats)

            if stats["n_trades"] > 0:
                logger.info(
                    f"  {label}: "
                    f"交易{stats['n_trades']}笔/{stats['n_days']}天, "
                    f"日均{stats['avg_daily_count']:.1f}笔, "
                    f"胜率{stats['win_rate']:.1%}, "
                    f"毛收益{stats['avg_raw_bps']:.1f}bps, "
                    f"净收益{stats['avg_net_bps']:.1f}bps, "
                    f"Sharpe={stats['sharpe']:.2f}"
                )
            else:
                logger.info(f"  {label}: 无交易信号")

    return all_results


# ==================== 报告打印 ====================
def print_report(all_results: list):
    """打印参数扫描报告"""
    print("\n" + "=" * 130)
    print(f"高X1动量 + 成交量确认 策略回测报告")
    print(f"  入场时间: {ENTRY_TIME} | 持仓: {HOLD_BARS*5}分钟 | 成本: {COST_BPS} bps")
    print("=" * 130)

    header = (
        f"{'X1阈值':>8}  {'成交量':>10}  {'交易笔数':>8}  {'交易天数':>8}  "
        f"{'日均笔数':>8}  {'笔胜率':>8}  {'日胜率':>8}  "
        f"{'毛收益bps':>10}  {'净收益bps':>10}  {'Sharpe':>8}  "
        f"{'最大回撤':>10}  {'获利因子':>8}"
    )
    print(f"\n{header}")
    print("-" * 130)

    for r in all_results:
        vol_desc = f">{r['vol_thresh']}" if r['vol_thresh'] > 0 else "无"
        if r["n_trades"] == 0:
            print(f"{r['x1_thresh']:>8.1f}  {vol_desc:>10}  {'N/A':>8}")
            continue

        print(
            f"{r['x1_thresh']:>8.1f}  {vol_desc:>10}  "
            f"{r['n_trades']:>8,}  {r['n_days']:>8}  "
            f"{r['avg_daily_count']:>8.1f}  {r['win_rate']:>7.1%}  "
            f"{r['day_win_rate']:>7.1%}  "
            f"{r['avg_raw_bps']:>10.2f}  {r['avg_net_bps']:>10.2f}  "
            f"{r['sharpe']:>8.2f}  "
            f"{r['max_drawdown']:>9.2%}  "
            f"{r['profit_factor']:>8.2f}"
        )

    print("-" * 130)

    # 找最优参数
    valid = [r for r in all_results if r["n_trades"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n最优参数 (按Sharpe): {best['label']}")
        print(
            f"  Sharpe={best['sharpe']:.2f}, 日胜率={best['day_win_rate']:.1%}, "
            f"净收益={best['avg_net_bps']:.2f}bps, 获利因子={best['profit_factor']:.2f}"
        )

    print("=" * 130)


def plot_equity_curves(all_results: list):
    """绘制资金曲线"""
    valid = [r for r in all_results if r["n_trades"] > 0 and "daily_ret" in r]
    if not valid:
        logger.warning("无有效结果可绘图")
        return

    # 按Sharpe排序，取前8个
    valid_sorted = sorted(valid, key=lambda x: x["sharpe"], reverse=True)[:8]

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 资金曲线
    for r in valid_sorted:
        daily_ret = r["daily_ret"]
        cum_ret = daily_ret.cumsum() * 10000  # 转为bps
        label = f"{r['label']} (S={r['sharpe']:.2f})"
        axes[0].plot(range(len(cum_ret)), cum_ret.values, label=label, alpha=0.8)

    axes[0].set_title(
        f"资金曲线: 高X1动量+成交量确认 (入场{ENTRY_TIME}, 持仓{HOLD_BARS*5}m, 成本{COST_BPS}bps)",
        fontsize=12,
    )
    axes[0].set_xlabel("交易日序号")
    axes[0].set_ylabel("累计净收益 (bps)")
    axes[0].legend(fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)

    # Sharpe 热力图
    x1_list = sorted(set(r["x1_thresh"] for r in valid))
    vol_list = sorted(set(r["vol_thresh"] for r in valid))

    sharpe_matrix = np.full((len(x1_list), len(vol_list)), np.nan)
    for r in valid:
        i = x1_list.index(r["x1_thresh"])
        j = vol_list.index(r["vol_thresh"])
        sharpe_matrix[i, j] = r["sharpe"]

    im = axes[1].imshow(sharpe_matrix, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(len(vol_list)))
    axes[1].set_xticklabels([f"vol>{v}" if v > 0 else "无" for v in vol_list])
    axes[1].set_yticks(range(len(x1_list)))
    axes[1].set_yticklabels([f"X1>{t}" for t in x1_list])
    axes[1].set_xlabel("成交量过滤")
    axes[1].set_ylabel("X1阈值")
    axes[1].set_title("Sharpe比率热力图", fontsize=12)

    for i in range(len(x1_list)):
        for j in range(len(vol_list)):
            val = sharpe_matrix[i, j]
            if not np.isnan(val):
                axes[1].text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="black" if abs(val) < 1.5 else "white",
                )

    plt.colorbar(im, ax=axes[1], label="Sharpe")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "backtest_momentum_vol.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"回测图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("高X1动量 + 成交量确认 策略回测")
    logger.info("=" * 60)

    # 1. 加载数据
    df = load_features(YEAR)

    # 2. 计算相对成交量
    df = calc_relative_volume(df)

    # 3. 预过滤
    mask = np.isfinite(df["X1_zscore"]) & np.isfinite(df["X2_zscore"])
    df = df[mask].copy()
    logger.info(f"有效样本: {len(df):,} 行")

    # 4. 参数扫描
    all_results = parameter_scan(df)

    # 5. 打印报告
    print_report(all_results)

    # 6. 绘制图表
    plot_equity_curves(all_results)

    # 7. 保存最优参数交易明细
    valid = [r for r in all_results if r["n_trades"] > 0]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        trades_df = best["trades_df"]
        if not trades_df.empty:
            trade_file = OUTPUT_DIR / "backtest_momentum_vol_trades.csv"
            trades_df.to_csv(trade_file, index=False)
            logger.success(f"最优参数交易明细已保存: {trade_file}")


if __name__ == "__main__":
    main()
