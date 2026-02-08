"""
X2信号回测脚本

职责：读取特征生产脚本(validate_x1_x2.py)输出的特征数据，进行信号回测
输入：df_features_{year}.pkl（包含特征+标签的完整数据）

策略逻辑：
  1. 在entry_time时刻，检查X2_zscore是否低于阈值
  2. 若触发信号，在下一个Bar以VWAP买入
  3. 持仓N个Bar后以VWAP卖出
  4. 扣除交易成本后计算净收益

依赖：validate_x1_x2.py 生产的特征数据

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

# 回测参数
COST_RATE = 0.0015  # 双边交易成本（佣金+印花税+滑点）
HOLD_BARS_LIST = [6, 12, 24]  # 持仓Bar数
THRESHOLD_LIST = [-2.0, -1.5, -1.0]  # X2_zscore 阈值
SIGNAL_COL = "X2_zscore"  # 信号列

# 交易时间窗口（剔除开盘和尾盘）
TRADE_START = "10:30"
TRADE_END = "14:00"


def load_features(year: int) -> pd.DataFrame:
    """加载特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 validate_x1_x2.py"
        )

    logger.info(f"加载特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    # 显式排序，确保 groupby+shift 的正确性
    df = df.sort_values(["SecuCode", "date", "entry_time"]).reset_index(drop=True)
    logger.success(f"加载完成: {len(df):,} 行")
    return df


# ==================== 回测核心 ====================
def run_single_backtest(
    df: pd.DataFrame,
    signal_col: str,
    threshold: float,
    hold_bars: int,
    cost_rate: float,
) -> pd.DataFrame:
    """
    单次回测

    参数：
      signal_col: 信号列名
      threshold: 信号阈值（< threshold 时买入）
      hold_bars: 持仓Bar数
      cost_rate: 交易成本

    返回：交易明细DataFrame
    """
    # 1. 筛选交易时间窗口
    df_trade = df[
        (df["entry_time"] >= TRADE_START) & (df["entry_time"] <= TRADE_END)
    ].copy()

    # 2. 生成买入信号
    buy_mask = df_trade[signal_col] < threshold
    df_buy = df_trade[buy_mask].copy()

    if df_buy.empty:
        return pd.DataFrame()

    # 3. 去重叠：同一只股票同一天，持仓期间跳过新信号
    # 冷却期 = hold_bars + 1（买入延迟1个Bar + 持仓hold_bars个Bar）
    cooldown = hold_bars + 1
    final_indices = []
    last_end = {}  # (stock, date) -> 上一笔交易结束的行位置

    for idx in df_buy.index:
        stock = df_buy.loc[idx, "SecuCode"]
        date = df_buy.loc[idx, "date"]
        key = (stock, date)

        # 获取当前信号在全量df中的行位置
        pos = df.index.get_loc(idx)

        if key not in last_end or pos >= last_end[key]:
            final_indices.append(idx)
            # 标记这笔交易的结束位置
            last_end[key] = pos + cooldown

    df_buy = df_buy.loc[final_indices]
    logger.debug(
        f"去重叠: 阈值={threshold}, 持仓={hold_bars}bars, "
        f"原始信号{buy_mask.sum()}笔 -> 去重后{len(df_buy)}笔"
    )

    if df_buy.empty:
        return pd.DataFrame()

    # 4. 计算买卖价格
    grp = df.groupby(["SecuCode", "date"])

    # 买入价：下一个Bar的VWAP（决策后执行）
    df["buy_price"] = grp["vwap_5m"].shift(-1)

    # 卖出价：持仓期间整段VWAP（与相关性测试V_next_Xm一致）
    # 即买入Bar之后hold_bars个Bar的加权均价
    cum_amt_s1 = grp["cum_amount"].shift(-1)
    cum_vol_s1 = grp["cum_volume"].shift(-1)
    cum_amt_sN = grp["cum_amount"].shift(-(hold_bars + 1))
    cum_vol_sN = grp["cum_volume"].shift(-(hold_bars + 1))
    df["sell_price"] = ((cum_amt_sN - cum_amt_s1) / (cum_vol_sN - cum_vol_s1)).replace(
        [np.inf, -np.inf], np.nan
    )

    # 卖出时间（持仓结束Bar的时间，仅用于展示）
    df["sell_time"] = grp["entry_time"].shift(-(hold_bars + 1))

    # 4. 提取买入信号行的买卖价格
    trades = df.loc[
        df_buy.index,
        [
            "SecuCode",
            "date",
            "entry_time",
            "close",
            signal_col,
            "buy_price",
            "sell_price",
            "sell_time",
        ],
    ].copy()

    # 5. 过滤无效交易（买卖价格缺失）
    trades = trades.dropna(subset=["buy_price", "sell_price"])

    if trades.empty:
        return pd.DataFrame()

    # 6. 计算收益
    trades["raw_ret"] = (trades["sell_price"] - trades["buy_price"]) / trades[
        "buy_price"
    ]
    trades["net_ret"] = trades["raw_ret"] - cost_rate

    return trades


def analyze_trades(trades: pd.DataFrame, label: str) -> dict:
    """分析交易结果"""
    if trades.empty:
        return {"label": label, "n_trades": 0}

    n_trades = len(trades)
    win_trades = (trades["net_ret"] > 0).sum()
    win_rate = win_trades / n_trades

    avg_raw = trades["raw_ret"].mean()
    avg_net = trades["net_ret"].mean()
    total_net = trades["net_ret"].sum()

    # 按日汇总（假设每天等权分配）
    daily_ret = trades.groupby("date")["net_ret"].mean()
    sharpe = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
    )
    max_dd = (daily_ret.cumsum() - daily_ret.cumsum().cummax()).min()

    # 获利因子
    gross_profit = trades.loc[trades["net_ret"] > 0, "net_ret"].sum()
    gross_loss = abs(trades.loc[trades["net_ret"] < 0, "net_ret"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "label": label,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_raw_ret": avg_raw,
        "avg_net_ret": avg_net,
        "total_net_ret": total_net,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "n_days": daily_ret.shape[0],
        "daily_ret": daily_ret,
    }


# ==================== 参数扫描 ====================
def parameter_scan(df: pd.DataFrame) -> list:
    """多参数组合扫描"""
    logger.info("开始参数扫描...")
    all_results = []

    for threshold in THRESHOLD_LIST:
        for hold_bars in HOLD_BARS_LIST:
            hold_mins = hold_bars * 5
            label = f"阈值={threshold}, 持仓={hold_mins}m"

            trades = run_single_backtest(
                df,
                signal_col=SIGNAL_COL,
                threshold=threshold,
                hold_bars=hold_bars,
                cost_rate=COST_RATE,
            )

            stats = analyze_trades(trades, label)
            stats["threshold"] = threshold
            stats["hold_bars"] = hold_bars
            stats["hold_mins"] = hold_mins
            all_results.append(stats)

            if stats["n_trades"] > 0:
                logger.info(
                    f"  {label}: "
                    f"交易{stats['n_trades']}笔, "
                    f"胜率{stats['win_rate']:.1%}, "
                    f"均净收益{stats['avg_net_ret']:.4%}, "
                    f"获利因子{stats['profit_factor']:.2f}"
                )
            else:
                logger.info(f"  {label}: 无交易信号")

    return all_results


# ==================== 报告和绘图 ====================
def print_report(all_results: list):
    """打印参数扫描报告"""
    header = (
        f"{'Thresh':>7} {'Hold':>6} {'Trades':>8} {'WinRate':>8} "
        f"{'AvgGross':>10} {'AvgNet':>10} {'TotalNet':>12} "
        f"{'Sharpe':>8} {'MaxDD':>10} {'PF':>8}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("X2信号回测参数扫描报告")
    print(f"  交易成本: {COST_RATE*100:.2f}%  |  信号: {SIGNAL_COL}")
    print(f"  交易窗口: {TRADE_START} ~ {TRADE_END}")
    print("=" * len(header))

    print(f"\n{header}")
    print(sep)

    for r in all_results:
        if r["n_trades"] == 0:
            print(f"{r['threshold']:>7.1f} {r['hold_mins']:>5}m {'N/A':>8}")
            continue

        print(
            f"{r['threshold']:>7.1f} {r['hold_mins']:>5}m "
            f"{r['n_trades']:>8} {r['win_rate']:>7.1%} "
            f"{r['avg_raw_ret']:>9.4%} {r['avg_net_ret']:>9.4%} "
            f"{r['total_net_ret']:>11.2%} "
            f"{r['sharpe']:>8.2f} {r['max_drawdown']:>9.2%} "
            f"{r['profit_factor']:>8.2f}"
        )

    print(sep)

    # 找最优参数
    valid_results = [r for r in all_results if r["n_trades"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["sharpe"])
        print(f"\n最优参数(按Sharpe): {best['label']}")
        print(f"  Sharpe:   {best['sharpe']:.2f}")
        print(f"  WinRate:  {best['win_rate']:.1%}")
        print(f"  PF:       {best['profit_factor']:.2f}")
        print(f"  TotalNet: {best['total_net_ret']:.2%}")

    print("=" * len(header))


def plot_equity_curves(all_results: list):
    """绘制资金曲线"""
    valid_results = [r for r in all_results if r["n_trades"] > 0 and "daily_ret" in r]

    if not valid_results:
        logger.warning("无有效结果可绘图")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 资金曲线
    for r in valid_results:
        daily_ret = r["daily_ret"]
        cum_ret = daily_ret.cumsum()
        label = f"T={r['threshold']}, H={r['hold_mins']}m (S={r['sharpe']:.2f})"
        axes[0].plot(cum_ret.index, cum_ret.values, label=label, alpha=0.8)

    axes[0].set_title(
        f"资金曲线对比 (信号: {SIGNAL_COL}, 成本: {COST_RATE*100:.2f}%)",
        fontsize=13,
    )
    axes[0].set_ylabel("累计收益率")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)

    # 热力图：阈值 x 持仓时间 -> 夏普比率
    thresholds = sorted(set(r["threshold"] for r in valid_results))
    hold_mins_list = sorted(set(r["hold_mins"] for r in valid_results))

    sharpe_matrix = np.full((len(thresholds), len(hold_mins_list)), np.nan)
    for r in valid_results:
        i = thresholds.index(r["threshold"])
        j = hold_mins_list.index(r["hold_mins"])
        sharpe_matrix[i, j] = r["sharpe"]

    im = axes[1].imshow(sharpe_matrix, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(len(hold_mins_list)))
    axes[1].set_xticklabels([f"{m}m" for m in hold_mins_list])
    axes[1].set_yticks(range(len(thresholds)))
    axes[1].set_yticklabels([f"{t:.1f}" for t in thresholds])
    axes[1].set_xlabel("持仓时间")
    axes[1].set_ylabel("阈值")
    axes[1].set_title("夏普比率热力图", fontsize=13)

    for i in range(len(thresholds)):
        for j in range(len(hold_mins_list)):
            val = sharpe_matrix[i, j]
            if not np.isnan(val):
                axes[1].text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black" if abs(val) < 1.5 else "white",
                )

    plt.colorbar(im, ax=axes[1], label="夏普比率")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "backtest_x2_signal.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"回测图表已保存: {save_path}")
    plt.close()


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("X2信号回测")
    logger.info("=" * 60)

    # 1. 加载特征数据（直接读取，无需重新计算）
    df = load_features(YEAR)

    # 2. 参数扫描回测
    all_results = parameter_scan(df)

    # 3. 打印报告
    print_report(all_results)

    # 4. 绘制图表
    plot_equity_curves(all_results)

    # 5. 保存最优参数交易明细
    valid_results = [r for r in all_results if r["n_trades"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["sharpe"])
        best_trades = run_single_backtest(
            df,
            signal_col=SIGNAL_COL,
            threshold=best["threshold"],
            hold_bars=best["hold_bars"],
            cost_rate=COST_RATE,
        )
        trade_file = OUTPUT_DIR / "backtest_x2_trades.csv"
        best_trades.to_csv(trade_file, index=False)
        logger.success(f"最优参数交易明细已保存: {trade_file}")

    return all_results


if __name__ == "__main__":
    results = main()
