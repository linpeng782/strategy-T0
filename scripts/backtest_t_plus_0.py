#!/usr/bin/env python3
"""
日内做T回测模拟器
==================
对比两组策略的净值曲线：
1. 对照组（Buy & Hold）：持有股票不动
2. 实验组（Buy & Hold + T Engine）：持有股票 + 做T收益

核心逻辑：
- 从中证2000随机抽取10只股票
- 触发条件：昨日高波动 + Z_final ∈ [-5, -1.5] + 新高不买滤网
- 成交价：入场价=当前Bar收盘价，出场价=30分钟后收盘价
- 成本：每笔扣除20bp（0.2%）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import random
import sys
import time as time_module

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import load_config
from services.data_loader import DataLoader

# ==================== 配置参数 ====================
DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_unadjusted"
)
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/backtest")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 回测参数
NUM_STOCKS = 500  # 随机抽取的股票数量
YEAR = 2025
ZSCORE_WINDOW = 20
ROLLING_WINDOW = 12

# 交易参数
Z_LOWER = -5.0  # Z_final 下限
Z_UPPER = -1.5  # Z_final 上限
HOLDING_PERIOD = 60  # 最大持有时间（60分钟）
TRADE_COST = 0.0002 + 0.0005 + 0.0008  # 交易成本（20bp）
T_LEVERAGE = 0.5  # 做T资金占比（50%底仓）

# 止盈止损参数
TAKE_PROFIT_BP = 80  # 止盈目标（60bp）
STOP_LOSS_BP = 40  # 止损目标（50bp）

# 设置字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_random_stocks(num_stocks: int = 10, seed: int = None) -> list:
    """从中证2000成分股中随机抽取股票"""
    if seed is not None:
        random.seed(seed)

    logger.info(f"从中证2000成分股中随机抽取 {num_stocks} 只股票...")

    config_path = project_root / "config" / "config.yaml"
    config = load_config(config_path)

    data_loader = DataLoader(config)
    all_stocks = data_loader.get_index_components("932000.INDX")

    if len(all_stocks) == 0:
        raise ValueError("未获取到中证2000成分股")

    # 排序确保每次顺序一致，这样 seed 才能保证采样结果可重复
    all_stocks = sorted(all_stocks)
    sample_stocks = random.sample(all_stocks, min(num_stocks, len(all_stocks)))
    logger.success(f"随机抽取 {len(sample_stocks)} 只股票: {sample_stocks}")
    return sample_stocks


def load_minute_data(stock_codes: list, year: int) -> pd.DataFrame:
    """加载1分钟数据"""
    pkl_path = DATA_DIR / f"{year}.pkl"
    logger.info(f"加载 {year} 年数据...")

    df = pd.read_pickle(pkl_path)
    df = df[df["SecuCode"].isin(stock_codes)]
    df["date"] = pd.to_datetime(df["TradingDay"]).dt.date

    logger.success(f"加载完成: {len(df):,} 条, {df['SecuCode'].nunique()} 只股票")
    return df


def aggregate_to_5min_bars(df: pd.DataFrame) -> pd.DataFrame:
    """聚合为5分钟Bar"""
    logger.info("聚合为5分钟Bar...")

    df["bar_time"] = pd.to_datetime(df["TradingDay"]).dt.floor("5min")

    agg_df = (
        df.groupby(["SecuCode", "date", "bar_time"])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        .reset_index()
    )

    agg_df = agg_df.sort_values(["SecuCode", "date", "bar_time"])
    agg_df["vwap_5m"] = agg_df["amount"] / agg_df["volume"]

    logger.success(f"聚合完成: {len(agg_df):,} 个5分钟Bar")
    return agg_df


def calculate_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """向量化计算特征"""
    logger.info("计算特征...")

    df = df.sort_values(["SecuCode", "date", "bar_time"]).copy()
    group_keys = ["SecuCode", "date"]

    # 累积成交量和成交额
    df["cum_volume"] = df.groupby(group_keys)["volume"].cumsum()
    df["cum_amount"] = df.groupby(group_keys)["amount"].cumsum()

    # 滚动窗口VWAP和TWAP
    def rolling_sum(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).sum()

    def rolling_mean(x):
        return x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    df["rolling_amount"] = df.groupby(group_keys)["amount"].transform(rolling_sum)
    df["rolling_volume"] = df.groupby(group_keys)["volume"].transform(rolling_sum)
    df["cum_vwap"] = df["rolling_amount"] / df["rolling_volume"]
    df["cum_twap"] = df.groupby(group_keys)["close"].transform(rolling_mean)

    # X1 和 X2
    df["X1"] = df["close"] / df["cum_vwap"] - 1
    df["X2"] = df["cum_vwap"] / df["cum_twap"] - 1

    # 日内新高
    def expanding_max(x):
        return x.expanding().max()

    df["day_high"] = df.groupby(group_keys)["close"].transform(expanding_max)
    df = df.drop(columns=["rolling_amount", "rolling_volume"])

    logger.success("特征计算完成")
    return df


def calculate_zscore_vectorized(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """向量化计算Z-Score"""
    logger.info(f"计算 Z-Score (窗口={window}天)...")

    df = df.sort_values(["SecuCode", "bar_time"]).copy()
    df["time_str"] = df["bar_time"].dt.strftime("%H:%M")

    group_keys = ["SecuCode", "time_str"]

    # 修正：加上 shift(1)，确保只用历史同时间点的数据计算基准，避免未来函数泄露
    def rolling_mean_shifted(x):
        return x.rolling(window=window, min_periods=5).mean().shift(1)

    def rolling_std_shifted(x):
        return x.rolling(window=window, min_periods=5).std().shift(1)

    df["X2_mean"] = df.groupby(group_keys)["X2"].transform(rolling_mean_shifted)
    df["X2_std"] = df.groupby(group_keys)["X2"].transform(rolling_std_shifted)
    df["X1_mean"] = df.groupby(group_keys)["X1"].transform(rolling_mean_shifted)
    df["X1_std"] = df.groupby(group_keys)["X1"].transform(rolling_std_shifted)

    df["X2_zscore"] = (df["X2"] - df["X2_mean"]) / df["X2_std"]
    df["X1_zscore"] = (df["X1"] - df["X1_mean"]) / df["X1_std"]
    df["Z_final"] = df["X2_zscore"] - 0.5 * df["X1_zscore"]

    logger.success(f"Z-Score 计算完成")
    return df


def calculate_yesterday_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算昨日波动率"""
    logger.info("计算昨日波动率...")

    daily_summary = (
        df.groupby(["SecuCode", "date"])["close"]
        .agg(["max", "min", "mean"])
        .reset_index()
    )
    daily_summary["daily_range"] = (
        daily_summary["max"] - daily_summary["min"]
    ) / daily_summary["mean"]
    daily_summary["yesterday_range"] = daily_summary.groupby("SecuCode")[
        "daily_range"
    ].shift(1)

    df = df.merge(
        daily_summary[["SecuCode", "date", "yesterday_range"]],
        on=["SecuCode", "date"],
        how="left",
    )

    logger.success("昨日波动率计算完成")
    return df


def prepare_future_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    预计算未来价格和止盈止损触碰信息

    使用向量化方式计算未来N个Bar内的最高价、最低价
    """
    logger.info("预计算未来价格和止盈止损信息...")

    df = df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    n_bars = HOLDING_PERIOD // 5  # 60分钟 = 12个5分钟Bar

    # 1. 未来N个Bar后的收盘价（超时平仓用）
    df["close_exit"] = df.groupby(["SecuCode", "date"])["close"].shift(-n_bars)

    # 2. 计算未来N个Bar内的最高价和最低价（止盈止损触碰检测用）
    # 使用反向rolling的技巧：先reverse，rolling，再reverse回来
    def calc_future_high_low(group):
        """计算每个时间点未来N个Bar的最高/最低价"""
        # 反转数据
        high_rev = group["high"].iloc[::-1]
        low_rev = group["low"].iloc[::-1]

        # rolling计算（min_periods=1允许不足N个Bar的情况）
        future_high_rev = high_rev.rolling(window=n_bars, min_periods=1).max()
        future_low_rev = low_rev.rolling(window=n_bars, min_periods=1).min()

        # 再反转回来，并shift使其代表"未来"而非"过去"
        future_high = future_high_rev.iloc[::-1].shift(-1)
        future_low = future_low_rev.iloc[::-1].shift(-1)

        return pd.DataFrame(
            {"future_high": future_high.values, "future_low": future_low.values},
            index=group.index,
        )

    # 按股票和日期分组计算
    future_hl = df.groupby(["SecuCode", "date"], group_keys=False).apply(
        calc_future_high_low
    )
    df["future_high"] = future_hl["future_high"]
    df["future_low"] = future_hl["future_low"]

    logger.success(f"未来价格计算完成 (持有期={HOLDING_PERIOD}分钟, {n_bars}个Bar)")
    return df


def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日收益率（Buy & Hold 基础收益）"""
    logger.info("计算每日基础收益率...")

    # 每日收盘价
    daily_close = (
        df.groupby(["SecuCode", "date"])["close"]
        .last()
        .reset_index()
        .rename(columns={"close": "daily_close"})
    )

    # 计算日收益率
    daily_close = daily_close.sort_values(["SecuCode", "date"])
    daily_close["daily_return"] = daily_close.groupby("SecuCode")[
        "daily_close"
    ].pct_change()

    logger.success("每日基础收益率计算完成")
    return daily_close


def run_backtest(df: pd.DataFrame, daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    运行回测模拟

    返回每日的 Hold_Only 和 With_T_Engine 收益
    """
    logger.info("开始回测模拟...")

    # 1. 昨日高波动滤网阈值（全局90%分位数，与深度分析保持一致）
    vol_threshold = df["yesterday_range"].quantile(0.9)
    logger.info(f"昨日波动率阈值（90%分位数）: {vol_threshold*100:.2f}%")

    # 2. 识别做T信号
    df["is_trade"] = (
        np.isfinite(df["Z_final"])
        & (df["Z_final"] > Z_LOWER)
        & (df["Z_final"] < Z_UPPER)
        & (df["close"] < df["day_high"])  # 新高不买
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > vol_threshold)  # 昨日高波动
        & (df["time_str"] < "14:15")  # 避免尾盘
        & np.isfinite(df["close_exit"])  # 有出场价
    )

    trade_count = df["is_trade"].sum()
    logger.info(f"识别到 {trade_count:,} 个做T信号")

    # 3. 计算单次做T净收益（使用止盈止损触碰逻辑）
    tp_ratio = TAKE_PROFIT_BP / 10000  # 止盈比例
    sl_ratio = STOP_LOSS_BP / 10000  # 止损比例

    # 计算止盈价和止损价
    df["tp_price"] = df["close"] * (1 + tp_ratio)
    df["sl_price"] = df["close"] * (1 - sl_ratio)

    # 判断触碰情况（向量化）
    df["hit_tp"] = df["future_high"] >= df["tp_price"]  # 触及止盈
    df["hit_sl"] = df["future_low"] <= df["sl_price"]  # 触及止损

    # 计算收益：止盈 > 止损 > 超时平仓
    # 注意：如果同时触及，优先止盈（假设先触及止盈）
    df["t_gross_return"] = np.where(
        df["hit_tp"],
        tp_ratio,  # 止盈成功
        np.where(
            df["hit_sl"],
            -sl_ratio,  # 止损离场
            (df["close_exit"] / df["close"]) - 1,  # 超时平仓
        ),
    )
    df["t_net_return"] = df["t_gross_return"] - TRADE_COST

    # 统计止盈止损情况
    df_signals = df[df["is_trade"]]
    n_tp = df_signals["hit_tp"].sum()
    n_sl = (~df_signals["hit_tp"] & df_signals["hit_sl"]).sum()
    n_timeout = (~df_signals["hit_tp"] & ~df_signals["hit_sl"]).sum()
    logger.info(
        f"止盈: {n_tp:,} ({n_tp/trade_count*100:.1f}%) | 止损: {n_sl:,} ({n_sl/trade_count*100:.1f}%) | 超时: {n_timeout:,} ({n_timeout/trade_count*100:.1f}%)"
    )

    # 4. 计算每日做T收益贡献
    # 假设每只股票等权，每次做T用 T_LEVERAGE 比例的资金
    df_trades = df[df["is_trade"]].copy()

    # 每日每只股票的做T净收益汇总
    daily_t_alpha = (
        df_trades.groupby(["date", "SecuCode"])["t_net_return"]
        .sum()
        .reset_index()
        .rename(columns={"t_net_return": "t_alpha"})
    )

    # 做T收益乘以杠杆比例（占底仓的比例）
    daily_t_alpha["t_alpha"] = daily_t_alpha["t_alpha"] * T_LEVERAGE

    # 5. 合并基础收益和做T收益
    results = daily_returns.merge(daily_t_alpha, on=["date", "SecuCode"], how="left")
    results["t_alpha"] = results["t_alpha"].fillna(0)

    # 6. 计算每日组合收益（等权）
    daily_portfolio = (
        results.groupby("date")
        .agg(
            {
                "daily_return": "mean",  # 持有收益（等权平均）
                "t_alpha": "mean",  # 做T收益（等权平均）
            }
        )
        .reset_index()
    )

    daily_portfolio["hold_only"] = daily_portfolio["daily_return"]
    daily_portfolio["with_t_engine"] = (
        daily_portfolio["daily_return"] + daily_portfolio["t_alpha"]
    )

    # 7. 去掉第一天（无收益率）
    daily_portfolio = daily_portfolio.dropna(subset=["hold_only"])

    # 8. 计算累计净值
    daily_portfolio["cum_hold"] = (1 + daily_portfolio["hold_only"]).cumprod()
    daily_portfolio["cum_with_t"] = (1 + daily_portfolio["with_t_engine"]).cumprod()

    # 统计信息
    n_days = len(daily_portfolio)
    total_trades = trade_count
    avg_trades_per_day = total_trades / n_days if n_days > 0 else 0

    logger.success(f"回测完成: {n_days} 个交易日, 共 {total_trades:,} 笔做T交易")
    logger.info(f"平均每天 {avg_trades_per_day:.1f} 笔交易")

    return daily_portfolio, df_trades


def calculate_metrics(results: pd.DataFrame) -> dict:
    """计算绩效指标"""
    metrics = {}

    for strategy in ["hold_only", "with_t_engine"]:
        returns = results[strategy]
        cum_col = "cum_hold" if strategy == "hold_only" else "cum_with_t"

        # 年化收益率
        total_return = results[cum_col].iloc[-1] - 1
        n_days = len(results)
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率（假设无风险利率2%）
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cum_max = results[cum_col].cummax()
        drawdown = (results[cum_col] - cum_max) / cum_max
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).mean()

        metrics[strategy] = {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    return metrics


def plot_results(results: pd.DataFrame, metrics: dict, save_path: Path = None):
    """绘制净值曲线对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 子图1：净值曲线
    ax1 = axes[0]
    ax1.plot(
        results["date"],
        results["cum_hold"],
        label="Buy & Hold",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        results["date"],
        results["cum_with_t"],
        label="Buy & Hold + T Engine",
        color="red",
        linewidth=2,
        alpha=0.8,
    )

    ax1.fill_between(
        results["date"],
        results["cum_hold"],
        results["cum_with_t"],
        where=results["cum_with_t"] > results["cum_hold"],
        color="green",
        alpha=0.2,
        label="T Engine Alpha",
    )
    ax1.fill_between(
        results["date"],
        results["cum_hold"],
        results["cum_with_t"],
        where=results["cum_with_t"] <= results["cum_hold"],
        color="red",
        alpha=0.2,
    )

    ax1.set_title(
        "Net Value Comparison: Buy & Hold vs Buy & Hold + T Engine", fontsize=14
    )
    ax1.set_ylabel("Cumulative Net Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 子图2：每日收益对比
    ax2 = axes[1]
    width = 0.35
    x = np.arange(len(results))

    # 只显示部分日期标签
    step = max(1, len(results) // 20)

    ax2.bar(
        x[::step],
        results["hold_only"].values[::step] * 100,
        width,
        label="Hold Only",
        color="blue",
        alpha=0.6,
    )
    ax2.bar(
        x[::step] + width,
        results["with_t_engine"].values[::step] * 100,
        width,
        label="With T Engine",
        color="red",
        alpha=0.6,
    )

    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Daily Return (%)")
    ax2.set_title("Daily Return Comparison (Sampled)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存: {save_path}")

    plt.close()


def print_report(
    metrics: dict, results: pd.DataFrame, trades: pd.DataFrame, stock_list: list
):
    """打印回测报告"""
    print("\n" + "=" * 80)
    print("日内做T回测模拟报告")
    print("=" * 80)

    print(f"\n📋 回测设置:")
    print(f"   - 股票池: {len(stock_list)} 只股票")
    print(f"   - Z_final 阈值: [{Z_LOWER}, {Z_UPPER}]")
    print(f"   - 最大持有时间: {HOLDING_PERIOD} 分钟")
    print(f"   - 止盈目标: +{TAKE_PROFIT_BP}bp | 止损目标: -{STOP_LOSS_BP}bp")
    print(f"   - 交易成本: {TRADE_COST * 10000:.0f}bp")
    print(f"   - 做T资金比例: {T_LEVERAGE * 100:.0f}%")

    print(f"\n📊 绩效对比:")
    print(
        f"   {'指标':>15} | {'Buy & Hold':>12} | {'With T Engine':>14} | {'差异':>10}"
    )
    print(f"   {'-' * 60}")

    hold = metrics["hold_only"]
    with_t = metrics["with_t_engine"]

    print(
        f"   {'总收益率':>15} | {hold['total_return']*100:>11.2f}% | {with_t['total_return']*100:>13.2f}% | {(with_t['total_return']-hold['total_return'])*100:>9.2f}%"
    )
    print(
        f"   {'年化收益率':>15} | {hold['annual_return']*100:>11.2f}% | {with_t['annual_return']*100:>13.2f}% | {(with_t['annual_return']-hold['annual_return'])*100:>9.2f}%"
    )
    print(
        f"   {'年化波动率':>15} | {hold['annual_vol']*100:>11.2f}% | {with_t['annual_vol']*100:>13.2f}% | {(with_t['annual_vol']-hold['annual_vol'])*100:>9.2f}%"
    )
    print(
        f"   {'夏普比率':>15} | {hold['sharpe']:>12.2f} | {with_t['sharpe']:>14.2f} | {with_t['sharpe']-hold['sharpe']:>10.2f}"
    )
    print(
        f"   {'最大回撤':>15} | {hold['max_drawdown']*100:>11.2f}% | {with_t['max_drawdown']*100:>13.2f}% | {(with_t['max_drawdown']-hold['max_drawdown'])*100:>9.2f}%"
    )
    print(
        f"   {'日胜率':>15} | {hold['win_rate']*100:>11.1f}% | {with_t['win_rate']*100:>13.1f}% | {(with_t['win_rate']-hold['win_rate'])*100:>9.1f}%"
    )

    # 做T统计
    print(f"\n📈 做T交易统计:")
    n_trades = len(trades)
    n_days = results["date"].nunique()
    avg_gross = trades["t_gross_return"].mean() * 10000
    avg_net = trades["t_net_return"].mean() * 10000
    win_rate = (trades["t_net_return"] > 0).mean()

    # 止盈止损统计
    n_tp = trades["hit_tp"].sum()
    n_sl = (~trades["hit_tp"] & trades["hit_sl"]).sum()
    n_timeout = (~trades["hit_tp"] & ~trades["hit_sl"]).sum()

    print(f"   - 总交易笔数: {n_trades:,}")
    print(f"   - 交易天数: {n_days}")
    print(f"   - 平均每天交易: {n_trades / n_days:.1f} 笔")
    print(f"   - 止盈次数: {n_tp:,} ({n_tp/n_trades*100:.1f}%)")
    print(f"   - 止损次数: {n_sl:,} ({n_sl/n_trades*100:.1f}%)")
    print(f"   - 超时平仓: {n_timeout:,} ({n_timeout/n_trades*100:.1f}%)")
    print(f"   - 平均毛收益: {avg_gross:.2f}bp")
    print(f"   - 平均净收益: {avg_net:.2f}bp (扣除{TRADE_COST*10000:.0f}bp成本)")
    print(f"   - 做T胜率: {win_rate*100:.1f}%")

    # 结论
    alpha = with_t["total_return"] - hold["total_return"]
    sharpe_improve = with_t["sharpe"] - hold["sharpe"]

    print(f"\n🎯 结论:")
    if alpha > 0:
        print(
            f"   ✅ T引擎有效！总收益提升 {alpha*100:.2f}%，夏普比率提升 {sharpe_improve:.2f}"
        )
    else:
        print(f"   ⚠️ T引擎未能带来正向收益，需调整参数")


def deep_analyze_trades(trades: pd.DataFrame, save_dir: Path = None):
    """
    深度交易分析

    包含：
    1. 时间分布分析 - 检测交易是否聚集在特定时段
    2. MFE/MAE 分析 - 评估止盈止损设置是否合理
    3. 单票贡献度分析 - 确认收益来源是否分散
    4. 盈亏分布分析 - 查看收益的分布特征
    """
    print("\n" + "=" * 80)
    print("深度交易分析报告")
    print("=" * 80)

    n_trades = len(trades)

    # ==================== 1. 时间分布分析 ====================
    print(f"\n📊 1. 交易时间分布分析")

    # 按时间段统计交易笔数
    time_dist = trades.groupby("time_str").size()

    # 找出交易最密集的时段
    top_times = time_dist.nlargest(5)
    print(f"   交易最频繁的5个时段:")
    for t, count in top_times.items():
        pct = count / n_trades * 100
        print(f"      {t}: {count:,} 笔 ({pct:.1f}%)")

    # 按小时汇总
    trades["hour"] = trades["time_str"].str[:2]
    hour_dist = trades.groupby("hour").size()
    print(f"\n   按小时分布:")
    for h, count in hour_dist.items():
        pct = count / n_trades * 100
        bar = "█" * int(pct / 2)
        print(f"      {h}:00 | {bar} {count:,} ({pct:.1f}%)")

    # 开盘集中度（9:30-10:00）
    morning_rush = trades[trades["time_str"] < "10:00"]
    morning_pct = len(morning_rush) / n_trades * 100
    print(
        f"\n   ⚠️ 开盘拥堵检测 (9:30-10:00): {len(morning_rush):,} 笔 ({morning_pct:.1f}%)"
    )
    if morning_pct > 40:
        print(f"      警告: 开盘时段交易过于集中，实盘需注意下单节奏")
    else:
        print(f"      正常: 交易分布较为均匀")

    # ==================== 2. MFE/MAE 分析 ====================
    print(f"\n📈 2. MFE/MAE 分析 (获利空间透视)")

    # MFE: Maximum Favorable Excursion - 入场后最高涨幅
    trades["mfe_bp"] = (trades["future_high"] / trades["close"] - 1) * 10000
    # MAE: Maximum Adverse Excursion - 入场后最大跌幅
    trades["mae_bp"] = (1 - trades["future_low"] / trades["close"]) * 10000

    avg_mfe = trades["mfe_bp"].mean()
    avg_mae = trades["mae_bp"].mean()
    median_mfe = trades["mfe_bp"].median()
    median_mae = trades["mae_bp"].median()

    print(f"   MFE (入场后最高涨幅):")
    print(f"      平均值: {avg_mfe:.1f} bp | 中位数: {median_mfe:.1f} bp")
    print(f"      当前止盈设置: {TAKE_PROFIT_BP} bp")
    if avg_mfe > TAKE_PROFIT_BP * 1.5:
        print(f"      💡 建议: MFE远高于止盈点，可考虑提高止盈目标")
    elif avg_mfe < TAKE_PROFIT_BP:
        print(f"      ⚠️ 警告: MFE低于止盈点，止盈设置可能过高")
    else:
        print(f"      ✅ 止盈设置合理")

    print(f"\n   MAE (入场后最大跌幅):")
    print(f"      平均值: {avg_mae:.1f} bp | 中位数: {median_mae:.1f} bp")
    print(f"      当前止损设置: {STOP_LOSS_BP} bp")
    if avg_mae < STOP_LOSS_BP * 0.5:
        print(f"      💡 建议: MAE远低于止损点，可考虑收紧止损")
    elif avg_mae > STOP_LOSS_BP:
        print(f"      ⚠️ 警告: MAE高于止损点，止损设置可能过紧")
    else:
        print(f"      ✅ 止损设置合理")

    # MFE分位数分布
    print(f"\n   MFE 分位数分布:")
    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
        val = trades["mfe_bp"].quantile(q)
        print(f"      {int(q*100)}%分位: {val:.1f} bp")

    # ==================== 3. 单票贡献度分析 ====================
    print(f"\n🎯 3. 单票贡献度分析")

    # 按股票统计收益
    ticker_stats = trades.groupby("SecuCode").agg(
        {"t_net_return": ["sum", "count", "mean"], "yesterday_range": "mean"}
    )
    ticker_stats.columns = ["total_pnl", "trade_count", "avg_pnl", "avg_vol"]
    ticker_stats["total_pnl_bp"] = ticker_stats["total_pnl"] * 10000
    ticker_stats["avg_pnl_bp"] = ticker_stats["avg_pnl"] * 10000
    ticker_stats["avg_vol_pct"] = ticker_stats["avg_vol"] * 100
    ticker_stats = ticker_stats.sort_values("total_pnl_bp", ascending=False)

    # Top 10 贡献者
    print(f"   Top 10 盈利贡献股票:")
    print(
        f"   {'股票代码':>10} | {'总收益(bp)':>10} | {'交易次数':>8} | {'单笔均值(bp)':>12} | {'昨日振幅':>8}"
    )
    print(f"   {'-' * 60}")
    for code, row in ticker_stats.head(10).iterrows():
        print(
            f"   {code:>10} | {row['total_pnl_bp']:>10.1f} | {int(row['trade_count']):>8} | {row['avg_pnl_bp']:>12.2f} | {row['avg_vol_pct']:>7.2f}%"
        )

    # Bottom 10 亏损者
    print(f"\n   Top 10 亏损股票:")
    print(
        f"   {'股票代码':>10} | {'总收益(bp)':>10} | {'交易次数':>8} | {'单笔均值(bp)':>12} | {'昨日振幅':>8}"
    )
    print(f"   {'-' * 60}")
    for code, row in ticker_stats.tail(10).iterrows():
        print(
            f"   {code:>10} | {row['total_pnl_bp']:>10.1f} | {int(row['trade_count']):>8} | {row['avg_pnl_bp']:>12.2f} | {row['avg_vol_pct']:>7.2f}%"
        )

    # 收益集中度
    total_profit = ticker_stats[ticker_stats["total_pnl_bp"] > 0]["total_pnl_bp"].sum()
    total_loss = ticker_stats[ticker_stats["total_pnl_bp"] < 0]["total_pnl_bp"].sum()
    top5_profit = ticker_stats.head(5)["total_pnl_bp"].sum()
    top5_pct = top5_profit / total_profit * 100 if total_profit > 0 else 0

    print(f"\n   收益集中度:")
    print(f"      总盈利: {total_profit:.1f} bp | 总亏损: {total_loss:.1f} bp")
    print(
        f"      Top 5 股票贡献: {top5_profit:.1f} bp ({top5_pct:.1f}% of total profit)"
    )
    if top5_pct > 50:
        print(f"      ⚠️ 警告: 收益过于集中在少数股票，风险较高")
    else:
        print(f"      ✅ 收益来源分散，策略稳健")

    # ==================== 4. 盈亏分布分析 ====================
    print(f"\n📉 4. 盈亏分布分析")

    pnl_bp = trades["t_net_return"] * 10000

    print(f"   单笔收益统计:")
    print(f"      平均值: {pnl_bp.mean():.2f} bp")
    print(f"      中位数: {pnl_bp.median():.2f} bp")
    print(f"      标准差: {pnl_bp.std():.2f} bp")
    print(f"      最大盈利: {pnl_bp.max():.2f} bp")
    print(f"      最大亏损: {pnl_bp.min():.2f} bp")

    # 盈亏比
    avg_win = pnl_bp[pnl_bp > 0].mean()
    avg_loss = abs(pnl_bp[pnl_bp < 0].mean())
    profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf

    print(f"\n   盈亏比分析:")
    print(f"      平均盈利: {avg_win:.2f} bp")
    print(f"      平均亏损: {avg_loss:.2f} bp")
    print(f"      盈亏比 (Profit Factor): {profit_factor:.2f}")

    # ==================== 5. 绘制分析图表 ====================
    if save_dir:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 时间分布
        ax1 = axes[0, 0]
        hour_dist.plot(kind="bar", ax=ax1, color="steelblue", alpha=0.7)
        ax1.set_title("Trade Distribution by Hour")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Number of Trades")
        ax1.tick_params(axis="x", rotation=0)

        # 图2: MFE分布
        ax2 = axes[0, 1]
        trades["mfe_bp"].hist(
            bins=50, ax=ax2, color="green", alpha=0.7, edgecolor="white"
        )
        ax2.axvline(
            TAKE_PROFIT_BP, color="red", linestyle="--", label=f"TP={TAKE_PROFIT_BP}bp"
        )
        ax2.axvline(avg_mfe, color="blue", linestyle="--", label=f"Avg={avg_mfe:.0f}bp")
        ax2.set_title("MFE Distribution (Maximum Favorable Excursion)")
        ax2.set_xlabel("MFE (bp)")
        ax2.legend()

        # 图3: 单笔收益分布
        ax3 = axes[1, 0]
        pnl_bp.hist(bins=50, ax=ax3, color="purple", alpha=0.7, edgecolor="white")
        ax3.axvline(0, color="black", linestyle="-", linewidth=2)
        ax3.axvline(
            pnl_bp.mean(),
            color="red",
            linestyle="--",
            label=f"Mean={pnl_bp.mean():.1f}bp",
        )
        ax3.set_title("PnL Distribution per Trade")
        ax3.set_xlabel("Net PnL (bp)")
        ax3.legend()

        # 图4: 单票贡献度
        ax4 = axes[1, 1]
        top_tickers = ticker_stats.head(15)
        colors = ["green" if x > 0 else "red" for x in top_tickers["total_pnl_bp"]]
        ax4.barh(
            range(len(top_tickers)),
            top_tickers["total_pnl_bp"],
            color=colors,
            alpha=0.7,
        )
        ax4.set_yticks(range(len(top_tickers)))
        ax4.set_yticklabels(top_tickers.index)
        ax4.set_title("Top 15 Ticker Contribution")
        ax4.set_xlabel("Total PnL (bp)")
        ax4.invert_yaxis()

        plt.tight_layout()
        save_path = save_dir / "deep_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"深度分析图表已保存: {save_path}")
        plt.close()

    return {
        "time_dist": time_dist,
        "ticker_stats": ticker_stats,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae,
        "profit_factor": profit_factor,
    }


def main():
    """主函数"""
    total_start = time_module.time()

    logger.info("=" * 80)
    logger.info("日内做T回测模拟器")
    logger.info("=" * 80)

    # 1. 随机抽取股票
    stock_list = get_random_stocks(NUM_STOCKS, seed=42)

    # 2. 加载数据
    df_1m = load_minute_data(stock_list, YEAR)

    # 3. 聚合为5分钟Bar
    df_5m = aggregate_to_5min_bars(df_1m)

    # 4. 计算特征
    df_5m = calculate_features_vectorized(df_5m)

    # 5. 计算 Z-Score
    df_5m = calculate_zscore_vectorized(df_5m, window=ZSCORE_WINDOW)

    # 6. 计算昨日波动率
    df_5m = calculate_yesterday_volatility(df_5m)

    # 7. 预计算未来价格
    df_5m = prepare_future_prices(df_5m)

    # 8. 计算每日基础收益率
    daily_returns = calculate_daily_returns(df_5m)

    # 9. 运行回测
    results, trades = run_backtest(df_5m, daily_returns)

    # 10. 计算绩效指标
    metrics = calculate_metrics(results)

    # 11. 绘图
    plot_results(results, metrics, save_path=OUTPUT_DIR / "backtest_comparison.png")

    # 12. 打印报告
    print_report(metrics, results, trades, stock_list)

    # 13. 深度交易分析
    deep_stats = deep_analyze_trades(trades, save_dir=OUTPUT_DIR)

    total_time = time_module.time() - total_start
    logger.success(f"\n总耗时: {total_time:.1f}s")

    return results, trades, metrics, deep_stats


if __name__ == "__main__":
    results, trades, metrics, deep_stats = main()
