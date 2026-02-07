#!/usr/bin/env python3
"""
ML驱动版日内做T回测模拟器（缓存版）
==================================
对比三组策略的净值曲线：
1. 对照组（Buy & Hold）：持有股票不动
2. 规则版（Rule-Based）：传统Z阈值过滤
3. AI过滤版（ML-Optimized）：规则 + LGBM模型概率过滤

数据加载方式：
- 直接从本地缓存读取预处理好的数据
- 缓存由 prepare_backtest_data.py 生成
- 避免每次回测都重新从米筐拉取和计算
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import lightgbm as lgb
import json
import time as time_module

# ==================== 配置参数 ====================
# 缓存目录（由 prepare_backtest_data.py 生成）
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")

# 输出目录
OUTPUT_DIR = Path(
    "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/backtest_ml"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径
MODEL_PATH = Path(
    "/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/model/lgbm_signal_classifier.txt"
)

# 交易参数
Z_LOWER = -5.0
Z_UPPER = -1.5
HOLDING_PERIOD = 60
TRADE_COST = 0.0015  # 15bp（与notebook保持一致）

# 止盈止损参数（与模型训练保持一致）
TAKE_PROFIT_BP = 70  # 止盈目标 +80bp
STOP_LOSS_BP = 40  # 止损目标 -40bp
T_LEVERAGE = 0.5  # 做T资金占底仓一半

# ML参数（V6完整版：12个特征，与train_lgbm.py保持一致）
ML_FEATURES = [
    "mkt_oversold_ratio",  # 大盘超跌占比
    "mkt_avg_x1",  # 大盘价格偏离
    "mkt_avg_z",  # 大盘引力中枢
    "mkt_ret_15m",  # 大盘15分钟动量
    "time_val",  # 时间特征
    "yesterday_range",  # 昨日波动率
    "X1_zscore",  # 个股价格偏离
    "vol_burst",  # 成交量爆发力
    "x1_slope",  # 价格动量斜率
    "Z_final",  # 个股引力
    "relative_z",  # 个股相对强度（独立超跌）
    "volatility_ratio",  # 波动率放大比
]
ML_PROB_THRESHOLD = 0.3  # 概率阈值（根据概率分布中位数0.475调整，取中位数以上）

# 设置字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_cached_data():
    """从本地缓存加载预处理好的数据"""
    logger.info(f"从缓存加载数据: {CACHE_DIR}")

    # 检查缓存是否存在
    if not CACHE_DIR.exists():
        raise FileNotFoundError(
            f"缓存目录不存在: {CACHE_DIR}\n"
            "请先运行 prepare_backtest_data.py 生成缓存数据"
        )

    # 加载元数据
    metadata_path = CACHE_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"缓存创建时间: {metadata.get('created_at', 'unknown')}")
        logger.info(f"股票数量: {metadata.get('n_stocks', 'unknown')}")

    # 加载5分钟数据
    df_5m_path = CACHE_DIR / "df_5m.pkl"
    logger.info(f"加载 df_5m.pkl...")
    df_5m = pd.read_pickle(df_5m_path)
    logger.success(f"  - {len(df_5m):,} 行, {df_5m['SecuCode'].nunique()} 只股票")

    # 加载每日收益率
    daily_returns_path = CACHE_DIR / "daily_returns.pkl"
    logger.info(f"加载 daily_returns.pkl...")
    daily_returns = pd.read_pickle(daily_returns_path)
    logger.success(f"  - {len(daily_returns):,} 行")

    # 加载股票列表
    stock_list_path = CACHE_DIR / "stock_list.pkl"
    logger.info(f"加载 stock_list.pkl...")
    stock_list = pd.read_pickle(stock_list_path)
    logger.success(f"  - {len(stock_list)} 只股票")

    logger.success("缓存数据加载完成!")
    return df_5m, daily_returns, stock_list


def load_ml_model(model_path: Path):
    """加载LGBM模型"""
    logger.info(f"加载 LGBM 模型: {model_path}")
    bst = lgb.Booster(model_file=str(model_path))
    logger.success("模型加载完成")
    return bst


def add_ml_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    """为每个Bar添加ML概率分数"""
    logger.info("计算 ML 信号概率分数...")

    # 检查特征是否存在
    missing_features = [f for f in ML_FEATURES if f not in df.columns]
    if missing_features:
        logger.warning(f"缺失特征: {missing_features}")
        # 如果time_val不存在，则计算
        if "time_val" in missing_features:
            df["time_val"] = df["bar_time"].dt.hour + df["bar_time"].dt.minute / 60.0
            missing_features.remove("time_val")
        if missing_features:
            raise ValueError(f"缺失必要特征: {missing_features}")

    # 提取特征矩阵
    X = df[ML_FEATURES].copy()

    # 处理缺失值（模型预测时需要）
    X = X.fillna(0)

    # 预测概率
    df["ml_prob"] = model.predict(X)

    # 查看概率分布
    valid_probs = df["ml_prob"][df["ml_prob"].notna()]
    logger.info(
        f"ML概率分布: min={valid_probs.min():.3f}, max={valid_probs.max():.3f}, "
        f"mean={valid_probs.mean():.3f}, median={valid_probs.median():.3f}"
    )

    logger.success("ML 信号打分完成")
    return df


def run_backtest_comparison(
    df: pd.DataFrame, daily_returns: pd.DataFrame, prob_threshold: float
):
    """
    运行对比回测：规则版 vs ML版（进攻型动态杠杆 V3）

    动态杠杆逻辑（3档分级，进攻型）：
    - ml_prob >= 0.50: 杠杆 1.2（顶级+优质信号，超额押注）
    - ml_prob >= 0.46: 杠杆 0.8（中等信号，主力头寸）
    - ml_prob >= 0.43: 杠杆 0.4（边缘信号，贡献频率）
    - ml_prob < 0.43:  杠杆 0.0（垃圾信号，绝对拦截）
    """
    logger.info(f"运行对比回测（进攻型动态杠杆版）...")

    # 1. 昨日高波动滤网
    vol_threshold = df["yesterday_range"].quantile(0.9)
    logger.info(f"昨日波动率阈值: {vol_threshold*100:.2f}%")

    # 2. 基础规则条件
    rule_mask = (
        np.isfinite(df["Z_final"])
        & (df["Z_final"] > Z_LOWER)
        & (df["Z_final"] < Z_UPPER)
        & (df["close"] < df["day_high"])
        & np.isfinite(df["yesterday_range"])
        & (df["yesterday_range"] > vol_threshold)
        & (df["time_str"] < "14:15")
        & np.isfinite(df["close_exit"])
    )

    # 3. 规则版信号（固定杠杆 0.5）
    df["is_trade_rule"] = rule_mask

    # 4. ML版：计算动态杠杆（3档分级，进攻型）
    # 注意：新模型的概率分布非常窄（都在0.22左右），需要使用百分位数作为阈值
    # 计算实际的百分位数阈值
    p25 = df["ml_prob"].quantile(0.25)
    p50 = df["ml_prob"].quantile(0.50)
    p75 = df["ml_prob"].quantile(0.75)

    logger.info(f"  - 动态杠杆阈值: 25%={p25:.4f}, 50%={p50:.4f}, 75%={p75:.4f}")

    conditions = [
        df["ml_prob"] >= p75,  # 顶级+优质信号：前25%，超额押注
        df["ml_prob"] >= p50,  # 中等信号：中位数以上，主力头寸
        df["ml_prob"] >= p25,  # 边缘信号：25%分位以上，贡献频率
    ]
    choices = [1.2, 0.8, 0.4]  # 进攻型杠杆配置
    df["dynamic_leverage"] = np.select(conditions, choices, default=0.0)

    # ML版信号：规则满足 且 动态杠杆 > 0
    df["is_trade_ml"] = rule_mask & (df["dynamic_leverage"] > 0)

    n_rule = df["is_trade_rule"].sum()
    n_ml = df["is_trade_ml"].sum()
    n_top = (rule_mask & (df["ml_prob"] >= 0.50)).sum()
    n_mid = (rule_mask & (df["ml_prob"] >= 0.46) & (df["ml_prob"] < 0.50)).sum()
    n_low = (rule_mask & (df["ml_prob"] >= 0.3) & (df["ml_prob"] < 0.46)).sum()
    n_skip = (rule_mask & (df["ml_prob"] < 0.3)).sum()

    logger.info(
        f"规则版信号: {n_rule:,} | ML版信号: {n_ml:,} | 过滤比例: {(1-n_ml/n_rule)*100:.1f}%"
    )
    logger.info(f"  - 顶级信号(杠杆1.2): {n_top:,} ({n_top/n_rule*100:.1f}%)")
    logger.info(f"  - 中等信号(杠杆0.8): {n_mid:,} ({n_mid/n_rule*100:.1f}%)")
    logger.info(f"  - 边缘信号(杠杆0.4): {n_low:,} ({n_low/n_rule*100:.1f}%)")
    logger.info(f"  - 放弃交易(杠杆0.0): {n_skip:,} ({n_skip/n_rule*100:.1f}%)")

    # 5. 计算收益（止盈止损逻辑）
    tp_ratio = TAKE_PROFIT_BP / 10000
    sl_ratio = STOP_LOSS_BP / 10000

    df["tp_price"] = df["close"] * (1 + tp_ratio)
    df["sl_price"] = df["close"] * (1 - sl_ratio)

    df["hit_tp"] = df["future_high"] >= df["tp_price"]
    df["hit_sl"] = df["future_low"] <= df["sl_price"]

    df["t_gross_return"] = np.where(
        df["hit_tp"],
        tp_ratio,
        np.where(
            df["hit_sl"],
            -sl_ratio,
            (df["close_exit"] / df["close"]) - 1,
        ),
    )
    df["t_net_return"] = df["t_gross_return"] - TRADE_COST

    # 6. 分别计算两版的每日收益（核心差异：动态杠杆 vs 固定杠杆）
    results_list = []

    for version, is_trade_col in [("rule", "is_trade_rule"), ("ml", "is_trade_ml")]:
        df_trades = df[df[is_trade_col]].copy()

        if version == "ml":
            # ML版：使用动态杠杆计算收益贡献
            df_trades["t_contribution"] = (
                df_trades["t_net_return"] * df_trades["dynamic_leverage"]
            )
        else:
            # 规则版：使用固定杠杆 0.5
            df_trades["t_contribution"] = df_trades["t_net_return"] * T_LEVERAGE

        daily_t_alpha = (
            df_trades.groupby(["date", "SecuCode"])["t_contribution"]
            .sum()
            .reset_index()
            .rename(columns={"t_contribution": f"t_alpha_{version}"})
        )

        results_list.append(daily_t_alpha)

    # 7. 合并结果
    results = daily_returns.copy()
    for r in results_list:
        col = [c for c in r.columns if c.startswith("t_alpha")][0]
        results = results.merge(r, on=["date", "SecuCode"], how="left")
        results[col] = results[col].fillna(0)

    # 8. 计算组合收益
    daily_portfolio = (
        results.groupby("date")
        .agg(
            {
                "daily_return": "mean",
                "t_alpha_rule": "mean",
                "t_alpha_ml": "mean",
            }
        )
        .reset_index()
    )

    daily_portfolio["hold_only"] = daily_portfolio["daily_return"]
    daily_portfolio["with_rule"] = (
        daily_portfolio["daily_return"] + daily_portfolio["t_alpha_rule"]
    )
    daily_portfolio["with_ml"] = (
        daily_portfolio["daily_return"] + daily_portfolio["t_alpha_ml"]
    )

    daily_portfolio = daily_portfolio.dropna(subset=["hold_only"])

    # 9. 累计净值
    daily_portfolio["cum_hold"] = (1 + daily_portfolio["hold_only"]).cumprod()
    daily_portfolio["cum_rule"] = (1 + daily_portfolio["with_rule"]).cumprod()
    daily_portfolio["cum_ml"] = (1 + daily_portfolio["with_ml"]).cumprod()

    # 10. 提取交易明细
    trades_rule = df[df["is_trade_rule"]].copy()
    trades_ml = df[df["is_trade_ml"]].copy()

    logger.success(f"回测完成: {len(daily_portfolio)} 个交易日")

    return daily_portfolio, trades_rule, trades_ml


def calculate_metrics(results: pd.DataFrame) -> dict:
    """计算绩效指标"""
    metrics = {}

    strategies = [
        ("hold_only", "cum_hold"),
        ("with_rule", "cum_rule"),
        ("with_ml", "cum_ml"),
    ]

    for strategy, cum_col in strategies:
        returns = results[strategy]

        total_return = results[cum_col].iloc[-1] - 1
        n_days = len(results)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0

        cum_max = results[cum_col].cummax()
        drawdown = (results[cum_col] - cum_max) / cum_max
        max_drawdown = drawdown.min()

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


def calculate_trade_stats(trades: pd.DataFrame, name: str) -> dict:
    """计算交易统计"""
    n_trades = len(trades)
    if n_trades == 0:
        return {"name": name, "n_trades": 0}

    n_tp = trades["hit_tp"].sum()
    n_sl = (~trades["hit_tp"] & trades["hit_sl"]).sum()
    n_timeout = (~trades["hit_tp"] & ~trades["hit_sl"]).sum()

    avg_gross = trades["t_gross_return"].mean() * 10000
    avg_net = trades["t_net_return"].mean() * 10000
    win_rate = (trades["t_net_return"] > 0).mean()

    # 获利因子
    gross_profit = trades[trades["t_net_return"] > 0]["t_net_return"].sum()
    gross_loss = abs(trades[trades["t_net_return"] < 0]["t_net_return"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "name": name,
        "n_trades": n_trades,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_timeout": n_timeout,
        "tp_rate": n_tp / n_trades,
        "sl_rate": n_sl / n_trades,
        "avg_gross_bp": avg_gross,
        "avg_net_bp": avg_net,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }


def plot_comparison(results: pd.DataFrame, metrics: dict, save_path: Path = None):
    """绘制三线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1: 净值曲线对比
    ax1 = axes[0, 0]
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
        results["cum_rule"],
        label="Rule-Based T",
        color="orange",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
    )
    ax1.plot(
        results["date"],
        results["cum_ml"],
        label="ML-Optimized T",
        color="red",
        linewidth=2.5,
    )

    ax1.set_title(
        "Net Value Comparison: Buy & Hold vs Rule-Based vs ML-Optimized", fontsize=12
    )
    ax1.set_ylabel("Cumulative Net Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 图2: 每日Alpha对比
    ax2 = axes[0, 1]
    ax2.bar(
        range(len(results)),
        results["t_alpha_rule"] * 10000,
        alpha=0.5,
        label="Rule Alpha",
        color="orange",
        width=1.0,
    )
    ax2.bar(
        range(len(results)),
        results["t_alpha_ml"] * 10000,
        alpha=0.7,
        label="ML Alpha",
        color="red",
        width=0.5,
    )
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_title("Daily T-Alpha Comparison (bp)")
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Daily Alpha (bp)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 累计Alpha对比
    ax3 = axes[1, 0]
    cum_alpha_rule = (results["t_alpha_rule"].cumsum()) * 10000
    cum_alpha_ml = (results["t_alpha_ml"].cumsum()) * 10000
    ax3.plot(
        results["date"],
        cum_alpha_rule,
        label="Rule Cumulative Alpha",
        color="orange",
        linewidth=2,
    )
    ax3.plot(
        results["date"],
        cum_alpha_ml,
        label="ML Cumulative Alpha",
        color="red",
        linewidth=2,
    )
    ax3.fill_between(
        results["date"],
        cum_alpha_rule,
        cum_alpha_ml,
        where=cum_alpha_ml > cum_alpha_rule,
        color="green",
        alpha=0.3,
        label="ML Advantage",
    )
    ax3.set_title("Cumulative T-Alpha (bp)")
    ax3.set_ylabel("Cumulative Alpha (bp)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 绩效指标对比
    ax4 = axes[1, 1]
    labels = ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"]
    rule_vals = [
        metrics["with_rule"]["total_return"] * 100,
        metrics["with_rule"]["sharpe"],
        metrics["with_rule"]["max_drawdown"] * 100,
        metrics["with_rule"]["win_rate"] * 100,
    ]
    ml_vals = [
        metrics["with_ml"]["total_return"] * 100,
        metrics["with_ml"]["sharpe"],
        metrics["with_ml"]["max_drawdown"] * 100,
        metrics["with_ml"]["win_rate"] * 100,
    ]

    x = np.arange(len(labels))
    width = 0.35
    ax4.bar(
        x - width / 2, rule_vals, width, label="Rule-Based", color="orange", alpha=0.7
    )
    ax4.bar(x + width / 2, ml_vals, width, label="ML-Optimized", color="red", alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.set_title("Performance Metrics Comparison")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"对比图表已保存: {save_path}")

    plt.close()


def print_comparison_report(metrics: dict, stats_rule: dict, stats_ml: dict):
    """打印对比报告"""
    print("\n" + "=" * 90)
    print("ML驱动版回测对比报告")
    print("=" * 90)

    print(f"\n📊 策略绩效对比:")
    print(
        f"   {'指标':>15} | {'Buy & Hold':>12} | {'规则版':>12} | {'ML版':>12} | {'ML提升':>10}"
    )
    print(f"   {'-' * 75}")

    hold = metrics["hold_only"]
    rule = metrics["with_rule"]
    ml = metrics["with_ml"]

    print(
        f"   {'总收益率':>15} | {hold['total_return']*100:>11.2f}% | {rule['total_return']*100:>11.2f}% | {ml['total_return']*100:>11.2f}% | {(ml['total_return']-rule['total_return'])*100:>9.2f}%"
    )
    print(
        f"   {'年化收益率':>15} | {hold['annual_return']*100:>11.2f}% | {rule['annual_return']*100:>11.2f}% | {ml['annual_return']*100:>11.2f}% | {(ml['annual_return']-rule['annual_return'])*100:>9.2f}%"
    )
    print(
        f"   {'年化波动率':>15} | {hold['annual_vol']*100:>11.2f}% | {rule['annual_vol']*100:>11.2f}% | {ml['annual_vol']*100:>11.2f}% | {(ml['annual_vol']-rule['annual_vol'])*100:>9.2f}%"
    )
    print(
        f"   {'夏普比率':>15} | {hold['sharpe']:>12.2f} | {rule['sharpe']:>12.2f} | {ml['sharpe']:>12.2f} | {ml['sharpe']-rule['sharpe']:>10.2f}"
    )
    print(
        f"   {'最大回撤':>15} | {hold['max_drawdown']*100:>11.2f}% | {rule['max_drawdown']*100:>11.2f}% | {ml['max_drawdown']*100:>11.2f}% | {(ml['max_drawdown']-rule['max_drawdown'])*100:>9.2f}%"
    )

    print(f"\n📈 交易统计对比:")
    print(f"   {'指标':>20} | {'规则版':>15} | {'ML版':>15} | {'变化':>12}")
    print(f"   {'-' * 70}")

    print(
        f"   {'交易笔数':>20} | {stats_rule['n_trades']:>15,} | {stats_ml['n_trades']:>15,} | {stats_ml['n_trades']-stats_rule['n_trades']:>+12,}"
    )
    print(
        f"   {'止盈率':>20} | {stats_rule['tp_rate']*100:>14.1f}% | {stats_ml['tp_rate']*100:>14.1f}% | {(stats_ml['tp_rate']-stats_rule['tp_rate'])*100:>+11.1f}%"
    )
    print(
        f"   {'止损率':>20} | {stats_rule['sl_rate']*100:>14.1f}% | {stats_ml['sl_rate']*100:>14.1f}% | {(stats_ml['sl_rate']-stats_rule['sl_rate'])*100:>+11.1f}%"
    )
    print(
        f"   {'平均毛收益(bp)':>20} | {stats_rule['avg_gross_bp']:>15.2f} | {stats_ml['avg_gross_bp']:>15.2f} | {stats_ml['avg_gross_bp']-stats_rule['avg_gross_bp']:>+12.2f}"
    )
    print(
        f"   {'平均净收益(bp)':>20} | {stats_rule['avg_net_bp']:>15.2f} | {stats_ml['avg_net_bp']:>15.2f} | {stats_ml['avg_net_bp']-stats_rule['avg_net_bp']:>+12.2f}"
    )
    print(
        f"   {'交易胜率':>20} | {stats_rule['win_rate']*100:>14.1f}% | {stats_ml['win_rate']*100:>14.1f}% | {(stats_ml['win_rate']-stats_rule['win_rate'])*100:>+11.1f}%"
    )
    print(
        f"   {'获利因子(PF)':>20} | {stats_rule['profit_factor']:>15.2f} | {stats_ml['profit_factor']:>15.2f} | {stats_ml['profit_factor']-stats_rule['profit_factor']:>+12.2f}"
    )

    # 结论
    print(f"\n🎯 核心结论:")
    trade_reduction = (1 - stats_ml["n_trades"] / stats_rule["n_trades"]) * 100
    avg_net_improve = stats_ml["avg_net_bp"] - stats_rule["avg_net_bp"]
    pf_improve = stats_ml["profit_factor"] - stats_rule["profit_factor"]

    print(
        f"   - 交易频率降低: {trade_reduction:.1f}% (从{stats_rule['n_trades']:,}笔减少到{stats_ml['n_trades']:,}笔)"
    )
    print(f"   - 单笔净收益提升: {avg_net_improve:+.2f}bp")
    print(f"   - 获利因子提升: {pf_improve:+.2f}")

    if (
        stats_ml["profit_factor"] > stats_rule["profit_factor"]
        and stats_ml["avg_net_bp"] > stats_rule["avg_net_bp"]
    ):
        print(f"\n   ✅ ML过滤器有效！通过剔除低质量信号，提升了交易效率和盈利能力")
    else:
        print(f"\n   ⚠️ ML过滤器效果有限，可能需要调整概率阈值或重新训练模型")


def main():
    """主函数"""
    total_start = time_module.time()

    logger.info("=" * 80)
    logger.info("ML驱动版日内做T回测模拟器（缓存版）")
    logger.info("=" * 80)

    # 1. 从缓存加载预处理好的数据
    df_5m, daily_returns, stock_list = load_cached_data()

    # 2. 加载LGBM模型
    model = load_ml_model(MODEL_PATH)

    # 3. 添加ML分数
    df_5m = add_ml_scores(df_5m, model)

    # 4. 查看ML概率分布
    print(f"\n📊 ML概率分布统计:")
    probs = df_5m["ml_prob"]
    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"   {int(q*100)}%分位数: {probs.quantile(q):.3f}")

    # 5. 运行对比回测
    results, trades_rule, trades_ml = run_backtest_comparison(
        df_5m, daily_returns, ML_PROB_THRESHOLD
    )

    # 6. 计算绩效指标
    metrics = calculate_metrics(results)

    # 7. 计算交易统计
    stats_rule = calculate_trade_stats(trades_rule, "规则版")
    stats_ml = calculate_trade_stats(trades_ml, "ML版")

    # 8. 绘图
    plot_comparison(results, metrics, save_path=OUTPUT_DIR / "ml_comparison.png")

    # 9. 打印报告
    print_comparison_report(metrics, stats_rule, stats_ml)

    total_time = time_module.time() - total_start
    logger.success(f"\n总耗时: {total_time:.1f}s")

    return results, trades_rule, trades_ml, metrics


if __name__ == "__main__":
    results, trades_rule, trades_ml, metrics = main()
