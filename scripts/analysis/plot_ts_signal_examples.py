"""
时序模型信号案例可视化脚本

职责：找出典型的高得分信号，画出全天价格/VWAP/TWAP/成交量走势
输入：ts_lgbm_predictions.pkl + df_features_{year}.pkl
输出：output/ts_signal_examples.png

图表格式（参考 accumulation_pattern_examples.png）：
  - 2x2 网格，每个子图一只股票一天
  - 左轴: 收盘价(灰线) + 累计VWAP(蓝线) + 累计TWAP(橙虚线)
  - 右轴: 5分钟成交量(浅蓝柱)
  - 标注: BUY(红箭头10:35) + SELL(绿箭头~14:05)
  - 标题: 股票代码 | 日期 | 特征值 | 收益

依赖：ts_lgbm_train.py + calc_feature_x1_x2.py

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
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")

PRED_PATH = OUTPUT_DIR / "ts_lgbm_predictions.pkl"
RAW_PATH = CACHE_DIR / "df_features_2025.pkl"

# 入场/出场 bar 索引 (0-based in day)
ENTRY_BAR_IDX = 12  # 10:30
BUY_BAR_IDX = 13    # 10:35 (买入价)
SELL_END_IDX = 37   # 14:05 (卖出VWAP结束)

# 选取案例数量
N_EXAMPLES = 4


# ==================== 数据加载 ====================
def load_data():
    """加载预测数据和原始bar数据"""
    pred = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测数据: {len(pred):,} 行")

    raw = pd.read_pickle(RAW_PATH)
    logger.success(f"加载原始bar数据: {len(raw):,} 行")

    return pred, raw


# ==================== 选取典型案例 ====================
def select_examples(pred: pd.DataFrame) -> pd.DataFrame:
    """
    选取4只典型的高得分+赚钱的股票

    标准：pred_prob 排名前5% 且 excess_ret > 0
    从中挑选4只特征各异的案例
    """
    # Top-50 (每天前50) 中赚钱最多的
    pred = pred.copy()
    pred["rank"] = pred.groupby("date")["pred_prob"].rank(ascending=False, method="first")
    top = pred[pred["rank"] <= 50].copy()

    # 按 excess_ret 降序，取赚钱的
    profitable = top[top["excess_ret"] > 0].sort_values("excess_ret", ascending=False)

    if len(profitable) < N_EXAMPLES:
        logger.warning(f"可选案例不足 {N_EXAMPLES} 只，放宽条件")
        profitable = top.sort_values("excess_ret", ascending=False)

    # 选取不同日期的案例，增加多样性
    selected = []
    seen_dates = set()
    for _, row in profitable.iterrows():
        d = str(row["date"])
        if d not in seen_dates:
            selected.append(row)
            seen_dates.add(d)
        if len(selected) >= N_EXAMPLES:
            break

    df_sel = pd.DataFrame(selected)
    logger.info(f"选取 {len(df_sel)} 只案例股票:")
    for _, row in df_sel.iterrows():
        logger.info(
            f"  {row['SecuCode']} | {row['date']} | "
            f"prob={row['pred_prob']:.3f} excess={row['excess_ret']*10000:.0f}bps"
        )

    return df_sel


# ==================== 绘制单只股票 ====================
def plot_single_stock(ax, raw_day: pd.DataFrame, info: pd.Series):
    """
    绘制单只股票的全天走势

    参数:
      ax: matplotlib axes
      raw_day: 该股票当天全部48bar的原始数据
      info: 预测结果中的一行（含特征和收益）
    """
    raw_day = raw_day.sort_values("entry_time").copy()
    n_bars = len(raw_day)

    # X轴: bar索引
    x = np.arange(n_bars)
    times = raw_day["entry_time"].values

    # ===== 左轴: 价格 =====
    # 收盘价
    ax.plot(x, raw_day["close"].values, color="gray", linewidth=1.0, alpha=0.8, label="Close")

    # 累计VWAP
    ax.plot(x, raw_day["cum_vwap"].values, color="royalblue", linewidth=1.8, label="VWAP (cum)")

    # 累计TWAP
    ax.plot(x, raw_day["cum_twap"].values, color="darkorange", linewidth=1.3, linestyle="--", label="TWAP (cum)")

    ax.set_ylabel("Price", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    # ===== 右轴: 成交量 =====
    ax2 = ax.twinx()
    ax2.bar(x, raw_day["volume"].values, alpha=0.25, color="steelblue", width=0.8)
    ax2.set_ylabel("Volume", fontsize=8, color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

    # 成交量轴范围调大，让柱子矮一些
    ax2.set_ylim(0, raw_day["volume"].max() * 3.5)

    # ===== 标注买卖点 =====
    # 午休分割线
    if n_bars > 24:
        ax.axvline(x=23.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    # BUY 标注 (bar 13 = 10:35)
    buy_idx = min(BUY_BAR_IDX, n_bars - 1)
    buy_price = raw_day.iloc[buy_idx]["close"]
    ax.annotate(
        f"BUY\n{times[buy_idx]}\n{buy_price:.2f}",
        xy=(buy_idx, buy_price),
        xytext=(buy_idx, buy_price - (raw_day["high"].max() - raw_day["low"].min()) * 0.15),
        fontsize=7,
        color="red",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # SELL 标注 (bar 37 = ~14:05)
    sell_idx = min(SELL_END_IDX, n_bars - 1)
    sell_price = raw_day.iloc[sell_idx]["close"]
    ax.annotate(
        f"SELL\n{times[sell_idx]}\n{sell_price:.2f}",
        xy=(sell_idx, sell_price),
        xytext=(sell_idx, sell_price + (raw_day["high"].max() - raw_day["low"].min()) * 0.15),
        fontsize=7,
        color="green",
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
    )

    # ===== X轴刻度 =====
    tick_interval = 4
    tick_pos = list(range(0, n_bars, tick_interval))
    tick_labels = [times[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

    # ===== 标题 =====
    raw_bps = info["raw_ret"] * 10000
    excess_bps = info["excess_ret"] * 10000
    ax.set_title(
        f"{info['SecuCode']} | {info['date']}\n"
        f"X1={info['X1']:.3f}  X2={info['X2']:.4f}  RelVol={info['rel_vol']:.1f}  "
        f"Prob={info['pred_prob']:.3f}\n"
        f"MorningRet={info['morning_ret']*100:.2f}%  "
        f"Mom5d={info['momentum_5d']*100:.1f}%  "
        f"RawRet={raw_bps:.0f}bps  ExcessRet={excess_bps:.0f}bps",
        fontsize=8,
        pad=8,
    )

    ax.grid(True, alpha=0.2)


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("时序模型信号案例可视化")
    logger.info("=" * 60)

    # 1. 加载数据
    pred, raw = load_data()

    # 2. 选取案例
    examples = select_examples(pred)

    # 3. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        'Time-Series Model Signal Examples\n'
        '"Low-Vol Momentum with Volume Expansion"',
        fontsize=13,
        fontweight="bold",
    )

    for idx, (_, info) in enumerate(examples.iterrows()):
        if idx >= N_EXAMPLES:
            break

        ax = axes[idx // 2, idx % 2]

        # 提取该股票当天的全部bar数据
        stock_day = raw[
            (raw["SecuCode"] == info["SecuCode"])
            & (raw["date"] == info["date"])
        ]

        if len(stock_day) == 0:
            logger.warning(f"未找到 {info['SecuCode']} {info['date']} 的bar数据")
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
            continue

        plot_single_stock(ax, stock_day, info)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = OUTPUT_DIR / "ts_signal_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"案例图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
