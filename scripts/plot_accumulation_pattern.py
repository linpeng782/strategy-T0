"""
Quiet Institutional Accumulation Pattern Visualization

Plot 4 example stocks showing the "Others" pattern:
  - Price slightly above VWAP (X1 > 0)
  - Volume concentrated at low prices (X2 < -1)
  - Moderate volume expansion (rel_vol ~ 1-3)
  - Net return > 50bps after cost

Each subplot shows:
  - 5-min bar close price (thin gray line)
  - Cumulative VWAP (blue line)
  - Cumulative TWAP (orange dashed line)
  - Buy signal (red up arrow)
  - Sell point (green down arrow)
  - Volume bars at bottom

Input: df_features_2025.pkl + lgbm_test_predictions.pkl
Output: output/accumulation_pattern_examples.png
"""

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
PRED_PATH = OUTPUT_DIR / "lgbm_test_predictions.pkl"
FEAT_PATH = CACHE_DIR / "df_features_2025.pkl"

COST_BPS = 15
COST_RATE = COST_BPS / 10000
HOLD_BARS = 24  # 120分钟 / 5分钟 = 24 bars

# 选定的4只股票（严格符合 Others 画像：X1>0 价格略高于VWAP + X2<-1 成交偏低位）
EXAMPLES = [
    ("601083", "2025-04-09", "11:10"),  # X1=+0.62, X2=-3.55, rv=1.4
    ("002068", "2025-04-09", "11:05"),  # X1=+0.57, X2=-1.13, rv=2.2
    ("301071", "2025-10-13", "09:55"),  # X1=+1.08, X2=-2.50, rv=2.2
    ("600395", "2025-10-13", "09:50"),  # X1=+1.16, X2=-2.43, rv=2.1
]


def load_data():
    """加载特征数据和预测数据"""
    logger.info("加载 2025 年特征数据...")
    feat = pd.read_pickle(FEAT_PATH)
    logger.success(f"加载完成: {len(feat):,} 行")

    pred = pd.read_pickle(PRED_PATH)
    logger.success(f"加载预测结果: {len(pred):,} 行")

    return feat, pred


def get_stock_day_data(
    feat: pd.DataFrame, secu_code: str, date_str: str
) -> pd.DataFrame:
    """获取某只股票某天的全天5分钟数据"""
    # date 列是 datetime.date 类型
    y, m, d = map(int, date_str.split("-"))
    date_obj = datetime.date(y, m, d)
    mask = (feat["SecuCode"] == secu_code) & (feat["date"] == date_obj)
    day_data = feat[mask].copy()
    day_data = day_data.sort_values("entry_time").reset_index(drop=True)
    return day_data


def plot_examples(feat: pd.DataFrame, pred: pd.DataFrame):
    """绘制4只股票的日内交易画像"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes_flat = axes.flatten()

    for idx, (secu_code, date, entry_time) in enumerate(EXAMPLES):
        ax = axes_flat[idx]

        # 获取全天数据
        day_data = get_stock_day_data(feat, secu_code, date)
        if day_data.empty:
            logger.warning(f"未找到数据: {secu_code} {date}")
            continue

        n_bars = len(day_data)
        times = day_data["entry_time"].values
        close_prices = day_data["close"].values
        vwap_prices = day_data["cum_vwap"].values
        twap_prices = day_data["cum_twap"].values
        volumes = day_data["volume"].values

        # 找到买入和卖出的 bar index
        buy_idx = day_data[day_data["entry_time"] == entry_time].index
        if len(buy_idx) == 0:
            logger.warning(f"未找到入场时间: {secu_code} {date} {entry_time}")
            continue

        buy_bar_pos = day_data.index.get_loc(buy_idx[0])
        # 买入价: 下一个 bar 的 vwap
        buy_exec_pos = buy_bar_pos + 1
        # 卖出价: 买入 bar 之后第 HOLD_BARS 个 bar 的 vwap
        sell_exec_pos = buy_bar_pos + 1 + HOLD_BARS

        if buy_exec_pos >= n_bars or sell_exec_pos >= n_bars:
            logger.warning(f"Bar 不足: {secu_code} {date} {entry_time}")
            continue

        buy_price = day_data.iloc[buy_exec_pos]["vwap_5m"]
        sell_price = day_data.iloc[sell_exec_pos]["vwap_5m"]
        buy_time = day_data.iloc[buy_exec_pos]["entry_time"]
        sell_time = day_data.iloc[sell_exec_pos]["entry_time"]
        raw_ret = (sell_price - buy_price) / buy_price
        net_ret = raw_ret - COST_RATE

        # 获取信号特征
        y, m, d = map(int, date.split("-"))
        date_obj_pred = datetime.date(y, m, d)
        sig_mask = (
            (pred["SecuCode"] == secu_code)
            & (pred["date"] == date_obj_pred)
            & (pred["entry_time"] == entry_time)
        )
        sig_row = pred[sig_mask]
        x1_val = sig_row["X1_zscore"].values[0] if len(sig_row) > 0 else np.nan
        x2_val = sig_row["X2_zscore"].values[0] if len(sig_row) > 0 else np.nan
        rv_val = sig_row["rel_vol"].values[0] if len(sig_row) > 0 else np.nan
        prob_val = sig_row["pred_prob"].values[0] if len(sig_row) > 0 else np.nan

        # --- 绘制价格 ---
        x_range = np.arange(n_bars)

        # 成交量柱状图（底部，降低透明度避免遮挡价格线）
        ax2 = ax.twinx()
        ax2.bar(x_range, volumes, alpha=0.25, color="#6a9fd8", width=0.8)
        ax2.set_ylabel("Volume", fontsize=8, color="steelblue")
        ax2.tick_params(axis="y", labelsize=7, colors="steelblue")

        # 价格线（加粗加深）
        ax.plot(
            x_range,
            close_prices,
            color="#555555",
            alpha=0.7,
            linewidth=1.2,
            label="Close",
        )
        ax.plot(
            x_range, vwap_prices, color="#0050d0", linewidth=2.5, label="VWAP (cum)"
        )
        ax.plot(
            x_range,
            twap_prices,
            color="#e05000",
            linewidth=2.0,
            linestyle="--",
            label="TWAP (cum)",
        )

        # 买入箭头（红色向上）
        ax.annotate(
            "",
            xy=(buy_exec_pos, buy_price),
            xytext=(
                buy_exec_pos,
                buy_price - (close_prices.max() - close_prices.min()) * 0.15,
            ),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        )
        ax.text(
            buy_exec_pos,
            buy_price - (close_prices.max() - close_prices.min()) * 0.18,
            f"BUY\n{buy_time}\n{buy_price:.3f}",
            ha="center",
            va="top",
            fontsize=7,
            color="red",
            fontweight="bold",
        )

        # 卖出箭头（绿色向下）
        ax.annotate(
            "",
            xy=(sell_exec_pos, sell_price),
            xytext=(
                sell_exec_pos,
                sell_price + (close_prices.max() - close_prices.min()) * 0.15,
            ),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5),
        )
        ax.text(
            sell_exec_pos,
            sell_price + (close_prices.max() - close_prices.min()) * 0.18,
            f"SELL\n{sell_time}\n{sell_price:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="green",
            fontweight="bold",
        )

        # 信号时刻竖线
        ax.axvline(buy_bar_pos, color="red", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(
            buy_bar_pos,
            close_prices.max(),
            f"Signal\n{entry_time}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="red",
            alpha=0.7,
        )

        # X 轴刻度
        tick_interval = max(1, n_bars // 12)
        tick_positions = list(range(0, n_bars, tick_interval))
        tick_labels = [times[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)

        # 标题
        ax.set_title(
            f"{secu_code} | {date}\n"
            f"X1={x1_val:.2f}  X2={x2_val:.2f}  RelVol={rv_val:.1f}  Prob={prob_val:.3f}\n"
            f"NetRet={net_ret*10000:.0f}bps  RawRet={raw_ret*10000:.0f}bps",
            fontsize=10,
        )
        ax.set_ylabel("Price", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)

    plt.suptitle(
        "Quiet Institutional Accumulation Pattern\n"
        '"Price slightly above VWAP + Volume concentrated at low prices + Moderate expansion"',
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    save_path = OUTPUT_DIR / "accumulation_pattern_examples.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.success(f"图表已保存: {save_path}")
    plt.close()


def main():
    logger.info("=" * 60)
    logger.info("Quiet Institutional Accumulation Pattern Visualization")
    logger.info("=" * 60)

    feat, pred = load_data()
    plot_examples(feat, pred)

    logger.success("绘图完成!")


if __name__ == "__main__":
    main()
