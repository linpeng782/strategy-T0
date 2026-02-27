"""
标签分布诊断脚本

研究三种标签的统计特性，为后续训练方案选择提供依据：
  1. 原始 fwd_ret：纯价格收益率（已去掉成本扣除）
  2. ret_ex：截面去均值后的超额收益（剔除大盘共同成分）
  3. target_zscore：基于ret_ex的个股时序滚动Z-Score

诊断内容：
  - 分布形态（直方图 + KDE + QQ-Plot）
  - 偏度、峰度、均值、标准差
  - 日内平稳性（各时刻均值是否为0）
  - 月度稳定性（分月箱线图）
  - 三种标签的IC对比（与原始特征）

运行：
  source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
  python label_distribution_analysis.py
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from scipy import stats
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

# ==================== 路径配置 ====================
BASE_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research")
FEAT_PKLS = [
    BASE_DIR / "output/features_csi1000_2024.pkl",
    BASE_DIR / "output/features_csi1000_2025.pkl",
]
OUTPUT_DIR = BASE_DIR / "output/label_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 参数 ====================
HOLD_PERIOD = "30min"  # 持有期
BARS_PER_DAY = 48  # 每天5分钟bar数量
ZSCORE_WINDOW_DAYS = 20  # Z-Score滚动窗口（交易日）
ZSCORE_MIN_DAYS = 5  # 最少需要的历史交易日

TRAIN_CUTOFF = pd.Timestamp("2025-01-01")

# ==================== 字体配置（英文）====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ==================== 数据加载 ====================
def load_features() -> pd.DataFrame:
    parts = []
    for p in FEAT_PKLS:
        if not p.exists():
            logger.warning(f"特征文件不存在，跳过: {p}")
            continue
        logger.info(f"加载特征文件: {p}")
        parts.append(pd.read_pickle(p))
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    logger.success(f"数据加载完成，shape={df.shape}，股票数={df['SecuCode'].nunique()}")
    return df


# ==================== 收益率计算 ====================
def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """计算纯价格收益率（不扣成本）"""
    logger.info("计算未来收益率（不含成本）...")
    if "bar_vwap" not in df.columns:
        df["bar_vwap"] = (df["amount"] / df["volume"]).replace(
            [np.inf, -np.inf], np.nan
        )

    grp = df.groupby(["SecuCode", "date"])
    df["buy_open"] = grp["bar_vwap"].shift(-1)

    hold_bars = {"15min": 3, "30min": 6, "45min": 9, "60min": 12}
    n_bars = hold_bars[HOLD_PERIOD]
    sell_vwap = grp["bar_vwap"].shift(-(1 + n_bars))
    df[f"fwd_ret_{HOLD_PERIOD}"] = sell_vwap / df["buy_open"] - 1

    logger.success(f"收益率计算完成: fwd_ret_{HOLD_PERIOD}")
    return df


# ==================== 标签构造 ====================
def build_labels(df: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    """
    构造三种标签：
      1. fwd_ret：原始纯价格收益率
      2. ret_ex：截面去均值超额收益（剔除大盘）
      3. target_zscore：个股时序滚动Z-Score（基于ret_ex，全量滚动窗口）
    """
    logger.info("构造标签...")
    t0 = time.time()

    # --- 1. 截面去均值（剔除大盘共同成分）---
    # 同一时刻（date+bar_time）截面均值代表大盘共同涨跌
    df["ret_ex"] = df[ret_col] - df.groupby(["date", "bar_time"])[ret_col].transform(
        "mean"
    )
    logger.info(f"截面去均值完成，ret_ex均值={df['ret_ex'].mean():.6f}")

    # --- 2. 个股时序滚动Z-Score（基于ret_ex，全量bar滚动，不按时刻分组）---
    # 窗口：20交易日 × 48bar = 960个点
    window_size = ZSCORE_WINDOW_DAYS * BARS_PER_DAY
    min_periods = ZSCORE_MIN_DAYS * BARS_PER_DAY

    logger.info(
        f"计算时序Z-Score（窗口={window_size}bar，min_periods={min_periods}bar）..."
    )

    # 按股票时序排列，对每只股票做全量rolling
    df = df.sort_values(["SecuCode", "date", "bar_time"]).copy()
    grp = df.groupby("SecuCode", sort=False)["ret_ex"]

    roll_mean = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).mean()
    )
    roll_std = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).std()
    )

    # shift(1)确保无前视：当前bar的Z-Score用的是历史均值/标准差
    roll_mean_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).mean().shift(1)
    )
    roll_std_lag = grp.transform(
        lambda x: x.rolling(window_size, min_periods=min_periods).std().shift(1)
    )

    df["target_zscore"] = (df["ret_ex"] - roll_mean_lag) / (roll_std_lag + 1e-8)

    logger.success(f"标签构造完成，耗时={time.time()-t0:.1f}s")
    logger.info(f"  fwd_ret: {df[ret_col].notna().sum():,} 非NaN")
    logger.info(f"  ret_ex:  {df['ret_ex'].notna().sum():,} 非NaN")
    logger.info(f"  zscore:  {df['target_zscore'].notna().sum():,} 非NaN")
    return df


# ==================== 统计摘要 ====================
def print_stats(series: pd.Series, name: str):
    """打印关键统计量"""
    s = series.dropna()
    logger.info(f"\n{'='*40}")
    logger.info(f"  {name} 统计摘要")
    logger.info(f"{'='*40}")
    logger.info(f"  样本量:     {len(s):,}")
    logger.info(f"  均值:       {s.mean():.6f}")
    logger.info(f"  标准差:     {s.std():.6f}")
    logger.info(f"  偏度:       {s.skew():.4f}  (|>1|需注意)")
    logger.info(f"  峰度:       {s.kurt():.4f}  (|>3|代表厚尾)")
    logger.info(f"  1%分位:     {s.quantile(0.01):.6f}")
    logger.info(f"  99%分位:    {s.quantile(0.99):.6f}")
    logger.info(f"  0.1%分位:   {s.quantile(0.001):.6f}")
    logger.info(f"  99.9%分位:  {s.quantile(0.999):.6f}")
    logger.info(f"  >0比例:     {(s > 0).mean():.3f}")


# ==================== 分布诊断图 ====================
def plot_label_distributions(df: pd.DataFrame, ret_col: str, split: str = "train"):
    """
    绘制三种标签的完整诊断图：
    - Row1: 分布直方图
    - Row2: QQ-Plot
    - Row3: 日内均值（平稳性）
    - Row4: 月度箱线图
    """
    logger.info(f"\n绘制标签分布诊断图（{split}集）...")

    labels = {
        f"fwd_ret ({HOLD_PERIOD})": df[ret_col].dropna(),
        "ret_ex (cross-section demeaned)": df["ret_ex"].dropna(),
        "target_zscore (rolling z-score)": df["target_zscore"].dropna(),
    }

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle(f"Label Distribution Diagnosis ({split} set)", fontsize=16, y=0.98)

    for col_idx, (label_name, series) in enumerate(labels.items()):
        s = series.dropna()
        # 为日内和月度图需要保留index，单独处理
        df_col = df[[ret_col, "ret_ex", "target_zscore", "date", "bar_time"]].copy()
        if "time" not in df_col.columns:
            df_col["time"] = pd.to_datetime(df_col["bar_time"]).dt.strftime("%H:%M")

        # ---- Row1: 分布直方图 + KDE ----
        ax = axes[0, col_idx]
        ax.hist(
            s.clip(s.quantile(0.005), s.quantile(0.995)),
            bins=100,
            density=True,
            alpha=0.6,
            color="royalblue",
            label="Histogram",
        )
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 300)
        mu, sigma = s.mean(), s.std()
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal PDF")
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.set_title(
            f"{label_name}\nSkew={s.skew():.2f}  Kurt={s.kurt():.2f}", fontsize=9
        )
        ax.set_xlabel("Value")
        ax.legend(fontsize=7)

        # ---- Row2: QQ-Plot ----
        ax2 = axes[1, col_idx]
        clip_s = s.clip(s.quantile(0.001), s.quantile(0.999))
        osm, osr = stats.probplot(clip_s, dist="norm")
        ax2.scatter(osm[0], osm[1], s=0.5, alpha=0.3, color="steelblue")
        ax2.plot(osm[0], osm[0] * osr[0] + osr[1], "r-", lw=1.5)
        ax2.set_title("QQ-Plot vs Normal", fontsize=9)
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Sample Quantiles")

        # ---- Row3: 日内均值（稳定性检查）----
        ax3 = axes[2, col_idx]
        col_key = [ret_col, "ret_ex", "target_zscore"][col_idx]
        intraday = (
            df_col.dropna(subset=[col_key])
            .groupby("time")[col_key]
            .mean()
            .reset_index()
        )
        ax3.bar(range(len(intraday)), intraday[col_key], color="teal", alpha=0.7)
        ax3.axhline(0, color="red", linestyle="--", lw=1)
        ax3.set_title("Intraday Mean (Stationarity Check)", fontsize=9)
        ax3.set_xlabel("Bar Time (index)")
        ax3.set_ylabel("Mean Value")
        # 每隔8个时刻显示一次标签
        tick_idx = list(range(0, len(intraday), 8))
        ax3.set_xticks(tick_idx)
        ax3.set_xticklabels(intraday["time"].iloc[tick_idx], rotation=45, fontsize=7)

        # ---- Row4: 月度箱线图 ----
        ax4 = axes[3, col_idx]
        df_col["month"] = pd.to_datetime(df_col["date"]).dt.to_period("M").astype(str)
        months = sorted(df_col["month"].dropna().unique())
        month_data = [
            df_col.loc[df_col["month"] == m, col_key]
            .dropna()
            .clip(s.quantile(0.01), s.quantile(0.99))
            .values
            for m in months
        ]
        bp = ax4.boxplot(month_data, patch_artist=True, flierprops={"markersize": 1})
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax4.axhline(0, color="red", linestyle="--", lw=1)
        ax4.set_title("Monthly Stability", fontsize=9)
        ax4.set_xticks(range(1, len(months) + 1))
        ax4.set_xticklabels(months, rotation=45, fontsize=6)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"label_distribution_{split}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"诊断图已保存: {save_path}")


# ==================== 日内平稳性对比图 ====================
def plot_intraday_comparison(df: pd.DataFrame, ret_col: str):
    """对比三种标签在各时刻的均值和标准差"""
    logger.info("绘制日内平稳性对比图...")

    if "time" not in df.columns:
        df = df.copy()
        df["time"] = pd.to_datetime(df["bar_time"]).dt.strftime("%H:%M")

    cols = {
        f"fwd_ret_{HOLD_PERIOD}": f"fwd_ret ({HOLD_PERIOD})",
        "ret_ex": "ret_ex (cross-section demeaned)",
        "target_zscore": "target_zscore (rolling z-score)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Intraday Stationarity: Mean & Std by Time", fontsize=14)

    for col_idx, (col_key, col_name) in enumerate(cols.items()):
        grp = df.dropna(subset=[col_key]).groupby("time")[col_key]
        intraday_mean = grp.mean()
        intraday_std = grp.std()

        ax_mean = axes[0, col_idx]
        ax_mean.bar(
            range(len(intraday_mean)),
            intraday_mean.values,
            color="steelblue",
            alpha=0.7,
        )
        ax_mean.axhline(0, color="red", linestyle="--", lw=1)
        ax_mean.set_title(f"{col_name}\nMean by Time", fontsize=9)
        ax_mean.set_xlabel("Bar index")
        tick_idx = list(range(0, len(intraday_mean), 8))
        ax_mean.set_xticks(tick_idx)
        ax_mean.set_xticklabels(intraday_mean.index[tick_idx], rotation=45, fontsize=7)

        ax_std = axes[1, col_idx]
        ax_std.bar(
            range(len(intraday_std)), intraday_std.values, color="darkorange", alpha=0.7
        )
        ax_std.set_title(f"{col_name}\nStd by Time", fontsize=9)
        ax_std.set_xlabel("Bar index")
        ax_std.set_xticks(tick_idx)
        ax_std.set_xticklabels(intraday_std.index[tick_idx], rotation=45, fontsize=7)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "intraday_stationarity.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"日内平稳性图已保存: {save_path}")


# ==================== 标签与特征的IC对比 ====================
def compare_label_ic(
    df: pd.DataFrame, ret_col: str, feat_cols: list, n_sample: int = 200000
):
    """
    对比三种标签下，代表性特征的IC，
    验证截面去均值和Z-Score标签是否能提升信号质量。
    """
    logger.info("计算三种标签下的代表性特征IC...")

    df_valid = df.dropna(subset=[ret_col, "ret_ex", "target_zscore"] + feat_cols).copy()

    # 随机采样避免计算过慢
    if len(df_valid) > n_sample:
        df_valid = df_valid.sample(n=n_sample, random_state=42)
        logger.info(f"采样 {n_sample:,} 行用于IC计算")

    label_map = {
        f"fwd_ret ({HOLD_PERIOD})": ret_col,
        "ret_ex": "ret_ex",
        "target_zscore": "target_zscore",
    }

    rows = []
    for feat in feat_cols:
        row = {"feature": feat}
        for label_name, label_col in label_map.items():
            corr, _ = stats.spearmanr(df_valid[feat], df_valid[label_col])
            row[label_name] = round(corr, 4)
        rows.append(row)

    df_ic = pd.DataFrame(rows).set_index("feature")
    logger.info(f"\n{'='*60}")
    logger.info("特征 vs 各标签的Spearman IC（随机采样）：")
    logger.info(f"\n{df_ic.to_string()}")

    df_ic.to_csv(OUTPUT_DIR / "label_ic_comparison.csv")
    logger.success(f"IC对比结果已保存: {OUTPUT_DIR / 'label_ic_comparison.csv'}")
    return df_ic


# ==================== 主流程 ====================
def main():
    logger.info("=" * 60)
    logger.info("标签分布诊断分析")
    logger.info("=" * 60)

    # 加载数据
    df = load_features()
    df = compute_forward_returns(df)

    ret_col = f"fwd_ret_{HOLD_PERIOD}"

    # 过滤有效收益率
    df = df[df[ret_col].notna()].copy()
    logger.info(f"有效样本（含收益率）: {len(df):,} 行")

    # 构造三种标签
    df = build_labels(df, ret_col)

    # 打印统计摘要
    print_stats(df[ret_col], f"fwd_ret_{HOLD_PERIOD}")
    print_stats(df["ret_ex"], "ret_ex（截面去均值）")
    print_stats(df["target_zscore"], "target_zscore（时序Z-Score）")

    # 分训练集和测试集分别诊断（date列可能是datetime.date类型，统一转为datetime64）
    df["date"] = pd.to_datetime(df["date"])
    df_train = df[df["date"] < TRAIN_CUTOFF].copy()
    df_test = df[df["date"] >= TRAIN_CUTOFF].copy()
    logger.info(
        f"\n训练集: {len(df_train):,}行，{df_train['date'].min().date()} ~ {df_train['date'].max().date()}"
    )
    logger.info(
        f"测试集: {len(df_test):,}行，{df_test['date'].min().date()} ~ {df_test['date'].max().date()}"
    )

    # 分布诊断图
    logger.info("\n对训练集绘制分布诊断图...")
    plot_label_distributions(df_train, ret_col, split="train")

    logger.info("\n对测试集绘制分布诊断图...")
    plot_label_distributions(df_test, ret_col, split="test")

    # 日内平稳性对比（全量）
    if "time" not in df.columns:
        df["time"] = pd.to_datetime(df["bar_time"]).dt.strftime("%H:%M")
    plot_intraday_comparison(df, ret_col)

    # 特征IC对比（代表性特征）
    feat_cols_sample = [
        "vwap_bias_norm",
        "range_pos_norm",
        "dif_norm",
        "dea_norm",
        "rsv20_norm",
        "ret_5bar_norm",
        "cord20_norm",
        "buy_pressure_norm",
    ]
    feat_cols_valid = [c for c in feat_cols_sample if c in df.columns]
    if feat_cols_valid:
        compare_label_ic(df, ret_col, feat_cols_valid)

    # 打印关键结论
    logger.info("\n" + "=" * 60)
    logger.info("诊断结论摘要")
    logger.info("=" * 60)

    for label_col, label_name in [
        (ret_col, "fwd_ret"),
        ("ret_ex", "ret_ex"),
        ("target_zscore", "target_zscore"),
    ]:
        s = df[label_col].dropna()
        skew_ok = abs(s.skew()) < 1
        kurt_ok = abs(s.kurt()) < 3
        # 日内平稳性：各时刻均值的绝对值均值
        if "time" in df.columns:
            intraday_bias = (
                df.dropna(subset=[label_col])
                .groupby("time")[label_col]
                .mean()
                .abs()
                .mean()
            )
            stable_ok = intraday_bias < 0.05 * s.std()
        else:
            intraday_bias = None
            stable_ok = None

        bias_str = f"{intraday_bias:.6f}" if intraday_bias is not None else "N/A"
        logger.info(
            f"  {label_name:30s} | skew={s.skew():+.2f}({'OK' if skew_ok else 'WARN'}) "
            f"| kurt={s.kurt():+.2f}({'OK' if kurt_ok else 'WARN'}) "
            f"| intraday_bias={bias_str}"
        )

    logger.success(f"\n所有诊断图和数据已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
