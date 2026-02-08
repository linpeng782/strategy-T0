"""
相关性测试脚本：特征与标签的相关性分析

职责：读取特征数据，按 (入场时间, 持有时间) 组合计算相关性
输入：df_features_{year}.pkl

分析维度：
  - 入场时间: ENTRY_TIMES 指定的时间点
  - 持有时间: HOLD_HORIZONS 指定的标签列
  - 全样本 + 极值样本 (|X2_zscore| > 阈值)

依赖：calc_feature_x1_x2.py 生产的特征数据

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
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(exist_ok=True)

YEAR = 2025

# 入场时间列表
# ENTRY_TIMES = ["09:45", "10:00", "10:30", "11:00", "13:05", "13:30", "14:00"]
ENTRY_TIMES = ["10:30"]


# 持有时间列表: (标签列名, 显示名)
# HOLD_HORIZONS = [
#     ("Y_15m", "15m"),
#     ("Y_30m", "30m"),
#     ("Y_60m", "60m"),
#     ("Y_90m", "90m"),
#     ("Y_120m", "120m"),
#     ("Y", "rest"),
# ]
HOLD_HORIZONS = [("Y_120m", "120m")]

# 极值阈值列表
EXTREME_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# 特征列表: (列名, 显示名)
FEATURES = [
    ("X2_zscore", "X2"),
    ("X1_zscore", "X1"),
    ("Z_final", "Zf"),
]


def load_features(year: int) -> pd.DataFrame:
    """加载特征数据"""
    pkl_path = CACHE_DIR / f"df_features_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"特征数据不存在: {pkl_path}\n请先运行 calc_feature_x1_x2.py"
        )

    logger.info(f"加载特征数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    n_stocks = df["SecuCode"].nunique()
    n_days = df["date"].nunique()
    logger.success(f"加载完成: {len(df):,} 行, {n_stocks} 只股票, {n_days} 个交易日")
    return df


def prefilter(df: pd.DataFrame) -> pd.DataFrame:
    """预过滤：去除特征为NaN/Inf的行"""
    mask = (
        np.isfinite(df["X2_zscore"])
        & np.isfinite(df["X1_zscore"])
        & np.isfinite(df["Z_final"])
    )
    df_valid = df[mask].copy()
    logger.info(f"预过滤: {len(df):,} -> {len(df_valid):,} 行")
    return df_valid


def calc_corr_for_subset(df_sub: pd.DataFrame, label_col: str) -> dict:
    """计算一个子集上各特征与指定标签的相关性"""
    # 过滤标签异常值
    mask = df_sub[label_col].notna() & np.isfinite(df_sub[label_col])
    mask &= (df_sub[label_col] > 0.9) & (df_sub[label_col] < 1.1)
    ds = df_sub[mask]
    n = len(ds)
    if n < 50:
        return {"n": n}
    result = {"n": n}
    result["Y_mean"] = ds[label_col].mean()
    result["Y_std"] = ds[label_col].std()
    for feat_col, feat_name in FEATURES:
        result[feat_name] = ds[feat_col].corr(ds[label_col])
    return result


def analyze_correlation(df: pd.DataFrame) -> list:
    """
    核心分析：按 (入场时间, 持有时间, 样本类型) 组合计算相关性

    返回: list of dict，每行是一个组合的结果
    """
    logger.info("开始相关性分析...")
    rows = []

    for entry_time in ENTRY_TIMES:
        df_entry = df[df["entry_time"] == entry_time]
        if df_entry.empty:
            continue

        for label_col, horizon_name in HOLD_HORIZONS:
            if label_col not in df_entry.columns:
                continue

            # 全样本
            r = calc_corr_for_subset(df_entry, label_col)
            if r["n"] >= 50:
                rows.append(
                    {
                        "入场时间": entry_time,
                        "持有时间": horizon_name,
                        "样本类型": "all",
                        "样本数": r["n"],
                        "Y_mean": r.get("Y_mean", np.nan),
                        "Y_std": r.get("Y_std", np.nan),
                        **{f"corr_{fn}": r.get(fn, np.nan) for _, fn in FEATURES},
                    }
                )

            # 极值样本
            for threshold in EXTREME_THRESHOLDS:
                extreme_mask = np.abs(df_entry["X2_zscore"]) > threshold
                df_ext = df_entry[extreme_mask]
                r_ext = calc_corr_for_subset(df_ext, label_col)
                if r_ext["n"] >= 50:
                    rows.append(
                        {
                            "入场时间": entry_time,
                            "持有时间": horizon_name,
                            "样本类型": f"|X2|>{threshold}",
                            "样本数": r_ext["n"],
                            "Y_mean": r_ext.get("Y_mean", np.nan),
                            "Y_std": r_ext.get("Y_std", np.nan),
                            **{
                                f"corr_{fn}": r_ext.get(fn, np.nan)
                                for _, fn in FEATURES
                            },
                        }
                    )

        logger.debug(f"  入场时间 {entry_time} 分析完成")

    logger.success(f"相关性分析完成: {len(rows)} 个组合")
    return rows


def print_report(df_result: pd.DataFrame):
    """打印报告"""
    print("\n" + "=" * 100)
    print("特征-标签相关性分析报告")
    print(f"  特征: {', '.join(fn for _, fn in FEATURES)}")
    print(f"  入场时间: {', '.join(ENTRY_TIMES)}")
    print(f"  持有时间: {', '.join(hn for _, hn in HOLD_HORIZONS)}")
    print(f"  极值阈值: {EXTREME_THRESHOLDS}")
    print("=" * 100)

    corr_cols = [f"corr_{fn}" for _, fn in FEATURES]
    feat_names = [fn for _, fn in FEATURES]

    # 列宽定义
    W_HOLD = 8
    W_FILTER = 12
    W_COUNT = 12
    W_STAT = 10
    W_CORR = 10

    def fmt_header():
        h = f"  {'hold':>{W_HOLD}}  {'filter':>{W_FILTER}}  {'count':>{W_COUNT}}"
        h += f"  {'Y_mean':>{W_STAT}}  {'Y_std':>{W_STAT}}"
        for fn in feat_names:
            h += f"  {'corr_'+fn:>{W_CORR}}"
        return h

    def fmt_row(row):
        line = f"  {row['持有时间']:>{W_HOLD}}  {row['样本类型']:>{W_FILTER}}  {row['样本数']:>{W_COUNT},}"
        for sc in ["Y_mean", "Y_std"]:
            v = row.get(sc, np.nan)
            if pd.notna(v):
                line += f"  {v:>{W_STAT}.6f}"
            else:
                line += f"  {'N/A':>{W_STAT}}"
        for cc in corr_cols:
            v = row[cc]
            if pd.notna(v):
                line += f"  {v:>{W_CORR}.4f}"
            else:
                line += f"  {'N/A':>{W_CORR}}"
        return line

    # 按入场时间分组打印
    for entry_time in ["ALL"] + ENTRY_TIMES:
        df_et = df_result[df_result["入场时间"] == entry_time]
        if df_et.empty:
            continue

        label = (
            "ALL entry times" if entry_time == "ALL" else f"entry_time: {entry_time}"
        )
        print(f"\n--- {label} ---")
        header = fmt_header()
        print(header)
        print("  " + "-" * (len(header) - 2))

        for _, row in df_et.iterrows():
            print(fmt_row(row))

    print("\n" + "=" * 100)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("相关性测试：特征与标签的相关性分析")
    logger.info("=" * 60)

    # 1. 加载并预过滤
    df = load_features(YEAR)
    df = prefilter(df)

    # 3. 按入场时间分析
    rows_by_time = analyze_correlation(df)

    # 4. 合并结果
    df_result = pd.DataFrame(rows_by_time)

    # 5. 打印报告
    print_report(df_result)

    return df_result


if __name__ == "__main__":
    df_result = main()
