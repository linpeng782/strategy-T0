"""
汇总脚本：合并多个标签的训练指标为综合表格（转置视图，label_type 为列）

用法：
  python summarize_metrics.py

输出：
  - 综合表格打印到控制台（转置，指标为行，label_type 为列，按测试集IC降序）
  - 保存到 /nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/csv/metrics_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

CSV_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output/csv")
LABEL_TYPES = ["fwdret", "cs_demean", "refined", "wf_rank"]


def _load_layer_stats(label_type: str, result_csv_dir: Path = CSV_DIR) -> dict:
    """
    从 layer_analysis csv 读取 Q20-Q1 spread。
    从 lgbm_result csv 重新计算日度 spread 胜率（Q20-Q1日均spread>0的天数比例）。
    """
    layer_path = result_csv_dir / f"layer_analysis_{label_type}.csv"
    result_path = result_csv_dir / f"lgbm_result_{label_type}.csv"

    if not layer_path.exists():
        logger.warning(f"layer_analysis 文件不存在: {layer_path}")
        return {"Q20-Q1 spread(bps)": "N/A", "spread胜率(日度)": "N/A"}

    df = pd.read_csv(layer_path)
    q1 = df[df["quintile"] == 1]["平均收益(bps)"].values
    q20 = df[df["quintile"] == 20]["平均收益(bps)"].values
    if len(q1) == 0 or len(q20) == 0:
        return {"Q20-Q1 spread(bps)": "N/A", "spread胜率(日度)": "N/A"}

    spread = q20[0] - q1[0]

    # 日度 spread 胜率：从 lgbm_result 重新计算（>0的天数比例）
    win_rate_str = "N/A"
    if result_path.exists():
        res = pd.read_csv(result_path, parse_dates=["date"])
        res["pred_rank"] = res.groupby(["date", "bar_time"])["pred"].rank(pct=True)
        n_bins = 20
        res["quintile"] = pd.cut(
            res["pred_rank"], bins=n_bins, labels=range(1, n_bins + 1)
        )
        ret_col = [c for c in res.columns if c.startswith("fwd_ret")][0]
        daily_layer = (
            res.groupby(["date", "quintile"], observed=False)[ret_col]
            .mean()
            .unstack("quintile")
        )
        daily_spread = (daily_layer[n_bins] - daily_layer[1]) * 10000
        win_rate_str = f"{(daily_spread > 0).mean():.1%}"

    return {
        "Q20-Q1 spread(bps)": f"{spread:.2f}",
        "spread胜率(日度)": win_rate_str,
    }


def summarize_metrics():
    """汇总所有标签的指标，转置为 label_type 做列"""
    logger.info("开始汇总指标...")

    all_rows = []
    for label_type in LABEL_TYPES:
        metrics_path = CSV_DIR / f"metrics_{label_type}.csv"
        if not metrics_path.exists():
            logger.warning(f"指标文件不存在，跳过: {metrics_path}")
            continue

        row = pd.read_csv(metrics_path).iloc[0].to_dict()

        # 补充 layer_analysis 中的 spread 和胜率
        layer = _load_layer_stats(label_type)
        row.update(layer)

        all_rows.append(row)
        logger.info(f"加载 {label_type} 指标完成")

    if not all_rows:
        logger.error("未找到任何指标文件")
        return

    summary_df = pd.DataFrame(all_rows)

    # 按测试集IC降序排列（label_type 将成为列名）
    summary_df["_ic_float"] = summary_df["测试集IC"].astype(float)
    summary_df = summary_df.sort_values("_ic_float", ascending=False).drop(
        columns=["_ic_float"]
    )
    summary_df = summary_df.reset_index(drop=True)

    # 只保留特征名称列（去掉重要性数值列）
    feat_name_cols = [
        c for c in summary_df.columns if c.startswith("Top") and c.endswith("_特征")
    ]
    drop_cols = [
        c for c in summary_df.columns if c.startswith("Top") and c.endswith("_重要性")
    ]
    summary_df = summary_df.drop(columns=drop_cols)

    # 重命名特征列：Top1_特征 → Top1特征，更简洁
    rename_map = {c: c.replace("_特征", "") for c in feat_name_cols}
    summary_df = summary_df.rename(columns=rename_map)

    # 定义行顺序（指标顺序）
    ordered_rows = [
        "label_type",
        "训练集IC",
        "测试集IC",
        "测试集ICIR",
        "IC_std",
        "IC>0比例",
        "IC_Gap",
        "Q20-Q1 spread(bps)",
        "spread胜率(日度)",
        "训练样本数",
        "测试样本数",
        "Top1",
        "Top2",
        "Top3",
        "Top4",
        "Top5",
    ]

    # 转置：label_type 做列名
    summary_df = summary_df.set_index("label_type").T
    # 按 ordered_rows 顺序（去掉 label_type 本身）
    row_order = [r for r in ordered_rows if r != "label_type" and r in summary_df.index]
    summary_df = summary_df.reindex(row_order)

    # 打印
    print("\n" + "=" * 100)
    print("四标签综合对比表（按测试集IC降序）")
    print("=" * 100)
    print(summary_df.to_string())
    print("=" * 100)

    # 保存
    summary_path = CSV_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_path)
    logger.success(f"综合表格已保存: {summary_path}")


if __name__ == "__main__":
    summarize_metrics()
