"""
信号阈值敏感度分析脚本
目标：找到预测值在不同阈值下的平均收益，判断是否存在能覆盖成本的"击球区"
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

# ==================== 配置 ====================
BASE_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research")
RESULT_CSV = BASE_DIR / "output/lgbm_result_refined_label.csv"
RET_COL = "fwd_ret_30min"
COST_LINE = 0.0010  # 双边成本 10 bps（统一口径）


def analyze_prediction_thresholds(csv_path: Path):
    """分析不同预测阈值下的收益分布，寻找能覆盖成本的击球区"""
    if not csv_path.exists():
        logger.error(f"找不到结果文件: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=["date", "bar_time"])
    logger.info(f"加载完成，shape={df.shape}")

    # 【实时截面排名】按每根bar（date+bar_time）做截面排名
    # 模拟实战：每5分钟bar结束时，在当前时刻所有股票中选出最强/最弱
    df["pred_pct"] = df.groupby(["date", "bar_time"])["pred"].rank(pct=True)

    total = len(df)
    logger.info(f"总样本数: {total:,}")

    # ==================== 做多阈值分析（Top N%） ====================
    logger.info("\n" + "=" * 80)
    logger.info("做多阈值分析（取每日预测值最高的 Top N% 样本）")
    logger.info("=" * 80)
    logger.info(
        f"{'阈值(Top%)':<15} | {'样本数':<10} | {'毛收益(bps)':<13} | {'净收益(bps)':<13} | "
        f"{'胜率(>0)':<10} | {'日均净收益':<12} | {'年化IR':<10}"
    )
    logger.info("-" * 95)

    long_thresholds = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
    long_results = []

    for thres in long_thresholds:
        top_slice = df[df["pred_pct"] >= thres].copy()
        if len(top_slice) == 0:
            continue

        gross_ret = top_slice[RET_COL].mean() * 10000
        net_ret = gross_ret - COST_LINE * 10000  # 扣除双边成本
        win_rate = (top_slice[RET_COL] > 0).mean()
        sample_count = len(top_slice)

        # 日度稳定性
        daily_gross = top_slice.groupby("date")[RET_COL].mean() * 10000
        daily_net = daily_gross - COST_LINE * 10000
        daily_std = daily_net.std()
        daily_ir = (daily_net.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        daily_avg_net = daily_net.mean()

        pct_label = f"Top {(1 - thres) * 100:.1f}%"
        logger.info(
            f"{pct_label:<15} | {sample_count:>10,} | {gross_ret:>11.2f} | {net_ret:>11.2f} | "
            f"{win_rate:>8.1%} | {daily_avg_net:>10.2f} | {daily_ir:>8.2f}"
        )

        long_results.append(
            {
                "方向": "做多",
                "阈值": pct_label,
                "样本数": sample_count,
                "毛收益_bps": round(gross_ret, 3),
                "净收益_bps": round(net_ret, 3),
                "胜率_0": round(win_rate, 4),
                "日均净收益_bps": round(daily_avg_net, 3),
                "年化IR": round(daily_ir, 3),
            }
        )

    # ==================== 做空阈值分析（Bottom N%） ====================
    logger.info("\n" + "=" * 80)
    logger.info(
        "做空阈值分析（取每日预测值最低的 Bottom N% 样本，净收益 = -fwd_ret - cost）"
    )
    logger.info("=" * 80)
    logger.info(
        f"{'阈值(Bot%)':<15} | {'样本数':<10} | {'做空毛收益(bps)':<15} | {'做空净收益(bps)':<15} | "
        f"{'胜率(>0)':<10} | {'日均净收益':<12} | {'年化IR':<10}"
    )
    logger.info("-" * 95)

    short_thresholds = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
    short_results = []

    for thres in short_thresholds:
        bot_slice = df[df["pred_pct"] <= thres].copy()
        if len(bot_slice) == 0:
            continue

        # 做空毛收益 = -fwd_ret，净收益 = 毛收益 - 双边成本
        short_ret = -bot_slice[RET_COL]
        gross_ret = short_ret.mean() * 10000
        net_ret = gross_ret - COST_LINE * 10000
        win_rate = (short_ret > 0).mean()
        sample_count = len(bot_slice)

        daily_gross = short_ret.groupby(bot_slice["date"]).mean() * 10000
        daily_net = daily_gross - COST_LINE * 10000
        daily_std = daily_net.std()
        daily_ir = (daily_net.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0
        daily_avg_net = daily_net.mean()

        pct_label = f"Bot {thres * 100:.1f}%"
        logger.info(
            f"{pct_label:<15} | {sample_count:>10,} | {gross_ret:>13.2f} | {net_ret:>13.2f} | "
            f"{win_rate:>8.1%} | {daily_avg_net:>10.2f} | {daily_ir:>8.2f}"
        )

        short_results.append(
            {
                "方向": "做空",
                "阈值": pct_label,
                "样本数": sample_count,
                "毛收益_bps": round(gross_ret, 3),
                "净收益_bps": round(net_ret, 3),
                "胜率_0": round(win_rate, 4),
                "日均净收益_bps": round(daily_avg_net, 3),
                "年化IR": round(daily_ir, 3),
            }
        )

    # ==================== 多空联合分析（同时持有Top+Bottom） ====================
    logger.info("\n" + "=" * 80)
    logger.info("多空联合分析（同时做多Top N% + 做空Bottom N%，等权合并，不含成本）")
    logger.info("=" * 80)
    logger.info(
        f"{'阈值':<15} | {'样本数':<10} | {'多空均值(bps)':<15} | {'日均spread(bps)':<16} | {'年化IR':<10}"
    )
    logger.info("-" * 75)

    combined_results = []
    for thres_top, thres_bot in zip(long_thresholds, short_thresholds):
        top_slice = df[df["pred_pct"] >= thres_top].copy()
        bot_slice = df[df["pred_pct"] <= thres_bot].copy()
        if len(top_slice) == 0 or len(bot_slice) == 0:
            continue

        top_daily = top_slice.groupby("date")[RET_COL].mean() * 10000
        bot_daily = (-bot_slice.groupby("date")[RET_COL].mean()) * 10000
        combined_daily = (top_daily + bot_daily) / 2

        mean_val = combined_daily.mean()
        std_val = combined_daily.std()
        daily_ir = (mean_val / std_val * np.sqrt(252)) if std_val > 0 else 0
        total_n = len(top_slice) + len(bot_slice)

        pct_label = f"Top/Bot {(1 - thres_top) * 100:.1f}%"
        logger.info(
            f"{pct_label:<15} | {total_n:>10,} | {mean_val:>13.2f} | {mean_val:>14.2f} | {daily_ir:>8.2f}"
        )

        combined_results.append(
            {
                "阈值": pct_label,
                "样本数": total_n,
                "多空均值_bps": round(mean_val, 3),
                "年化IR": round(daily_ir, 3),
            }
        )

    # ==================== 保存结果 ====================
    all_results = pd.DataFrame(long_results + short_results)
    save_path = csv_path.parent / "signal_threshold_report.csv"
    all_results.to_csv(save_path, index=False)
    logger.success(f"阈值分析报告已保存: {save_path}")

    combined_df = pd.DataFrame(combined_results)
    save_path2 = csv_path.parent / "signal_combined_report.csv"
    combined_df.to_csv(save_path2, index=False)
    logger.success(f"多空联合报告已保存: {save_path2}")

    return all_results, combined_df


if __name__ == "__main__":
    analyze_prediction_thresholds(RESULT_CSV)
