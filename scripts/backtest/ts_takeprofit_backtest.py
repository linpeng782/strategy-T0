"""
止盈/止损策略回测

逻辑:
  - 10:30 入场（买入价 = 10:35 bar 的 VWAP，即 next_vwap_5m）
  - 逐bar监控浮盈: running_pnl = bar_close / entry_price - 1
  - 若浮盈 >= take_profit → 以该bar的close退出
  - 若浮亏 <= -stop_loss → 以该bar的close退出
  - 若到收盘未触发 → 以收盘VWAP(V_rest)退出（原策略）

测试矩阵:
  止盈: 0.5%, 1.0%, 1.5%, 2.0%, 3.0%, 无止盈
  止损: 0.5%, 1.0%, 1.5%, 无止损
  组合: 最优止盈+最优止损

作者：量化研究
日期：2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

COST_RATE = 0.0015  # 单边15bps
ENTRY_TIME = "10:30"

# 止盈/止损阈值
TP_THRESHOLDS = [None, 0.005, 0.01, 0.015, 0.02, 0.03]  # None=不止盈
SL_THRESHOLDS = [None, 0.005, 0.01, 0.015, 0.02]          # None=不止损


def load_bar_data(year: int) -> pd.DataFrame:
    """加载全量bar数据"""
    path = CACHE_DIR / f"df_features_{year}.pkl"
    logger.info(f"加载全量bar数据: {path}")
    df = pd.read_pickle(path)
    logger.success(f"加载完成: {len(df):,} 行, {df['SecuCode'].nunique()} 只股票")
    return df


def simulate_takeprofit(
    bar_data: pd.DataFrame,
    predictions: pd.DataFrame,
    top_n: int = 50,
    buffer_ratio: float = 2.0,
    take_profit: float = None,
    stop_loss: float = None,
):
    """
    带止盈止损的组合回测

    参数:
      bar_data: 全量5分钟bar数据 (48 bars/天/股)
      predictions: 模型预测结果 (entry_time=10:30, 含 pred_prob)
      top_n: 持仓数量
      buffer_ratio: 换手缓冲
      take_profit: 止盈阈值 (如0.01=1%), None=不止盈
      stop_loss: 止损阈值 (如0.01=1%), None=不止损
    """
    buffer_n = int(top_n * buffer_ratio)

    # 预处理: 每只股票每天的10:30入场价和后续bar的close
    pred_df = predictions.copy()
    pred_df["rank"] = pred_df.groupby("date")["pred_prob"].rank(
        ascending=False, method="first"
    )

    dates = sorted(pred_df["date"].unique())
    prev_holdings = set()
    daily_records = []

    # 预构建bar_data索引加速查询
    bar_data_sorted = bar_data.sort_values(["SecuCode", "date", "entry_time"])

    # 为每只股票每天提取10:30入场价和后续close序列
    logger.info("预计算入场价和后续价格路径...")
    entry_bars = bar_data_sorted[bar_data_sorted["entry_time"] == ENTRY_TIME].copy()
    entry_prices = entry_bars.set_index(["SecuCode", "date"])["next_vwap_5m"]

    # 后续bars: 10:35到15:00
    post_entry = bar_data_sorted[bar_data_sorted["entry_time"] > ENTRY_TIME].copy()

    # 预构建每只股票每天的close路径 {(SecuCode, date): np.array of closes}
    close_paths = {}
    vwap_rest = {}  # 原策略的退出价 V_rest

    for (code, date), group in post_entry.groupby(["SecuCode", "date"]):
        close_paths[(code, date)] = group["close"].values
        # V_rest = 入场后所有bar的VWAP (amount_total / volume_total)
        total_amt = group["amount"].sum()
        total_vol = group["volume"].sum()
        if total_vol > 0:
            vwap_rest[(code, date)] = total_amt / total_vol
        else:
            vwap_rest[(code, date)] = group["close"].iloc[-1]

    # 日收盘价 (当天最后一个bar的close)
    last_bar = bar_data_sorted.groupby(["SecuCode", "date"]).last()
    close_prices = last_bar["close"]

    logger.info(f"开始回测: TopN={top_n}, Buffer={buffer_ratio}x, "
                f"TP={take_profit}, SL={stop_loss}")

    tp_trigger_counts = 0
    sl_trigger_counts = 0
    total_trades = 0

    for date in dates:
        day_pred = pred_df[pred_df["date"] == date]
        if len(day_pred) == 0:
            continue

        # 选股逻辑（带缓冲）
        if buffer_ratio > 1.0 and len(prev_holdings) > 0:
            kept_mask = day_pred["SecuCode"].isin(prev_holdings) & (
                day_pred["rank"] <= buffer_n
            )
            kept = set(day_pred.loc[kept_mask, "SecuCode"])
            if len(kept) > top_n:
                kept = set(
                    day_pred[day_pred["SecuCode"].isin(kept)]
                    .nsmallest(top_n, "rank")["SecuCode"]
                )
            n_fill = top_n - len(kept)
            if n_fill > 0:
                fills = set(
                    day_pred[
                        (day_pred["rank"] <= top_n)
                        & (~day_pred["SecuCode"].isin(kept))
                    ].nsmallest(n_fill, "rank")["SecuCode"]
                )
            else:
                fills = set()
            holdings = kept | fills
            n_new = len(holdings - prev_holdings)
        else:
            holdings = set(day_pred[day_pred["rank"] <= top_n]["SecuCode"])
            n_new = len(holdings)

        if len(holdings) == 0:
            prev_holdings = set()
            continue

        # 计算每只股票的收益（含止盈止损）
        stock_returns = []
        stock_close_returns = []

        for code in holdings:
            key = (code, date)
            if key not in entry_prices.index or pd.isna(entry_prices.loc[key]):
                continue
            ep = entry_prices.loc[key]

            # 判断是否触发止盈/止损
            triggered = False
            exit_price = None

            if (take_profit is not None or stop_loss is not None) and key in close_paths:
                path = close_paths[key]
                for bar_close in path:
                    pnl = bar_close / ep - 1
                    if take_profit is not None and pnl >= take_profit:
                        exit_price = bar_close
                        tp_trigger_counts += 1
                        triggered = True
                        break
                    if stop_loss is not None and pnl <= -stop_loss:
                        exit_price = bar_close
                        sl_trigger_counts += 1
                        triggered = True
                        break

            if not triggered:
                # 未触发 → 原策略退出 (V_rest VWAP)
                if key in vwap_rest:
                    exit_price = vwap_rest[key]
                else:
                    continue

            raw_ret = exit_price / ep - 1
            stock_returns.append(raw_ret)

            # 同时记录close离场的收益（用于计算市场均值）
            if key in close_prices.index:
                close_ret = close_prices.loc[key] / ep - 1
                stock_close_returns.append(close_ret)

            total_trades += 1

        if len(stock_returns) == 0:
            prev_holdings = set()
            continue

        # 市场均值 (用当天所有股票的VWAP退出收益)
        day_all = day_pred.copy()
        mkt_rets = []
        for code in day_all["SecuCode"]:
            mkey = (code, date)
            if mkey in entry_prices.index and mkey in vwap_rest:
                ep_m = entry_prices.loc[mkey]
                if not pd.isna(ep_m) and ep_m > 0:
                    mkt_rets.append(vwap_rest[mkey] / ep_m - 1)
        market_ret = np.mean(mkt_rets) if mkt_rets else 0

        raw_mean = np.mean(stock_returns)
        turnover = n_new / len(holdings)
        cost = turnover * COST_RATE
        net_ret = raw_mean - cost
        excess_ret = raw_mean - market_ret

        daily_records.append({
            "date": date,
            "raw_ret": raw_mean,
            "net_ret": net_ret,
            "excess_ret": excess_ret,
            "market_ret": market_ret,
            "turnover": turnover,
            "n_stocks": len(holdings),
        })
        prev_holdings = holdings

    if not daily_records:
        return {}

    dr = pd.DataFrame(daily_records)
    n_days = len(dr)

    tp_pct = tp_trigger_counts / total_trades * 100 if total_trades > 0 else 0
    sl_pct = sl_trigger_counts / total_trades * 100 if total_trades > 0 else 0

    return {
        "tp": take_profit,
        "sl": stop_loss,
        "top_n": top_n,
        "n_days": n_days,
        "ann_net": dr["net_ret"].mean() * 242 * 100,
        "ann_exc": dr["excess_ret"].mean() * 242 * 100,
        "ann_raw": dr["raw_ret"].mean() * 242 * 100,
        "avg_net_bps": dr["net_ret"].mean() * 10000,
        "avg_exc_bps": dr["excess_ret"].mean() * 10000,
        "sharpe_net": (
            dr["net_ret"].mean() / dr["net_ret"].std() * np.sqrt(242)
            if dr["net_ret"].std() > 0 else 0
        ),
        "sharpe_exc": (
            dr["excess_ret"].mean() / dr["excess_ret"].std() * np.sqrt(242)
            if dr["excess_ret"].std() > 0 else 0
        ),
        "avg_turnover": dr["turnover"].mean(),
        "tp_trigger_pct": tp_pct,
        "sl_trigger_pct": sl_pct,
        "max_dd_net": (
            (1 + dr["net_ret"]).cumprod().cummax()
            - (1 + dr["net_ret"]).cumprod()
        ).max() * 100,
    }


def main():
    # 加载数据
    bar_data = load_bar_data(2025)

    # 加载最优模型预测 (0.7V+0.3C)
    pred_path = OUTPUT_DIR / "ts_lgbm_predictions_label_w73.pkl"
    if not pred_path.exists():
        logger.warning(f"0.7V+0.3C预测不存在，尝试加载VWAP基线")
        pred_path = OUTPUT_DIR / "ts_lgbm_predictions.pkl"
    predictions = pd.read_pickle(pred_path)
    logger.info(f"加载预测: {pred_path.name}, {len(predictions):,} 行")

    # ===== 表1: 纯止盈测试 (Top-50, Buffer=2.0) =====
    print("\n" + "=" * 120)
    print("表1: 止盈策略测试 (Top-50, Buffer=2.0x, 无止损)")
    print("=" * 120)

    header = (
        f"{'止盈阈值':>10s} | {'年化净%':>7s} {'年化超额%':>8s} "
        f"{'ShpNet':>6s} {'ShpExc':>6s} | {'NetBps':>6s} {'ExcBps':>6s} "
        f"{'换手':>5s} {'触发率':>6s} {'MaxDD%':>6s}"
    )
    print(header)
    print("-" * 100)

    tp_results = []
    for tp in TP_THRESHOLDS:
        r = simulate_takeprofit(
            bar_data, predictions,
            top_n=50, buffer_ratio=2.0,
            take_profit=tp, stop_loss=None,
        )
        tp_results.append(r)
        tp_str = f"{tp*100:.1f}%" if tp else "无止盈"
        print(
            f"{tp_str:>10s} | {r['ann_net']:>+7.1f} {r['ann_exc']:>+8.1f} "
            f"{r['sharpe_net']:>6.2f} {r['sharpe_exc']:>6.2f} | "
            f"{r['avg_net_bps']:>+6.2f} {r['avg_exc_bps']:>+6.2f} "
            f"{r['avg_turnover']:>4.0%} {r['tp_trigger_pct']:>5.1f}% "
            f"{r['max_dd_net']:>5.1f}%"
        )

    # ===== 表2: 纯止损测试 =====
    print("\n" + "=" * 120)
    print("表2: 止损策略测试 (Top-50, Buffer=2.0x, 无止盈)")
    print("=" * 120)
    print(header)
    print("-" * 100)

    sl_results = []
    for sl in SL_THRESHOLDS:
        r = simulate_takeprofit(
            bar_data, predictions,
            top_n=50, buffer_ratio=2.0,
            take_profit=None, stop_loss=sl,
        )
        sl_results.append(r)
        sl_str = f"{sl*100:.1f}%" if sl else "无止损"
        print(
            f"{sl_str:>10s} | {r['ann_net']:>+7.1f} {r['ann_exc']:>+8.1f} "
            f"{r['sharpe_net']:>6.2f} {r['sharpe_exc']:>6.2f} | "
            f"{r['avg_net_bps']:>+6.2f} {r['avg_exc_bps']:>+6.2f} "
            f"{r['avg_turnover']:>4.0%} {r['sl_trigger_pct']:>5.1f}% "
            f"{r['max_dd_net']:>5.1f}%"
        )

    # ===== 表3: 止盈+止损组合 =====
    print("\n" + "=" * 120)
    print("表3: 止盈+止损组合 (Top-50, Buffer=2.0x)")
    print("=" * 120)

    combo_header = (
        f"{'止盈':>6s} {'止损':>6s} | {'年化净%':>7s} {'年化超额%':>8s} "
        f"{'ShpNet':>6s} {'ShpExc':>6s} | {'NetBps':>6s} "
        f"{'TP触发':>6s} {'SL触发':>6s} {'MaxDD%':>6s}"
    )
    print(combo_header)
    print("-" * 100)

    # 测试关键组合
    tp_candidates = [0.01, 0.015, 0.02]
    sl_candidates = [0.005, 0.01, 0.015]

    combo_results = []
    for tp in tp_candidates:
        for sl in sl_candidates:
            r = simulate_takeprofit(
                bar_data, predictions,
                top_n=50, buffer_ratio=2.0,
                take_profit=tp, stop_loss=sl,
            )
            combo_results.append(r)
            print(
                f"{tp*100:.1f}% {sl*100:.1f}% | {r['ann_net']:>+7.1f} {r['ann_exc']:>+8.1f} "
                f"{r['sharpe_net']:>6.2f} {r['sharpe_exc']:>6.2f} | "
                f"{r['avg_net_bps']:>+6.2f} "
                f"{r['tp_trigger_pct']:>5.1f}% {r['sl_trigger_pct']:>5.1f}% "
                f"{r['max_dd_net']:>5.1f}%"
            )

    # ===== 表4: 最优参数在不同TopN下 =====
    # 找止盈单独最优
    best_tp_r = max(
        [r for r in tp_results if r.get("tp") is not None],
        key=lambda x: x["sharpe_net"],
    )
    best_tp = best_tp_r["tp"]

    # 找组合最优
    best_combo = max(combo_results, key=lambda x: x["sharpe_net"])
    best_tp_c, best_sl_c = best_combo["tp"], best_combo["sl"]

    print(f"\n最优止盈: {best_tp*100:.1f}%")
    print(f"最优组合: TP={best_tp_c*100:.1f}% + SL={best_sl_c*100:.1f}%")

    print("\n" + "=" * 120)
    print(f"表4: 最优参数 vs 基线 — 不同TopN (Buffer=2.0x)")
    print("=" * 120)

    header4 = (
        f"{'策略':>25s} {'TopN':>4s} | {'年化净%':>7s} {'年化超额%':>8s} "
        f"{'ShpNet':>6s} {'ShpExc':>6s} {'MaxDD%':>6s}"
    )
    print(header4)
    print("-" * 80)

    for tn in [20, 50, 100]:
        # 基线
        r0 = simulate_takeprofit(
            bar_data, predictions,
            top_n=tn, buffer_ratio=2.0,
            take_profit=None, stop_loss=None,
        )
        # 最优止盈
        r1 = simulate_takeprofit(
            bar_data, predictions,
            top_n=tn, buffer_ratio=2.0,
            take_profit=best_tp, stop_loss=None,
        )
        # 最优组合
        r2 = simulate_takeprofit(
            bar_data, predictions,
            top_n=tn, buffer_ratio=2.0,
            take_profit=best_tp_c, stop_loss=best_sl_c,
        )
        for label, r in [("基线(无TP/SL)", r0),
                          (f"TP={best_tp*100:.1f}%", r1),
                          (f"TP={best_tp_c*100:.1f}%+SL={best_sl_c*100:.1f}%", r2)]:
            print(
                f"{label:>25s} {tn:>4d} | {r['ann_net']:>+7.1f} {r['ann_exc']:>+8.1f} "
                f"{r['sharpe_net']:>6.2f} {r['sharpe_exc']:>6.2f} {r['max_dd_net']:>5.1f}%"
            )
        print()

    logger.success("止盈止损策略回测完成!")


if __name__ == "__main__":
    main()
