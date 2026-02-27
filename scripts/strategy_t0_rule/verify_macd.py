"""
MACD计算对齐验证脚本
对比 compute_macd_continuous（当前实现）与 TA-Lib MACD 的差异
"""

import numpy as np
import pandas as pd
import talib
from pathlib import Path
from loguru import logger

# ==================== 参数 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
MACD_FAST = 9
MACD_SLOW = 12
MACD_SIGNAL = 7
TEST_YEAR = 2025
TEST_STOCK = "002344"
WARMUP_SKIP = 50  # 跳过前N根bar（初始化差异区）


# ==================== 当前实现（原版：close[0]初始化） ====================
def compute_macd_v1(close: np.ndarray) -> tuple:
    """原版实现：第一根bar用close[0]初始化EMA"""
    n = len(close)
    alpha_fast = 2.0 / (MACD_FAST + 1)
    alpha_slow = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)

    ema_fast = np.empty(n)
    ema_slow = np.empty(n)
    dif = np.empty(n)
    dea = np.empty(n)

    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    dif[0] = 0.0
    dea[0] = 0.0

    for i in range(1, n):
        ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]
        ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]
        dif[i] = ema_fast[i] - ema_slow[i]
        dea[i] = alpha_sig * dif[i] + (1 - alpha_sig) * dea[i - 1]

    macd = 2.0 * (dif - dea)
    return dif, dea, macd


# ==================== pandas ewm实现（adjust=False对齐talib） ====================
def compute_macd_pandas(close: np.ndarray) -> tuple:
    """
    用pandas.ewm(adjust=False)计算，adjust=False等价于递推公式，与talib一致。
    talib的EMA初始化方式：第一个有效EMA值 = 前N个收盘价的SMA，
    pandas ewm(adjust=False)的初始值 = close[0]，与talib不同。
    但经过足够warmup后两者收敛。
    """
    s = pd.Series(close)
    ema_fast = s.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = s.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd = 2.0 * (dif - dea)
    return dif.values, dea.values, macd.values


# ==================== talib实现 ====================
def compute_macd_talib(close: np.ndarray) -> tuple:
    """talib MACD：SMA初始化EMA，前N个返回NaN"""
    dif, dea, macd = talib.MACD(
        close.astype(np.float64),
        fastperiod=MACD_FAST,
        slowperiod=MACD_SLOW,
        signalperiod=MACD_SIGNAL,
    )
    return dif, dea, macd


# ==================== SMA初始化版本（对齐talib） ====================
def compute_macd_sma_init(close: np.ndarray) -> tuple:
    """
    用SMA初始化EMA，对齐talib的行为。
    talib逻辑：
      - 第MACD_SLOW个bar开始才有效（前面为NaN）
      - ema_slow的第一个有效值 = SMA(close[:MACD_SLOW])
      - ema_fast的第一个有效值 = SMA(close[:MACD_FAST])
      - dea的第一个有效值 = SMA(dif[:MACD_SIGNAL])
    """
    n = len(close)
    alpha_fast = 2.0 / (MACD_FAST + 1)
    alpha_slow = 2.0 / (MACD_SLOW + 1)
    alpha_sig = 2.0 / (MACD_SIGNAL + 1)

    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    dif = np.full(n, np.nan)
    dea = np.full(n, np.nan)

    if n < MACD_SLOW:
        return dif, dea, np.full(n, np.nan)

    # SMA初始化：ema_fast第一个有效值在bar MACD_FAST-1处
    ema_fast[MACD_FAST - 1] = close[:MACD_FAST].mean()
    for i in range(MACD_FAST, n):
        ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]

    # SMA初始化：ema_slow第一个有效值在bar MACD_SLOW-1处
    ema_slow[MACD_SLOW - 1] = close[:MACD_SLOW].mean()
    for i in range(MACD_SLOW, n):
        ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]

    # DIF从 MACD_SLOW-1 开始有效
    for i in range(MACD_SLOW - 1, n):
        dif[i] = ema_fast[i] - ema_slow[i]

    # DEA用SMA初始化：dif的前 MACD_SIGNAL 个有效值的均値
    dea_start = MACD_SLOW - 1 + MACD_SIGNAL - 1  # 第一个DEA有效索引
    if dea_start >= n:
        return dif, dea, np.full(n, np.nan)
    dea[dea_start] = dif[MACD_SLOW - 1 : MACD_SLOW - 1 + MACD_SIGNAL].mean()
    for i in range(dea_start + 1, n):
        dea[i] = alpha_sig * dif[i] + (1 - alpha_sig) * dea[i - 1]

    macd = 2.0 * (dif - dea)
    return dif, dea, macd


# ==================== 对比分析 ====================
def compare_implementations(close: np.ndarray, label: str = ""):
    """对比四种实现，打印详细报告"""
    dif_v1, dea_v1, macd_v1 = compute_macd_v1(close)
    dif_pd, dea_pd, macd_pd = compute_macd_pandas(close)
    dif_sma, dea_sma, macd_sma = compute_macd_sma_init(close)
    dif_tl, dea_tl, macd_tl = compute_macd_talib(close)

    # talib有效区间（非NaN）
    valid_tl = ~np.isnan(dif_tl)
    first_valid = int(np.argmax(valid_tl)) if valid_tl.any() else len(close)
    logger.info(f"TA-Lib首个有效bar索引: {first_valid}（前{first_valid}个为NaN）")

    # 跳过warmup后的有效区间
    start = max(first_valid, WARMUP_SKIP)
    n_valid = len(close) - start
    if n_valid <= 0:
        logger.warning(
            f"数据量不足，跳过详细对比（总{len(close)}根bar，warmup需{start}根）"
        )
        return
    logger.info(f"对比区间: [{start}, {len(close)})，共{n_valid}根bar")

    logger.info(f"\n{'='*60}")
    logger.info(f"对比报告 {label}")
    logger.info(f"{'='*60}")

    for name, (arr_v1, arr_pd, arr_sma, arr_tl) in [
        ("DIF", (dif_v1, dif_pd, dif_sma, dif_tl)),
        ("DEA", (dea_v1, dea_pd, dea_sma, dea_tl)),
        ("MACD", (macd_v1, macd_pd, macd_sma, macd_tl)),
    ]:
        v1_vs_tl = np.abs(arr_v1[start:] - arr_tl[start:])
        pd_vs_tl = np.abs(arr_pd[start:] - arr_tl[start:])
        sma_vs_tl = np.abs(arr_sma[start:] - arr_tl[start:])

        logger.info(f"\n  [{name}]")
        logger.info(
            f"    v1(close[0]初始) vs talib → 最大差: {v1_vs_tl.max():.4e}, 均値差: {v1_vs_tl.mean():.4e}"
        )
        logger.info(
            f"    pandas(ewm)      vs talib → 最大差: {pd_vs_tl.max():.4e}, 均値差: {pd_vs_tl.mean():.4e}"
        )
        logger.info(
            f"    SMA初始化         vs talib → 最大差: {sma_vs_tl.max():.4e}, 均値差: {sma_vs_tl.mean():.4e}"
        )
        logger.info(
            f"    v1对齐talib: {'✅' if v1_vs_tl.max() < 1e-6 else '❌'} | "
            f"pandas对齐talib: {'✅' if pd_vs_tl.max() < 1e-6 else '❌'} | "
            f"SMA对齐talib: {'✅' if sma_vs_tl.max() < 1e-6 else '❌'}"
        )

    # 收敛分析：不同warmup下的最大差异
    logger.info(f"\n{'='*60}")
    logger.info("收敛分析：v1 vs talib 的DIF最大差异随 warmup 的变化:")
    logger.info(f"{'Warmup':>8} {'DIF最大差':>14} {'DEA最大差':>14} {'MACD最大差':>14}")
    for w in [20, 50, 100, 200, 500, 1000]:
        if w >= len(close) - first_valid:
            break
        s = max(first_valid, w)
        d_dif = np.abs(dif_v1[s:] - dif_tl[s:]).max()
        d_dea = np.abs(dea_v1[s:] - dea_tl[s:]).max()
        d_macd = np.abs(macd_v1[s:] - macd_tl[s:]).max()
        logger.info(f"  {w:>8} {d_dif:>14.4e} {d_dea:>14.4e} {d_macd:>14.4e}")

    # warmup结束后抽样对比
    logger.info(f"\n{'='*60}")
    logger.info(f"warmup后（bar {start}附近）的DIF数値对比:")
    logger.info(
        f"{'Bar':>6} {'v1_DIF':>12} {'sma_DIF':>12} {'tl_DIF':>12} {'v1-tl':>12} {'sma-tl':>12}"
    )
    for i in range(start, min(start + 10, len(close))):
        logger.info(
            f"  {i:>6} {dif_v1[i]:>12.6f} {dif_sma[i]:>12.6f} "
            f"{dif_tl[i]:>12.6f} {dif_v1[i]-dif_tl[i]:>12.2e} {dif_sma[i]-dif_tl[i]:>12.2e}"
        )

    return dif_v1, dea_v1, macd_v1, dif_pd, dea_pd, macd_pd, dif_tl, dea_tl, macd_tl


# ==================== 主函数 ====================
def main():
    logger.info("加载真实数据...")
    pkl_path = CACHE_DIR / f"df_5m_{TEST_YEAR}.pkl"
    df = pd.read_pickle(pkl_path)
    df = df[df["SecuCode"] == TEST_STOCK].copy()
    df = df.sort_values(["date", "bar_time"]).reset_index(drop=True)
    close = df["close"].values.astype(np.float64)
    logger.info(f"股票: {TEST_STOCK}, 年份: {TEST_YEAR}, 总bar数: {len(close)}")

    # ---- 测试1：全年连续数据 ----
    logger.info("\n【测试1】全年连续数据对比")
    compare_implementations(close, label=f"{TEST_STOCK} {TEST_YEAR}年全量数据")

    # ---- 测试2：单日数据（模拟每日重置的极端情况） ----
    logger.info("\n【测试2】单日数据对比（仅取第一个交易日）")
    first_date = df["date"].iloc[0]
    df_day = df[df["date"] == first_date]
    close_day = df_day["close"].values.astype(np.float64)
    logger.info(f"第一个交易日: {first_date}, bar数: {len(close_day)}")
    compare_implementations(close_day, label=f"单日数据({first_date})")

    # ---- 测试3：合成数据（排除数据问题） ----
    logger.info("\n【测试3】合成数据（线性递增，排除数据干扰）")
    np.random.seed(42)
    synthetic = np.cumsum(np.random.randn(500)) + 100.0
    compare_implementations(synthetic, label="合成随机游走数据(500根)")

    logger.info("\n验证完成！")


if __name__ == "__main__":
    main()
