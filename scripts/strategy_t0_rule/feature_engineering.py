"""
特征工程脚本
职责：从原始5分钟K线数据构造所有特征，并对每只股票做时间序列去极值+标准化处理。

特征列表（MACD系列）：
  - vwap_bias    : close / 日内累计VWAP - 1
  - dif          : MACD快线（EMA9 - EMA12，跨日连续）
  - dea          : MACD信号线（DIF的EMA7，跨日连续）
  - macd         : 2*(dif - dea)，国内标准
  - macd_area    : 当前红/绿柱段的累计面积（每次交叉后重置）
  - dif_slope3   : DIF最近3根bar的变化速度

特征列表（量价类，新增）：
  - vol_ratio    : 当前bar成交量 / 过去20bar均量（放量/缩量信号）
  - buy_pressure : (close - low) / (high - low)，买方压力
  - amplitude    : (high - low) / open，振幅（波动率代理）
  - ret_1bar     : 1根bar收益率（短期动量）
  - ret_3bar     : 3根bar收益率（中短期动量）
  - ret_5bar     : 5根bar收益率（中期动量）
  - range_pos    : (close - 日内最低) / (日内最高 - 日内最低)，日内价格位置

标准化流程（跨日同时刻，无未来函数）：
  1. 2倍MAD去极值（基于过去20天同一时刻历史值的中位数和std）
  2. z-score标准化（基于过去20天同一时刻历史均值和std）
  严格无未来函数：今天09:40的参数 = 过去20天09:40历史值的统计量

输出：
  - 返回含原始特征列和 _norm 标准化列的DataFrame
  - 可直接保存为pickle供ic_analysis.py使用
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MACD参数
MACD_FAST = 9
MACD_SLOW = 12
MACD_SIGNAL = 7

# 所有原始特征列名（按类别组织）
RAW_FEATURE_COLS = [
    # MACD系列（IC验证有效）
    "vwap_bias",
    "dif",
    "dea",
    "macd",
    "macd_area",
    "dif_slope3",
    # 量价类
    "vol_ratio",  # 全天放量（日内rolling5）
    "vol_ratio_open",  # 早盘放量（09:30-10:30）
    "vol_ratio_mid",  # 午盘放量（10:30-13:30）
    "vol_ratio_close",  # 尾盘放量（13:30-15:00）
    "buy_pressure",
    "amplitude",
    "ret_1bar",
    "ret_3bar",
    "ret_5bar",
    "range_pos",
    "vol_price_corr",  # 日内rolling10
    # alpha158精选
    "kmid",
    "kup2",
    "klow2",
    "rsv10",  # 日内rolling10
    "rsv20",  # 日内rolling20
    "cord20",  # 日内rolling10
    "wvma10",  # 日内rolling5
    "vsumd10",  # 日内rolling5
]


# ==================== MACD跨日连续计算 ====================
def _compute_macd_continuous(close: np.ndarray) -> tuple:
    """
    跨日连续计算MACD，EMA不在每天开盘时重置。
    返回: (dif, dea, macd) 均为ndarray，长度与close相同
    """
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


# ==================== 单股特征构造 ====================
def compute_features_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的全年数据构造所有原始特征。
    输入df须已按 date, bar_time 排序，且只含一只股票。
    """
    df = df.copy()

    # ---- vwap_bias ----
    grp = df.groupby("date")
    df["cum_amount"] = grp["amount"].transform("cumsum")
    df["cum_volume"] = grp["volume"].transform("cumsum")
    df["vwap"] = (df["cum_amount"] / df["cum_volume"]).replace(
        [np.inf, -np.inf], np.nan
    )
    df["vwap_bias"] = df["close"] / df["vwap"] - 1

    # ---- 跨日连续MACD ----
    close_arr = df["close"].values.astype(np.float64)
    dif, dea, macd = _compute_macd_continuous(close_arr)
    df["dif"] = dif
    df["dea"] = dea
    df["macd"] = macd
    df["macd_diff"] = df["macd"].diff().fillna(0.0)

    # ---- 交叉距离 & 能量积蓄（需要循环，数据量小可接受） ----
    macd_sign = np.sign(df["macd"].values)
    cross_event = np.zeros(len(df), dtype=np.int8)
    cross_event[1:] = (macd_sign[1:] != macd_sign[:-1]).astype(np.int8)

    cross_dist = np.zeros(len(df), dtype=np.float64)
    macd_area = np.zeros(len(df), dtype=np.float64)
    cnt = 0
    running_sum = 0.0
    macd_vals = df["macd"].values
    for i in range(len(df)):
        if i > 0 and cross_event[i] == 1:
            cnt = 0
            running_sum = 0.0
        cross_dist[i] = cnt
        running_sum += macd_vals[i]
        macd_area[i] = running_sum
        cnt += 1

    df["cross_dist"] = cross_dist
    df["macd_area"] = macd_area

    # ---- DIF斜率（最近3根bar） ----
    df["dif_slope3"] = df["dif"].diff(3).fillna(0.0)

    # ---- 量价类特征 ----
    grp = df.groupby("date")

    # 全天放量：当前bar成交量 / 日内前5根bar均量（rolling5，更及时）
    vol_ma_intraday = grp["volume"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    )
    df["vol_ratio"] = np.where(vol_ma_intraday > 0, df["volume"] / vol_ma_intraday, 1.0)

    # 放量三段拆分：早盘/午盘/尾盘（消除日内U型成交量结构的干扰）
    # 早盘：09:30-10:30（前12根bar），午盘：10:30-13:30（中间段），尾盘：13:30-15:00（后18根bar）
    # 各段内独立rolling，只在本段内比较，段外bar填1.0（无信号）
    time_col = df["bar_time"].dt.strftime("%H:%M")
    open_mask = time_col.between("09:30", "10:25")  # 早盘12根
    mid_mask = time_col.between("10:30", "13:25")  # 午盘
    close_mask = time_col.between("13:30", "14:55")  # 尾盘

    for seg_col, seg_mask in [
        ("vol_ratio_open", open_mask),
        ("vol_ratio_mid", mid_mask),
        ("vol_ratio_close", close_mask),
    ]:
        seg_vol = df["volume"].where(seg_mask)
        grp_seg = df.groupby("date")
        vol_ma_seg = grp_seg["volume"].transform(
            lambda x, m=seg_mask: x.where(m.loc[x.index])
            .shift(1)
            .rolling(3, min_periods=2)
            .mean()
        )
        # 段外bar填NaN，避免污染标准化；IC计算时na_option='keep'自动排除
        ratio = np.where(vol_ma_seg > 0, df["volume"] / vol_ma_seg, np.nan)
        df[seg_col] = np.where(seg_mask, ratio, np.nan)

    # 买方压力：(close - low) / (high - low)，反映bar内买卖力量对比
    hl_range = df["high"] - df["low"]
    df["buy_pressure"] = np.where(
        hl_range > 0, (df["close"] - df["low"]) / hl_range, 0.5
    )

    # 振幅：(high - low) / open，反映当根bar的波动强度
    df["amplitude"] = np.where(
        df["open"] > 0, (df["high"] - df["low"]) / df["open"], 0.0
    )

    # 短期价格动量（按日分组，避免跨日隔夜收益污染）
    df["ret_1bar"] = grp["close"].pct_change(1).fillna(0.0)
    df["ret_3bar"] = grp["close"].pct_change(3).fillna(0.0)
    df["ret_5bar"] = grp["close"].pct_change(5).fillna(0.0)

    # 日内价格位置：(close - 日内最低) / (日内最高 - 日内最低)
    cum_high = grp["high"].transform("cummax")
    cum_low = grp["low"].transform("cummin")
    intraday_range = cum_high - cum_low
    df["range_pos"] = np.where(
        intraday_range > 0, (df["close"] - cum_low) / intraday_range, 0.5
    )

    # 量价相关性：close收益率与log(volume)的滚动相关（日内窗口10bar，缩短以提升及时性）
    # 正值=量价同向（趋势），负值=量价背离（反转预警）
    close_ret = grp["close"].pct_change(1).fillna(0.0)
    log_vol = np.log(df["volume"].clip(lower=1))
    df["vol_price_corr"] = (
        grp["close"]
        .transform(
            lambda x: close_ret.loc[x.index]
            .rolling(10, min_periods=5)
            .corr(log_vol.loc[x.index])
        )
        .fillna(0.0)
    )

    # ---- alpha158精选因子 ----
    hl = df["high"] - df["low"]
    hl_safe = hl.clip(lower=1e-8)  # 避免除零

    # K线形态因子：bar内价格方向强度
    # kmid: (close-open)/(high-low)，归一化到振幅的涨跌幅
    df["kmid"] = (df["close"] - df["open"]) / hl_safe

    # kup2: 上影线比例，反映上方抛压
    df["kup2"] = (df["high"] - np.maximum(df["open"], df["close"])) / hl_safe

    # klow2: 下影线比例，反映下方支撑
    df["klow2"] = (np.minimum(df["open"], df["close"]) - df["low"]) / hl_safe

    # RSV：近N根bar内的价格位置（KDJ的K值前身）
    # 改为日内rolling，不跨日，避免隔夜跳空污染
    low10_intra = grp["low"].transform(lambda x: x.rolling(10, min_periods=3).min())
    high10_intra = grp["high"].transform(lambda x: x.rolling(10, min_periods=3).max())
    low20_intra = grp["low"].transform(lambda x: x.rolling(20, min_periods=5).min())
    high20_intra = grp["high"].transform(lambda x: x.rolling(20, min_periods=5).max())
    df["rsv10"] = np.where(
        (high10_intra - low10_intra) > 0,
        (df["close"] - low10_intra) / (high10_intra - low10_intra),
        0.5,
    )
    df["rsv20"] = np.where(
        (high20_intra - low20_intra) > 0,
        (df["close"] - low20_intra) / (high20_intra - low20_intra),
        0.5,
    )

    # cord20: close收益率与log(成交量变化率)的滚动相关
    # 改为日内rolling10，避免跨日量价关系被隔夜跳空干扰
    close_ret_cont = grp["close"].pct_change(1).fillna(0.0)
    vol_prev = grp["volume"].shift(1).clip(lower=1)
    log_vol_ret = np.log((df["volume"].clip(lower=1) / vol_prev).clip(lower=1e-8))
    df["cord20"] = (
        grp["close"]
        .transform(
            lambda x: close_ret_cont.loc[x.index]
            .rolling(10, min_periods=5)
            .corr(log_vol_ret.loc[x.index])
        )
        .fillna(0.0)
    )

    # wvma10: 成交量加权价格变化波动率 = Std(|ret|*vol, 5) / Mean(|ret|*vol, 5)
    # 改为日内rolling5，缩短窗口提升及时性
    abs_ret = close_ret_cont.abs()
    weighted = abs_ret * df["volume"]
    wvma_std = grp["close"].transform(
        lambda x: weighted.loc[x.index].rolling(5, min_periods=3).std()
    )
    wvma_mean = grp["close"].transform(
        lambda x: weighted.loc[x.index].rolling(5, min_periods=3).mean()
    )
    df["wvma10"] = np.where(wvma_mean > 0, wvma_std / wvma_mean, 0.0)

    # vsumd10: 成交量方向性指标（放量占比 - 缩量占比）
    # 改为日内rolling5，缩短窗口
    vol_chg = grp["volume"].diff(1)
    sum_pos = grp["close"].transform(
        lambda x: vol_chg.loc[x.index].clip(lower=0).rolling(5, min_periods=3).sum()
    )
    sum_neg = grp["close"].transform(
        lambda x: (-vol_chg.loc[x.index]).clip(lower=0).rolling(5, min_periods=3).sum()
    )
    sum_abs = grp["close"].transform(
        lambda x: vol_chg.loc[x.index].abs().rolling(5, min_periods=3).sum()
    )
    df["vsumd10"] = np.where(sum_abs > 0, (sum_pos - sum_neg) / sum_abs, 0.0)

    # 清理中间列
    df.drop(columns=["cum_amount", "cum_volume", "vwap"], inplace=True)

    return df


# ==================== 跨日同时刻标准化（无未来函数） ====================
def normalize_features_time_series(
    df: pd.DataFrame, window: int = 20, min_periods: int = 10
) -> pd.DataFrame:
    """
    按 (SecuCode, time) 分组，用过去N个交易日同一时刻的历史因子值
    计算标准化参数，对当天同一时刻做去极值+z-score标准化。

    严格无未来函数：今天09:40的标准化参数 = 过去N天09:40历史值的统计量，
    通过 shift(1) 确保不包含当天任何信息。

    实现方式（全市场一次pivot，消除逐股票循环）：
      1. 对每个特征，pivot成 (date × [SecuCode, time]) 宽表
      2. shift(1)+rolling(mean/std) 一次计算所有股票所有时刻的统计量
      3. stack回长表，用 reindex 对齐到原始df行顺序（比merge快）
      4. 去极值（用历史mean±2*std近似2倍MAD）+ z-score

    性能说明：
      - rolling.mean/std 比 rolling.median 快3倍，且对同时刻历史值（近正态）近似误差小
      - 全市场一次pivot消除逐股票循环，1000只股票约1-2min

    参数：
      window      : 回看天数，默认20个交易日（约1个月）
      min_periods : 最小有效天数，不足时输出NaN
    """
    logger.info(
        f"开始跨日同时刻标准化（回看={window}天，min_periods={min_periods}）..."
    )

    if "time" not in df.columns:
        df = df.copy()
        df["time"] = df["bar_time"].dt.strftime("%H:%M")

    norm_cols = [f + "_norm" for f in RAW_FEATURE_COLS]

    # 预建 (date, SecuCode, time) MultiIndex，用于 reindex 对齐
    key_idx = pd.MultiIndex.from_arrays(
        [df["date"], df["SecuCode"], df["time"]], names=["date", "SecuCode", "time"]
    )

    for feat in RAW_FEATURE_COLS:
        col_norm = feat + "_norm"

        # ---- 全市场一次pivot：行=date，列=(SecuCode, time) ----
        pivot = df.pivot_table(
            index="date", columns=["SecuCode", "time"], values=feat, aggfunc="first"
        )
        shifted = pivot.shift(1)

        # ---- 历史统计量（shift(1)后rolling，严格不含当天） ----
        hist_median = shifted.rolling(window, min_periods=min_periods).median()
        hist_mean = shifted.rolling(window, min_periods=min_periods).mean()
        hist_std = shifted.rolling(window, min_periods=min_periods).std()

        # ---- stack回长表，reindex对齐到原始df行顺序 ----
        # stack后 index = (date, SecuCode, time)
        median_s = hist_median.stack([0, 1], dropna=False)
        mean_s = hist_mean.stack([0, 1], dropna=False)
        std_s = hist_std.stack([0, 1], dropna=False)

        median_aligned = median_s.reindex(key_idx).values
        mean_aligned = mean_s.reindex(key_idx).values
        std_aligned = std_s.reindex(key_idx).values

        # ---- 步骤1：2倍MAD去极值（MAD ≈ 0.6745*std，正态近似） ----
        mad_approx = 0.6745 * std_aligned
        lower = median_aligned - 2.0 * mad_approx
        upper = median_aligned + 2.0 * mad_approx
        winsorized = np.clip(df[feat].values, lower, upper)

        # ---- 步骤2：z-score标准化 ----
        df[col_norm] = np.where(
            std_aligned > 0,
            (winsorized - mean_aligned) / std_aligned,
            np.nan,
        )

    nan_counts = df[norm_cols].isna().sum()
    logger.info(f"  NaN数量（前{min_periods}天无历史数据）: {nan_counts.to_dict()}")
    logger.success(f"跨日同时刻标准化完成，新增列: {norm_cols}")
    return df


# ==================== 全市场批量处理 ====================
def build_features(year: int, stock_list: list = None) -> pd.DataFrame:
    """
    加载指定年份数据，对所有（或指定）股票构造特征，并做时间序列标准化。

    参数:
        year       : 数据年份
        stock_list : 指定股票列表，None表示全部

    返回:
        含原始特征列和 _norm 标准化列的完整DataFrame
    """
    logger.info(f"加载 {year} 年5分钟数据...")
    pkl_path = CACHE_DIR / f"df_5m_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {pkl_path}")

    df_all = pd.read_pickle(pkl_path)

    if stock_list is not None:
        df_all = df_all[df_all["SecuCode"].isin(stock_list)].copy()

    df_all = df_all.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    df_all["time"] = df_all["bar_time"].dt.strftime("%H:%M")

    stocks = df_all["SecuCode"].unique()
    logger.info(f"共 {len(stocks)} 只股票，开始逐股构造原始特征...")

    parts = []
    for i, code in enumerate(stocks):
        df_s = df_all[df_all["SecuCode"] == code].copy()
        df_s = compute_features_single(df_s)
        parts.append(df_s)
        if (i + 1) % 100 == 0:
            logger.info(f"  已处理 {i+1}/{len(stocks)} 只股票")

    df_feat = pd.concat(parts, ignore_index=True)
    logger.success(f"原始特征构造完成，shape: {df_feat.shape}")

    # 时间序列标准化（每只股票独立，日内滚动窗口）
    df_feat = normalize_features_time_series(df_feat)

    return df_feat


# ==================== 主函数（单独运行时使用） ====================
def main():
    """
    示例：构造2024年全市场特征并保存
    手动设置参数后运行
    """
    YEAR = 2024
    # 如只想跑部分股票，在此填入列表；None表示全部
    STOCK_LIST = None

    df_feat = build_features(year=YEAR, stock_list=STOCK_LIST)

    out_path = OUTPUT_DIR / f"features_{YEAR}.pkl"
    df_feat.to_pickle(out_path)
    logger.success(f"特征数据已保存: {out_path}")
    logger.info(f"列名: {df_feat.columns.tolist()}")
    logger.info(f"shape: {df_feat.shape}")


if __name__ == "__main__":
    main()
