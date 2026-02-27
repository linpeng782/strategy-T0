"""
公共配置与数据加载
"""

from pathlib import Path
import pandas as pd
from loguru import logger

# ==================== 路径配置 ====================
CACHE_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/vwap-research/backtest_cache")
OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")
FEAT_CACHE_DIR = OUTPUT_DIR / "feat_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 因子分组定义 ====================
# 各组原始特征列名，供标准化和ic_analysis使用
FEATURE_GROUPS = {
    # 经典技术指标：MACD双参数、KDJ、布林带、RSV
    "classic": [
        "dif",  # MACD快线组(9,12,7) DIF
        "dea",  # MACD快线组 DEA
        "macd",  # MACD快线组 柱状值
        "macd_area",  # MACD快线组 累计面积
        "dif_slope3",  # MACD快线组 DIF斜率
        "dif_std",  # MACD标准组(12,26,9) DIF
        "dea_std",  # MACD标准组 DEA
        "macd_std",  # MACD标准组 柱状值
        "kdj_k",  # KDJ(9,3,3) K值
        "kdj_d",  # KDJ D值
        "kdj_j",  # KDJ J值（放大超买超卖）
        "kdj_kd",  # KD差值（类似DIF-DEA）
        "boll_pct_b",  # 布林带%B（价格在带内位置）
        "boll_bw",  # 布林带带宽（波动率收缩/扩张）
        "rsv10",  # 近10bar价格位置
        "rsv20",  # 近20bar价格位置
    ],
    "price_volume": [
        "vol_ratio",
        "buy_pressure",
        "amplitude",
        "ret_1bar",
        "ret_3bar",
        "ret_5bar",
        "range_pos",
    ],
    "alpha158": [
        "vwap_bias",  # close/日内累计VWAP-1（价格位置类）
        "x2",  # cum_vwap/cum_twap-1（量加权 vs 时间加权均价偏离，正向因子）
        "kmid",
        "kup2",
        "klow2",
        "cord20",
        "wvma10",
        "vsumd10",
    ],
}

# 所有原始特征列（按组顺序合并）
RAW_FEATURE_COLS = (
    FEATURE_GROUPS["classic"]
    + FEATURE_GROUPS["price_volume"]
    + FEATURE_GROUPS["alpha158"]
)


def load_raw_data(year: int, stock_list: list = None) -> pd.DataFrame:
    """加载指定年份5分钟K线数据，可选过滤股票列表"""
    pkl_path = CACHE_DIR / f"df_5m_{year}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {pkl_path}")

    logger.info(f"加载 {year} 年5分钟数据: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    if stock_list is not None:
        df = df[df["SecuCode"].isin(stock_list)].copy()

    df = df.sort_values(["SecuCode", "date", "bar_time"]).reset_index(drop=True)
    df["time"] = df["bar_time"].dt.strftime("%H:%M")
    logger.success(f"数据加载完成，shape={df.shape}，股票数={df['SecuCode'].nunique()}")
    return df
