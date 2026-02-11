"""
Q1-Q4 综合分析脚本
Q1: 回归模型赚钱画像
Q2: 分类标签偏差验证
Q3: D7-D9单调下降分析
Q4: 回归信号 close离场 vs VWAP离场
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ts_portfolio_backtest import run_topn_backtest, run_topn_backtest_turnover

OUTPUT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/vwap-research/output")

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)

# 加载回归模型预测（含close数据）
df = pd.read_pickle(OUTPUT_DIR / "ts_lgbm_predictions_with_close.pkl")
print(f"数据: {len(df):,} 行, {df['date'].nunique()} 天")

# ================================================================
# Q1: 回归模型赚钱画像
# ================================================================
print("\n" + "=" * 100)
print("Q1: 回归模型赚钱画像 — 每天 Top-50 vs Bottom-50 的特征差异")
print("=" * 100)

df["daily_rank"] = df.groupby("date")["pred_prob"].rank(ascending=False)
df["group"] = "middle"
df.loc[df["daily_rank"] <= 50, "group"] = "top50"
max_rank = df.groupby("date")["daily_rank"].transform("max")
df.loc[df["daily_rank"] > max_rank - 50, "group"] = "bottom50"

profile_feats = [
    "X1", "X2", "X1_zscore", "X2_zscore", "rel_vol", "vol_accel",
    "vol_concentration", "morning_ret", "price_position", "morning_range",
    "intraday_vol", "overnight_gap", "prev_day_ret", "momentum_5d",
    "hist_vol_20d", "X1_diff1", "X1_morning_std",
]

header = f"{'特征':>20s} | {'Top50':>10s} {'Bottom50':>10s} {'全样本':>10s} | {'T-B':>8s} | z值"
print(f"\n{header}")
print("-" * 85)

top50 = df[df["group"] == "top50"]
bot50 = df[df["group"] == "bottom50"]

for feat in profile_feats:
    t = top50[feat].mean()
    b = bot50[feat].mean()
    a = df[feat].mean()
    diff = t - b
    std = df[feat].std()
    z = diff / std if std > 0 else 0
    arrow = "↑" if diff > 0 else "↓"
    star = "***" if abs(z) > 0.5 else "**" if abs(z) > 0.3 else "*" if abs(z) > 0.1 else ""
    print(f"{feat:>20s} | {t:>10.4f} {b:>10.4f} {a:>10.4f} | {diff:>+8.4f}{arrow}| z={z:>+5.2f} {star}")

print(f"\n各组实际收益:")
mid = df[df["group"] == "middle"]
for name, grp in [("Top50", top50), ("Middle", mid), ("Bottom50", bot50)]:
    print(
        f"  {name:>10s}: raw={grp['raw_ret'].mean()*10000:>6.2f}bps, "
        f"excess={grp['excess_ret'].mean()*10000:>6.2f}bps, "
        f"close_ret={grp['close_ret'].mean()*10000:>6.2f}bps"
    )

# ================================================================
# Q2: 分类标签偏差验证
# ================================================================
print("\n\n" + "=" * 100)
print("Q2: 分类标签 正类(label=1, vol_adj_ret>0.5) 的特征画像")
print("=" * 100)

pos = df[df["label"] == 1]
neg = df[df["label"] == 0]
print(f"正类: {len(pos):,} ({len(pos)/len(df):.1%}), 负类: {len(neg):,} ({len(neg)/len(df):.1%})")

header2 = f"{'特征':>20s} | {'正类':>10s} {'负类':>10s} {'差值':>10s} | 正类偏向"
print(f"\n{header2}")
print("-" * 80)

for feat in profile_feats:
    p = pos[feat].mean()
    n = neg[feat].mean()
    diff = p - n
    std = df[feat].std()
    z = diff / std if std > 0 else 0
    direction = "偏高↑" if diff > 0 else "偏低↓"
    star = "***" if abs(z) > 0.3 else "**" if abs(z) > 0.15 else "*" if abs(z) > 0.05 else ""
    print(f"{feat:>20s} | {p:>10.4f} {n:>10.4f} {diff:>+10.4f} | {direction} z={z:>+.2f} {star}")

print(f"\n正类 vs 负类的VWAP/Close收益对比:")
print(f"  正类(label=1): raw_ret(VWAP)={pos['raw_ret'].mean()*10000:.2f}bps, close_ret={pos['close_ret'].mean()*10000:.2f}bps, 差={((pos['raw_ret']-pos['close_ret']).mean())*10000:.2f}bps")
print(f"  负类(label=0): raw_ret(VWAP)={neg['raw_ret'].mean()*10000:.2f}bps, close_ret={neg['close_ret'].mean()*10000:.2f}bps, 差={((neg['raw_ret']-neg['close_ret']).mean())*10000:.2f}bps")
print(f"  → 正类VWAP-Close差 vs 负类差: 正类冲高回落是否更严重?")

# 正类的 rel_vol 分布
print(f"\n正类 rel_vol 分布:")
for pct in [10, 25, 50, 75, 90]:
    v = pos["rel_vol"].quantile(pct / 100)
    v_all = df["rel_vol"].quantile(pct / 100)
    print(f"  P{pct}: 正类={v:.2f}, 全样本={v_all:.2f}")

# ================================================================
# Q3: D7-D9 单调下降分析
# ================================================================
print("\n\n" + "=" * 100)
print("Q3: 预测分位 D7-D9 单调下降深度分析")
print("=" * 100)

df["pred_decile"] = pd.qcut(df["pred_prob"], 10, labels=False, duplicates="drop")

print("\n各分位详细特征 (D7/D8/D9):")
header3 = f"{'指标':>20s} | {'D7':>10s} {'D8':>10s} {'D9':>10s} | {'D9-D7':>8s}"
print(header3)
print("-" * 70)

d7 = df[df["pred_decile"] == 7]
d8 = df[df["pred_decile"] == 8]
d9 = df[df["pred_decile"] == 9]

for col in ["pred_prob", "excess_ret", "raw_ret", "close_ret", "morning_ret",
            "morning_range", "rel_vol", "hist_vol_20d", "momentum_5d",
            "overnight_gap", "prev_day_ret", "intraday_vol", "price_position",
            "vol_accel", "X1", "X1_diff1"]:
    v7 = d7[col].mean()
    v8 = d8[col].mean()
    v9 = d9[col].mean()
    diff = v9 - v7
    mult = 10000 if col in ["pred_prob", "excess_ret", "raw_ret", "close_ret",
                             "morning_ret", "overnight_gap", "prev_day_ret",
                             "X1", "X1_diff1"] else 1
    unit = "bps" if mult == 10000 else ""
    print(f"{col:>20s} | {v7*mult:>10.2f} {v8*mult:>10.2f} {v9*mult:>10.2f} | {diff*mult:>+8.2f} {unit}")

# D9的VWAP vs Close差距
print(f"\n各分位 VWAP vs Close 离场差距:")
for d in range(10):
    grp = df[df["pred_decile"] == d]
    vwap = grp["raw_ret"].mean() * 10000
    close = grp["close_ret"].mean() * 10000
    diff = vwap - close
    bar = "█" * int(abs(diff))
    print(f"  D{d}: VWAP={vwap:>6.2f}bps, Close={close:>6.2f}bps, 差={diff:>+5.2f}bps {bar}")

# "运气"成分分析: bootstrap检验
print(f"\nD7-D9 超额的统计显著性 (bootstrap 95% CI):")
np.random.seed(42)
for d_label, d_data in [("D7", d7), ("D8", d8), ("D9", d9)]:
    daily_exc = d_data.groupby("date")["excess_ret"].mean() * 10000
    n_days = len(daily_exc)
    boot_means = []
    for _ in range(1000):
        sample = daily_exc.sample(n=n_days, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    t_stat = daily_exc.mean() / (daily_exc.std() / np.sqrt(n_days))
    print(f"  {d_label}: mean={daily_exc.mean():.2f}bps, 95%CI=[{ci_lo:.2f}, {ci_hi:.2f}], t={t_stat:.2f}")

# ================================================================
# Q4: 回归信号 close离场 vs VWAP离场 回测
# ================================================================
print("\n\n" + "=" * 100)
print("Q4: 回归模型 VWAP离场 vs Close离场 回测对比")
print("=" * 100)

COST_BPS = 15
COST_RATE = COST_BPS / 10000

# VWAP版（已有的 raw_ret）
print("\n--- VWAP离场 (原版 raw_ret) ---")
for top_n in [20, 50, 100]:
    r = run_topn_backtest(df, top_n=top_n)
    print(f"  Top-{top_n:>3d}: RawBps={r['avg_raw_bps']:>6.2f}, NetBps={r['avg_net_bps']:>6.2f}, ExcBps={r['avg_excess_bps']:>6.2f}, Sharpe(Exc)={r['sharpe_excess']:.2f}")

# Close版: 用 close_ret 替代 raw_ret 做回测
print("\n--- Close离场 (用 close_ret 替代 raw_ret) ---")
df_close = df.copy()
# 保存原始 raw_ret
df_close["orig_raw_ret"] = df_close["raw_ret"]
df_close["orig_excess_ret"] = df_close["excess_ret"]
# 替换为close版收益
df_close["raw_ret"] = df_close["close_ret"]
# close版的市场收益: 全样本close_ret的截面均值
close_market = df_close.groupby("date")["close_ret"].transform("mean")
df_close["excess_ret"] = df_close["close_ret"] - close_market
df_close["market_ret"] = close_market

for top_n in [20, 50, 100]:
    r = run_topn_backtest(df_close, top_n=top_n)
    print(f"  Top-{top_n:>3d}: RawBps={r['avg_raw_bps']:>6.2f}, NetBps={r['avg_net_bps']:>6.2f}, ExcBps={r['avg_excess_bps']:>6.2f}, Sharpe(Exc)={r['sharpe_excess']:.2f}")

# 带换手控制的对比
print("\n--- VWAP离场 + 2x缓冲换手控制 ---")
for top_n in [20, 50, 100]:
    r = run_topn_backtest_turnover(df, top_n=top_n, buffer_ratio=2.0)
    print(f"  Top-{top_n:>3d}: RawBps={r['avg_raw_bps']:>6.2f}, NetBps={r['avg_net_bps']:>6.2f}, ExcBps={r['avg_excess_bps']:>6.2f}, Turnover={r['avg_turnover']:.1%}, Sharpe(Net)={r['sharpe_net']:.2f}")

print("\n--- Close离场 + 2x缓冲换手控制 ---")
for top_n in [20, 50, 100]:
    r = run_topn_backtest_turnover(df_close, top_n=top_n, buffer_ratio=2.0)
    print(f"  Top-{top_n:>3d}: RawBps={r['avg_raw_bps']:>6.2f}, NetBps={r['avg_net_bps']:>6.2f}, ExcBps={r['avg_excess_bps']:>6.2f}, Turnover={r['avg_turnover']:.1%}, Sharpe(Net)={r['sharpe_net']:.2f}")

# 分位对比: VWAP vs Close
print("\n--- 各分位 VWAP vs Close 超额 ---")
header4 = f"{'分位':>5s} | {'VWAP ExcBps':>12s} {'Close ExcBps':>13s} {'差(VWAP-Close)':>15s}"
print(header4)
print("-" * 55)
for d in range(10):
    grp = df[df["pred_decile"] == d]
    vwap_exc = grp["excess_ret"].mean() * 10000
    close_exc = (grp["close_ret"] - close_market.loc[grp.index]).mean() * 10000
    diff = vwap_exc - close_exc
    print(f"  D{d:>2d}  | {vwap_exc:>12.2f} {close_exc:>13.2f} {diff:>+15.2f}")

print("\n分析完成!")
