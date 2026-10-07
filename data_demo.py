"""
data_demo.py — Part 3 demo: what dirty data does to your backtest.

1. Adjusted vs unadjusted closes: same strategy, same period —
   how much return "disappears" when dividends aren't reinvested.
2. Survivorship bias: a toy demo of why testing only today's
   survivors overstates the past.

Run:  python data_demo.py
Needs: internet (Yahoo Finance). Falls back to synthetic data offline.
"""
import warnings

import numpy as np
import pandas as pd

from data import adjusted_vs_raw, load_and_clean, align_symbols

try:
    from backtest import run_backtest, ma_crossover_signals, summary
except ImportError:  # pragma: no cover
    from quant_toolkit.backtest import run_backtest, ma_crossover_signals, summary

warnings.filterwarnings("ignore")


def demo_adjusted_vs_raw(symbol="SPY", start="2020-01-01", end="2026-10-06"):
    print(f"=== 1. Adjusted vs unadjusted: {symbol} ===")
    adj, raw = adjusted_vs_raw(symbol, start=start, end=end)
    print(f"bars: {len(adj)}  ({adj.index[0].date()} -> {adj.index[-1].date()})")
    print(f"adjusted total return: {(adj.iloc[-1] / adj.iloc[0] - 1):.1%}")
    print(f"raw      total return: {(raw.iloc[-1] / raw.iloc[0] - 1):.1%}")
    gap = (adj.iloc[-1] / adj.iloc[0]) - (raw.iloc[-1] / raw.iloc[0])
    print(f"gap (dividends you pretended didn't exist): {gap:.1%}")

    # Same MA(20,50) strategy from Part 1, both price series.
    results = {}
    for name, px in (("adjusted", adj), ("raw", raw)):
        entries, exits = ma_crossover_signals(px, fast=20, slow=50)
        pf = run_backtest(px, entries, exits, freq="1D")
        m = summary(pf)
        results[name] = m
        print(f"\n[{name}] Sharpe {m['sharpe']:.2f} | "
              f"return {m['total_return']:.1%} | "
              f"max DD {m['max_drawdown']:.1%} | "
              f"trades {m['n_trades']}")
    d = results["adjusted"]["total_return"] - results["raw"]["total_return"]
    print(f"\nBacktest return gap (adjusted - raw): {d:+.1%}")
    print("The strategy is identical. The data isn't.")
    return adj, raw, results


def demo_survivorship_bias():
    print("\n=== 2. Survivorship bias (toy demo) ===")
    # Ten fake stocks, 5 years of daily returns. Three of them "die"
    # (go to zero) in year 3 — like real delistings. A researcher who
    # only tests today's survivors never sees the corpses.
    rng = np.random.default_rng(42)
    n_days = 252 * 5
    names = [f"STK{i}" for i in range(10)]
    prices = {}
    for i, name in enumerate(names):
        rets = rng.normal(0.0004, 0.02, n_days)  # ~10%/yr drift, 32% vol
        px = 100 * np.exp(np.cumsum(rets))
        if i < 3:  # these three get delisted in year 3
            px[252 * 3:] = np.nan
        prices[name] = px
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B", tz="UTC")
    full = pd.DataFrame(prices, index=idx)

    # "Survivor" universe: only stocks alive at the end.
    survivors = full.dropna(axis=1)
    # Honest universe: point-in-time — each day, hold what's alive then.
    # (Toy version: equal-weight the alive set, rebalance daily.)
    alive_rets = full.pct_change().where(full.notna())
    n_alive = full.notna().sum(axis=1)
    honest_daily = (alive_rets.sum(axis=1) / n_alive).fillna(0)
    surv_rets = survivors.pct_change().mean(axis=1).fillna(0)

    honest_cagr = (1 + honest_daily).prod() ** (252 / n_days) - 1
    surv_cagr = (1 + surv_rets).prod() ** (252 / n_days) - 1
    print(f"stocks: 10 total, {len(survivors.columns)} survivors, "
          f"3 delisted in year 3")
    print(f"honest (point-in-time) CAGR:    {honest_cagr:.1%}")
    print(f"survivors-only CAGR:            {surv_cagr:.1%}")
    print(f"survivorship bias:               {surv_cagr - honest_cagr:+.1%} per year")
    print("Testing only survivors rewrites history in your favor.")
    return honest_cagr, surv_cagr


def demo_alignment():
    print("\n=== 3. Multi-symbol alignment ===")
    frames = {
        s: load_and_clean(s, start="2024-01-01", end="2024-06-30")
        for s in ("SPY", "QQQ")
    }
    for s, df in frames.items():
        print(f"{s}: {len(df)} bars, tz={df.index.tz}")
    aligned = align_symbols(frames, how="inner")
    idx = next(iter(aligned.values())).index
    print(f"inner join: {len(idx)} common bars "
          f"({idx[0].date()} -> {idx[-1].date()})")
    return aligned


if __name__ == "__main__":
    try:
        adj, raw, results = demo_adjusted_vs_raw()
    except Exception as e:  # offline fallback
        print(f"download failed ({e}); skipping live demos 1 & 3")
        adj = raw = None
    demo_survivorship_bias()
    if adj is not None:
        try:
            demo_alignment()
        except Exception as e:
            print(f"alignment demo failed: {e}")
