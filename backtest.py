"""
backtest.py — Part 1 of "Build Your Own Quant Research System".

Thin, honest wrapper around vectorbt's Portfolio.from_signals.
Sensible defaults (fees, slippage, freq) so you can't accidentally
run a fantasy backtest.

Tutorial: https://goosos.com/vectorbt-tutorial
Code: https://github.com/goosos/vectorbt-tutorial
"""
import pandas as pd
import vectorbt as vbt


DEFAULT_FEES = 0.001      # 10 bps per side
DEFAULT_SLIPPAGE = 0.0005  # 5 bps


def run_backtest(price, entries, exits,
                 fees=DEFAULT_FEES,
                 slippage=DEFAULT_SLIPPAGE,
                 freq="1D",
                 init_cash=100.0,
                 **kwargs):
    """
    Run a vectorized backtest with honest defaults.

    Parameters
    ----------
    price : pd.Series
        Asset close prices, datetime-indexed.
    entries, exits : pd.Series of bool
        Entry/exit signals. SHIFT THEM ONE BAR BEFORE CALLING
        (entries.vbt.signals.fshift(1)) to avoid lookahead bias.
    fees : float
        Proportional commission per side. Default 0.1%.
    slippage : float
        Proportional slippage against the trader. Default 0.05%.
    freq : str
        Bar frequency ("1D", "1h"). REQUIRED for correct annualization.
    init_cash : float
        Starting capital.
    **kwargs
        Passed through to vbt.Portfolio.from_signals.

    Returns
    -------
    vbt.Portfolio
    """
    return vbt.Portfolio.from_signals(
        price, entries, exits,
        fees=fees,
        slippage=slippage,
        freq=freq,
        init_cash=init_cash,
        **kwargs,
    )


def ma_crossover_signals(price, fast=20, slow=50):
    """
    Classic MA crossover signals, shifted one bar (no lookahead).

    Returns (entries, exits) as boolean Series.
    """
    fast_ma = vbt.MA.run(price, fast)
    slow_ma = vbt.MA.run(price, slow)
    entries = fast_ma.ma_crossed_above(slow_ma).vbt.signals.fshift(1)
    exits = fast_ma.ma_crossed_below(slow_ma).vbt.signals.fshift(1)
    return entries, exits


def summary(pf):
    """One-line dict of the metrics that matter."""
    return {
        "total_return": float(pf.total_return()),
        "sharpe": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": float(pf.trades.win_rate()),
        "n_trades": int(pf.trades.count()),
        "profit_factor": float(pf.trades.profit_factor()),
    }
