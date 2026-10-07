"""
data.py — Part 3 of "Build Your Own Quant Research System".

Honest data loading: corporate-action adjustment, missing-data handling,
and multi-symbol alignment. Garbage in, garbage out — this module makes
sure it's not garbage.

Tutorial: https://goosos.com/data-cleaning-alignment
Code: https://github.com/goosos/data-cleaning-tutorial
"""
import warnings

import pandas as pd

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:  # pragma: no cover
    _HAS_YFINANCE = False


def load_and_clean(symbol, start=None, end=None, interval="1d",
                   adjust=True, fill="none", tz="UTC"):
    """
    Download and clean OHLCV data for one symbol.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "SPY".
    start, end : str or None
        Date bounds, e.g. "2023-10-01". None = yfinance default.
    interval : str
        Bar size, e.g. "1d", "1h".
    adjust : bool
        True (default): use split-and-dividend-adjusted closes
        (yfinance auto_adjust=True). This is what you want for
        backtesting total returns — unadjusted prices understate
        long-run performance because dividends vanish into thin air.
        False: raw exchange prices (use only if you model corporate
        actions yourself).
    fill : {"none", "ffill", "drop"}
        How to handle missing bars:
        - "none" (default): leave NaNs in place and warn. Safest —
          forces you to notice gaps instead of silently inventing data.
        - "ffill": forward-fill. Convenient but DANGEROUS on daily
          bars: a halted stock flatlines instead of showing a gap,
          and your backtest trades a price that never existed.
        - "drop": drop rows with any NaN.
    tz : str
        Target timezone for the index. "UTC" (default) keeps everything
        comparable across symbols.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV with tz-aware DatetimeIndex. Columns:
        Open, High, Low, Close, Volume (+ Adj Close dropped when
        adjust=True since Close is already adjusted).
    """
    if not _HAS_YFINANCE:
        raise ImportError("yfinance is required: pip install yfinance")
    if fill not in ("none", "ffill", "drop"):
        raise ValueError("fill must be one of: none, ffill, drop")

    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=adjust, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol!r}")

    # yfinance >= 1.0 returns MultiIndex columns for single tickers;
    # flatten to plain OHLCV.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize timezone: everything in one tz so multi-symbol
    # alignment can't silently compare 4pm New York to 4pm London.
    if df.index.tz is None:
        warnings.warn(
            f"{symbol}: naive index, localizing to {tz}. "
            "Check your data source's documented timezone.",
            UserWarning, stacklevel=2,
        )
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    n_missing = int(df["Close"].isna().sum())
    if n_missing:
        msg = (f"{symbol}: {n_missing} missing Close bars "
               f"({n_missing / len(df):.1%} of {len(df)} rows)")
        if fill == "none":
            warnings.warn(msg + " — left as NaN (fill='none').",
                          UserWarning, stacklevel=2)
        elif fill == "ffill":
            warnings.warn(
                msg + " — forward-filled. On daily bars this invents "
                "prices for halted/suspended sessions; prefer fill='none' "
                "and handle gaps explicitly.",
                UserWarning, stacklevel=2,
            )
            df = df.ffill()
        else:  # drop
            warnings.warn(msg + " — rows dropped.", UserWarning,
                          stacklevel=2)
            df = df.dropna(subset=["Close"])

    # Sanity: prices must be positive; a zero/negative close means
    # corrupt data, not a buying opportunity.
    bad = df["Close"] <= 0
    if bad.any():
        raise ValueError(
            f"{symbol}: {int(bad.sum())} non-positive closes — "
            "data is corrupt, refusing to continue."
        )
    return df


def align_symbols(frames, how="inner", freq=None):
    """
    Align multiple OHLCV frames to a common datetime index.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        {symbol: cleaned OHLCV frame} as returned by load_and_clean.
    how : {"inner", "outer"}
        "inner" (default): keep only timestamps present in EVERY symbol.
        Safest for backtesting — no invented bars.
        "outer": union of timestamps (NaNs where a symbol didn't trade;
        combine with an explicit fill policy downstream).
    freq : str or None
        If given, reindex each frame to a regular grid of this frequency
        first (e.g. "1D" for daily). Useful when sources disagree on
        whether a half-day session counts as a bar.

    Returns
    -------
    dict[str, pd.DataFrame]
        Aligned frames sharing one DatetimeIndex.
    """
    if not frames:
        raise ValueError("frames is empty")
    if how not in ("inner", "outer"):
        raise ValueError("how must be 'inner' or 'outer'")

    work = {}
    for sym, df in frames.items():
        if df.index.tz is None:
            raise ValueError(
                f"{sym}: naive index — run load_and_clean first so all "
                "frames share one timezone."
            )
        work[sym] = df.asfreq(freq) if freq else df

    if how == "inner":
        common = None
        for df in work.values():
            common = df.index if common is None else common.intersection(df.index)
        common = common.sort_values()
    else:
        common = None
        for df in work.values():
            common = df.index if common is None else common.union(df.index)
        common = common.sort_values()

    dropped = {s: len(d) - len(common) if how == "inner" else 0
               for s, d in work.items()}
    if how == "inner" and any(v for v in dropped.values()):
        warnings.warn(
            "align_symbols(inner): dropped bars per symbol: " +
            ", ".join(f"{s}={v}" for s, v in dropped.items() if v) +
            ". Bars a symbol didn't trade are excluded from ALL symbols.",
            UserWarning, stacklevel=2,
        )
    return {s: d.reindex(common) for s, d in work.items()}


def adjusted_vs_raw(symbol, start=None, end=None):
    """
    Convenience helper for the tutorial's core demo: download both the
    adjusted and the raw (unadjusted) close series for one symbol so you
    can see exactly what corporate actions do to a backtest.

    Returns
    -------
    (adjusted, raw) : tuple of pd.Series
        Two Close series on the same tz-aware index.
    """
    adj = load_and_clean(symbol, start=start, end=end, adjust=True)
    raw = load_and_clean(symbol, start=start, end=end, adjust=False)
    common = adj.index.intersection(raw.index).sort_values()
    return adj["Close"].reindex(common), raw["Close"].reindex(common)
