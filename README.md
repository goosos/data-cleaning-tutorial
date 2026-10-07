# Data Cleaning Tutorial

**Part 3 of [_Build Your Own Quant Research System_](https://github.com/goosos/quant-toolkit)** — the article: [Data Cleaning & Alignment: Garbage In, Garbage Out](https://goosos.com/data-cleaning-alignment) *(draft)*

Your backtest is only as honest as your data. This tutorial shows you what corporate actions, missing bars, and survivorship bias do to your numbers — and contributes `data.py` to [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit).

## What it does

Same MA(20,50) strategy, same SPY, same 2020–2026 period — two different datasets:

**Answer from the actual run:** adjusted prices give **80.2%** backtest return; unadjusted give **65.0%**. A 15.2-point gap from one boolean flag. Survivorship bias adds another +1.2%/year in the toy demo.

## Run it

```bash
pip install -r requirements.txt
python data_demo.py
```

Needs internet for Yahoo Finance data.

## Files

| File | What |
|---|---|
| `data.py` | The module: `load_and_clean()` + `align_symbols()` + `adjusted_vs_raw()` |
| `backtest.py` | Part 1's module (vendored so this repo runs standalone) |
| `data_demo.py` | Three demos: adjusted vs raw, survivorship bias, multi-symbol alignment |
| `diagram.webp` / `survivorship.webp` | Charts (WebP versions; PNGs kept locally) |
| `article.md` | Full tutorial text |

## The module

```python
from data import load_and_clean, align_symbols

# One call: adjusted, tz-aware, sanity-checked
df = load_and_clean("SPY", start="2020-01-01")

# Align two symbols to a common clock
aligned = align_symbols({
    "SPY": load_and_clean("SPY", start="2024-01-01"),
    "QQQ": load_and_clean("QQQ", start="2024-01-01"),
}, how="inner")
```

`data.py` was merged into [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit) as its third module.

## Tested

yfinance 1.7.0 · Python 3.12 · pandas 3.0.6 · Last verified: 2026-10-07
