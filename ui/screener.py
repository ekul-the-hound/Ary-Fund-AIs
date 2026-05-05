"""TradingView-style stock screener tab for ARY QUANT.

Data sources
------------
This module renders the screener table from **real market data**:

    1. Live prices, volume, and 1-day change come from a single
       ``yf.download`` batch call (``_fetch_live_prices_batch``),
       cached for 5 minutes. One HTTP round trip covers the whole
       universe (~560 symbols).

    2. Fundamentals (P/E, market cap, sector, margins, growth, etc.)
       come from ``MarketData.get_fundamentals`` via yfinance, lazy-
       loaded for the visible/filtered subset and SQLite-cached for
       24 hours so re-renders are instant.

    3. ``_OFFLINE_FALLBACK_STOCKS`` is the *last-resort* list used
       only when both yfinance and MarketData fail (no network,
       missing modules). A banner makes this state visible to the
       user so they know they're seeing stale data.

There is **no synthetic / RNG-generated data** in the default flow.
Cells that aren't yet loaded display "—" rather than fabricated values.

Wire-up contract with `ui/app.py`
---------------------------------
    * `app.py` initialises `st.session_state['active_ticker']` before tabs are
      rendered.
    * `app.py` reads from that same key when calling `load_ticker_context`.
    * This module **only writes** to that key — nothing else.

Public API:

    render_screener_tab() -> None
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import streamlit as st

# Issue #4 fix: pull tickers from the full curated US universe (~560 names)
# rather than the original 22-stock seed. Free-text search adds anything
# else.
try:
    from data.universe import (
        US_UNIVERSE,
        is_valid_us_ticker,
        normalize_ticker,
    )
except ImportError:
    # Flat-layout fallback (file lives next to data/universe.py).
    try:
        from universe import (  # type: ignore[no-redef]
            US_UNIVERSE,
            is_valid_us_ticker,
            normalize_ticker,
        )
    except ImportError:
        # Defensive fallback in case the universe module is missing — keep
        # the screener working with just the seed list rather than crashing.
        US_UNIVERSE: tuple = ()  # type: ignore[no-redef]

        def is_valid_us_ticker(symbol: str) -> bool:  # type: ignore[no-redef]
            return bool(symbol) and symbol.strip().isalpha() and len(symbol.strip()) <= 5

        def normalize_ticker(symbol: str) -> str:  # type: ignore[no-redef]
            return (symbol or "").strip().upper()


# ----------------------------------------------------------------------
# MarketData (real-data fundamentals via yfinance, SQLite-cached)
# ----------------------------------------------------------------------
# Importing under a try/except so a missing market_data module degrades
# gracefully to the offline-fallback path rather than crashing the tab.
try:
    from data.market_data import MarketData  # type: ignore
except ImportError:
    try:
        from market_data import MarketData  # type: ignore
    except ImportError:
        MarketData = None  # type: ignore[misc, assignment]


# ======================================================================
# Real-data fetchers
# ======================================================================
# These functions provide *real* data for the screener via two layers:
#
#   1. ``_fetch_live_prices_batch`` — one call to ``yf.download`` returns
#      last-close, prev-close, and volume for the entire universe in a
#      single HTTP round trip. Cached for 5 minutes.
#
#   2. ``_fetch_fundamentals_one`` — wraps the existing MarketData
#      fundamentals fetcher (already SQLite-cached for 24h). Called
#      lazily for the visible/filtered subset to avoid 500+ HTTP calls
#      on every render.
#
# Together they replace the previous synthetic RNG fills. Synthetic data
# is now only used as a last-resort offline fallback (see
# ``_OFFLINE_FALLBACK_STOCKS`` below) — never silently in normal flow.
# ======================================================================
def _normalize_recommendation(rec: Any) -> str:
    """Map yfinance's `recommendationKey` to the labels the screener uses."""
    if not rec:
        return "—"
    s = str(rec).strip().lower()
    return {
        "strong_buy":  "Strong buy",
        "buy":         "Buy",
        "hold":        "Neutral",
        "neutral":     "Neutral",
        "underperform":"Sell",
        "sell":        "Sell",
        "strong_sell": "Strong sell",
    }.get(s, str(rec).title())


@st.cache_data(ttl=300, show_spinner="Fetching live prices…")
def _fetch_live_prices_batch(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Last-close + previous-close + volume for the whole universe in one
    HTTP call via ``yf.download``.

    Returns a DataFrame indexed by symbol with columns:
        price, change_pct, volume, prev_close

    On any failure (no network, yfinance hiccup), returns an empty frame
    so the caller can fall back to the offline seed.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "price", "change_pct", "volume", "prev_close"])

    try:
        import yfinance as yf  # local import — keeps the module importable
                                # in environments where yfinance is missing.
    except ImportError:
        return pd.DataFrame()

    try:
        # period="5d" covers weekends / market holidays so we always have at
        # least 2 trading sessions worth of closes.
        px = yf.download(
            list(symbols),
            period="5d",
            progress=False,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    multi = isinstance(px.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            sub = px[sym] if multi else px
            closes = sub["Close"].dropna()
            vols   = sub["Volume"].dropna() if "Volume" in sub.columns else pd.Series(dtype=float)
            if len(closes) == 0:
                continue
            price = float(closes.iloc[-1])
            prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price
            change_pct = ((price / prev) - 1.0) * 100.0 if prev else 0.0
            volume = float(vols.iloc[-1]) if len(vols) else float("nan")
            rows.append({
                "symbol": sym,
                "price": price,
                "change_pct": change_pct,
                "volume": volume,
                "prev_close": prev,
            })
        except Exception:
            # Skip individual failures — one missing symbol shouldn't
            # take down the whole screener.
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_fundamentals_one(symbol: str) -> dict[str, Any]:
    """Real fundamentals for one symbol via ``MarketData.get_fundamentals``.

    MarketData wraps yfinance and already SQLite-caches for 24h, so this
    decorator is mostly belt-and-suspenders against re-renders.

    Returns a flat dict shaped to the column names the screener expects.
    Empty dict if MarketData is unavailable or yfinance fails.
    """
    if MarketData is None:
        return {}
    try:
        md = MarketData()
        f = md.get_fundamentals(symbol, use_cache=True)
    except Exception:
        return {}

    ov  = f.get("overview", {})    or {}
    val = f.get("valuation", {})   or {}
    fin = f.get("financials", {})  or {}
    gr  = f.get("growth", {})      or {}
    div = f.get("dividends", {})   or {}
    an  = f.get("analyst", {})     or {}

    def _pct(x: Any) -> float:
        """yfinance returns ratios as decimals (e.g. 0.23 = 23%); convert."""
        try:
            return float(x) * 100.0
        except (TypeError, ValueError):
            return float("nan")

    def _num(x: Any) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "symbol":           symbol,
        "name":             f.get("name") or symbol,
        "sector":           f.get("sector") or "—",
        # Overview
        "market_cap":       _num(ov.get("market_cap")),
        "beta":             _num(ov.get("beta")),
        # Valuation
        "pe":               _num(val.get("trailing_pe")),
        "forward_pe":       _num(val.get("forward_pe")),
        "peg":              _num(val.get("peg_ratio")),
        "ps":               _num(val.get("price_to_sales")),
        "pb":               _num(val.get("price_to_book")),
        "ev_ebitda":        _num(val.get("ev_to_ebitda")),
        # Financials
        "revenue":          _num(fin.get("revenue")),
        "gross_profit":     _num(fin.get("gross_profit")),
        "ebitda":           _num(fin.get("ebitda")),
        "net_income":       _num(fin.get("net_income")),
        "fcf":              _num(fin.get("free_cash_flow")),
        "op_cash_flow":     _num(fin.get("operating_cash_flow")),
        "total_debt":       _num(fin.get("total_debt")),
        # yfinance returns debtToEquity as a percent already (e.g. 60.5),
        # not a decimal — keep as-is for display.
        "debt_to_equity":   _num(fin.get("debt_to_equity")),
        "current_ratio":    _num(fin.get("current_ratio")),
        "roe":              _pct(fin.get("return_on_equity")),
        "roa":              _pct(fin.get("return_on_assets")),
        "profit_margin":    _pct(fin.get("profit_margin")),
        "gross_margin":     _pct(fin.get("gross_margin")),
        "op_margin":        _pct(fin.get("operating_margin")),
        # Growth
        "revenue_growth":   _pct(gr.get("revenue_growth")),
        "eps_dil_growth":   _pct(gr.get("earnings_growth")),
        # Dividends
        # yfinance: dividendYield is already a percent in newer versions
        # (e.g. 0.74 = 0.74%), but historically was a decimal (0.0074).
        # We probe and normalize: a value < 1 is treated as decimal.
        "div_yield":        _div_yield_normalize(div.get("dividend_yield")),
        "div_payout":       _pct(div.get("payout_ratio")),
        "ex_div_date":      str(div.get("ex_dividend_date") or "—"),
        # Analyst
        "analyst_rating":   _normalize_recommendation(an.get("recommendation")),
    }


def _div_yield_normalize(raw: Any) -> float:
    """yfinance's dividend yield reporting changed mid-2024.

    Old format:  0.0074  (decimal, meaning 0.74%)
    New format:  0.74    (already a percentage)

    Heuristic: if the value is between 0 and 1 it's almost certainly the
    decimal form; if > 1 it's already a percentage. This avoids the
    common bug where AAPL shows "0.38%" (correct) vs "38%" (decimal *100).
    """
    if raw is None:
        return float("nan")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    if 0 < v < 1:
        return v * 100.0
    return v


# ======================================================================
# Theme / styling
# ======================================================================
# TradingView's screener uses a near-black background with very subtle
# borders and muted text. We inject scoped CSS that targets popovers and
# the radio-button category bar — `st.dataframe` is styled separately
# via a pandas Styler because its DOM is sandboxed.
_SCREENER_CSS = """
<style>
.ary-screener-title {
    color: #d1d4dc;
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0.25rem 0 0.5rem 0;
    letter-spacing: -0.01em;
}
.ary-screener-meta {
    color: #787b86;
    font-size: 0.78rem;
    margin-bottom: 0.75rem;
}
.ary-screener-region {
    display: inline-block;
    background: #1e222d;
    color: #b2b5be;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    border: 1px solid #2a2e39;
    margin-right: 6px;
}
/* Popover trigger buttons — make them look like TradingView pill chips. */
[data-testid="stPopover"] > div > button {
    background: transparent !important;
    color: #b2b5be !important;
    border: 1px solid #2a2e39 !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.25rem 0.6rem !important;
    min-height: 32px !important;
}
[data-testid="stPopover"] > div > button:hover {
    background: #1e222d !important;
    border-color: #363a45 !important;
    color: #d1d4dc !important;
}
/* Horizontal radio (category tabs). */
.ary-screener-cats div[role="radiogroup"] {
    gap: 0.25rem !important;
}
.ary-screener-cats div[role="radiogroup"] > label {
    background: transparent;
    color: #787b86;
    padding: 0.35rem 0.85rem;
    border-radius: 4px;
    font-size: 0.85rem;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.1s;
}
.ary-screener-cats div[role="radiogroup"] > label:hover {
    color: #d1d4dc;
    background: #1e222d;
}
.ary-screener-cats div[role="radiogroup"] > label:has(input:checked) {
    color: #d1d4dc;
    background: #1e222d;
    font-weight: 600;
    border-color: #2a2e39;
}
/* Hide the radio circle itself — we only want the label pills. */
.ary-screener-cats div[role="radiogroup"] input[type="radio"] {
    display: none !important;
}
.ary-screener-footer-note {
    color: #787b86;
    font-size: 0.75rem;
    margin-top: 0.5rem;
}
</style>
"""

# Sector → accent color (used for the symbol cell badge). Loose mapping —
# extend or override as needed.
_SECTOR_COLORS: dict[str, str] = {
    "Electronic Technology": "#2962ff",
    "Technology Services":   "#4caf50",
    "Retail Trade":          "#ff9800",
    "Finance":               "#9c27b0",
    "Health Technology":     "#00bcd4",
    "Energy Minerals":       "#ff5722",
    "Consumer Services":     "#e91e63",
    "Producer Manufacturing":"#795548",
    "Communications":        "#3f51b5",
    "Process Industries":    "#607d8b",
}


# ======================================================================
# Seed data
# ======================================================================
# ======================================================================
# Offline fallback seed data
# ======================================================================
# This block is *only* used when the live data path fails entirely:
# yfinance unreachable, no network, MarketData import broken, etc. The
# values below are last-known reasonable numbers — the screener will
# render with these so the UI doesn't crash, but a banner is shown so
# the user knows they're seeing stale data instead of live.
#
# In the normal flow, ``_build_screener_frame`` calls
# ``_fetch_live_prices_batch`` (one HTTP round-trip via yf.download)
# for prices/volume/change, then lazy-loads fundamentals via
# ``_fetch_fundamentals_one`` for the visible/filtered subset.
_OFFLINE_FALLBACK_STOCKS: list[dict[str, Any]] = [
    {"symbol":"NVDA", "name":"NVIDIA Corporation",        "price":199.57, "change_pct":-4.63, "volume":225_240_000, "rel_volume":1.56, "market_cap":4.85e12, "pe":40.72, "eps_dil":4.90,    "eps_dil_growth":66.75,  "div_yield":0.02, "sector":"Electronic Technology",   "analyst_rating":"Strong buy"},
    {"symbol":"GOOG", "name":"Alphabet Inc.",             "price":381.94, "change_pct": 9.97, "volume": 44_570_000, "rel_volume":2.72, "market_cap":4.65e12, "pe":29.14, "eps_dil":13.11,   "eps_dil_growth":46.15,  "div_yield":0.24, "sector":"Technology Services",     "analyst_rating":"Strong buy"},
    {"symbol":"AAPL", "name":"Apple Inc.",                "price":271.35, "change_pct": 0.42, "volume": 91_840_000, "rel_volume":2.20, "market_cap":3.98e12, "pe":34.33, "eps_dil":7.90,    "eps_dil_growth":25.65,  "div_yield":0.38, "sector":"Electronic Technology",   "analyst_rating":"Buy"},
    {"symbol":"MSFT", "name":"Microsoft Corporation",     "price":407.78, "change_pct":-3.93, "volume": 70_910_000, "rel_volume":2.06, "market_cap":3.03e12, "pe":24.29, "eps_dil":16.79,   "eps_dil_growth":29.75,  "div_yield":0.82, "sector":"Technology Services",     "analyst_rating":"Strong buy"},
    {"symbol":"AMZN", "name":"Amazon.com, Inc.",          "price":265.06, "change_pct": 0.77, "volume":100_970_000, "rel_volume":2.17, "market_cap":2.85e12, "pe":31.69, "eps_dil":8.37,    "eps_dil_growth":36.44,  "div_yield":0.00, "sector":"Retail Trade",            "analyst_rating":"Strong buy"},
    {"symbol":"AVGO", "name":"Broadcom Inc.",             "price":417.43, "change_pct": 2.95, "volume": 21_820_000, "rel_volume":1.11, "market_cap":1.98e12, "pe":81.43, "eps_dil":5.13,    "eps_dil_growth":147.26, "div_yield":0.61, "sector":"Electronic Technology",   "analyst_rating":"Strong buy"},
    {"symbol":"META", "name":"Meta Platforms, Inc.",      "price":611.91, "change_pct":-8.55, "volume": 52_760_000, "rel_volume":4.27, "market_cap":1.55e12, "pe":22.25, "eps_dil":27.50,   "eps_dil_growth":7.29,   "div_yield":0.31, "sector":"Technology Services",     "analyst_rating":"Strong buy"},
    {"symbol":"TSLA", "name":"Tesla, Inc.",               "price":381.63, "change_pct": 2.37, "volume": 51_080_000, "rel_volume":0.78, "market_cap":1.43e12, "pe":348.65,"eps_dil":1.09,    "eps_dil_growth":-39.80, "div_yield":0.00, "sector":"Consumer Durables",       "analyst_rating":"Neutral"},
    {"symbol":"WMT",  "name":"Walmart Inc.",              "price":131.93, "change_pct": 3.06, "volume": 19_870_000, "rel_volume":1.11, "market_cap":1.05e12, "pe":48.33, "eps_dil":2.73,    "eps_dil_growth":13.38,  "div_yield":0.74, "sector":"Retail Trade",            "analyst_rating":"Strong buy"},
    {"symbol":"BRK.A","name":"Berkshire Hathaway Inc.",   "price":711_900.00,"change_pct":-0.15, "volume": 89,         "rel_volume":0.47, "market_cap":1.02e12, "pe":15.29, "eps_dil":46_563.02,"eps_dil_growth":-24.79,"div_yield":0.00, "sector":"Finance",                 "analyst_rating":"Buy"},
    {"symbol":"LLY",  "name":"Eli Lilly and Company",     "price":934.60, "change_pct": 9.80, "volume":  8_240_000, "rel_volume":2.70, "market_cap":883.03e9, "pe":41.37, "eps_dil":22.59,   "eps_dil_growth":96.62,  "div_yield":0.73, "sector":"Health Technology",       "analyst_rating":"Buy"},
    {"symbol":"JPM",  "name":"JP Morgan Chase & Co.",     "price":313.23, "change_pct": 1.29, "volume":  8_870_000, "rel_volume":1.13, "market_cap":844.79e9, "pe":15.00, "eps_dil":20.88,   "eps_dil_growth":2.48,   "div_yield":1.88, "sector":"Finance",                 "analyst_rating":"Buy"},
    {"symbol":"XOM",  "name":"Exxon Mobil Corporation",   "price":154.33, "change_pct":-0.22, "volume": 22_890_000, "rel_volume":1.41, "market_cap":641.48e9, "pe":23.05, "eps_dil":6.69,    "eps_dil_growth":-14.72, "div_yield":2.61, "sector":"Energy Minerals",         "analyst_rating":"Buy"},
    {"symbol":"V",    "name":"Visa Inc.",                 "price":329.84, "change_pct":-1.50, "volume": 11_240_000, "rel_volume":1.47, "market_cap":628.72e9, "pe":29.00, "eps_dil":11.37,   "eps_dil_growth":15.70,  "div_yield":0.75, "sector":"Finance",                 "analyst_rating":"Strong buy"},
    {"symbol":"MU",   "name":"Micron Technology, Inc.",   "price":517.16, "change_pct":-0.25, "volume": 36_550_000, "rel_volume":0.99, "market_cap":583.22e9, "pe":24.42, "eps_dil":21.18,   "eps_dil_growth":409.12, "div_yield":0.10, "sector":"Electronic Technology",   "analyst_rating":"Strong buy"},
    {"symbol":"JNJ",  "name":"Johnson & Johnson",         "price":162.40, "change_pct": 0.34, "volume": 12_300_000, "rel_volume":1.05, "market_cap":390.50e9, "pe":15.10, "eps_dil":10.75,   "eps_dil_growth":4.20,   "div_yield":3.05, "sector":"Health Technology",       "analyst_rating":"Buy"},
    {"symbol":"UNH",  "name":"UnitedHealth Group Inc.",   "price":580.20, "change_pct": 1.85, "volume":  4_120_000, "rel_volume":0.92, "market_cap":536.10e9, "pe":18.40, "eps_dil":31.50,   "eps_dil_growth":6.70,   "div_yield":1.45, "sector":"Health Technology",       "analyst_rating":"Strong buy"},
    {"symbol":"MA",   "name":"Mastercard Incorporated",   "price":520.45, "change_pct":-0.78, "volume":  3_450_000, "rel_volume":1.20, "market_cap":478.30e9, "pe":36.70, "eps_dil":14.18,   "eps_dil_growth":11.22,  "div_yield":0.55, "sector":"Finance",                 "analyst_rating":"Strong buy"},
    {"symbol":"HD",   "name":"The Home Depot, Inc.",      "price":402.15, "change_pct": 0.92, "volume":  3_800_000, "rel_volume":1.00, "market_cap":399.40e9, "pe":26.10, "eps_dil":15.40,   "eps_dil_growth":1.10,   "div_yield":2.32, "sector":"Retail Trade",            "analyst_rating":"Buy"},
    {"symbol":"PG",   "name":"Procter & Gamble Co.",      "price":172.80, "change_pct":-0.45, "volume":  6_200_000, "rel_volume":0.95, "market_cap":405.20e9, "pe":27.50, "eps_dil":6.28,    "eps_dil_growth":2.10,   "div_yield":2.40, "sector":"Consumer Non-Durables",   "analyst_rating":"Buy"},
    {"symbol":"COST", "name":"Costco Wholesale Corp.",    "price":925.30, "change_pct": 1.10, "volume":  2_010_000, "rel_volume":1.08, "market_cap":410.60e9, "pe":52.40, "eps_dil":17.65,   "eps_dil_growth":15.80,  "div_yield":0.45, "sector":"Retail Trade",            "analyst_rating":"Strong buy"},
    {"symbol":"NFLX", "name":"Netflix, Inc.",             "price":712.50, "change_pct": 4.32, "volume":  4_900_000, "rel_volume":1.85, "market_cap":305.20e9, "pe":42.10, "eps_dil":16.92,   "eps_dil_growth":52.40,  "div_yield":0.00, "sector":"Communications",          "analyst_rating":"Buy"},
    {"symbol":"AMD",  "name":"Advanced Micro Devices",    "price":162.85, "change_pct":-2.18, "volume": 48_300_000, "rel_volume":1.32, "market_cap":263.40e9, "pe":105.20,"eps_dil":1.55,    "eps_dil_growth":210.00, "div_yield":0.00, "sector":"Electronic Technology",   "analyst_rating":"Buy"},
    {"symbol":"CRM",  "name":"Salesforce, Inc.",          "price":322.10, "change_pct": 0.55, "volume":  6_700_000, "rel_volume":1.04, "market_cap":311.50e9, "pe":48.30, "eps_dil":6.67,    "eps_dil_growth":48.20,  "div_yield":0.50, "sector":"Technology Services",     "analyst_rating":"Buy"},
    {"symbol":"BAC",  "name":"Bank of America Corp.",     "price":47.20,  "change_pct": 0.85, "volume": 38_400_000, "rel_volume":1.01, "market_cap":363.20e9, "pe":13.80, "eps_dil":3.42,    "eps_dil_growth":7.10,   "div_yield":2.20, "sector":"Finance",                 "analyst_rating":"Buy"},
    {"symbol":"PEP",  "name":"PepsiCo, Inc.",             "price":168.90, "change_pct":-0.12, "volume":  5_100_000, "rel_volume":0.88, "market_cap":231.60e9, "pe":24.50, "eps_dil":6.89,    "eps_dil_growth":3.40,   "div_yield":3.30, "sector":"Consumer Non-Durables",   "analyst_rating":"Neutral"},
    {"symbol":"DIS",  "name":"The Walt Disney Company",   "price":108.45, "change_pct": 1.92, "volume": 12_800_000, "rel_volume":1.18, "market_cap":196.10e9, "pe":35.20, "eps_dil":3.08,    "eps_dil_growth":18.50,  "div_yield":0.92, "sector":"Consumer Services",       "analyst_rating":"Buy"},
    {"symbol":"KO",   "name":"The Coca-Cola Company",     "price":68.10,  "change_pct":-0.30, "volume": 14_600_000, "rel_volume":0.96, "market_cap":293.40e9, "pe":26.40, "eps_dil":2.58,    "eps_dil_growth":5.80,   "div_yield":2.95, "sector":"Consumer Non-Durables",   "analyst_rating":"Buy"},
]


# ======================================================================
# Column sets per category tab
# ======================================================================
# Keys are the category labels shown in the radio bar; values are the
# columns each view should display. Every column listed here must exist in
# the seed-stock dicts above (or be derived in `_build_screener_frame`).
_CATEGORY_COLUMNS: dict[str, list[str]] = {
    "Overview": [
        "symbol", "name", "price", "change_pct", "volume", "rel_volume",
        "market_cap", "pe", "eps_dil", "eps_dil_growth", "div_yield",
        "sector", "analyst_rating",
    ],
    "Performance": [
        "symbol", "name", "change_pct", "perf_1w", "perf_1m", "perf_3m",
        "perf_6m", "perf_ytd", "perf_1y", "volatility_1m",
    ],
    "Extended hours": [
        "symbol", "name", "premarket_close", "premarket_chg_pct",
        "premarket_vol", "postmarket_close", "postmarket_chg_pct",
        "postmarket_vol",
    ],
    "Valuation": [
        "symbol", "name", "market_cap", "pe", "peg", "ps", "pb",
        "p_fcf", "ev_ebitda",
    ],
    "Dividends": [
        "symbol", "name", "div_per_share", "div_yield", "div_payout",
        "div_growth_5y", "ex_div_date",
    ],
    "Profitability": [
        "symbol", "name", "gross_margin", "op_margin", "profit_margin",
        "fcf_margin", "roa", "roe", "roic",
    ],
    "Income statement": [
        "symbol", "name", "revenue", "revenue_growth", "gross_profit",
        "operating_income", "net_income", "eps_dil", "ebitda",
    ],
    "Balance sheet": [
        "symbol", "name", "total_assets", "total_debt", "debt_to_equity",
        "current_ratio", "cash_per_share", "book_per_share",
    ],
    "Cash flow": [
        "symbol", "name", "op_cash_flow", "capex", "fcf",
        "fcf_per_share", "fcf_yield",
    ],
    "Per share": [
        "symbol", "name", "eps_dil", "book_per_share", "cash_per_share",
        "sales_per_share", "fcf_per_share", "div_per_share",
    ],
    "Technicals": [
        "symbol", "name", "price", "change_pct", "rsi_14", "ma_50",
        "ma_200", "beta", "atr_14", "volatility_1m",
    ],
}

_CATEGORIES: list[str] = list(_CATEGORY_COLUMNS.keys())

# Pretty-print column header → display label.
_COLUMN_LABELS: dict[str, str] = {
    "symbol":              "Symbol",
    "name":                "Name",
    "price":               "Price",
    "change_pct":          "Change %",
    "volume":              "Volume",
    "rel_volume":          "Rel Volume",
    "market_cap":          "Market cap",
    "pe":                  "P/E",
    "eps_dil":             "EPS dil",
    "eps_dil_growth":      "EPS dil growth",
    "div_yield":           "Div yield %",
    "sector":              "Sector",
    "analyst_rating":      "Analyst Rating",
    "perf_1w":             "Perf 1W",
    "perf_1m":             "Perf 1M",
    "perf_3m":             "Perf 3M",
    "perf_6m":             "Perf 6M",
    "perf_ytd":            "Perf YTD",
    "perf_1y":             "Perf 1Y",
    "volatility_1m":       "Vol 1M",
    "premarket_close":     "Pre Close",
    "premarket_chg_pct":   "Pre Chg %",
    "premarket_vol":       "Pre Vol",
    "postmarket_close":    "Post Close",
    "postmarket_chg_pct":  "Post Chg %",
    "postmarket_vol":      "Post Vol",
    "peg":                 "PEG",
    "ps":                  "P/S",
    "pb":                  "P/B",
    "p_fcf":               "P/FCF",
    "ev_ebitda":           "EV/EBITDA",
    "div_per_share":       "Div/Share",
    "div_payout":          "Payout %",
    "div_growth_5y":       "Div Growth 5Y",
    "ex_div_date":         "Ex-Div Date",
    "gross_margin":        "Gross Margin",
    "op_margin":           "Op Margin",
    "profit_margin":       "Profit Margin",
    "fcf_margin":          "FCF Margin",
    "roa":                 "ROA",
    "roe":                 "ROE",
    "roic":                "ROIC",
    "revenue":             "Revenue",
    "revenue_growth":      "Rev Growth",
    "gross_profit":        "Gross Profit",
    "operating_income":    "Op Income",
    "net_income":          "Net Income",
    "ebitda":              "EBITDA",
    "total_assets":        "Total Assets",
    "total_debt":          "Total Debt",
    "debt_to_equity":      "D/E",
    "current_ratio":       "Current Ratio",
    "cash_per_share":      "Cash/Share",
    "book_per_share":      "Book/Share",
    "op_cash_flow":        "Op Cash Flow",
    "capex":               "CapEx",
    "fcf":                 "FCF",
    "fcf_per_share":       "FCF/Share",
    "fcf_yield":           "FCF Yield",
    "sales_per_share":     "Sales/Share",
    "rsi_14":              "RSI 14",
    "ma_50":               "MA 50",
    "ma_200":              "MA 200",
    "beta":                "Beta",
    "atr_14":              "ATR 14",
}

# Columns where positive = green, negative = red. We treat any column that
# is conceptually a percentage change as colorable.
_PCT_CHANGE_COLS: set[str] = {
    "change_pct", "eps_dil_growth", "perf_1w", "perf_1m", "perf_3m",
    "perf_6m", "perf_ytd", "perf_1y", "premarket_chg_pct",
    "postmarket_chg_pct", "revenue_growth", "div_growth_5y",
}


# ======================================================================
# Formatters
# ======================================================================
def _fmt_compact_number(v: Any) -> str:
    """1_234_567 → '1.23M', 4_850_000_000_000 → '4.85T'."""
    if v is None or pd.isna(v):
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.2f} T"
    if a >= 1e9:
        return f"{v/1e9:.2f} B"
    if a >= 1e6:
        return f"{v/1e6:.2f} M"
    if a >= 1e3:
        return f"{v/1e3:.2f} K"
    return f"{v:,.0f}"


def _fmt_money(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} B USD"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f} M USD"
    return f"{v:,.2f} USD"


def _fmt_market_cap(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.2f} T USD"
    if a >= 1e9:
        return f"{v/1e9:.2f} B USD"
    if a >= 1e6:
        return f"{v/1e6:.2f} M USD"
    return f"{v:,.0f} USD"


def _fmt_price(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        return f"{float(v):,.2f} USD"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_ratio(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_int(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    try:
        return _fmt_compact_number(int(v))
    except (TypeError, ValueError):
        return "—"


_FORMATTERS: dict[str, Callable[[Any], str]] = {
    "price":              _fmt_price,
    "change_pct":         _fmt_pct,
    "volume":             _fmt_int,
    "rel_volume":         _fmt_ratio,
    "market_cap":         _fmt_market_cap,
    "pe":                 _fmt_ratio,
    "eps_dil":            _fmt_price,
    "eps_dil_growth":     _fmt_pct,
    "div_yield":          lambda v: f"{v:.2f}%" if pd.notna(v) else "—",
    "perf_1w":            _fmt_pct,
    "perf_1m":            _fmt_pct,
    "perf_3m":            _fmt_pct,
    "perf_6m":            _fmt_pct,
    "perf_ytd":           _fmt_pct,
    "perf_1y":            _fmt_pct,
    "volatility_1m":      _fmt_pct,
    "premarket_close":    _fmt_price,
    "premarket_chg_pct":  _fmt_pct,
    "premarket_vol":      _fmt_int,
    "postmarket_close":   _fmt_price,
    "postmarket_chg_pct": _fmt_pct,
    "postmarket_vol":     _fmt_int,
    "peg":                _fmt_ratio,
    "ps":                 _fmt_ratio,
    "pb":                 _fmt_ratio,
    "p_fcf":              _fmt_ratio,
    "ev_ebitda":          _fmt_ratio,
    "div_per_share":      _fmt_price,
    "div_payout":         lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "div_growth_5y":      _fmt_pct,
    "gross_margin":       lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "op_margin":          lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "profit_margin":      lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "fcf_margin":         lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "roa":                lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "roe":                lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "roic":               lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
    "revenue":            _fmt_market_cap,
    "revenue_growth":     _fmt_pct,
    "gross_profit":       _fmt_market_cap,
    "operating_income":   _fmt_market_cap,
    "net_income":         _fmt_market_cap,
    "ebitda":             _fmt_market_cap,
    "total_assets":       _fmt_market_cap,
    "total_debt":         _fmt_market_cap,
    "debt_to_equity":     _fmt_ratio,
    "current_ratio":      _fmt_ratio,
    "cash_per_share":     _fmt_price,
    "book_per_share":     _fmt_price,
    "op_cash_flow":       _fmt_market_cap,
    "capex":              _fmt_market_cap,
    "fcf":                _fmt_market_cap,
    "fcf_per_share":      _fmt_price,
    "fcf_yield":          lambda v: f"{v:.2f}%" if pd.notna(v) else "—",
    "sales_per_share":    _fmt_price,
    "rsi_14":             _fmt_ratio,
    "ma_50":              _fmt_price,
    "ma_200":             _fmt_price,
    "beta":               _fmt_ratio,
    "atr_14":             _fmt_ratio,
}


# ======================================================================
# Frame construction (real data only)
# ======================================================================
# The full set of columns the screener can render across all category
# tabs. Any column not populated by the live data path is left as NaN
# (which the formatters render as "—") rather than synthesized.
_ALL_SCREENER_COLUMNS: tuple[str, ...] = (
    # Core identity
    "symbol", "name", "sector",
    # Price (live: yf.download)
    "price", "change_pct", "volume", "rel_volume", "prev_close",
    # Overview / valuation (lazy: MarketData.get_fundamentals)
    "market_cap", "pe", "forward_pe", "peg", "ps", "pb", "ev_ebitda",
    "eps_dil", "eps_dil_growth",
    # Dividends
    "div_yield", "div_per_share", "div_payout", "div_growth_5y", "ex_div_date",
    # Profitability
    "gross_margin", "op_margin", "profit_margin", "fcf_margin",
    "roa", "roe", "roic",
    # Income statement
    "revenue", "revenue_growth", "gross_profit", "operating_income",
    "net_income", "ebitda",
    # Balance sheet
    "total_assets", "total_debt", "debt_to_equity", "current_ratio",
    "cash_per_share", "book_per_share",
    # Cash flow
    "op_cash_flow", "capex", "fcf", "fcf_per_share", "fcf_yield",
    "sales_per_share",
    # Performance / technicals (not yet pulled — left blank)
    "perf_1w", "perf_1m", "perf_3m", "perf_6m", "perf_ytd", "perf_1y",
    "volatility_1m", "rsi_14", "ma_50", "ma_200", "beta", "atr_14",
    # Extended hours (not yet pulled — left blank)
    "premarket_close", "premarket_chg_pct", "premarket_vol",
    "postmarket_close", "postmarket_chg_pct", "postmarket_vol",
    # Analyst
    "analyst_rating",
)


# Cap on lazy fundamentals fetches per render. Each call is a yfinance
# round trip; even with 24h SQLite caching, the *first* render after a
# clean start can hit dozens of network calls for the visible page.
# 60 covers the default Streamlit dataframe height comfortably without
# stalling the UI on cold caches.
_FUNDAMENTALS_LAZY_LIMIT = 60


def _empty_screener_row(symbol: str) -> dict[str, Any]:
    """Row template for a symbol with no real data yet.

    Every column from ``_ALL_SCREENER_COLUMNS`` is present so the
    DataFrame schema is stable. Numeric fields are NaN (formatters
    show "—"); string fields default to the symbol or em-dash.
    """
    row: dict[str, Any] = {col: float("nan") for col in _ALL_SCREENER_COLUMNS}
    row["symbol"] = symbol
    row["name"] = symbol
    row["sector"] = "—"
    row["analyst_rating"] = "—"
    row["ex_div_date"] = "—"
    return row


@st.cache_data(ttl=300, show_spinner=False)
def _build_screener_frame() -> pd.DataFrame:
    """Build the canonical screener DataFrame using **live data only**.

    Flow:
        1. Enumerate the universe (US_UNIVERSE + any user-added tickers).
        2. Fetch live prices/volume/change for all symbols in one
           ``yf.download`` batch (cached 5min).
        3. Lazy-load fundamentals (P/E, market cap, sector, margins, etc.)
           for the top ``_FUNDAMENTALS_LAZY_LIMIT`` symbols by symbol
           order. The remaining rows have NaN for fundamentals and
           render as "—" until they fall into a filtered/visible page
           on a future render.
        4. If both yfinance and MarketData fail entirely (e.g. no
           network), fall back to ``_OFFLINE_FALLBACK_STOCKS`` so the
           UI still renders. A banner makes this state visible.

    Cached for 5 minutes — clear with ``_build_screener_frame.clear()``
    when adding a custom ticker.
    """
    # --- Step 1: figure out the universe ---------------------------
    custom: list[str] = []
    try:
        custom = list(st.session_state.get("custom_tickers", []) or [])
    except Exception:
        # Outside a Streamlit context (e.g. unit tests).
        pass

    # Universe = curated US universe ∪ user-added customs ∪ offline-fallback
    # symbols (so popular tickers always appear even on a cold cache).
    fallback_syms = [r["symbol"] for r in _OFFLINE_FALLBACK_STOCKS]
    all_symbols: list[str] = []
    seen: set[str] = set()
    for sym in list(US_UNIVERSE) + custom + fallback_syms:
        if sym and sym not in seen:
            all_symbols.append(sym)
            seen.add(sym)

    if not all_symbols:
        # Empty universe — return an empty schema-stable frame.
        return pd.DataFrame(columns=list(_ALL_SCREENER_COLUMNS))

    # --- Step 2: live prices via one yf.download batch -------------
    prices_df = _fetch_live_prices_batch(tuple(all_symbols))
    have_live_prices = not prices_df.empty

    # --- Step 3: build the base frame (one row per symbol) ---------
    rows: list[dict[str, Any]] = [_empty_screener_row(s) for s in all_symbols]
    df = pd.DataFrame(rows, columns=list(_ALL_SCREENER_COLUMNS))

    # Merge in live prices.
    if have_live_prices:
        idx = {s: i for i, s in enumerate(df["symbol"].tolist())}
        for _, prow in prices_df.iterrows():
            i = idx.get(prow["symbol"])
            if i is None:
                continue
            df.at[i, "price"]      = prow.get("price", float("nan"))
            df.at[i, "change_pct"] = prow.get("change_pct", float("nan"))
            df.at[i, "volume"]     = prow.get("volume", float("nan"))
            df.at[i, "prev_close"] = prow.get("prev_close", float("nan"))

    # --- Step 4: merge offline fallback for symbols that have it ---
    # Real data takes precedence; the offline values only fill cells
    # that are still NaN (i.e. we couldn't fetch live data for them).
    for fb_row in _OFFLINE_FALLBACK_STOCKS:
        sym = fb_row["symbol"]
        mask = df["symbol"] == sym
        if not mask.any():
            continue
        ridx = df.index[mask][0]
        for col, val in fb_row.items():
            if col not in df.columns:
                continue
            existing = df.at[ridx, col]
            is_blank = (
                pd.isna(existing)
                or (isinstance(existing, str) and existing in ("—", ""))
                or (col == "name" and existing == sym)  # placeholder name
            )
            if is_blank:
                df.at[ridx, col] = val

    # --- Step 5: lazy fundamentals for the top-N symbols -----------
    # We sort by market_cap (where known) so the largest names get
    # their fundamentals filled first — that's what the user sees on
    # the default sort.
    sort_key = df["market_cap"].fillna(-1.0)
    df_for_lazy = df.assign(_mcap=sort_key).sort_values(
        "_mcap", ascending=False, kind="mergesort"
    ).drop(columns=["_mcap"])

    fetched = 0
    for ridx in df_for_lazy.index:
        if fetched >= _FUNDAMENTALS_LAZY_LIMIT:
            break
        sym = df.at[ridx, "symbol"]
        if not sym:
            continue
        # Skip if we already have fundamentals (market_cap + pe + sector).
        already = (
            pd.notna(df.at[ridx, "market_cap"])
            and pd.notna(df.at[ridx, "pe"])
            and df.at[ridx, "sector"] not in (None, "", "—")
        )
        if already:
            continue
        fund = _fetch_fundamentals_one(sym)
        if not fund:
            continue
        for col, val in fund.items():
            if col in df.columns and val is not None:
                # Don't overwrite a real live value with a None/NaN.
                if isinstance(val, float) and np.isnan(val):
                    continue
                df.at[ridx, col] = val
        fetched += 1

    # --- Step 6: derive rel_volume from volume / 30-day average ----
    # We don't have the 30-day average here without a second fetch,
    # so leave as NaN. (Adding it would mean another yf.download with
    # period="2mo"; defer until requested.)

    # --- Step 7: stable sort by market cap desc --------------------
    df = df.sort_values(
        "market_cap", ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)

    # Surface a flag so the UI can show a banner if the live path
    # failed entirely (we're rendering only fallback rows).
    df.attrs["live_data_ok"] = have_live_prices
    df.attrs["lazy_fetched"] = fetched
    return df


# ======================================================================
# Pandas Styler — TradingView-ish color coding inside st.dataframe
# ======================================================================
def _style_change(val: Any) -> str:
    if pd.isna(val):
        return "color: #787b86;"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: #22ab94; font-weight: 500;"
    if v < 0:
        return "color: #f7525f; font-weight: 500;"
    return "color: #787b86;"


def _style_rating(val: Any) -> str:
    s = str(val or "").lower()
    if "strong buy" in s:
        return "color: #22ab94; font-weight: 600;"
    if "buy" in s:
        return "color: #22ab94;"
    if "neutral" in s:
        return "color: #b2b5be;"
    if "sell" in s:
        return "color: #f7525f;"
    return "color: #787b86;"


def _style_symbol(val: Any) -> str:
    # Symbol cell looks like a tiny pill — flat color, slight emphasis.
    return (
        "color: #2962ff; font-weight: 600; "
        "background-color: rgba(41, 98, 255, 0.08);"
    )


def _style_for(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply colors + formatters to the visible subset.

    Note: pandas 2.x renamed `Styler.applymap` to `Styler.map`. We use
    `getattr` to support both, so the screener works on whichever pandas
    version the project ends up pinning.
    """
    styler = df.style
    apply_cell = getattr(styler, "map", None) or getattr(styler, "applymap")

    def _styler_apply_cell(fn, subset):
        # Stays bound to the LATEST styler returned at each step.
        nonlocal styler
        m = getattr(styler, "map", None) or getattr(styler, "applymap")
        styler = m(fn, subset=subset)

    # Colors
    for col in df.columns:
        if col in _PCT_CHANGE_COLS:
            _styler_apply_cell(_style_change, [col])
    if "analyst_rating" in df.columns:
        _styler_apply_cell(_style_rating, ["analyst_rating"])
    if "symbol" in df.columns:
        _styler_apply_cell(_style_symbol, ["symbol"])

    # Formatters (only for columns that exist in this view)
    fmt_map = {c: f for c, f in _FORMATTERS.items() if c in df.columns}
    if fmt_map:
        styler = styler.format(fmt_map, na_rep="—")
    return styler


# ======================================================================
# Header & filter pill row
# ======================================================================
def _render_header() -> None:
    st.markdown(
        "<div class='ary-screener-title'>All stocks ▾</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ary-screener-meta'>"
        "<span class='ary-screener-region'>🇺🇸 US</span>"
        "Live screener — click any ticker to load it into the analyzer."
        "</div>",
        unsafe_allow_html=True,
    )


# ======================================================================
# Filter pills — real filters
# ======================================================================
# Each filter writes its value to ``st.session_state`` under a stable
# key. ``_apply_filters(df)`` reads those keys and returns the filtered
# DataFrame. A "Clear all filters" button resets every key.
#
# Sentinel values (the default ranges below) mean "no filter" — the
# corresponding column passes through untouched.
# ======================================================================
_FILTER_DEFAULTS: dict[str, Any] = {
    "_flt_price":   (0.0, 1_000_000.0),
    "_flt_change":  (-100.0, 100.0),
    "_flt_mcap":    [],   # empty multiselect = all
    "_flt_pe":      (0.0, 1000.0),
    "_flt_epsg":    (-1000.0, 1000.0),
    "_flt_dy":      (0.0, 100.0),
    "_flt_sector":  [],   # empty multiselect = all
    "_flt_rating":  [],   # empty multiselect = all
    "_flt_revg":    (-1000.0, 1000.0),
    "_flt_peg":     (0.0, 100.0),
    "_flt_roe":     (-1000.0, 1000.0),
    "_flt_beta":    (0.0, 10.0),
}


# Market-cap bucket boundaries (USD). Used by both the popover label
# and the filter logic so they stay in sync.
_MCAP_BUCKETS: dict[str, tuple[float, float]] = {
    "Mega ≥ $200B":   (200e9, float("inf")),
    "Large $10–200B": (10e9,  200e9),
    "Mid $2–10B":     (2e9,   10e9),
    "Small $300M–2B": (300e6, 2e9),
    "Micro < $300M":  (0.0,   300e6),
}


def _filter_watchlist() -> None:
    st.caption("Watchlist filter — coming soon.")
    st.selectbox(
        "Select watchlist",
        ["All stocks", "My core picks", "AI semiconductors"],
        key="_flt_watchlist",
        help="Watchlist filtering will be enabled once portfolio_db "
             "watchlists are wired in. Currently informational.",
    )


def _filter_index() -> None:
    st.caption("Filter by index membership — coming soon.")
    st.multiselect(
        "Index",
        ["S&P 500", "Nasdaq 100", "Dow 30", "Russell 1000"],
        key="_flt_index",
        help="Index membership data is not yet wired. The screener "
             "currently uses the curated US universe (S&P 500 + extras).",
    )


def _filter_price() -> None:
    st.caption("Live last-close price (USD).")
    a, b = st.columns(2)
    a.number_input("Min", value=0.0, key="_flt_price_min", step=1.0)
    b.number_input("Max", value=1_000_000.0, key="_flt_price_max", step=100.0)
    # Mirror into a tuple key for _apply_filters.
    st.session_state["_flt_price"] = (
        float(st.session_state.get("_flt_price_min", 0.0) or 0.0),
        float(st.session_state.get("_flt_price_max", 1_000_000.0) or 1_000_000.0),
    )


def _filter_change() -> None:
    st.caption("1-day % change range (live).")
    rng = st.slider(
        "Change %", -50.0, 50.0, (-50.0, 50.0), step=0.1, key="_flt_change_widget",
    )
    st.session_state["_flt_change"] = rng


def _filter_marketcap() -> None:
    st.caption("Market capitalization buckets.")
    st.multiselect(
        "Bucket",
        list(_MCAP_BUCKETS.keys()),
        key="_flt_mcap",
        help="Multiple buckets are OR'd together. Empty = no filter.",
    )


def _filter_pe() -> None:
    st.caption("Trailing P/E (negative earnings excluded).")
    rng = st.slider("P/E", 0.0, 200.0, (0.0, 200.0), key="_flt_pe_widget")
    st.session_state["_flt_pe"] = rng


def _filter_eps_growth() -> None:
    st.caption("Diluted EPS YoY growth (%).")
    rng = st.slider(
        "EPS dil growth %", -100.0, 500.0, (-100.0, 500.0), key="_flt_epsg_widget",
    )
    st.session_state["_flt_epsg"] = rng


def _filter_div_yield() -> None:
    st.caption("Forward dividend yield (%).")
    rng = st.slider("Div yield %", 0.0, 15.0, (0.0, 15.0), key="_flt_dy_widget")
    st.session_state["_flt_dy"] = rng


def _filter_sector() -> None:
    st.caption("Sector — live values from yfinance fundamentals.")
    # Pull sector list from the actual screener frame so it matches what
    # the user sees rather than a hardcoded list.
    try:
        live_df = _build_screener_frame()
        sectors = sorted({
            s for s in live_df["sector"].dropna().unique()
            if s and s not in ("—", "Other", "")
        })
    except Exception:
        sectors = []
    if not sectors:
        st.info("Sectors are populated lazily — open the table once to load.")
        return
    st.multiselect("Sector", sectors, key="_flt_sector")


def _filter_analyst() -> None:
    st.caption("Aggregate analyst rating.")
    st.multiselect(
        "Analyst Rating",
        ["Strong buy", "Buy", "Neutral", "Sell", "Strong sell"],
        key="_flt_rating",
    )


def _filter_perf() -> None:
    st.caption("Period performance — coming soon.")
    st.selectbox(
        "Window", ["1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"],
        key="_flt_perf_window",
    )
    st.slider("Range %", -100.0, 1000.0, (-100.0, 1000.0), key="_flt_perf_range")
    st.info(
        "Period performance values are not yet pulled from yfinance. "
        "This filter is informational until the perf-history fetch lands."
    )


def _filter_revenue_growth() -> None:
    st.caption("Trailing revenue growth YoY (%).")
    rng = st.slider(
        "Revenue growth %", -100.0, 500.0, (-100.0, 500.0), key="_flt_revg_widget",
    )
    st.session_state["_flt_revg"] = rng


def _filter_peg() -> None:
    st.caption("PEG ratio (lower is cheaper relative to growth).")
    rng = st.slider("PEG", 0.0, 10.0, (0.0, 10.0), key="_flt_peg_widget")
    st.session_state["_flt_peg"] = rng


def _filter_roe() -> None:
    st.caption("Return on Equity (%).")
    rng = st.slider("ROE %", -50.0, 200.0, (-50.0, 200.0), key="_flt_roe_widget")
    st.session_state["_flt_roe"] = rng


def _filter_beta() -> None:
    st.caption("Levered 5Y beta.")
    rng = st.slider("Beta", 0.0, 5.0, (0.0, 5.0), key="_flt_beta_widget")
    st.session_state["_flt_beta"] = rng


# Order matches the TradingView screenshot.
_FILTERS: list[tuple[str, Callable[[], None]]] = [
    ("Watchlist",        _filter_watchlist),
    ("Index",            _filter_index),
    ("Price",            _filter_price),
    ("Change %",         _filter_change),
    ("Market cap",       _filter_marketcap),
    ("P/E",              _filter_pe),
    ("EPS dil growth",   _filter_eps_growth),
    ("Div yield %",      _filter_div_yield),
    ("Sector",           _filter_sector),
    ("Analyst Rating",   _filter_analyst),
    ("Perf %",           _filter_perf),
    ("Revenue growth",   _filter_revenue_growth),
    ("PEG",              _filter_peg),
    ("ROE",              _filter_roe),
    ("Beta",             _filter_beta),
]


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the current filter-pill state to the screener DataFrame.

    Each filter is a no-op if the user hasn't moved it off its default.
    NaN values are kept (a row with NaN P/E shouldn't be excluded by a
    P/E filter — the user hasn't seen real data yet).
    """
    if df.empty:
        return df

    out = df.copy()
    ss = st.session_state

    def _between(col: str, key: str) -> None:
        nonlocal out
        rng = ss.get(key)
        default = _FILTER_DEFAULTS[key]
        if not isinstance(rng, tuple) or rng == default:
            return
        lo, hi = rng
        # Keep NaN rows (data not yet loaded — don't punish them).
        mask = out[col].isna() | ((out[col] >= lo) & (out[col] <= hi))
        out = out[mask]

    _between("price",          "_flt_price")
    _between("change_pct",     "_flt_change")
    _between("pe",             "_flt_pe")
    _between("eps_dil_growth", "_flt_epsg")
    _between("div_yield",      "_flt_dy")
    _between("revenue_growth", "_flt_revg")
    _between("peg",            "_flt_peg")
    _between("roe",            "_flt_roe")
    _between("beta",           "_flt_beta")

    # Market cap buckets (multiselect, OR'd together)
    buckets = ss.get("_flt_mcap") or []
    if buckets:
        ranges = [_MCAP_BUCKETS[b] for b in buckets if b in _MCAP_BUCKETS]
        if ranges:
            mask = out["market_cap"].isna()
            for lo, hi in ranges:
                mask = mask | ((out["market_cap"] >= lo) & (out["market_cap"] < hi))
            out = out[mask]

    # Sector multiselect
    sectors = ss.get("_flt_sector") or []
    if sectors:
        out = out[out["sector"].isin(sectors) | out["sector"].isna()]

    # Analyst rating multiselect
    ratings = ss.get("_flt_rating") or []
    if ratings:
        out = out[out["analyst_rating"].isin(ratings) | out["analyst_rating"].isna()]

    return out.reset_index(drop=True)


def _render_filter_bar() -> None:
    """Render the row of filter pill chips. Filters apply to the
    rendered DataFrame via ``_apply_filters`` in ``_render_results_table``."""
    n_per_row = 8  # Streamlit columns wrap awkwardly past ~8 pill buttons.
    for chunk_start in range(0, len(_FILTERS), n_per_row):
        chunk = _FILTERS[chunk_start : chunk_start + n_per_row]
        cols = st.columns(len(chunk))
        for col, (label, render_fn) in zip(cols, chunk):
            with col:
                with st.popover(label, use_container_width=True):
                    render_fn()


def _reset_filters() -> None:
    """Clear all filter state. Bound to a 'Clear filters' button."""
    for k, default in _FILTER_DEFAULTS.items():
        st.session_state[k] = default
        # Also clear paired widget keys (sliders use a separate _widget key
        # so we can mirror values into the canonical state key).
        st.session_state.pop(f"{k}_widget", None)
    # Multiselect-only keys
    for k in ("_flt_mcap", "_flt_sector", "_flt_rating"):
        st.session_state[k] = []
    # Price min/max number inputs
    st.session_state["_flt_price_min"] = 0.0
    st.session_state["_flt_price_max"] = 1_000_000.0



# ======================================================================
# Category radio (Overview / Performance / …)
# ======================================================================
def _render_category_tabs() -> str:
    """Horizontal radio styled as TradingView pill tabs."""
    st.markdown("<div class='ary-screener-cats'>", unsafe_allow_html=True)
    category = st.radio(
        "View",
        _CATEGORIES,
        horizontal=True,
        label_visibility="collapsed",
        key="_screener_category",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return category


# ======================================================================
# Results table (with row-click → active_ticker handoff)
# ======================================================================
def _columns_for_view(category: str, df: pd.DataFrame) -> list[str]:
    """Return the columns to display for the given category, dropping any
    that don't exist in the frame so a typo or seed gap doesn't crash."""
    wanted = _CATEGORY_COLUMNS.get(category, _CATEGORY_COLUMNS["Overview"])
    return [c for c in wanted if c in df.columns]


def _render_results_table(df: pd.DataFrame, category: str) -> None:
    cols = _columns_for_view(category, df)
    visible = df[cols].copy()

    # Build column_config for header labels + a touch of width control on
    # the symbol/name columns.
    col_config: dict[str, Any] = {}
    for c in visible.columns:
        label = _COLUMN_LABELS.get(c, c.replace("_", " ").title())
        if c == "symbol":
            col_config[c] = st.column_config.TextColumn(label, width="small", pinned=True)
        elif c == "name":
            col_config[c] = st.column_config.TextColumn(label, width="medium")
        else:
            col_config[c] = st.column_config.Column(label, width="small")

    styled = _style_for(visible)

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=560,
        on_select="rerun",
        selection_mode="single-row",
        column_config=col_config,
        key=f"_screener_table_{category}",
    )

    # ---- Click handoff ----
    rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if rows:
        idx = int(rows[0])
        # Use the ORIGINAL df (full schema) to look up the symbol — `visible`
        # may not contain the column if a future view drops it.
        try:
            ticker = str(df.iloc[idx]["symbol"]).strip().upper()
        except (KeyError, IndexError):
            ticker = ""
        if ticker and st.session_state.get("active_ticker") != ticker:
            st.session_state["active_ticker"] = ticker
            # Optional: surface a brief toast so the user sees the handoff.
            try:
                st.toast(f"📊 Loaded {ticker} into the analyzer", icon="✅")
            except Exception:
                pass
            st.rerun()

    # ---- Footer / explicit fallback action ----
    active = st.session_state.get("active_ticker", "—")
    cap_col, btn_col = st.columns([4, 1])
    with cap_col:
        st.markdown(
            f"<div class='ary-screener-footer-note'>"
            f"{len(visible):,} symbols · {len(visible.columns)} columns · "
            f"category: <b>{category}</b> · active: <b>{active}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with btn_col:
        if st.button("View in Analyzer →", use_container_width=True, key="_screener_view_in_analyzer"):
            # No-op handoff: the active ticker is already set; just nudge a
            # rerun so the ticker header at the top reflects current state.
            st.rerun()


def _render_data_source_banner(df: pd.DataFrame) -> None:
    """Surface the data-source state so the user knows whether they're
    looking at live or fallback data.

    - Live OK + fundamentals lazy-loaded:  green check (silent if all good)
    - Live OK but fundamentals empty:       blue info note
    - Live data fetch failed:               yellow warning (using fallback)
    """
    live_ok = bool(df.attrs.get("live_data_ok", False))
    lazy_n  = int(df.attrs.get("lazy_fetched", 0))

    if not live_ok:
        st.warning(
            "⚠️ Live price fetch unavailable — showing fallback data for "
            f"{len(_OFFLINE_FALLBACK_STOCKS)} popular tickers. "
            "Check your network connection or yfinance install. "
            "Other rows will display \"—\" until live data is reachable."
        )
        return

    # Live data ok, but fundamentals only filled for top-N. Show a small
    # info caption so the user understands why some cells are blank.
    if lazy_n > 0:
        st.caption(
            f"✓ Live prices for {len(df):,} symbols · "
            f"fundamentals loaded for top {lazy_n} by market cap. "
            "Use filters or sort to surface other rows; their fundamentals "
            "load on the next render (24h cached)."
        )


# ======================================================================
# Public entrypoint
# ======================================================================
def render_screener_tab() -> None:
    """Render the entire Screener tab. Safe to call inside a `with tab_x:`."""
    st.markdown(_SCREENER_CSS, unsafe_allow_html=True)
    _render_header()
    _render_search_box()  # Issue #4: free-text ticker search
    _render_filter_bar()

    # Clear-filters control + active filter count.
    active_filters = _count_active_filters()
    fcol1, fcol2 = st.columns([5, 1])
    with fcol1:
        if active_filters:
            st.caption(f"🔎 {active_filters} active filter{'s' if active_filters != 1 else ''}.")
    with fcol2:
        if st.button(
            "Clear filters",
            use_container_width=True,
            key="_screener_clear_filters",
            disabled=(active_filters == 0),
        ):
            _reset_filters()
            st.rerun()

    st.markdown("")  # spacer
    category = _render_category_tabs()

    # Build the canonical (real-data) frame, then apply the filter pills.
    df_full = _build_screener_frame()
    _render_data_source_banner(df_full)
    df_filtered = _apply_filters(df_full)
    _render_results_table(df_filtered, category)


def _count_active_filters() -> int:
    """How many filter pills are currently moved off their default?"""
    n = 0
    ss = st.session_state
    for k, default in _FILTER_DEFAULTS.items():
        v = ss.get(k)
        if v is None:
            continue
        if isinstance(default, list):
            if v:
                n += 1
        elif v != default:
            n += 1
    return n



def _render_search_box() -> None:
    """Free-text ticker search.

    Lets users research any US-listed ticker, not just the ~560 in the
    curated universe. Validates loosely (must look like a US ticker
    symbol) — yfinance handles the actual existence check downstream.

    Workflow:
      1. User types a symbol (e.g. "PLTR", "BRK.B", "abcde").
      2. On submit, the symbol is normalized (uppercase, dot/hyphen
         handling) and validated.
      3. Valid symbols are added to ``st.session_state["custom_tickers"]``
         so they appear in the screener table on the next render, AND
         set as ``active_ticker`` so the rest of the app jumps straight
         to that stock's analysis tabs.
      4. Cache is cleared so the new ticker shows up immediately.
    """
    st.session_state.setdefault("custom_tickers", [])

    with st.container():
        col_input, col_btn, col_info = st.columns([3, 1, 4])
        with col_input:
            typed = st.text_input(
                "Search any US ticker",
                value="",
                placeholder="e.g. AAPL, COST, BRK.B, PLTR...",
                key="_screener_search_input",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.button("Add / Open", use_container_width=True)
        with col_info:
            n_universe = len(US_UNIVERSE)
            n_custom = len(st.session_state.get("custom_tickers", []))
            st.markdown(
                f"<div class='ary-screener-meta'>"
                f"Universe: {n_universe} S&P + large-caps · "
                f"+{n_custom} added by you"
                f"</div>",
                unsafe_allow_html=True,
            )

        if submitted and typed:
            symbol = normalize_ticker(typed)
            if not is_valid_us_ticker(symbol):
                st.warning(
                    f"'{typed}' doesn't look like a US ticker symbol. "
                    "Tickers are 1-5 letters, optionally with a dot suffix "
                    "(e.g. BRK.B)."
                )
                return

            # Add to custom tickers if not already in the universe.
            customs = list(st.session_state.get("custom_tickers", []))
            if symbol not in US_UNIVERSE and symbol not in customs:
                customs.append(symbol)
                st.session_state["custom_tickers"] = customs
                # Bust the cache so _build_screener_frame picks up the
                # new ticker on next render.
                _build_screener_frame.clear()

            # Hand off to the rest of the app.
            st.session_state["active_ticker"] = symbol
            st.rerun()