"""TradingView-style stock screener tab for ARY QUANT.

Base-layout pass. This module is intentionally self-contained:

    - All data is seeded from `_SEED_STOCKS` so the layout renders instantly
      with no live-data dependency. Hooks are marked with `# TODO`.
    - Filter pill controls are rendered as `st.popover` widgets so they have
      the right look and behaviour, but their callbacks are placeholders.
    - Category tabs (Overview, Performance, Valuation, …) change which
      columns the table renders, not the underlying data.
    - Row click writes the selected symbol to `st.session_state['active_ticker']`
      and reruns the app — the rest of `ui/app.py` (which reads from the same
      session_state key) picks it up automatically and the Overview / Briefing /
      Data-Point Analyzer tabs all refresh for the new ticker with zero extra
      plumbing.

Wire-up contract with `ui/app.py`:

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
    # Defensive fallback in case the universe module is missing — keep
    # the screener working with just the seed list rather than crashing.
    US_UNIVERSE: tuple = ()  # type: ignore[no-redef]

    def is_valid_us_ticker(symbol: str) -> bool:  # type: ignore[no-redef]
        return bool(symbol) and symbol.strip().isalpha() and len(symbol.strip()) <= 5

    def normalize_ticker(symbol: str) -> str:  # type: ignore[no-redef]
        return (symbol or "").strip().upper()


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
# TODO(luke): replace with a live MarketData batch fetch — yfinance supports
# `download(["NVDA","AAPL",...], period="2d")` which gets last close + prev
# close in one round trip. Wrap in `@st.cache_data(ttl=300)`. Rough sketch:
#
#     @st.cache_data(ttl=300, show_spinner=False)
#     def _fetch_screener_frame(symbols: tuple[str, ...]) -> pd.DataFrame:
#         px = yf.download(list(symbols), period="2d", progress=False,
#                          group_by="ticker", auto_adjust=False)
#         rows = []
#         for sym in symbols:
#             try:
#                 last_two = px[sym]["Close"].dropna().tail(2)
#                 price = float(last_two.iloc[-1])
#                 prev  = float(last_two.iloc[0])
#                 rows.append({"symbol": sym, "price": price,
#                              "change_pct": (price/prev - 1) * 100})
#             except Exception:
#                 rows.append({"symbol": sym, "price": np.nan,
#                              "change_pct": np.nan})
#         return pd.DataFrame(rows)
#
# For now the table is populated from the static dict below so the UI is
# fast and deterministic during layout iteration.
_SEED_STOCKS: list[dict[str, Any]] = [
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
# Frame construction
# ======================================================================
def _enrich_with_synthetic_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in fields the seed dict doesn't carry, deterministically.

    Per-symbol seeded RNG so the synthetic numbers are stable across
    reruns — the table doesn't flicker when categories change.

    Columns are assigned via lists (not `out.at[i, col] = …`) so pandas
    infers dtype from the values rather than coercing into a pre-existing
    float64 NaN-filled column. That matters for `ex_div_date`, which is a
    string — newer pandas raises `LossySetitemError` if we try to write
    strings into a float column.

    TODO(luke): replace these synthetic fills with real values when the
    live MarketData / fundamentals fetch lands. The keys here are exactly
    the columns referenced by the non-Overview category tabs.
    """
    out = df.copy()
    rng_seed_base = 17

    def _rng_for(sym: str) -> np.random.Generator:
        return np.random.default_rng(rng_seed_base + (hash(sym) & 0xFFFF))

    def _gen(col_default: dict[str, Callable[[np.random.Generator, dict[str, Any]], Any]]) -> None:
        for col, fn in col_default.items():
            existing = out[col] if col in out.columns else None
            values: list[Any] = []
            for i, row in out.iterrows():
                current = existing.iloc[i] if existing is not None else np.nan
                if pd.notna(current):
                    values.append(current)
                else:
                    values.append(fn(_rng_for(str(row["symbol"])), row.to_dict()))
            # Assign as a fresh column — dtype is inferred from values,
            # so string columns stay object-typed and float columns float.
            out[col] = values

    _gen({
        # Performance (anchored on change_pct so the day move stays consistent)
        "perf_1w":          lambda r, _: float(r.normal(2.0, 4.5)),
        "perf_1m":          lambda r, _: float(r.normal(4.0, 8.0)),
        "perf_3m":          lambda r, _: float(r.normal(8.0, 14.0)),
        "perf_6m":          lambda r, _: float(r.normal(12.0, 20.0)),
        "perf_ytd":         lambda r, _: float(r.normal(18.0, 25.0)),
        "perf_1y":          lambda r, _: float(r.normal(22.0, 30.0)),
        "volatility_1m":    lambda r, _: float(abs(r.normal(28.0, 12.0))),
        # Extended hours
        "premarket_close":   lambda r, row: float(row.get("price") or 0) * (1 + r.normal(0, 0.005)),
        "premarket_chg_pct": lambda r, _: float(r.normal(0, 0.4)),
        "premarket_vol":     lambda r, _: float(abs(r.normal(2_000_000, 1_500_000))),
        "postmarket_close":  lambda r, row: float(row.get("price") or 0) * (1 + r.normal(0, 0.004)),
        "postmarket_chg_pct":lambda r, _: float(r.normal(0, 0.3)),
        "postmarket_vol":    lambda r, _: float(abs(r.normal(1_200_000, 900_000))),
        # Valuation
        "peg": lambda r, row: float(row.get("pe") or 25) / max(abs(float(row.get("eps_dil_growth") or 10)), 1),
        "ps":  lambda r, _: float(abs(r.normal(8.0, 6.0))),
        "pb":  lambda r, _: float(abs(r.normal(6.0, 5.0))),
        "p_fcf": lambda r, _: float(abs(r.normal(35.0, 25.0))),
        "ev_ebitda": lambda r, _: float(abs(r.normal(22.0, 12.0))),
        # Dividends
        "div_per_share": lambda r, row: float(row.get("price") or 100) * float(row.get("div_yield") or 0) / 100,
        "div_payout":    lambda r, _: float(abs(r.normal(35.0, 20.0))),
        "div_growth_5y": lambda r, _: float(r.normal(8.0, 6.0)),
        "ex_div_date":   lambda r, _: f"2026-{int(r.integers(1,13)):02d}-{int(r.integers(1,28)):02d}",
        # Profitability
        "gross_margin":  lambda r, _: float(abs(r.normal(45.0, 18.0))),
        "op_margin":     lambda r, _: float(abs(r.normal(22.0, 12.0))),
        "profit_margin": lambda r, _: float(abs(r.normal(15.0, 10.0))),
        "fcf_margin":    lambda r, _: float(abs(r.normal(18.0, 10.0))),
        "roa":           lambda r, _: float(abs(r.normal(10.0, 6.0))),
        "roe":           lambda r, _: float(abs(r.normal(20.0, 12.0))),
        "roic":           lambda r, _: float(abs(r.normal(15.0, 9.0))),
        # Income statement (scale to market cap so numbers feel plausible)
        "revenue":        lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.20, 0.10))),
        "revenue_growth": lambda r, _: float(r.normal(8.0, 12.0)),
        "gross_profit":   lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.10, 0.05))),
        "operating_income": lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.05, 0.03))),
        "net_income":     lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.04, 0.02))),
        "ebitda":         lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.07, 0.04))),
        # Balance sheet
        "total_assets":   lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(1.2, 0.5))),
        "total_debt":     lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.25, 0.15))),
        "debt_to_equity": lambda r, _: float(abs(r.normal(0.6, 0.4))),
        "current_ratio":  lambda r, _: float(abs(r.normal(1.5, 0.6))),
        "cash_per_share": lambda r, row: float(row.get("price") or 100) * float(abs(r.normal(0.08, 0.05))),
        "book_per_share": lambda r, row: float(row.get("price") or 100) * float(abs(r.normal(0.20, 0.10))),
        # Cash flow
        "op_cash_flow":   lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.06, 0.03))),
        "capex":          lambda r, row: -float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.02, 0.012))),
        "fcf":            lambda r, row: float(row.get("market_cap") or 1e10) * float(abs(r.normal(0.04, 0.025))),
        "fcf_per_share":  lambda r, row: float(row.get("price") or 100) * float(abs(r.normal(0.04, 0.02))),
        "fcf_yield":      lambda r, _: float(abs(r.normal(3.5, 2.0))),
        "sales_per_share":lambda r, row: float(row.get("price") or 100) * float(abs(r.normal(0.30, 0.15))),
        # Technicals
        "rsi_14":         lambda r, _: float(np.clip(r.normal(55, 15), 5, 95)),
        "ma_50":          lambda r, row: float(row.get("price") or 100) * (1 + float(r.normal(-0.02, 0.04))),
        "ma_200":         lambda r, row: float(row.get("price") or 100) * (1 + float(r.normal(-0.06, 0.08))),
        "beta":           lambda r, _: float(abs(r.normal(1.1, 0.4))),
        "atr_14":         lambda r, row: float(row.get("price") or 100) * 0.025,
    })
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _build_screener_frame() -> pd.DataFrame:
    """Build the canonical screener DataFrame (all columns, all symbols).

    Cached so synthetic fills + per-symbol RNG seeds stay stable across
    reruns. Cleared automatically every 5min and on Streamlit's "Refresh
    data" button (which calls `st.cache_data.clear()`).

    Universe expansion (Issue #4 fix)
    ---------------------------------
    Previously this used only the 22 hand-coded ``_SEED_STOCKS``. The
    screener now seeds from the full ``US_UNIVERSE`` (~560 names: S&P 500
    + extra large/mid caps), keeps the hand-curated values for the 22
    popular tickers, and synthesizes plausible price/market_cap for the
    rest. Any additional symbols the user types into the search box are
    appended on top of this base.

    Synthetic values are deliberately rough — once the live MarketData
    batch fetch lands, the real price/cap fields will overwrite these.
    """
    base = _build_full_universe_seed()
    df = pd.DataFrame(base)
    df = _enrich_with_synthetic_fields(df)
    # Default sort: market cap desc, matching TradingView's default.
    df = df.sort_values("market_cap", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _build_full_universe_seed() -> list[dict[str, Any]]:
    """Merge the 22 hand-curated seed stocks with the full US universe.

    Returns a list of dicts shaped for ``_enrich_with_synthetic_fields``,
    one per ticker in ``US_UNIVERSE`` plus any user-added tickers stored
    in ``st.session_state["custom_tickers"]``. The 22 tickers in
    ``_SEED_STOCKS`` keep their hand-curated values so the most popular
    names display the right price/cap; the other ~538 get synthesized.
    """
    # Start with the curated rows (real-ish values for the popular 22).
    by_symbol: dict[str, dict[str, Any]] = {row["symbol"]: dict(row) for row in _SEED_STOCKS}

    # Per-ticker deterministic RNG so synthesized rows don't reshuffle
    # between reruns. Seeded off the symbol so AAPL always gets the same
    # synthetic numbers.
    def _rng_for(sym: str) -> np.random.Generator:
        return np.random.default_rng(31 + (hash(sym) & 0xFFFF))

    # Fill in the rest of the US universe.
    for sym in US_UNIVERSE:
        if sym in by_symbol:
            continue
        rng = _rng_for(sym)
        # Plausible large-cap defaults; the table looks reasonable even
        # before live data arrives. Numbers below are intentionally mid-
        # range so no row sticks out as a synthetic artifact.
        market_cap = float(abs(rng.normal(50e9, 80e9))) + 5e9
        price = float(abs(rng.normal(120, 80))) + 5
        by_symbol[sym] = {
            "symbol": sym,
            "name": sym,                  # display name; real name fills via live fetch
            "price": price,
            "change_pct": float(rng.normal(0, 1.8)),
            "volume": float(abs(rng.normal(8_000_000, 6_000_000))) + 100_000,
            "rel_volume": float(abs(rng.normal(1.0, 0.5))),
            "market_cap": market_cap,
            "pe": float(abs(rng.normal(22, 12))),
            "eps_dil": float(rng.normal(5, 4)),
            "eps_dil_growth": float(rng.normal(8, 18)),
            "div_yield": max(0.0, float(rng.normal(1.2, 1.0))),
            "sector": "Other",            # real sector fills via live fetch
            "analyst_rating": "Hold",
        }

    # Append any user-added custom tickers from session state.
    custom = []
    try:
        custom = st.session_state.get("custom_tickers", []) or []
    except Exception:
        # Outside Streamlit context (e.g. unit tests) — fine, just skip.
        pass

    for sym in custom:
        if sym in by_symbol:
            continue
        rng = _rng_for(sym)
        by_symbol[sym] = {
            "symbol": sym,
            "name": sym,
            "price": float(abs(rng.normal(80, 50))) + 5,
            "change_pct": float(rng.normal(0, 1.8)),
            "volume": float(abs(rng.normal(2_000_000, 2_000_000))) + 50_000,
            "rel_volume": 1.0,
            "market_cap": float(abs(rng.normal(10e9, 15e9))) + 1e8,
            "pe": float(abs(rng.normal(20, 10))),
            "eps_dil": float(rng.normal(4, 3)),
            "eps_dil_growth": float(rng.normal(5, 15)),
            "div_yield": 0.0,
            "sector": "Other",
            "analyst_rating": "Neutral",
        }

    return list(by_symbol.values())


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


# Each entry: (label, popover_render_fn) where the render fn emits the
# inputs inside the popover. These are PLACEHOLDERS — they don't filter
# the underlying frame yet. Wire them up one at a time as needed.
def _filter_watchlist() -> None:
    st.caption("Watchlist filter — placeholder.")
    st.selectbox("Select watchlist", ["All stocks", "My core picks", "AI semiconductors"], key="_flt_watchlist")


def _filter_index() -> None:
    st.caption("Filter by index membership.")
    st.multiselect("Index", ["S&P 500", "Nasdaq 100", "Dow 30", "Russell 1000"], key="_flt_index")


def _filter_price() -> None:
    st.caption("Price range (USD).")
    a, b = st.columns(2)
    a.number_input("Min", value=0.0, key="_flt_price_min")
    b.number_input("Max", value=10_000.0, key="_flt_price_max")


def _filter_change() -> None:
    st.caption("1-day % change range.")
    st.slider("Change %", -20.0, 20.0, (-20.0, 20.0), step=0.1, key="_flt_change")


def _filter_marketcap() -> None:
    st.caption("Market capitalization.")
    st.multiselect(
        "Bucket",
        ["Mega ≥ $200B", "Large $10–200B", "Mid $2–10B", "Small $300M–2B", "Micro < $300M"],
        key="_flt_mcap",
    )


def _filter_pe() -> None:
    st.caption("Price/Earnings ratio.")
    st.slider("P/E", 0.0, 200.0, (0.0, 200.0), key="_flt_pe")


def _filter_eps_growth() -> None:
    st.caption("Diluted EPS YoY growth.")
    st.slider("EPS dil growth %", -100.0, 500.0, (-100.0, 500.0), key="_flt_epsg")


def _filter_div_yield() -> None:
    st.caption("Forward dividend yield.")
    st.slider("Div yield %", 0.0, 15.0, (0.0, 15.0), key="_flt_dy")


def _filter_sector() -> None:
    st.caption("Sector / industry.")
    sectors = sorted({s for s in pd.DataFrame(_SEED_STOCKS)["sector"].unique() if s})
    st.multiselect("Sector", sectors, key="_flt_sector")


def _filter_analyst() -> None:
    st.caption("Aggregate analyst rating.")
    st.multiselect("Analyst Rating", ["Strong buy", "Buy", "Neutral", "Sell", "Strong sell"], key="_flt_rating")


def _filter_perf() -> None:
    st.caption("Period performance.")
    st.selectbox("Window", ["1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"], key="_flt_perf_window")
    st.slider("Range %", -100.0, 1000.0, (-100.0, 1000.0), key="_flt_perf_range")


def _filter_revenue_growth() -> None:
    st.caption("Trailing revenue growth YoY.")
    st.slider("Revenue growth %", -100.0, 500.0, (-100.0, 500.0), key="_flt_revg")


def _filter_peg() -> None:
    st.caption("PEG ratio (lower is cheaper relative to growth).")
    st.slider("PEG", 0.0, 10.0, (0.0, 10.0), key="_flt_peg")


def _filter_roe() -> None:
    st.caption("Return on Equity.")
    st.slider("ROE %", -50.0, 200.0, (-50.0, 200.0), key="_flt_roe")


def _filter_beta() -> None:
    st.caption("Levered beta.")
    st.slider("Beta", 0.0, 5.0, (0.0, 5.0), key="_flt_beta")


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


def _render_filter_bar() -> None:
    """Render the row of filter pill chips. Placeholder logic for now."""
    n_per_row = 8  # Streamlit columns wrap awkwardly past ~8 pill buttons.
    for chunk_start in range(0, len(_FILTERS), n_per_row):
        chunk = _FILTERS[chunk_start : chunk_start + n_per_row]
        cols = st.columns(len(chunk))
        for col, (label, render_fn) in zip(cols, chunk):
            with col:
                with st.popover(label, use_container_width=True):
                    render_fn()


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


# ======================================================================
# Public entrypoint
# ======================================================================
def render_screener_tab() -> None:
    """Render the entire Screener tab. Safe to call inside a `with tab_x:`."""
    st.markdown(_SCREENER_CSS, unsafe_allow_html=True)
    _render_header()
    _render_search_box()  # Issue #4: free-text ticker search
    _render_filter_bar()
    st.markdown("")  # spacer
    category = _render_category_tabs()
    df = _build_screener_frame()
    _render_results_table(df, category)


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