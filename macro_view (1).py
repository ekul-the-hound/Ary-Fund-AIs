"""
ui/macro_view.py
================

The **Macro** destination for ARY QUANT's top nav (app_v2.py dispatches
``macro_view.render_macro()`` when the Macro tab is selected).

Four sections, all fed by the OpenBB adapter (``data/openbb_provider.py``):

* **Rates** — latest Treasury yield curve + historical yields by tenor.
  (The same curve data is a natural feed for ``quant/yield_curve_3d.py``.)
* **Growth & inflation** — real GDP, CPI YoY, unemployment side by side.
* **Commodities** — WTI / Brent / gold / natural-gas spot selector.
* **FX** — daily history for any typed pair (EURUSD, USDJPY, ...).

Defensive contract (matches the rest of the UI layer):

* Every provider call is wrapped in ``st.cache_data`` (1-hour TTL) so
  Streamlit reruns never refetch.
* Every section degrades to a caption if its data source is empty or
  unreachable — nothing on this tab can crash the app.
* OpenBB's column shapes drift between versions, so each chart builder
  is wrapped: if shaping fails, the raw DataFrame is shown instead.
* THEME tokens come from ``ui.components``; a local copy of the same
  OpenBB-style palette is the fallback so this module renders even if
  components.py predates the theme patch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Path bootstrap — same convention as app_v2.py, so `data.*` imports
# resolve no matter how Streamlit was launched.
# ----------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ----------------------------------------------------------------------
# THEME — prefer the single source of truth in ui/components.py;
# fall back to an identical local copy (keys mirror fix_theme_openbb.py).
# ----------------------------------------------------------------------
_FALLBACK_THEME: dict[str, str] = {
    "bg":       "#0B0E14",
    "surface":  "#131722",
    "raised":   "#1A1F2B",
    "border":   "#232936",
    "text":     "#E6E8EB",
    "text_dim": "#8A919E",
    "accent":   "#2962FF",
    "good":     "#22C55E",
    "warn":     "#EAB308",
    "bad":      "#EF4444",
    "severe":   "#B91C1C",
    "muted":    "#7C8591",
    "card_bg":  "rgba(148,163,184,0.05)",
}

try:
    from ui.components import THEME as _THEME  # type: ignore
except Exception:  # noqa: BLE001 — pre-theme components.py or import issue
    _THEME = _FALLBACK_THEME


def _t(key: str) -> str:
    """Theme token with fallback — never KeyErrors on a stale THEME."""
    return _THEME.get(key, _FALLBACK_THEME.get(key, "#8A919E"))


# ----------------------------------------------------------------------
# OpenBB adapter — guarded import. If it's missing, the tab explains
# itself instead of taking the app down.
# ----------------------------------------------------------------------
try:
    from data import openbb_provider as obb  # type: ignore
except Exception:  # noqa: BLE001
    obb = None  # type: ignore[assignment]


# ----------------------------------------------------------------------
# Cached fetchers — 1-hour TTL. Underscore-free args only (hashable).
# show_spinner=False: sections manage their own spinners so the first
# OpenBB import (which builds its interface, a few seconds) is labeled.
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yield_curve() -> pd.DataFrame:
    return obb.yield_curve() if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_treasury_rates(start_date: str) -> pd.DataFrame:
    return obb.treasury_rates(start_date=start_date) if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_gdp(start_date: str) -> pd.DataFrame:
    return obb.gdp_real(start_date=start_date) if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cpi(start_date: str) -> pd.DataFrame:
    return obb.cpi(start_date=start_date) if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_unemployment(start_date: str) -> pd.DataFrame:
    return obb.unemployment(start_date=start_date) if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_commodity(commodity: str, start_date: str) -> pd.DataFrame:
    return obb.commodity_spot(commodity, start_date=start_date) if obb is not None else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fx(pair: str, start_date: str) -> pd.DataFrame:
    return obb.fx_historical(pair, start_date=start_date) if obb is not None else pd.DataFrame()


# ----------------------------------------------------------------------
# DataFrame shaping helpers — OpenBB column names drift by provider and
# version, so nothing below assumes an exact schema.
# ----------------------------------------------------------------------
def _with_date_column(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    """Return (df, date_col). Promotes a DatetimeIndex; sniffs common names."""
    if df.empty:
        return df, None
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.reset_index()
        out.rename(columns={out.columns[0]: "date"}, inplace=True)
        return out, "date"
    for cand in ("date", "Date", "as_of", "period", "index"):
        if cand in df.columns:
            return df, cand
    # last resort: any column that parses as datetime
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return df, col
    return df, None


def _numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> list[str]:
    skip = set(exclude)
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


_CHART_SERIES_COLORS = ("accent", "good", "warn", "bad", "muted", "severe")


def _themed_layout(fig, title: str):
    """Apply the ARY QUANT dark theme to a Plotly figure in place."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_t("text"))),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_t("surface"),
        font=dict(color=_t("text_dim"), size=11),
        margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=_t("border"), zerolinecolor=_t("border"))
    fig.update_yaxes(gridcolor=_t("border"), zerolinecolor=_t("border"))
    return fig


def _line_chart(df: pd.DataFrame, date_col: str, series: list[str],
                title: str) -> None:
    """Themed multi-series line chart; falls back to a raw table."""
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, col in enumerate(series):
            fig.add_trace(go.Scatter(
                x=df[date_col], y=df[col], mode="lines", name=str(col),
                line=dict(width=1.6,
                          color=_t(_CHART_SERIES_COLORS[i % len(_CHART_SERIES_COLORS)])),
            ))
        _themed_layout(fig, title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:  # noqa: BLE001 — shape drift: show the data anyway
        st.caption(f"{title} — chart unavailable, showing raw data.")
        st.dataframe(df, use_container_width=True, height=240)


def _section_empty(name: str, hint: str = "") -> None:
    msg = f"{name}: no data returned."
    if hint:
        msg += f" {hint}"
    st.caption(msg)


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
def _render_rates(start_date: str) -> None:
    st.subheader("Rates")

    # --- latest yield curve --------------------------------------------
    with st.spinner("Loading yield curve (first OpenBB call in a session "
                    "takes a few seconds while it builds its interface)…"):
        curve = _fetch_yield_curve()

    if curve.empty:
        _section_empty("Yield curve", "Treasury endpoints are keyless — "
                       "check that openbb is installed and network is up.")
    else:
        try:
            df = curve.copy()
            # Common shapes: (maturity, rate) columns, or one row of
            # tenor-named numeric columns.
            if {"maturity", "rate"}.issubset(df.columns):
                x, y = df["maturity"], df["rate"]
            else:
                num = _numeric_columns(df)
                if len(df) == 1 and num:          # single wide row
                    x, y = num, df.iloc[0][num].values
                elif len(num) == 1:               # tenor index + one column
                    x, y = df.index.astype(str), df[num[0]]
                else:
                    raise ValueError("unrecognized curve shape")
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(
                x=list(x), y=list(y), mode="lines+markers",
                line=dict(width=2, color=_t("accent")),
                marker=dict(size=6, color=_t("accent")),
                name="latest curve",
            ))
            _themed_layout(fig, "Latest Treasury yield curve")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:  # noqa: BLE001
            st.caption("Yield curve — chart unavailable, showing raw data.")
            st.dataframe(curve, use_container_width=True, height=240)
        st.caption("This is the same curve that can feed "
                   "`quant/yield_curve_3d.py`'s time × maturity surface.")

    # --- treasury history by tenor -------------------------------------
    hist = _fetch_treasury_rates(start_date)
    if hist.empty:
        _section_empty("Treasury history")
        return
    hist, date_col = _with_date_column(hist)
    tenors = _numeric_columns(hist, exclude=(date_col,) if date_col else ())
    if not date_col or not tenors:
        st.dataframe(hist, use_container_width=True, height=240)
        return
    preferred = [c for c in tenors
                 if any(k in str(c).lower()
                        for k in ("2", "10", "30", "year_2", "year_10"))]
    default = (preferred or tenors)[:3]
    picked = st.multiselect("Tenors", tenors, default=default,
                            key="macro_tenors")
    if picked:
        _line_chart(hist, date_col, picked, "Treasury yields by tenor")


def _render_growth_inflation(start_date: str) -> None:
    st.subheader("Growth & inflation")
    cols = st.columns(3)
    panels = (
        ("Real GDP", _fetch_gdp, cols[0]),
        ("CPI", _fetch_cpi, cols[1]),
        ("Unemployment", _fetch_unemployment, cols[2]),
    )
    for name, fetch, col in panels:
        with col:
            df = fetch(start_date)
            if df.empty:
                _section_empty(name)
                continue
            df, date_col = _with_date_column(df)
            series = _numeric_columns(df, exclude=(date_col,) if date_col else ())
            if not date_col or not series:
                st.dataframe(df, use_container_width=True, height=220)
                continue
            _line_chart(df, date_col, series[:1], name)


def _render_commodities(start_date: str) -> None:
    st.subheader("Commodities")
    label_to_arg = {
        "WTI crude": "wti",
        "Brent crude": "brent",
        "Gold": "gold",
        "Natural gas": "natural_gas",
    }
    choice = st.selectbox("Commodity", list(label_to_arg), index=0,
                          key="macro_commodity")
    df = _fetch_commodity(label_to_arg[choice], start_date)
    if df.empty:
        _section_empty(choice, "Commodity spot rides FRED_API_KEY — "
                       "check it's set in .env.")
        return
    df, date_col = _with_date_column(df)
    series = _numeric_columns(df, exclude=(date_col,) if date_col else ())
    if not date_col or not series:
        st.dataframe(df, use_container_width=True, height=240)
        return
    _line_chart(df, date_col, series[:1], f"{choice} spot")


def _render_fx(start_date: str) -> None:
    st.subheader("FX")
    pair = st.text_input("Pair (e.g. EURUSD, USDJPY, GBPUSD)", "EURUSD",
                         key="macro_fx_pair").strip().upper()
    if not pair:
        return
    df = _fetch_fx(pair, start_date)
    if df.empty:
        _section_empty(f"FX {pair}", "Unknown pair, or yfinance unreachable.")
        return
    df, date_col = _with_date_column(df)
    # Prefer the close; otherwise first numeric column.
    close = next((c for c in df.columns if str(c).lower() == "close"), None)
    series = [close] if close else _numeric_columns(
        df, exclude=(date_col,) if date_col else ())[:1]
    if not date_col or not series:
        st.dataframe(df, use_container_width=True, height=240)
        return
    _line_chart(df, date_col, series, f"{pair} daily")


# ----------------------------------------------------------------------
# Entry point — called by app_v2.py's destination dispatch.
# ----------------------------------------------------------------------
def render_macro() -> None:
    """Render the full Macro destination."""
    if obb is None:
        st.warning(
            "OpenBB adapter not importable — is `data/openbb_provider.py` "
            "in place (and `pip install openbb -c constraints_openbb.txt` "
            "done in this venv)?"
        )
        return

    # One shared lookback control for the whole tab.
    lookback = st.select_slider(
        "History", options=["6M", "1Y", "2Y", "5Y", "10Y"], value="2Y",
        key="macro_lookback",
    )
    months = {"6M": 6, "1Y": 12, "2Y": 24, "5Y": 60, "10Y": 120}[lookback]
    start_date = (pd.Timestamp.today() -
                  pd.DateOffset(months=months)).strftime("%Y-%m-%d")

    _render_rates(start_date)
    st.divider()
    _render_growth_inflation(start_date)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _render_commodities(start_date)
    with c2:
        _render_fx(start_date)


__all__ = ["render_macro"]

# D:\Ary Fund\ui\macro_view.py
