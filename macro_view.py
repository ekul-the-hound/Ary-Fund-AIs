"""
ui/macro_view.py
================

The "Macro" destination: a dashboard over the OpenBB-backed data adapter
(``data/openbb_provider.py``). Four sections:

    * Rates        — US treasury yield curve (latest) + rate history
    * Growth       — real GDP, CPI, unemployment
    * Commodities  — spot price history for a selectable commodity
    * FX           — daily history for a user-entered pair

Design constraints honored (same contract as the other ui/ views):
    * PURE UI: no SQLite writes, no Ollama. Reads only via the adapter.
    * DEFENSIVE: the adapter already returns empty DataFrames on any
      failure; every section here renders a neutral caption for empty
      data and never raises. A missing adapter (not yet copied into
      data/) degrades to a single explanatory message.
    * CACHED: every fetch is wrapped in ``st.cache_data`` (1h TTL) so
      Streamlit reruns don't refetch. First ever call in a session also
      pays OpenBB's one-time interface build (a few seconds) — the
      spinner says so, so it doesn't look like a hang.
    * THEME-AWARE: chart colors come from components.THEME, with
      fallbacks so this module works even if the theme patch is absent.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st

# Dual-import fallback (project convention).
try:
    from ui import components as C
except Exception:  # noqa: BLE001
    import components as C  # type: ignore

try:
    from data import openbb_provider as obp
except Exception:  # noqa: BLE001
    try:
        import openbb_provider as obp  # type: ignore
    except Exception:  # noqa: BLE001
        obp = None  # rendered as a friendly message, never a crash

logger = logging.getLogger("ary_quant.ui.macro_view")

_THEME: dict = getattr(C, "THEME", {}) or {}
_ACCENT = _THEME.get("accent", "#2962FF")
_GOOD = _THEME.get("good", "#22C55E")
_BAD = _THEME.get("bad", "#EF4444")
_DIM = _THEME.get("text_dim", "#8A919E")
_GRID = _THEME.get("border", "#232936")

_COMMODITIES = ["wti", "brent", "gold", "natural_gas"]


# ======================================================================
# Cached fetch layer — thin st.cache_data wrappers over the adapter.
# Module-level functions (not lambdas) so Streamlit can hash them.
# ======================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _yield_curve() -> pd.DataFrame:
    return obp.yield_curve() if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _treasury_rates(start: str) -> pd.DataFrame:
    return obp.treasury_rates(start_date=start) if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _gdp(start: str) -> pd.DataFrame:
    return obp.gdp_real(start_date=start) if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _cpi(start: str) -> pd.DataFrame:
    return obp.cpi(start_date=start) if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _unemployment(start: str) -> pd.DataFrame:
    return obp.unemployment(start_date=start) if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _commodity(name: str, start: str) -> pd.DataFrame:
    return obp.commodity_spot(name, start_date=start) if obp else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _fx(pair: str, start: str) -> pd.DataFrame:
    return obp.fx_historical(pair, start_date=start) if obp else pd.DataFrame()


# ======================================================================
# Chart helpers — defensive against unknown/drifting column shapes.
# ======================================================================
def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ("date", "period", "as_of", "timestamp"):
        if cand in df.columns:
            return cand
    return None


def _line_chart(df: pd.DataFrame, title: str, *,
                y_cols: Optional[list[str]] = None,
                color: str = _ACCENT) -> None:
    """Render a themed plotly line chart; fall back to st.dataframe if the
    shape can't be charted. Never raises."""
    try:
        import plotly.graph_objects as go

        d = df.copy()
        date_col = _find_date_col(d)
        if date_col is not None:
            d = d.set_index(date_col)
        num = d.select_dtypes("number")
        if y_cols:
            num = num[[c for c in y_cols if c in num.columns]]
        if num.empty:
            st.dataframe(df, use_container_width=True, height=240)
            return

        fig = go.Figure()
        palette = [_ACCENT, _GOOD, _BAD, _DIM, "#EAB308", "#8B5CF6",
                   "#14B8A6", "#F97316", "#64748B", "#E879F9", "#38BDF8"]
        for i, col in enumerate(num.columns):
            fig.add_trace(go.Scatter(
                x=num.index, y=num[col], name=str(col), mode="lines",
                line=dict(width=1.6, color=palette[i % len(palette)])))
        fig.update_layout(
            title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=_DIM, size=11),
            xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID),
            legend=dict(orientation="h", y=-0.2),
            showlegend=len(num.columns) > 1,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("chart fallback for %s: %s", title, e)
        st.dataframe(df, use_container_width=True, height=240)


def _empty_note(what: str, hint: str) -> None:
    st.caption(f"No {what} data — {hint}")


# ======================================================================
# Sections
# ======================================================================
def _section_rates() -> None:
    st.markdown("#### Rates")
    left, right = st.columns([1, 2])

    with left:
        yc = _yield_curve()
        if yc.empty:
            _empty_note("yield curve", "Federal Reserve source unreachable.")
        else:
            _line_chart(yc, "US yield curve (latest)")
            st.caption("Feed for quant/yield_curve_3d.py lives in "
                       "data.openbb_provider.yield_curve().")

    with right:
        tr = _treasury_rates("2024-01-01")
        if tr.empty:
            _empty_note("treasury rate", "Federal Reserve source unreachable.")
        else:
            _line_chart(tr, "Treasury rates by tenor (daily)")


def _section_growth() -> None:
    st.markdown("#### Growth & inflation")
    c1, c2, c3 = st.columns(3)
    with c1:
        df = _gdp("2019-01-01")
        _line_chart(df, "Real GDP (OECD)") if not df.empty else \
            _empty_note("GDP", "OECD source unreachable.")
    with c2:
        df = _cpi("2022-01-01")
        _line_chart(df, "CPI (FRED)") if not df.empty else \
            _empty_note("CPI", "check FRED_API_KEY in .env.")
    with c3:
        df = _unemployment("2022-01-01")
        _line_chart(df, "Unemployment (OECD)") if not df.empty else \
            _empty_note("unemployment", "OECD source unreachable.")


def _section_commodities() -> None:
    st.markdown("#### Commodities")
    name = st.selectbox("Commodity", _COMMODITIES, index=0,
                        key="macro_commodity")
    df = _commodity(name, "2024-01-01")
    if df.empty:
        _empty_note(f"{name} spot", "check FRED_API_KEY in .env.")
    else:
        _line_chart(df, f"{name.upper()} spot (FRED)")


def _section_fx() -> None:
    st.markdown("#### FX")
    pair = st.text_input("Pair", value="EURUSD", key="macro_fx_pair",
                         help="e.g. EURUSD, USDJPY, GBPUSD").strip().upper()
    if not pair:
        return
    df = _fx(pair, "2025-01-01")
    if df.empty:
        _empty_note(f"{pair}", "unknown pair or yfinance unreachable.")
    else:
        _line_chart(df, f"{pair} daily", y_cols=["close"])


# ======================================================================
# Entry point (called by app_v2's dispatcher)
# ======================================================================
def render_macro() -> None:
    """Render the Macro destination. Safe to call unconditionally."""
    st.markdown("### Macro")

    if obp is None:
        st.warning("data/openbb_provider.py is not importable — copy it "
                   "into data/ and install openbb (see the adapter's "
                   "docstring), then reload.")
        return

    st.caption("OpenBB-backed macro data · cached 1h · first load in a "
               "session builds the OpenBB interface (a few seconds).")

    with st.spinner("Loading macro data…"):
        _section_rates()
        _section_growth()
        _section_commodities()
        _section_fx()


__all__ = ["render_macro"]

# D:\Ary Fund\ui\macro_view.py
