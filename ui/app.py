"""Streamlit dashboard for the hedge-fund AI research system.

Sections:
    - Sidebar: ticker picker, lookback window, chart overlays, refresh.
    - Portfolio overview cards.
    - Selected ticker summary (price, change, thesis outlook, risk).
    - Price chart with optional MA / RSI / volatility overlays.
    - Risk & thesis panel (two columns).
    - Macro context panel.
    - Debug: raw agent context JSON.

Run with:
    streamlit run ui/app.py

The UI calls into backend modules defensively — missing modules or mock
agent output will not crash the app; fallback placeholders are shown
instead.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# Path setup — let this file be run directly with `streamlit run ui/app.py`
# by making the repo root importable.
# ----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ----------------------------------------------------------------------
# Backend imports — wrapped so a missing module shows a banner rather
# than breaking the whole dashboard.
# ----------------------------------------------------------------------
_BACKEND_ERRORS: list[str] = []

try:
    from data import pipeline as data_pipeline
except Exception as e:  # pragma: no cover - defensive UI boundary
    data_pipeline = None
    _BACKEND_ERRORS.append(f"data.pipeline unavailable: {e}")

try:
    from data import portfolio_db
except Exception as e:
    portfolio_db = None
    _BACKEND_ERRORS.append(f"data.portfolio_db unavailable: {e}")

try:
    from data import market_data
except Exception as e:
    market_data = None
    _BACKEND_ERRORS.append(f"data.market_data unavailable: {e}")

try:
    from data import macro_data
except Exception as e:
    macro_data = None
    _BACKEND_ERRORS.append(f"data.macro_data unavailable: {e}")

try:
    from quant import volatility as quant_vol
except Exception as e:
    quant_vol = None
    _BACKEND_ERRORS.append(f"quant.volatility unavailable: {e}")

try:
    import config as app_config
except Exception as e:
    app_config = None
    _BACKEND_ERRORS.append(f"config unavailable: {e}")

try:
    from data.market_data import MarketData
except Exception as e:
    MarketData = None
    _BACKEND_ERRORS.append(f"data.market_data.MarketData unavailable: {e}")

try:
    from data.macro_data import MacroData
except Exception as e:
    MacroData = None
    _BACKEND_ERRORS.append(f"data.macro_data.MacroData unavailable: {e}")

try:
    # Imported to trigger on-demand agent generation from the UI.
    # main._process_ticker() runs the full chain and persists to agent_opinions.
    import main as agent_main
except Exception as e:
    agent_main = None
    _BACKEND_ERRORS.append(f"main (agent chain) unavailable: {e}")

# Local UI modules — chat + essay. Import after backend so any failure here
# is still recoverable (the Overview tab works without chat/essay).
try:
    from ui import chat as ary_chat
    from ui import essay as ary_essay
except Exception as e:
    ary_chat = None
    ary_essay = None
    _BACKEND_ERRORS.append(f"ARY QUANT modules unavailable: {e}")

# Screener module — fully self-contained TradingView-style stock screener.
# Imports independently from chat/essay so a failure here doesn't take the
# rest of the dashboard down.
try:
    from ui import screener as ary_screener
except Exception as e:
    ary_screener = None
    _BACKEND_ERRORS.append(f"ARY QUANT screener unavailable: {e}")


# ======================================================================
# Page configuration
# ======================================================================
st.set_page_config(
    page_title="ARY QUANT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# Styling helpers
# ======================================================================
RISK_COLORS = {
    "low": "#16a34a",      # green-600
    "medium": "#ca8a04",   # yellow-600
    "moderate": "#ca8a04",
    "high": "#dc2626",     # red-600
    "severe": "#991b1b",   # red-800
    "unknown": "#6b7280",  # gray-500
}

OUTLOOK_COLORS = {
    "bullish": "#16a34a",
    "neutral": "#6b7280",
    "bearish": "#dc2626",
    "unknown": "#6b7280",
}


def _risk_badge(level: str) -> str:
    """Return an HTML span styled as a colored risk pill."""
    key = (level or "unknown").strip().lower()
    color = RISK_COLORS.get(key, RISK_COLORS["unknown"])
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.85em;"
        f"font-weight:600;text-transform:uppercase;'>{key}</span>"
    )


def _outlook_badge(direction: str) -> str:
    key = (direction or "unknown").strip().lower()
    color = OUTLOOK_COLORS.get(key, OUTLOOK_COLORS["unknown"])
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.85em;"
        f"font-weight:600;text-transform:uppercase;'>{key}</span>"
    )


# ======================================================================
# Backend-facing loaders (cached)
# ======================================================================
def _db_path() -> str | None:
    """Resolve the portfolio DB path from config, or None if unavailable."""
    if app_config is None:
        return None
    return getattr(app_config, "PORTFOLIO_DB_PATH", None)


@st.cache_data(ttl=300, show_spinner=False)
def load_portfolio_summary() -> dict[str, Any]:
    """Return a portfolio summary dict from PortfolioDB.

    Uses the class-based API:
        - get_portfolio_snapshot(market_data=...) for positions and totals
        - get_risk_metrics(market_data=...) for concentration-based risk level
        - get_cash() for the cash balance

    Live prices are pulled via MarketData when available; otherwise the
    snapshot falls back to stored entry prices (no unrealized P&L in that
    case). Falls back to an empty-but-well-shaped dict on any failure so
    the UI renders placeholders instead of crashing.
    """
    empty = {
        "holdings": pd.DataFrame(),
        "num_holdings": 0,
        "total_value": 0.0,
        "total_pnl": 0.0,
        "avg_risk": "unknown",
        "cash": None,
    }

    if portfolio_db is None or not hasattr(portfolio_db, "PortfolioDB"):
        return empty

    path = _db_path()
    if not path:
        return empty

    try:
        db = portfolio_db.PortfolioDB(db_path=path)
    except Exception as e:
        st.warning(f"Could not open portfolio DB: {e}")
        return empty

    # Optional: live prices for unrealized P&L. Passing None is safe —
    # the snapshot will just use stored entry prices as "current".
    md = None
    if MarketData is not None:
        try:
            md = MarketData(db_path=path)
        except Exception:
            md = None

    try:
        snap = db.get_portfolio_snapshot(market_data=md)
    except Exception as e:
        st.warning(f"Could not build portfolio snapshot: {e}")
        return empty

    positions = snap.get("positions", []) or []
    summary = snap.get("summary", {}) or {}

    # Positions is a list-of-dicts from get_portfolio_snapshot — it already
    # includes ticker, shares, avg_entry_price, current_price, market_value,
    # unrealized_pnl, unrealized_pct, portfolio_weight, sector, conviction.
    holdings_df = pd.DataFrame(positions) if positions else pd.DataFrame()

    # Risk level comes from the concentration bucket (LOW / MODERATE / HIGH),
    # which is already compatible with RISK_COLORS.
    avg_risk = "unknown"
    try:
        risk = db.get_risk_metrics(market_data=md) or {}
        concentration = str(risk.get("concentration", "")).lower()
        if concentration in {"low", "moderate", "high"}:
            avg_risk = concentration
    except Exception:
        pass

    # Cash: prefer the dedicated getter so an empty portfolio still shows
    # the starting balance from portfolio_meta.
    cash = summary.get("cash")
    try:
        cash = float(db.get_cash())
    except Exception:
        pass

    return {
        "holdings": holdings_df,
        "num_holdings": int(summary.get("num_positions", len(holdings_df))),
        "total_value": float(summary.get("total_value", 0.0) or 0.0),
        "total_pnl": float(summary.get("unrealized_pnl", 0.0) or 0.0),
        "avg_risk": avg_risk,
        "cash": cash,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_opinion(ticker: str) -> dict[str, Any]:
    """Fetch the most recent agent opinion for this ticker from portfolio.db.

    The opinions table is written by main.py via
    ``portfolio_db.save_agent_opinion``. Each row's ``payload_json`` column
    contains the full merged opinion dict (outlook, confidence, risks,
    thesis, rationale, etc.) produced by the agent pipeline.

    Returns an empty dict if the table is empty, the ticker has no opinions
    yet, or the DB is unreachable.
    """
    path = _db_path()
    if not path:
        return {}
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM agent_opinions "
                "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if not row:
            return {}
        return json.loads(row[0]) or {}
    except sqlite3.OperationalError:
        # agent_opinions table may not exist yet if main.py has never run.
        return {}
    except Exception as e:
        st.warning(f"Could not load opinion for {ticker}: {e}")
        return {}


def generate_opinion(ticker: str) -> dict[str, Any] | None:
    """Run the full agent chain for one ticker and persist the result.

    Calls ``main._process_ticker(ticker, db_path, cfg)``, which:
        1. Builds context (SEC filings, prices, fundamentals, macro).
        2. Runs filing_analyzer to shape filings and extract key metrics.
        3. Invokes the LLM agent (Qwen3 via Ollama).
        4. Computes rule-based risk flags.
        5. Generates the thesis.
        6. Writes the merged opinion to the agent_opinions table.

    This can take 30s–2min depending on Ollama speed and whether SEC /
    yfinance / FRED data is cached. Exceptions inside _process_ticker are
    already caught there; we treat a None return as a soft failure.

    Returns the opinion dict on success, None on failure.
    """
    if agent_main is None or not hasattr(agent_main, "_process_ticker"):
        st.error("Agent chain unavailable — `main._process_ticker` not importable.")
        return None
    if app_config is None:
        st.error("Config unavailable — cannot locate DB path.")
        return None
    path = _db_path()
    if not path:
        st.error("PORTFOLIO_DB_PATH is not set in config.")
        return None

    try:
        opinion = agent_main._process_ticker(ticker, path, app_config)
    except Exception as e:
        # _process_ticker already catches internally, but belt-and-suspenders
        # in case an import-time issue escapes.
        st.error(f"Agent chain raised for {ticker}: {e}")
        return None

    if opinion is None:
        st.warning(
            f"Agent chain returned no opinion for {ticker}. "
            "Check the Streamlit console / hedgefund_ai.log for details."
        )
        return None

    # Invalidate the caches that depend on agent_opinions / context so the
    # next read picks up the freshly-written row.
    load_latest_opinion.clear()
    load_ticker_context.clear()
    return opinion


def render_generate_cta(ticker: str) -> None:
    """Render the 'Generate analysis' call-to-action button.

    Shown inline inside render_risk_panel / render_thesis_panel when no
    opinion exists yet. Uses a unique key per ticker so Streamlit doesn't
    conflate button presses across reruns.

    The button lives inside a form-less flow: when clicked, we run the
    agent synchronously under a spinner, then st.rerun() so the panels
    re-render with the freshly-persisted data.
    """
    st.info(f"No agent opinion for **{ticker}** yet.")
    st.caption(
        "Running the agent chain fetches filings, pulls macro data, "
        "and invokes the local LLM. Typical runtime: 30s–2min."
    )
    clicked = st.button(
        f"🤖 Generate analysis for {ticker}",
        key=f"gen_opinion_{ticker}",
        type="primary",
        use_container_width=True,
    )
    if clicked:
        with st.spinner(f"Running agent chain for {ticker}… (this takes ~1min)"):
            opinion = generate_opinion(ticker)
        if opinion is not None:
            st.success(f"Opinion generated for {ticker}.")
            st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def load_ticker_context(ticker: str, lookback_days: int) -> dict[str, Any]:
    """Build the full per-ticker context for the dashboard.

    Merges two sources:
        1. pipeline.build_agent_context(ticker, db_path, cfg) — raw data.
        2. The latest row from agent_opinions — LLM-derived thesis & risk.
    """
    if data_pipeline is None or not hasattr(data_pipeline, "build_agent_context"):
        return {}
    path = _db_path()
    if not path or app_config is None:
        st.warning("Config or DB path missing — cannot build context.")
        return {}

    # 1) Raw data context from the pipeline.
    try:
        ctx = data_pipeline.build_agent_context(ticker, path, app_config)
        if not isinstance(ctx, dict):
            ctx = {"raw": ctx}
    except Exception as e:
        st.warning(f"Could not build context for {ticker}: {e}")
        ctx = {}

    # 2) Latest agent opinion (thesis + risk) from portfolio.db.
    opinion = load_latest_opinion(ticker)
    if opinion:
        # main._assemble_final_opinion flattens thesis keys at the top level,
        # so we reconstruct the nested dict the UI renders from.
        ctx["thesis"] = {
            "outlook": opinion.get("outlook"),
            "direction": opinion.get("price_direction"),
            "confidence": opinion.get("confidence"),
            "time_horizon": opinion.get("time_horizon"),
            "summary": opinion.get("rationale"),
            "key_risks": opinion.get("key_risks", []),
            "opportunities": opinion.get("key_opportunities", []),
        }

        risk_flags = opinion.get("risk_flags") or {}
        levels = risk_flags.get("levels") or {}
        combined = str(
            levels.get("combined")
            or risk_flags.get("combined_level")
            or "unknown"
        ).lower()
        ctx["risk"] = {
            "combined_level": combined,
            "fundamental_risk": str(levels.get("fundamental", "unknown")).lower(),
            "macro_risk": str(levels.get("macro", "unknown")).lower(),
            "market_risk": str(levels.get("market", "unknown")).lower(),
            "flags": (
                risk_flags.get("reasons")
                if isinstance(risk_flags.get("reasons"), list)
                else []
            ),
        }

    return ctx


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Load OHLCV price history via a MarketData instance.

    Returns an empty DataFrame on any failure. The returned frame has a
    DatetimeIndex and lowercase column names so render_price_chart works.
    """
    if MarketData is None:
        return pd.DataFrame()
    path = _db_path()
    if not path:
        return pd.DataFrame()

    # Map lookback days → yfinance period string.
    if lookback_days <= 35:
        period = "1mo"
    elif lookback_days <= 100:
        period = "3mo"
    elif lookback_days <= 200:
        period = "6mo"
    elif lookback_days <= 400:
        period = "1y"
    elif lookback_days <= 800:
        period = "2y"
    else:
        period = "5y"

    try:
        md = MarketData(db_path=path)
        df = md.get_prices(ticker, period=period, interval="1d")
    except Exception as e:
        st.warning(f"Could not load price history for {ticker}: {e}")
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # Normalise columns for downstream plotting.
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return pd.DataFrame()

    # Trim to the requested lookback window.
    try:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
        idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df = df[idx >= cutoff]
    except Exception:
        pass

    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_macro_snapshot() -> dict[str, Any]:
    """Pull the latest macro snapshot from a MacroData instance."""
    if MacroData is None:
        return {}
    path = _db_path()
    if not path:
        return {}
    try:
        mac = MacroData(db_path=path)
        snap = mac.get_macro_dashboard()
        return snap if isinstance(snap, dict) else {}
    except Exception as e:
        st.warning(f"Could not load macro snapshot: {e}")
        return {}


# ======================================================================
# Indicator math (display-only — kept out of the quant package)
# ======================================================================
def compute_moving_averages(prices: pd.Series, windows=(20, 50)) -> dict[int, pd.Series]:
    return {w: prices.rolling(window=w, min_periods=max(2, w // 4)).mean() for w in windows}


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_rolling_vol(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling volatility. Uses quant.volatility if available."""
    returns = prices.pct_change()
    if quant_vol is not None and hasattr(quant_vol, "rolling_volatility"):
        try:
            return quant_vol.rolling_volatility(returns, window=window)
        except Exception:
            pass
    return returns.rolling(window=window).std() * np.sqrt(252)


# ======================================================================
# Rendering helpers
# ======================================================================
def render_metric_cards(summary: dict[str, Any]) -> None:
    cols = st.columns(5)
    cols[0].metric("Holdings", summary.get("num_holdings", 0))
    cols[1].metric("Total Value", f"${summary.get('total_value', 0):,.0f}")
    pnl = summary.get("total_pnl", 0.0)
    cols[2].metric(
        "Total P&L",
        f"${pnl:,.0f}",
        delta=f"{pnl:+,.0f}",
        delta_color="normal",
    )
    cols[3].markdown(
        f"**Avg Risk**<br>{_risk_badge(summary.get('avg_risk', 'unknown'))}",
        unsafe_allow_html=True,
    )
    cash = summary.get("cash")
    cols[4].metric("Cash", f"${cash:,.0f}" if cash is not None else "—")


def render_ticker_header(ticker: str, prices: pd.DataFrame, context: dict[str, Any]) -> None:
    st.subheader(f"📊 {ticker}")
    cols = st.columns(4)

    if not prices.empty and "close" in prices.columns:
        latest = float(prices["close"].iloc[-1])
        prev = float(prices["close"].iloc[-2]) if len(prices) > 1 else latest
        change = latest - prev
        pct = (change / prev * 100) if prev else 0.0
        cols[0].metric(
            "Latest Close",
            f"${latest:,.2f}",
            delta=f"{change:+,.2f} ({pct:+.2f}%)",
        )
    else:
        cols[0].metric("Latest Close", "—")

    thesis = context.get("thesis") or {}
    outlook = thesis.get("outlook") or thesis.get("direction") or "unknown"
    confidence = thesis.get("confidence")
    cols[1].markdown(
        f"**Outlook**<br>{_outlook_badge(outlook)}",
        unsafe_allow_html=True,
    )
    cols[2].metric(
        "Confidence",
        f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—",
    )

    risk = context.get("risk") or {}
    combined = risk.get("combined_level") or risk.get("level") or "unknown"
    cols[3].markdown(
        f"**Risk Level**<br>{_risk_badge(combined)}",
        unsafe_allow_html=True,
    )


def render_price_chart(
    ticker: str,
    prices: pd.DataFrame,
    show_ma: bool,
    show_rsi: bool,
    show_vol: bool,
) -> None:
    if prices.empty or "close" not in prices.columns:
        st.info(f"No price history available for {ticker}.")
        return

    rows = 1 + int(show_rsi) + int(show_vol)
    row_heights = [0.6] + [0.4 / max(1, rows - 1)] * (rows - 1) if rows > 1 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    # --- Price (candlestick if OHLC present, else line) ---
    ohlc_cols = {"open", "high", "low", "close"}
    if ohlc_cols.issubset(prices.columns):
        fig.add_trace(
            go.Candlestick(
                x=prices.index,
                open=prices["open"],
                high=prices["high"],
                low=prices["low"],
                close=prices["close"],
                name="Price",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices["close"],
                mode="lines",
                name="Close",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

    if show_ma:
        for window, series in compute_moving_averages(prices["close"]).items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines",
                    name=f"MA{window}",
                    line=dict(width=1.2),
                ),
                row=1,
                col=1,
            )

    current_row = 1
    if show_rsi:
        current_row += 1
        rsi = compute_rsi(prices["close"])
        fig.add_trace(
            go.Scatter(
                x=rsi.index,
                y=rsi,
                mode="lines",
                name="RSI(14)",
                line=dict(color="#8b5cf6", width=1.5),
                showlegend=False,
            ),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)

    if show_vol:
        current_row += 1
        vol = compute_rolling_vol(prices["close"])
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol,
                mode="lines",
                name="Rolling Vol (20d, ann.)",
                line=dict(color="#f59e0b", width=1.5),
                showlegend=False,
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.1)",
            ),
            row=current_row,
            col=1,
        )
        fig.update_yaxes(title_text="Ann. Vol", row=current_row, col=1)

    fig.update_layout(
        title=f"{ticker} — Price & Indicators",
        height=300 + 150 * (rows - 1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_risk_panel(context: dict[str, Any], ticker: str) -> None:
    st.markdown("### ⚠️ Risk")
    risk = context.get("risk") or {}
    if not risk:
        render_generate_cta(ticker)
        return

    combined = risk.get("combined_level") or risk.get("level") or "unknown"
    st.markdown(
        f"**Combined:** {_risk_badge(combined)}",
        unsafe_allow_html=True,
    )

    # Sub-risk breakdown
    sub_rows = []
    for label, key in [
        ("Fundamental", "fundamental_risk"),
        ("Macro", "macro_risk"),
        ("Market", "market_risk"),
    ]:
        val = risk.get(key) or risk.get(key.replace("_risk", ""))
        if val:
            sub_rows.append((label, str(val)))

    if sub_rows:
        for label, val in sub_rows:
            st.markdown(
                f"- **{label}:** {_risk_badge(val)}",
                unsafe_allow_html=True,
            )

    flags = risk.get("flags") or risk.get("risk_flags") or []
    if flags:
        st.markdown("**Flags**")
        for f in flags:
            st.markdown(f"- {f}")


def render_thesis_panel(context: dict[str, Any]) -> None:
    st.markdown("### 💡 Thesis")
    thesis = context.get("thesis") or {}
    if not thesis:
        st.caption(
            "Thesis will appear here once the agent has generated an "
            "opinion (use the button in the Risk panel)."
        )
        return

    direction = thesis.get("outlook") or thesis.get("direction") or "unknown"
    st.markdown(
        f"**Direction:** {_outlook_badge(direction)}",
        unsafe_allow_html=True,
    )

    summary_text = thesis.get("summary") or thesis.get("narrative")
    if summary_text:
        st.markdown(summary_text)

    opportunities = thesis.get("opportunities") or thesis.get("catalysts") or []
    if opportunities:
        st.markdown("**Opportunities / Catalysts**")
        for op in opportunities:
            st.markdown(f"- {op}")

    key_risks = thesis.get("risks") or thesis.get("key_risks") or []
    if key_risks:
        st.markdown("**Key Risks**")
        for r in key_risks:
            st.markdown(f"- {r}")


def render_macro_panel(context: dict[str, Any], snapshot: dict[str, Any]) -> None:
    st.markdown("### 🌍 Macro Context")
    # Prefer context['macro'] if available; merge with global snapshot.
    macro = {**(snapshot or {}), **(context.get("macro") or {})}
    if not macro:
        st.info("No macro data available.")
        return

    # macro_data.get_macro_dashboard() returns a nested dict. Flatten the
    # pieces we care about so the cards below can just do flat lookups.
    rates = macro.get("interest_rates") or {}
    financial = macro.get("financial_conditions") or {}
    recession = macro.get("recession_signals") or {}

    recession_prob = (
        recession.get("recession_probability")
        or macro.get("recession_probability")
    )
    # FRED recession probability is already in % (e.g. 1.68), not a fraction.
    # The inline agent macro uses 0-1 fractions; handle both.
    if isinstance(recession_prob, (int, float)) and recession_prob > 1.0:
        recession_prob = recession_prob / 100.0

    term_spread = (
        rates.get("yield_spread_10y2y")
        or macro.get("term_spread")
        or macro.get("yield_curve_spread")
    )
    vix = (
        financial.get("vix")
        or macro.get("vix")
        or macro.get("volatility_index")
    )
    inverted = rates.get("yield_curve_inverted")
    if isinstance(inverted, bool):
        regime = "Inverted" if inverted else "Normal"
    else:
        regime = macro.get("regime") or macro.get("market_regime")

    cols = st.columns(4)

    def _show(col, label, value, fmt: str | None = None):
        if value is None:
            col.metric(label, "—")
            return
        if fmt and isinstance(value, (int, float)):
            col.metric(label, fmt.format(value))
        else:
            col.metric(label, str(value))

    _show(cols[0], "Recession Prob.", recession_prob, "{:.1%}")
    _show(cols[1], "Term Spread (10Y-2Y)", term_spread, "{:+.2f}%")
    _show(cols[2], "VIX", vix, "{:.1f}")
    _show(cols[3], "Regime", regime)


def render_events_panel(context: dict[str, Any]) -> None:
    events = context.get("recent_events") or context.get("filings") or []
    if not events:
        return
    st.markdown("### 📰 Recent Events / Filings")
    for e in events[:8]:
        if isinstance(e, dict):
            # SEC fetcher returns: filed_date, filing_type, accession_number,
            # primary_doc_url, description. Keep fallbacks in case the source
            # changes.
            when = (
                e.get("filed_date")
                or e.get("filing_date")
                or e.get("date")
                or e.get("filed_at")
                or "—"
            )
            form = (
                e.get("filing_type")
                or e.get("form_type")
                or e.get("form")
                or "Event"
            )
            desc = e.get("description") or e.get("title") or ""
            url = e.get("primary_doc_url") or e.get("url") or e.get("html_url")

            line = f"- **{when}** — `{form}`"
            if desc:
                line += f" · {desc}"
            if url:
                line += f" · [view]({url})"
            st.markdown(line)
        else:
            st.markdown(f"- {e}")


# ======================================================================
# ARY QUANT — Briefing + chat renderers
# ======================================================================
_CHAT_STATE_KEY = "ary_quant_chat_history"


def _chat_history(ticker: str) -> list[dict[str, str]]:
    """Per-ticker chat history, lazily initialised in session_state."""
    store = st.session_state.setdefault(_CHAT_STATE_KEY, {})
    return store.setdefault(ticker, [])


def _clear_chat_history(ticker: str) -> None:
    store = st.session_state.get(_CHAT_STATE_KEY, {})
    store.pop(ticker, None)


def render_essay_block(ticker: str, context: dict[str, Any]) -> dict[str, Any] | None:
    """Render the long-form briefing. Returns the essay dict (or None)
    so the chat can use it to ground its answers."""
    st.markdown(f"## 📝 Research Briefing — {ticker}")

    if ary_essay is None:
        st.warning(
            "Essay module unavailable — see backend import warnings."
        )
        return None

    essay = ary_essay.get_cached_essay(ticker)

    top_cols = st.columns([3, 1])
    with top_cols[1]:
        button_label = (
            "🔁 Regenerate briefing" if essay else "✨ Generate briefing"
        )
        if st.button(button_label, use_container_width=True, key=f"gen_essay_{ticker}"):
            if app_config is None:
                st.error("config module not importable — cannot call Ollama.")
            else:
                with st.spinner(
                    f"ARY QUANT is drafting the {ticker} briefing… "
                    "(30s–2min on local Qwen3-30B)"
                ):
                    essay = ary_essay.generate_essay(ticker, context, app_config)
                st.rerun()

    if essay is None:
        st.info(
            f"No briefing yet for **{ticker}**. Click **Generate briefing** to "
            "have ARY QUANT draft a 2+ page institutional-memo essay from the "
            "current data."
        )
        return None

    text = essay.get("text", "")

    # If generation produced an error marker, surface it prominently OUTSIDE
    # the 640px scroll container so the user sees it immediately rather than
    # having to scroll to notice.
    if essay.get("error"):
        st.error(text)
        return None

    meta_bits = []
    if essay.get("model_used"):
        meta_bits.append(f"model: `{essay['model_used']}`")
    if essay.get("word_count"):
        meta_bits.append(f"{essay['word_count']:,} words")
    if essay.get("elapsed_ms"):
        meta_bits.append(f"{essay['elapsed_ms'] / 1000:.1f}s")
    if essay.get("fallback"):
        meta_bits.append("⚠️ deterministic fallback")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    # Render essay in a scrollable container so a 2-page essay doesn't dominate
    # the viewport next to the chat pane.
    with st.container(border=True, height=640):
        st.markdown(text)

    return essay


def render_chat_panel(
    ticker: str,
    context: dict[str, Any],
    essay: dict[str, Any] | None,
) -> None:
    """ChatGPT-style conversation with ARY QUANT, grounded in the briefing."""
    header_cols = st.columns([4, 1])
    with header_cols[0]:
        st.markdown("## 💬 Ask ARY QUANT")
    with header_cols[1]:
        if st.button(
            "🧹 Clear chat",
            use_container_width=True,
            key=f"clear_chat_{ticker}",
            help="Remove this ticker's chat history.",
        ):
            _clear_chat_history(ticker)
            st.rerun()

    if ary_chat is None:
        st.error("Chat module unavailable — see backend import warnings.")
        return

    history = _chat_history(ticker)

    # Scrollable transcript — use the same height as the essay block so the
    # two columns line up visually on wide screens.
    with st.container(border=True, height=560):
        if not history:
            st.markdown(
                "<div style='color:#6b7280;padding:1rem 0.5rem;font-size:0.95em;"
                "line-height:1.55'>"
                f"👋 Hi — I'm <b>ARY QUANT</b>, your research analyst for "
                f"<b>{ticker}</b>.<br><br>"
                "Ask me about the briefing, the risk flags, valuation, the "
                "macro backdrop, or anything else in the context to the left. "
                "I'll anchor answers in the data you're looking at."
                "</div>",
                unsafe_allow_html=True,
            )
        for turn in history:
            role = turn["role"]
            avatar = "🧠" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.markdown(turn["content"])

    # The chat input is sticky at the bottom of the viewport by Streamlit
    # default — matches the ChatGPT pattern.
    user_msg = st.chat_input(f"Ask ARY QUANT about {ticker}…")

    if user_msg:
        # Append user turn immediately so the transcript shows before we wait
        # on the LLM.
        history.append({"role": "user", "content": user_msg})

        system_prompt = ary_chat.build_grounded_system_prompt(
            ticker=ticker,
            essay_text=(essay or {}).get("text"),
            context=context,
        )
        prompt = ary_chat.build_chat_prompt(
            system_prompt=system_prompt,
            history=history[:-1],  # exclude the just-appended user msg
            user_message=user_msg,
        )

        # Stream the response. We buffer into a string so we can persist the
        # final answer to history.
        if app_config is None:
            history.append(
                {
                    "role": "assistant",
                    "content": "⚠️ config module not importable; cannot reach Ollama.",
                }
            )
            st.rerun()
            return

        try:
            response_stream = ary_chat.stream_chat_response(
                prompt=prompt, config=app_config
            )

            # Accumulate as we stream so we can store the finished answer.
            collected: list[str] = []

            def _gen():
                for chunk in response_stream:
                    collected.append(chunk)
                    yield chunk

            with st.chat_message("assistant", avatar="🧠"):
                st.write_stream(_gen())

            history.append(
                {"role": "assistant", "content": "".join(collected).strip()}
            )
        except Exception as exc:
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"⚠️ ARY QUANT couldn't reach the LLM backend: "
                        f"`{exc}`\n\nCheck that Ollama is running and "
                        "`DEFAULT_AGENT_MODEL` is set in config.py."
                    ),
                }
            )
        st.rerun()


def render_briefing_tab(ticker: str, context: dict[str, Any]) -> None:
    """The 'ARY QUANT Briefing' tab — essay on the left, chat on the right."""
    essay_col, chat_col = st.columns([1, 1], gap="large")
    with essay_col:
        essay = render_essay_block(ticker, context)
    with chat_col:
        render_chat_panel(ticker, context, essay)


# ======================================================================
# Sidebar
# ======================================================================
def build_sidebar(portfolio_summary: dict[str, Any]) -> dict[str, Any]:
    st.sidebar.title("⚙️ Controls")

    # Ticker selection — prefer tickers from the portfolio if present.
    holdings = portfolio_summary.get("holdings", pd.DataFrame())
    tickers: list[str] = []
    if isinstance(holdings, pd.DataFrame) and "ticker" in holdings.columns:
        tickers = sorted(holdings["ticker"].astype(str).unique().tolist())
    if not tickers:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "SPY"]

    # ---- Active-ticker contract ------------------------------------------
    # `st.session_state["active_ticker"]` is the *single source of truth*
    # for which ticker the dashboard is analysing. main() initialises it
    # before tabs render. Both the sidebar widgets below AND the screener
    # row-click handler in ui.screener write to this key directly.
    #
    # We use on_change callbacks (rather than reading the widgets' return
    # values) so that the persisted text_input value doesn't override an
    # active_ticker that was just set by a screener row click. on_change
    # only fires when the user actually changes the widget, not on every
    # rerun, so the screener handoff survives intact.
    # ----------------------------------------------------------------------
    active = st.session_state.get("active_ticker") or tickers[0]
    if active not in tickers:
        # Make sure the dropdown can display whatever the screener picked,
        # even if that ticker isn't in the portfolio.
        tickers = [active] + tickers

    def _on_select_change() -> None:
        st.session_state["active_ticker"] = st.session_state["_sidebar_ticker_pick"]

    def _on_text_change() -> None:
        val = (st.session_state.get("_sidebar_ticker_text") or "").strip().upper()
        if val:
            st.session_state["active_ticker"] = val

    st.sidebar.selectbox(
        "Ticker",
        tickers,
        index=tickers.index(active),
        key="_sidebar_ticker_pick",
        on_change=_on_select_change,
    )
    st.sidebar.text_input(
        "…or enter another ticker",
        value="",
        key="_sidebar_ticker_text",
        on_change=_on_text_change,
        help="Type a symbol and press Enter.",
    )

    ticker = st.session_state["active_ticker"]

    lookback = st.sidebar.slider("Lookback (days)", 30, 730, 180, step=30)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chart Overlays**")
    show_ma = st.sidebar.checkbox("Moving averages (20 / 50)", value=True)
    show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
    show_vol = st.sidebar.checkbox("Rolling volatility", value=False)
    show_macro = st.sidebar.checkbox("Show macro panel", value=True)

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("Dashboard reads the latest stored data. Click refresh to reload.")

    return {
        "ticker": ticker,
        "lookback": lookback,
        "show_ma": show_ma,
        "show_rsi": show_rsi,
        "show_vol": show_vol,
        "show_macro": show_macro,
    }


# ======================================================================
# Main app
# ======================================================================
def main() -> None:
    st.title("🧠 ARY QUANT")
    st.caption(
        "AI research analyst · portfolio · per-ticker briefings · "
        "macro overlay · conversational Q&A"
    )

    # Initialise the single source of truth for the active ticker. Both the
    # sidebar widgets and the Screener tab write here; everything else reads
    # it indirectly via build_sidebar() → controls["ticker"]. Setting this
    # BEFORE any widget that uses it avoids the "value not yet in state on
    # first render" race.
    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = "NVDA"

    if _BACKEND_ERRORS:
        with st.expander("⚠️ Backend import warnings", expanded=False):
            for msg in _BACKEND_ERRORS:
                st.warning(msg)

    # ---- Portfolio ----
    summary = load_portfolio_summary()
    st.markdown("## Portfolio Overview")
    if summary["num_holdings"] == 0:
        st.info(
            "No holdings found. The portfolio DB appears empty, or "
            "`portfolio_db` is not exposing a supported accessor."
        )
    render_metric_cards(summary)

    # ---- Sidebar (needs portfolio summary for ticker list) ----
    controls = build_sidebar(summary)

    st.markdown("---")

    # ---- Per-ticker section ----
    ticker = controls["ticker"]
    lookback = controls["lookback"]

    with st.spinner(f"Loading {ticker}…"):
        context = load_ticker_context(ticker, lookback)
        prices = load_price_history(ticker, lookback)

    render_ticker_header(ticker, prices, context)

    tab_screener, tab_overview, tab_briefing, tab_analyzer, tab_debug = st.tabs(
        [
            "🔍 Screener",
            "📊 Overview",
            "📝 ARY QUANT Briefing",
            "🔎 Data-Point Analyzer",
            "🛠 Debug",
        ]
    )

    # ---------------- Screener tab ----------------
    # The screener writes the clicked ticker into st.session_state["active_ticker"]
    # and triggers a rerun. On the next render, build_sidebar() picks up the new
    # value, controls["ticker"] reflects it, and load_ticker_context / the
    # Overview / Briefing / Data-Point Analyzer tabs all refresh automatically.
    with tab_screener:
        if ary_screener is not None and hasattr(ary_screener, "render_screener_tab"):
            ary_screener.render_screener_tab()
        else:
            st.error(
                "Screener module is not importable. Check that "
                "`ui/screener.py` exists and that there are no import errors."
            )

    # ---------------- Overview tab ----------------
    with tab_overview:
        st.markdown("### 📉 Price & Indicators")
        render_price_chart(
            ticker,
            prices,
            show_ma=controls["show_ma"],
            show_rsi=controls["show_rsi"],
            show_vol=controls["show_vol"],
        )

        # ---- Risk & Thesis side-by-side ----
        col_r, col_t = st.columns(2)
        with col_r:
            render_risk_panel(context, ticker)
        with col_t:
            render_thesis_panel(context)

        # ---- Events / filings (if present) ----
        render_events_panel(context)

        # ---- Macro ----
        if controls["show_macro"]:
            st.markdown("---")
            macro_snapshot = load_macro_snapshot()
            render_macro_panel(context, macro_snapshot)

    # ---------------- ARY QUANT Briefing tab ----------------
    with tab_briefing:
        render_briefing_tab(ticker, context)

    # ---------------- Data-Point Analyzer tab ----------------
    with tab_analyzer:
        try:
            from ui.data_point_analyzer_section import render_data_point_analyzer_section
            render_data_point_analyzer_section(
                ticker=ticker,
                context=context,
                config=app_config,
            )
        except ImportError as e:
            st.error(
                f"Data-Point Analyzer module not available: {e}. "
                "Place `agent/data_point_analyzer.py` and "
                "`ui/data_point_analyzer_section.py` in your project."
            )
        except Exception as e:
            st.error(f"Analyzer section failed to render: {e}")

    # ---------------- Debug tab ----------------
    with tab_debug:
        db_path = _db_path()
        st.write(f"**DB path:** `{db_path}`")
        if db_path:
            try:
                with sqlite3.connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT ticker, created_at FROM agent_opinions "
                        "ORDER BY id DESC LIMIT 20"
                    ).fetchall()
                st.write(f"**agent_opinions rows ({len(rows)}):**")
                if rows:
                    st.table(
                        pd.DataFrame(rows, columns=["ticker", "created_at"])
                    )
                else:
                    st.write("Table empty — run `python main.py` to populate.")
            except sqlite3.OperationalError as e:
                st.write(f"Table missing or unreadable: {e}")
            except Exception as e:
                st.write(f"DB read failed: {e}")

        st.write("**Context dict:**")
        if context:
            st.json(context, expanded=False)
        else:
            st.write("No context returned.")


if __name__ == "__main__":
    main()