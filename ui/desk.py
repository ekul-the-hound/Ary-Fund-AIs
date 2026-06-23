"""
ui/desk.py
==========

The Analyst's Desk: a scrollable, single-name research workspace.

This is the deep-research surface of the v2 dashboard (Concept 5, embedded
as the Analyze/Memo stage of the Research Pipeline). It replaces the flat
tab row with one top-to-bottom working sheet: a 10-second snapshot above the
fold, then progressively deeper sections you scroll into.

Section order (each is an independent, defensive renderer)
----------------------------------------------------------
    Snapshot   : sticky header (price, bias, conviction, risk triplet) + chart
                 + three summary tiles (thesis / risk / quant).
    Thesis     : the institutional memo essay + self-review scorecard.
    Risk       : the fundamental/macro/market decomposition + sector z-scores
                 + Altman distress zone + per-axis reasons.
    Quant      : a compact strip of the highest-signal models (regime, Hurst,
                 realized vol, VaR/ES) computed from the loaded price series.
    Filings    : recent SEC filings (click-through) + sentiment block.
    Macro      : the macro context panel.
    Evidence   : the RAG retrieved_context chunks the thesis rests on.

Contracts honored
------------------
* Reads the registry-backed context dict from
  ``pipeline.build_agent_context`` exactly as the existing app builds it
  (the caller passes ``context`` in; this module does not fetch it).
* Long LLM work (generate opinion / generate memo) is dispatched through
  ``ui.state``'s job queue so the UI never blocks. The Desk shows in-flight
  status inline and applies results on the next poll.
* All visual encoding goes through ``ui.components`` so the Desk shares one
  vocabulary with the board, lab, and rail.
* Every section degrades to a neutral placeholder when its data is absent —
  matching the backend's explicit-absence contract.

The Desk does NOT own the sidebar, the global header, or navigation between
destinations — ``app_v2`` does. The Desk owns everything from the ticker
header down.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui import components as C
from ui import state as S

logger = logging.getLogger("ary_quant.ui.desk")


# ======================================================================
# Soft backend imports — the Desk must render even if a quant module or
# the agent chain is mid-refactor. Each missing piece degrades to a
# placeholder rather than crashing the page.
# ======================================================================
def _try(import_fn: Callable[[], Any], label: str) -> Any:
    try:
        return import_fn()
    except Exception as e:  # pragma: no cover - defensive UI boundary
        logger.warning("desk: %s unavailable: %s", label, e)
        return None


# Quant snapshot deps (pure functions; safe to call on a price Series).
_hurst = _try(lambda: __import__("quant.hurst", fromlist=["hurst_exponent"]).hurst_exponent,
              "quant.hurst")
_var_es_report = _try(
    lambda: __import__("quant.var_es", fromlist=["var_es_report"]).var_es_report,
    "quant.var_es")
_compute_returns = _try(
    lambda: __import__("quant.volatility", fromlist=["compute_returns"]).compute_returns,
    "quant.volatility.compute_returns")
_realized_vol = _try(
    lambda: __import__("quant.volatility", fromlist=["realized_volatility"]).realized_volatility,
    "quant.volatility.realized_volatility")
_annualize_vol = _try(
    lambda: __import__("quant.volatility", fromlist=["annualize_volatility"]).annualize_volatility,
    "quant.volatility.annualize_volatility")


# ======================================================================
# Indicator math (display-only, mirrors the original app's helpers)
# ======================================================================
def _moving_averages(prices: pd.Series, windows=(20, 50)) -> dict[int, pd.Series]:
    return {w: prices.rolling(window=w, min_periods=max(2, w // 4)).mean()
            for w in windows}


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


# ======================================================================
# SNAPSHOT — sticky header + chart + three summary tiles
# ======================================================================
def _render_sticky_header(ticker: str, prices: pd.DataFrame,
                          context: dict[str, Any]) -> None:
    """The always-visible-on-scroll header: price, bias, conviction, risk.

    Pulls the thesis/risk blocks the app already merges into context from
    the latest opinion. Everything here is a glance-level read; depth is
    below the fold.
    """
    thesis = context.get("thesis") or {}
    risk_flags = _risk_flags_from_context(context)

    # Price + change.
    last = prev = None
    if isinstance(prices, pd.DataFrame) and not prices.empty and "close" in prices.columns:
        last = float(prices["close"].iloc[-1])
        prev = float(prices["close"].iloc[-2]) if len(prices) > 1 else last
    change = (last - prev) if (last is not None and prev is not None) else None
    pct = (change / prev) if (change is not None and prev) else None

    price_str = C.fmt_money(last) if last is not None else "—"
    chg_str = ""
    if change is not None and pct is not None:
        color = C.OUTLOOK_COLORS["bullish"] if change >= 0 else C.OUTLOOK_COLORS["bearish"]
        chg_str = (
            f"<span style='color:{color};font-size:0.95em;margin-left:8px;"
            f"font-variant-numeric:tabular-nums;'>{change:+,.2f} "
            f"({pct*100:+.2f}%)</span>"
        )

    outlook = thesis.get("outlook") or thesis.get("direction") or "unknown"
    confidence = thesis.get("confidence")
    bias = (context.get("thesis") or {}).get("bias_score")
    # bias_score may live at opinion top-level rather than inside thesis;
    # the app flattens thesis from the opinion, so also check there.
    if bias is None:
        bias = context.get("bias_score")

    # Freshness for the price block, if provenance recorded it.
    price_chip = C.chip_from_provenance(context, "ticker.price.adj_close",
                                        label=None)

    st.markdown(
        f"<div style='position:sticky;top:0;z-index:6;padding:10px 0 8px 0;"
        f"border-bottom:1px solid {C._HAIRLINE};margin-bottom:6px;"
        f"backdrop-filter:blur(6px);'>"
        f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;'>"
        f"<span style='font-size:1.6em;font-weight:800;letter-spacing:0.02em;'>{ticker}</span>"
        f"<span style='font-size:1.25em;font-weight:700;"
        f"font-variant-numeric:tabular-nums;'>{price_str}</span>{chg_str}"
        f"<span style='margin-left:auto;'>{price_chip}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Signal row: outlook pill, bias bar, conviction meter, risk triplet.
    cols = st.columns([1.1, 1.3, 1.3, 1.6])
    with cols[0]:
        st.markdown("<div style='font-size:0.78em;color:#9ca3af;'>OUTLOOK</div>",
                    unsafe_allow_html=True)
        st.markdown(C.badge_outlook(outlook), unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div style='font-size:0.78em;color:#9ca3af;'>BIAS</div>",
                    unsafe_allow_html=True)
        st.markdown(C.meter_bias(bias), unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div style='font-size:0.78em;color:#9ca3af;'>CONVICTION</div>",
                    unsafe_allow_html=True)
        st.markdown(C.meter_conviction(confidence), unsafe_allow_html=True)
    with cols[3]:
        st.markdown("<div style='font-size:0.78em;color:#9ca3af;'>RISK (F·M·Mk)</div>",
                    unsafe_allow_html=True)
        st.markdown(C.inline_risk_triplet(risk_flags), unsafe_allow_html=True)


def _render_price_chart(ticker: str, prices: pd.DataFrame, *,
                        show_ma: bool, show_rsi: bool, show_vol: bool) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty or "close" not in prices.columns:
        st.info(f"No price history available for {ticker}.")
        return

    rows = 1 + int(show_rsi) + int(show_vol)
    row_heights = ([0.62] + [0.38 / max(1, rows - 1)] * (rows - 1)) if rows > 1 else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=row_heights)

    ohlc = {"open", "high", "low", "close"}
    if ohlc.issubset(prices.columns):
        fig.add_trace(go.Candlestick(
            x=prices.index, open=prices["open"], high=prices["high"],
            low=prices["low"], close=prices["close"], name="Price",
            showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=prices.index, y=prices["close"], mode="lines", name="Close",
            line=dict(width=2)), row=1, col=1)

    if show_ma:
        for w, series in _moving_averages(prices["close"]).items():
            fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines",
                                     name=f"MA{w}", line=dict(width=1.2)),
                          row=1, col=1)

    r = 1
    if show_rsi:
        r += 1
        rsi = _rsi(prices["close"])
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI(14)",
                                 line=dict(color="#8b5cf6", width=1.4),
                                 showlegend=False), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)

    if show_vol:
        r += 1
        returns = prices["close"].pct_change()
        vol = returns.rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=vol.index, y=vol, mode="lines",
                                 name="Ann. Vol (20d)",
                                 line=dict(color="#f59e0b", width=1.4),
                                 showlegend=False, fill="tozeroy",
                                 fillcolor="rgba(245,158,11,0.1)"), row=r, col=1)
        fig.update_yaxes(title_text="Ann. Vol", row=r, col=1)

    fig.update_layout(
        height=300 + 140 * (rows - 1), xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _render_summary_tiles(ticker: str, context: dict[str, Any],
                          prices: pd.DataFrame) -> None:
    """Three above-the-fold tiles: thesis headline, risk, quant glance."""
    thesis = context.get("thesis") or {}
    risk_flags = _risk_flags_from_context(context)
    ds = context.get("derived_signals") or {}

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Thesis**", unsafe_allow_html=True)
        outlook = thesis.get("outlook") or "unknown"
        st.markdown(C.badge_outlook(outlook), unsafe_allow_html=True)
        summary = thesis.get("summary") or thesis.get("rationale")
        if summary:
            txt = summary if len(summary) <= 180 else summary[:177] + "…"
            st.markdown(f"<div style='font-size:0.86em;margin-top:6px;color:#cbd5e1;'>"
                        f"{txt}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.86em;color:#9ca3af;margin-top:6px;'>"
                        "No thesis yet.</div>", unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    with c2:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Risk**", unsafe_allow_html=True)
        levels = (risk_flags or {}).get("levels") or {}
        if levels:
            st.markdown(C.inline_risk_triplet(risk_flags), unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.86em;color:#9ca3af;'>"
                        "Run analysis to compute risk.</div>",
                        unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    with c3:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Quant glance**", unsafe_allow_html=True)
        regime = ds.get("regime")
        rsi_14 = ds.get("rsi_14")
        rvol = ds.get("realized_vol_30d")
        bits = []
        if regime:
            bits.append(f"regime: <b>{regime}</b>")
        if C._is_num(rsi_14):
            bits.append(f"RSI: <b>{float(rsi_14):.0f}</b>")
        if C._is_num(rvol):
            bits.append(f"σ(30d): <b>{C.fmt_pct(rvol)}</b>")
        if bits:
            st.markdown("<div style='font-size:0.86em;color:#cbd5e1;line-height:1.7;'>"
                        + "<br>".join(bits) + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.86em;color:#9ca3af;'>"
                        "No derived signals yet.</div>", unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)


# ======================================================================
# THESIS — memo essay + review scorecard, with non-blocking generation
# ======================================================================
def _render_thesis_section(ticker: str, context: dict[str, Any], *,
                           essay_mod: Any, config: Any,
                           on_generate_opinion: Optional[Callable[[str], str]]) -> None:
    C.section_anchor("sec-thesis", "Thesis & Memo",
                     subtitle="Institutional memo with self-review scoring")

    thesis = context.get("thesis") or {}
    opinion = context.get("_opinion") or {}  # app may stash the raw opinion
    review = opinion.get("review") or context.get("review") or {}

    # --- Generation controls (non-blocking via job queue) ---------------
    jobs = S.poll_jobs(kinds={"opinion"}, ticker=ticker)
    in_flight = [j for j in jobs if j.state in (S.JobState.QUEUED, S.JobState.RUNNING)]

    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_r:
        if in_flight:
            j = in_flight[0]
            st.markdown(
                f"<div style='font-size:0.85em;color:#93c5fd;'>⏳ generating… "
                f"({j.elapsed():.0f}s)</div>", unsafe_allow_html=True)
        else:
            label = "🔁 Regenerate analysis" if opinion else "🤖 Generate analysis"
            if st.button(label, use_container_width=True, key=f"gen_op_{ticker}"):
                if on_generate_opinion is not None:
                    on_generate_opinion(ticker)
                    st.rerun()
                else:
                    st.warning("Opinion generation is not wired in this context.")

    # --- Essay body -----------------------------------------------------
    essay_text = thesis.get("essay") or opinion.get("essay")
    essay_meta = opinion.get("essay_meta") or {}
    is_fallback = bool(essay_meta.get("fallback"))

    if essay_text:
        C.panel_fallback_notice(is_fallback, what="memo")
        meta_bits = []
        if essay_meta.get("model"):
            meta_bits.append(f"model: `{essay_meta['model']}`")
        if essay_meta.get("word_count"):
            meta_bits.append(f"{essay_meta['word_count']:,} words")
        if meta_bits:
            st.caption(" · ".join(meta_bits))

        # Two-column: memo reading column + review scorecard margin.
        memo_col, score_col = st.columns([2.2, 1])
        with memo_col:
            with st.container(border=True, height=620):
                st.markdown(essay_text)
        with score_col:
            st.markdown("**Self-review**")
            C.panel_review_scorecard(review)
    else:
        # No memo yet — offer to generate via the on-demand essay module
        # (separate from the full opinion chain; lighter).
        st.info(
            f"No memo yet for **{ticker}**. Generate the full analysis above, "
            "or draft just the briefing essay below."
        )
        if essay_mod is not None and hasattr(essay_mod, "generate_essay"):
            cached = essay_mod.get_cached_essay(ticker) if hasattr(
                essay_mod, "get_cached_essay") else None
            if cached and cached.get("text"):
                C.panel_fallback_notice(bool(cached.get("fallback")), what="briefing")
                with st.container(border=True, height=520):
                    st.markdown(cached["text"])
            else:
                if st.button("✨ Draft briefing essay", key=f"gen_essay_{ticker}"):
                    if config is None:
                        st.error("config not importable — cannot call Ollama.")
                    else:
                        with st.spinner(f"Drafting {ticker} briefing… (local LLM)"):
                            essay_mod.generate_essay(ticker, context, config)
                        st.rerun()


# ======================================================================
# RISK — three-axis decomposition + sector z-scores + distress zone
# ======================================================================
def _risk_flags_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """Return a risk_flags-shaped dict from context, tolerant of both the
    merged ``context['risk']`` shape (from app.load_ticker_context) and a
    raw opinion's ``risk_flags``.
    """
    # The app merges a simplified ``risk`` dict; prefer the raw risk_flags
    # if the opinion is stashed, else reconstruct levels from ``risk``.
    opinion = context.get("_opinion") or {}
    rf = opinion.get("risk_flags")
    if isinstance(rf, dict) and rf.get("levels"):
        return rf

    risk = context.get("risk") or {}
    if not risk:
        return {}
    # Reconstruct the levels/reasons shape from the flattened ``risk`` dict.
    levels = {
        "fundamental": risk.get("fundamental_risk", "unknown"),
        "macro": risk.get("macro_risk", "unknown"),
        "market": risk.get("market_risk", "unknown"),
        "combined": risk.get("combined_level", "unknown"),
    }
    reasons = {"fundamental": [], "macro": [], "market": []}
    flags = risk.get("flags") or []
    if flags:
        # Without an axis breakdown, attach flat flags to fundamental so
        # they're at least visible. The full per-axis split is available
        # when the raw opinion is present.
        reasons["fundamental"] = list(flags)
    return {"levels": levels, "reasons": reasons}


def _render_risk_section(ticker: str, context: dict[str, Any]) -> None:
    C.section_anchor("sec-risk", "Risk Decomposition",
                     subtitle="Fundamental · Macro · Market, sector-relative")

    risk_flags = _risk_flags_from_context(context)
    if not (risk_flags or {}).get("levels"):
        st.info("No risk decomposition yet — generate analysis to compute the "
                "fundamental / macro / market levels and sector z-scores.")
        return

    left, right = st.columns([1, 1])
    with left:
        C.panel_risk_triplet(risk_flags, show_reasons=True, top_reasons=3)

    with right:
        # Sector-relative z-scores, if the opinion carried them. The
        # risk_scanner emits per-metric z-scores when peer stats are
        # available; surface any that are present.
        opinion = context.get("_opinion") or {}
        rf = opinion.get("risk_flags") or {}
        zscores = rf.get("zscores") or rf.get("sector_zscores") or {}
        if zscores:
            st.markdown("**Sector-relative (σ from peer mean)**")
            for label, z in list(zscores.items())[:8]:
                st.markdown(C.panel_zscore_bar(str(label), z),
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.88em;color:#9ca3af;'>"
                        "Sector z-scores appear here when peer statistics are "
                        "available for this name's sector.</div>",
                        unsafe_allow_html=True)

        # Altman distress zone, if present.
        distress = rf.get("distress") or rf.get("altman") or {}
        if distress and distress.get("zone"):
            st.markdown("<div style='margin-top:10px;'><b>Distress (Altman Z)</b></div>",
                        unsafe_allow_html=True)
            z = distress.get("z")
            zone = distress.get("zone")
            z_str = f"{float(z):.2f}" if C._is_num(z) else "—"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-top:4px;'>"
                f"{C.badge_zone(zone)}<span style='font-variant-numeric:tabular-nums;"
                f"font-size:0.9em;'>Z = {z_str}</span></div>",
                unsafe_allow_html=True)


# ======================================================================
# QUANT — compact strip of high-signal models from the price series
# ======================================================================
def _render_quant_section(ticker: str, prices: pd.DataFrame,
                          context: dict[str, Any]) -> None:
    C.section_anchor("sec-quant", "Quant Snapshot",
                     subtitle="Computed from the loaded price window")

    ds = context.get("derived_signals") or {}

    cols = st.columns(4)

    # 1) Regime (from derived_signals; HMM lives in the full Quant Lab).
    with cols[0]:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Regime**")
        regime = ds.get("regime")
        if regime:
            color = C.OUTLOOK_COLORS.get(str(regime).lower(), C._NEUTRAL_TEXT)
            st.markdown(f"<div style='font-size:1.1em;font-weight:700;color:{color};'>"
                        f"{regime}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9ca3af;'>—</div>", unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    # 2) Hurst exponent (compute on demand).
    with cols[1]:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Hurst (H)**")
        h_val, h_interp = _compute_hurst(prices)
        if h_val is not None:
            st.markdown(f"<div style='font-size:1.1em;font-weight:700;"
                        f"font-variant-numeric:tabular-nums;'>{h_val:.2f}</div>"
                        f"<div style='font-size:0.78em;color:#9ca3af;'>{h_interp}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9ca3af;'>insufficient data</div>",
                        unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    # 3) Realized vol (annualized).
    with cols[2]:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**Realized σ (ann.)**")
        rv = _compute_realized_vol(prices)
        if rv is not None:
            st.markdown(f"<div style='font-size:1.1em;font-weight:700;"
                        f"font-variant-numeric:tabular-nums;'>{rv*100:.1f}%</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9ca3af;'>—</div>", unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    # 4) VaR / ES (95%, historical).
    with cols[3]:
        st.markdown(C.card_open(), unsafe_allow_html=True)
        st.markdown("**VaR / ES 95%**")
        var, es = _compute_var_es(prices)
        if var is not None:
            st.markdown(f"<div style='font-size:1.0em;font-weight:700;"
                        f"font-variant-numeric:tabular-nums;'>"
                        f"{var*100:.2f}% / {es*100:.2f}%</div>"
                        f"<div style='font-size:0.72em;color:#9ca3af;'>daily loss</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#9ca3af;'>—</div>", unsafe_allow_html=True)
        st.markdown(C.card_close(), unsafe_allow_html=True)

    st.caption("Deeper probabilistic models (HMM regimes, OU mean-reversion, "
               "GBM Monte Carlo, RMT/MST structure) live in the Quant Lab.")


def _price_series(prices: pd.DataFrame) -> Optional[pd.Series]:
    if not isinstance(prices, pd.DataFrame) or prices.empty or "close" not in prices.columns:
        return None
    s = prices["close"].dropna()
    return s if len(s) >= 30 else None


def _compute_hurst(prices: pd.DataFrame) -> tuple[Optional[float], str]:
    s = _price_series(prices)
    if s is None or _hurst is None:
        return None, ""
    try:
        res = _hurst(s)
        if isinstance(res, dict) and res.get("available", True) and res.get("hurst") is not None:
            return float(res["hurst"]), str(res.get("interpretation") or "")
    except Exception as e:  # pragma: no cover
        logger.warning("desk hurst failed: %s", e)
    return None, ""


def _compute_realized_vol(prices: pd.DataFrame) -> Optional[float]:
    s = _price_series(prices)
    if s is None or _compute_returns is None or _realized_vol is None:
        return None
    try:
        rets = _compute_returns(s, kind="log")
        rv = _realized_vol(rets)
        if _annualize_vol is not None and C._is_num(rv):
            try:
                return float(_annualize_vol(rv))
            except Exception:
                pass
        # Fallback: annualize manually if annualize_volatility signature differs.
        if C._is_num(rv):
            return float(rv) * np.sqrt(252)
    except Exception as e:  # pragma: no cover
        logger.warning("desk realized_vol failed: %s", e)
    return None


def _compute_var_es(prices: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    s = _price_series(prices)
    if s is None or _compute_returns is None or _var_es_report is None:
        return None, None
    try:
        rets = _compute_returns(s, kind="log")
        rep = _var_es_report(rets, confidence=0.95, method="historical")
        if isinstance(rep, dict):
            var, es = rep.get("var"), rep.get("es")
            # sign_convention is "positive = loss"; values are already losses.
            if C._is_num(var) and C._is_num(es):
                return float(var), float(es)
    except Exception as e:  # pragma: no cover
        logger.warning("desk var_es failed: %s", e)
    return None, None


# ======================================================================
# FILINGS & SENTIMENT
# ======================================================================
def _render_filings_section(ticker: str, context: dict[str, Any]) -> None:
    C.section_anchor("sec-filings", "Filings & Sentiment")

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown("**Recent filings & events**")
        events = context.get("filings") or context.get("recent_events") or []
        if not events:
            st.markdown("<div style='color:#9ca3af;font-size:0.9em;'>"
                        "No recent filings loaded.</div>", unsafe_allow_html=True)
        else:
            for e in events[:8]:
                if isinstance(e, dict):
                    when = (e.get("filed_date") or e.get("filing_date")
                            or e.get("date") or "—")
                    form = (e.get("filing_type") or e.get("form_type")
                            or e.get("form") or "Event")
                    desc = e.get("description") or e.get("title") or ""
                    url = e.get("primary_doc_url") or e.get("url")
                    line = f"**{when}** · `{form}`"
                    if desc:
                        line += f" — {desc}"
                    if url:
                        line += f" · [view]({url})"
                    st.markdown(line)
                else:
                    st.markdown(f"- {e}")

    with right:
        st.markdown("**Sentiment**")
        sent = context.get("sentiment") or {}
        if not sent:
            st.markdown("<div style='color:#9ca3af;font-size:0.9em;'>"
                        "No sentiment data.</div>", unsafe_allow_html=True)
        else:
            rows = [
                ("WSB mentions (24h)", sent.get("wsb_mentions_24h"), "{:.0f}"),
                ("WSB score", sent.get("wsb_score"), "{:.2f}"),
                ("News count (7d)", sent.get("news_count_7d"), "{:.0f}"),
                ("News tone (7d)", sent.get("news_tone_7d"), "{:+.2f}"),
            ]
            for label, val, fmt in rows:
                disp = fmt.format(val) if C._is_num(val) else "—"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.88em;margin:2px 0;'>"
                    f"<span style='color:#9ca3af;'>{label}</span>"
                    f"<span style='font-variant-numeric:tabular-nums;'>{disp}</span></div>",
                    unsafe_allow_html=True)


# ======================================================================
# MACRO
# ======================================================================
def _render_macro_section(context: dict[str, Any],
                          macro_snapshot: Optional[dict[str, Any]]) -> None:
    C.section_anchor("sec-macro", "Macro Context")

    macro = {**(macro_snapshot or {}), **(context.get("macro") or {})}
    if not macro:
        st.markdown("<div style='color:#9ca3af;'>No macro data available.</div>",
                    unsafe_allow_html=True)
        return

    rates = macro.get("interest_rates") or {}
    financial = macro.get("financial_conditions") or {}
    recession = macro.get("recession_signals") or {}

    recession_prob = (recession.get("recession_probability")
                      or macro.get("recession_probability"))
    if C._is_num(recession_prob) and recession_prob > 1.0:
        recession_prob = recession_prob / 100.0  # FRED reports percent points

    term_spread = (rates.get("yield_spread_10y2y") or macro.get("term_spread")
                   or macro.get("yield_curve_spread"))
    vix = (financial.get("vix") or macro.get("vix") or macro.get("volatility_index"))
    inverted = rates.get("yield_curve_inverted")
    if isinstance(inverted, bool):
        regime = "Inverted" if inverted else "Normal"
    else:
        regime = macro.get("regime") or macro.get("market_regime")

    cols = st.columns(4)

    def _metric(col, label, value, fmt: Optional[str]):
        with col:
            if value is None:
                st.metric(label, "—")
            elif fmt and C._is_num(value):
                st.metric(label, fmt.format(value))
            else:
                st.metric(label, str(value))

    _metric(cols[0], "Recession Prob.", recession_prob, "{:.1%}")
    _metric(cols[1], "Term Spread (10Y-2Y)", term_spread, "{:+.2f}%")
    _metric(cols[2], "VIX", vix, "{:.1f}")
    _metric(cols[3], "Regime", regime, None)

    # Freshness chip for the macro section.
    fresh = (context.get("freshness") or {}).get("macro")
    if fresh:
        st.markdown(C.chip_freshness(fresh, label=None), unsafe_allow_html=True)


# ======================================================================
# EVIDENCE — RAG retrieved_context
# ======================================================================
def _render_evidence_section(context: dict[str, Any]) -> None:
    C.section_anchor("sec-evidence", "Evidence",
                     subtitle="Retrieved context the thesis is grounded in")
    chunks = context.get("retrieved_context") or []
    C.panel_evidence_list(chunks, limit=8)


# ======================================================================
# PUBLIC ENTRY POINT
# ======================================================================
def render_desk(
    ticker: str,
    context: dict[str, Any],
    prices: pd.DataFrame,
    *,
    controls: Optional[dict[str, Any]] = None,
    macro_snapshot: Optional[dict[str, Any]] = None,
    essay_mod: Any = None,
    config: Any = None,
    on_generate_opinion: Optional[Callable[[str], str]] = None,
) -> None:
    """Render the full Analyst's Desk for one ticker.

    Parameters
    ----------
    ticker:
        Active ticker symbol.
    context:
        The merged context dict (output of the app's ``load_ticker_context``:
        ``build_agent_context`` + the flattened latest opinion). If the app
        also stashes the raw opinion under ``context['_opinion']``, the Desk
        will surface the full per-axis risk reasons, sector z-scores, and
        distress zone; otherwise it degrades to the simplified risk view.
    prices:
        OHLCV DataFrame with a DatetimeIndex and a 'close' column.
    controls:
        Optional dict of chart toggles: ``show_ma``, ``show_rsi``,
        ``show_vol``. Defaults to MA+RSI on, vol off.
    macro_snapshot:
        Optional global macro dashboard dict to merge with ``context['macro']``.
    essay_mod:
        The ``ui.essay`` module (for the on-demand briefing fallback). Optional.
    config:
        The project config module (needed for on-demand essay generation).
    on_generate_opinion:
        Callback taking a ticker and dispatching the full agent chain
        (typically a closure that calls ``ui.state.submit_job`` with
        ``main._process_ticker``). When None, the regenerate button warns
        instead of running.

    The Desk owns everything from the ticker header down. The caller owns
    the sidebar, global header, and destination navigation.
    """
    controls = controls or {}
    show_ma = controls.get("show_ma", True)
    show_rsi = controls.get("show_rsi", True)
    show_vol = controls.get("show_vol", False)

    # In-page nav across sections.
    C.section_nav([
        ("sec-thesis", "Thesis"),
        ("sec-risk", "Risk"),
        ("sec-quant", "Quant"),
        ("sec-filings", "Filings"),
        ("sec-macro", "Macro"),
        ("sec-evidence", "Evidence"),
    ])

    # --- SNAPSHOT (above the fold) --------------------------------------
    _render_sticky_header(ticker, prices, context)
    _render_price_chart(ticker, prices, show_ma=show_ma, show_rsi=show_rsi,
                        show_vol=show_vol)
    _render_summary_tiles(ticker, context, prices)

    st.markdown("---")
    _render_thesis_section(ticker, context, essay_mod=essay_mod, config=config,
                           on_generate_opinion=on_generate_opinion)

    st.markdown("---")
    _render_risk_section(ticker, context)

    st.markdown("---")
    _render_quant_section(ticker, prices, context)

    st.markdown("---")
    _render_filings_section(ticker, context)

    st.markdown("---")
    _render_macro_section(context, macro_snapshot)

    st.markdown("---")
    _render_evidence_section(context)


__all__ = ["render_desk"]

# D:\Ary Fund\ui\desk.py
