"""
ui/pipeline_rail.py
===================

The Research Pipeline rail + the Decision stage.

Two pieces:

1. THE STAGE RAIL
   A compact, always-visible funnel showing where the active ticker sits in
   the investment process: Screen -> Analyze -> Memo -> Review -> Decision.
   Completion is DERIVED FROM REAL PERSISTED STATE via
   ``ui.state.compute_stage_status`` (an opinion exists? a memo? a review
   score? a recorded thesis / open position?), so the rail reflects work
   that genuinely happened, not a UI guess. The review score is surfaced
   inline so a weak memo is visible without opening it.

2. THE DECISION STAGE
   The piece the current dashboard is missing entirely: the step that turns
   a thesis into a sized, recorded decision and closes the loop back to
   outcomes. It:
     * suggests a position size from conviction + realized volatility using
       the project's own quant (``volatility_target_size`` as the safe
       default for equities, with ``kelly_continuous`` shown alongside as a
       reference — fractional-Kelly, never full);
     * lets the analyst record the thesis (``portfolio_db.record_thesis``)
       and optionally open a position (``portfolio_db.add_position``), using
       the exact verified signatures;
     * shows any open theses / outcomes already on the books for this name,
       which is what later feeds the RAG learning loop.

CONTRACTS (verified against portfolio_db.py)
--------------------------------------------
    add_position(ticker, shares, entry_price, *, sector=None, thesis=None,
                 conviction="MEDIUM", position_type="LONG") -> dict
    record_thesis(ticker, *, thesis_text=None, essay_text=None,
                  score=None, stance=None, author=None, model=None,
                  entry_price=None, shares=None, ...) -> int
        NOTE: ``score`` MUST be in [0, 1] or record_thesis raises. Review
        scores are /10, so we divide by 10 before passing.
    get_cash() -> float

QUANT (verified against quant/)
-------------------------------
    kelly_continuous(expected_return, variance, risk_free_rate=0.0,
                     fractional=1.0) -> float (or dict)
    volatility_target_size(capital, target_vol, asset_vol) -> float
        (dollar or fraction depending on impl; treated defensively)

Writes are GUARDED: nothing touches the DB until the analyst explicitly
confirms in a form, and every write is wrapped so a failure surfaces as an
error message rather than a crash.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ui import components as C
from ui import state as S
from ui.state import Stage, STAGE_ORDER, STAGE_LABELS

logger = logging.getLogger("ary_quant.ui.pipeline_rail")


# ======================================================================
# Soft imports for the quant sizing helpers
# ======================================================================
def _try(fn: Callable[[], Any], label: str) -> Any:
    try:
        return fn()
    except Exception as e:  # pragma: no cover
        logger.warning("pipeline_rail: %s unavailable: %s", label, e)
        return None


_kelly_continuous = _try(
    lambda: __import__("quant.kelly", fromlist=["kelly_continuous"]).kelly_continuous,
    "quant.kelly.kelly_continuous")
if _kelly_continuous is None:
    _kelly_continuous = _try(
        lambda: __import__("kelly", fromlist=["kelly_continuous"]).kelly_continuous,
        "kelly.kelly_continuous")

_vol_target_size = _try(
    lambda: __import__("quant.position_sizing",
                       fromlist=["volatility_target_size"]).volatility_target_size,
    "quant.position_sizing.volatility_target_size")
if _vol_target_size is None:
    _vol_target_size = _try(
        lambda: __import__("position_sizing",
                           fromlist=["volatility_target_size"]).volatility_target_size,
        "position_sizing.volatility_target_size")


def _portfolio_db(db_path: str):
    """Instantiate PortfolioDB, tolerating package or flat layout."""
    try:
        from data.portfolio_db import PortfolioDB
    except Exception:
        try:
            from portfolio_db import PortfolioDB  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning("pipeline_rail: portfolio_db unavailable: %s", e)
            return None
    try:
        return PortfolioDB(db_path=db_path)
    except Exception as e:  # pragma: no cover
        logger.warning("pipeline_rail: PortfolioDB init failed: %s", e)
        return None


# ======================================================================
# THE STAGE RAIL
# ======================================================================
def render_stage_rail(status: S.StageStatus, *,
                      on_stage_click: Optional[Callable[[Stage], None]] = None,
                      current: Optional[Stage] = None) -> None:
    """Render the horizontal funnel for one ticker.

    Each stage shows a state glyph (done / current / pending) and, for
    Review, the numeric score inline. Clicking a stage (if a handler is
    given) lets the analyst jump to it. The rail is compact enough to sit
    above the Desk without stealing vertical space.
    """
    cols = st.columns(len(STAGE_ORDER))
    for i, stage in enumerate(STAGE_ORDER):
        done = status.is_done(stage)
        is_current = (stage == current)

        if done:
            glyph, color = "✓", C.RISK_COLORS["low"]
        elif is_current:
            glyph, color = "▸", "#3b82f6"
        else:
            glyph, color = "○", C._NEUTRAL_TEXT

        label = STAGE_LABELS[stage]
        extra = ""
        if stage == Stage.REVIEW and status.review_score is not None:
            sc = status.review_score
            sc_color = (C.RISK_COLORS["low"] if sc >= 8
                        else C.RISK_COLORS["medium"] if sc >= 6
                        else C.RISK_COLORS["high"])
            extra = (f"<div style='font-size:0.72em;color:{sc_color};"
                     f"font-variant-numeric:tabular-nums;'>{sc:.1f}/10</div>")
        if stage == Stage.DECISION and status.held:
            extra = (f"<div style='font-size:0.72em;color:{C.RISK_COLORS['low']};'>"
                     f"held</div>")

        with cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:4px 0;"
                f"border-bottom:2px solid {color if (done or is_current) else 'transparent'};'>"
                f"<div style='font-size:1.1em;color:{color};'>{glyph}</div>"
                f"<div style='font-size:0.82em;font-weight:600;"
                f"color:{'#e5e7eb' if (done or is_current) else C._NEUTRAL_TEXT};'>"
                f"{label}</div>{extra}</div>",
                unsafe_allow_html=True)
            if on_stage_click is not None:
                if st.button("↧", key=f"rail_jump_{stage.value}",
                             help=f"Go to {label}", use_container_width=True):
                    on_stage_click(stage)


# ======================================================================
# THE DECISION STAGE
# ======================================================================
def _suggested_size(
    *,
    capital: float,
    confidence: Optional[float],
    realized_vol_annual: Optional[float],
    target_vol: float,
) -> dict[str, Any]:
    """Compute a suggested position size from conviction + volatility.

    Strategy
    --------
    Primary (shown as the recommendation): VOLATILITY TARGETING. Size so the
    position contributes ~``target_vol`` annualized — the defensible default
    for equities and far more stable than raw Kelly. We then SCALE that by
    conviction (a 0.5..1.0 multiplier from confidence) so higher conviction
    earns a larger fraction of the vol budget.

    Reference (shown alongside, never the default): FRACTIONAL KELLY via
    ``kelly_continuous``. We map confidence to an expected excess return and
    use realized variance, then apply a 0.25 fractional multiplier (quarter-
    Kelly) because full Kelly is far too aggressive for discretionary equity
    bets. This is informational — it is not what pre-fills the order.

    Returns a dict with both fractions/dollar amounts and the inputs used,
    all defensive against missing data.
    """
    out: dict[str, Any] = {
        "vol_target_fraction": None,
        "vol_target_dollars": None,
        "kelly_fraction": None,
        "conviction_mult": None,
        "inputs": {
            "capital": capital,
            "confidence": confidence,
            "realized_vol_annual": realized_vol_annual,
            "target_vol": target_vol,
        },
    }
    if not C._is_num(realized_vol_annual) or realized_vol_annual <= 0:
        return out

    # Conviction multiplier: confidence 0..1 -> 0.5..1.0 of the vol budget.
    conf = float(confidence) if C._is_num(confidence) else 0.5
    conf = max(0.0, min(1.0, conf))
    conviction_mult = 0.5 + 0.5 * conf
    out["conviction_mult"] = conviction_mult

    # --- Volatility targeting (primary) --------------------------------
    if _vol_target_size is not None:
        try:
            raw = _vol_target_size(capital, target_vol, realized_vol_annual)
            # The helper may return a dollar amount or a fraction; normalize.
            if C._is_num(raw):
                raw = float(raw)
                if raw > 1.5 * capital:
                    raw = capital  # cap pathological outputs at full capital
                if 0 <= raw <= 1.0 and capital > 1.0:
                    # Looks like a fraction.
                    frac = raw
                    dollars = frac * capital
                else:
                    dollars = raw
                    frac = dollars / capital if capital else None
                # Apply conviction scaling.
                if frac is not None:
                    frac *= conviction_mult
                    dollars = frac * capital
                out["vol_target_fraction"] = frac
                out["vol_target_dollars"] = dollars
        except Exception as e:  # pragma: no cover
            logger.warning("vol_target_size failed: %s", e)
    # Manual fallback if the helper is unavailable or returned junk.
    if out["vol_target_fraction"] is None:
        frac = (target_vol / realized_vol_annual) * conviction_mult
        frac = max(0.0, min(1.0, frac))
        out["vol_target_fraction"] = frac
        out["vol_target_dollars"] = frac * capital

    # --- Fractional Kelly (reference only) -----------------------------
    if _kelly_continuous is not None:
        try:
            # Map confidence to a modest expected annual excess return.
            # A neutral 0.5 confidence -> 0 edge; 1.0 -> +12% expected.
            exp_ret = (conf - 0.5) * 0.24
            variance = float(realized_vol_annual) ** 2
            k = _kelly_continuous(exp_ret, variance, 0.0, fractional=0.25)
            if isinstance(k, dict):
                k = k.get("fraction") or k.get("kelly_fraction")
            if C._is_num(k):
                out["kelly_fraction"] = max(0.0, min(1.0, float(k)))
        except Exception as e:  # pragma: no cover
            logger.warning("kelly_continuous failed: %s", e)

    return out


def render_decision_stage(
    ticker: str,
    context: dict[str, Any],
    prices: pd.DataFrame,
    *,
    db_path: str,
    target_vol: float = 0.15,
) -> None:
    """Render the Decision stage: size, record, and close the loop.

    Parameters
    ----------
    ticker, context, prices:
        Active name, its merged context (with thesis/opinion), and price frame.
    db_path:
        Portfolio SQLite path. Writes go here; reads (cash, open theses,
        current position) come from here.
    target_vol:
        Annualized volatility budget for the sizing recommendation
        (default 15%, a reasonable single-name equity target).

    This stage performs WRITES, but only inside explicit confirm forms.
    """
    C.section_anchor("sec-decision", "Decision",
                     subtitle="Size the position, record the thesis, close the loop")

    db = _portfolio_db(db_path)
    if db is None:
        st.error("Portfolio DB unavailable — cannot size or record decisions "
                 "in this context.")
        return

    thesis = context.get("thesis") or {}
    opinion = context.get("_opinion") or {}
    confidence = thesis.get("confidence")
    if confidence is None:
        confidence = opinion.get("confidence")
    outlook = thesis.get("outlook") or opinion.get("outlook") or "neutral"

    # Current cash + any existing position for this name.
    try:
        cash = float(db.get_cash())
    except Exception:
        cash = 0.0
    current_pos = _current_position(db, ticker)

    # Realized vol (annualized) from the price window, for sizing.
    realized_vol = _realized_vol_annual(prices)
    last_price = _last_price(prices)

    # --- Recommendation card -------------------------------------------
    st.markdown("#### Suggested size")
    if not C._is_num(realized_vol):
        st.info("Need price history to compute a volatility-based size.")
    else:
        sizing = _suggested_size(
            capital=cash if cash > 0 else 100_000.0,
            confidence=confidence,
            realized_vol_annual=realized_vol,
            target_vol=target_vol)

        cols = st.columns(4)
        vt_frac = sizing["vol_target_fraction"]
        vt_dollars = sizing["vol_target_dollars"]
        cols[0].metric(
            "Vol-target size",
            C.fmt_pct(vt_frac) if C._is_num(vt_frac) else "—",
            help=f"Sized to ~{target_vol:.0%} annualized contribution, scaled "
                 f"by conviction. This is the recommendation.")
        cols[1].metric(
            "≈ Dollars",
            C.fmt_big(vt_dollars) if C._is_num(vt_dollars) else "—",
            help="Vol-target fraction × available cash.")
        cols[2].metric(
            "Realized σ (ann.)",
            C.fmt_pct(realized_vol),
            help="Annualized realized volatility over the loaded window.")
        kelly = sizing["kelly_fraction"]
        cols[3].metric(
            "¼-Kelly (ref.)",
            C.fmt_pct(kelly) if C._is_num(kelly) else "—",
            help="Quarter-Kelly from conviction-implied edge. Reference only — "
                 "not the recommendation. Full Kelly is too aggressive for "
                 "discretionary equity bets.")

        st.caption(
            "Recommendation uses volatility targeting (stable for equities); "
            "Kelly is shown only as a sanity reference. Conviction "
            f"({C.fmt_pct(confidence) if C._is_num(confidence) else '—'}) scales "
            "the vol budget between 50% and 100%.")

    st.markdown("---")

    # --- Current position context --------------------------------------
    if current_pos:
        st.markdown(
            f"**Current position:** {current_pos.get('shares', '—')} shares"
            + (f" @ {C.fmt_money(current_pos.get('avg_price'))} cost"
               if C._is_num(current_pos.get('avg_price')) else "")
        )

    # --- Record decision form (GUARDED WRITE) --------------------------
    st.markdown("#### Record decision")
    _render_record_form(
        ticker, db,
        outlook=outlook,
        confidence=confidence,
        last_price=last_price,
        suggested_dollars=(sizing["vol_target_dollars"]
                           if C._is_num(realized_vol) else None),
        essay_text=(thesis.get("essay") or opinion.get("essay")),
        review=(opinion.get("review") or context.get("review") or {}),
        model=(opinion.get("essay_meta") or {}).get("model"))

    # --- Open theses / outcome loop ------------------------------------
    st.markdown("---")
    st.markdown("#### Theses on the books")
    _render_thesis_history(db, ticker)


def _render_record_form(
    ticker: str, db: Any, *,
    outlook: str, confidence: Optional[float],
    last_price: Optional[float],
    suggested_dollars: Optional[float],
    essay_text: Optional[str],
    review: dict[str, Any],
    model: Optional[str],
) -> None:
    """The explicit confirm form that writes a thesis (+ optional position)."""
    stance_default = {"bullish": "LONG", "bearish": "SHORT"}.get(
        str(outlook).lower(), "WATCH")
    conviction_default = (
        "HIGH" if (C._is_num(confidence) and confidence >= 0.66)
        else "LOW" if (C._is_num(confidence) and confidence < 0.33)
        else "MEDIUM")

    suggested_shares = 0.0
    if C._is_num(suggested_dollars) and C._is_num(last_price) and last_price > 0:
        suggested_shares = round(suggested_dollars / last_price, 2)

    with st.form(key=f"decision_form_{ticker}"):
        c1, c2, c3 = st.columns(3)
        stance = c1.selectbox(
            "Stance", ["LONG", "SHORT", "WATCH"],
            index=["LONG", "SHORT", "WATCH"].index(stance_default),
            key=f"dec_stance_{ticker}")
        conviction = c2.selectbox(
            "Conviction", ["LOW", "MEDIUM", "HIGH"],
            index=["LOW", "MEDIUM", "HIGH"].index(conviction_default),
            key=f"dec_conv_{ticker}")
        author = c3.text_input("Author", value="", key=f"dec_author_{ticker}",
                               placeholder="your initials")

        also_open = st.checkbox(
            "Also open a position", value=False,
            key=f"dec_open_{ticker}",
            help="Writes a position to the book in addition to recording the "
                 "thesis. Leave off to record the thesis only.")
        c4, c5 = st.columns(2)
        shares = c4.number_input(
            "Shares", min_value=0.0, value=float(suggested_shares),
            step=1.0, key=f"dec_shares_{ticker}",
            help="Pre-filled from the vol-target recommendation; override freely.")
        entry_price = c5.number_input(
            "Entry price", min_value=0.0,
            value=float(last_price) if C._is_num(last_price) else 0.0,
            step=0.01, key=f"dec_price_{ticker}")

        note = st.text_area(
            "Thesis note (optional)", value="", height=80,
            key=f"dec_note_{ticker}",
            placeholder="One-line rationale for the record…")

        submitted = st.form_submit_button("📝 Record decision",
                                          use_container_width=True)

    if not submitted:
        return

    # --- Perform the guarded writes ------------------------------------
    # Review score is /10; record_thesis requires [0,1].
    score01 = None
    overall = (review or {}).get("scores", {}).get("overall")
    if C._is_num(overall):
        score01 = max(0.0, min(1.0, float(overall) / 10.0))

    try:
        thesis_id = db.record_thesis(
            ticker,
            thesis_text=(note or None),
            essay_text=essay_text,
            score=score01,
            stance=stance,
            author=(author or None),
            model=model,
            entry_price=(float(entry_price) if entry_price else None),
            shares=(float(shares) if (also_open and shares) else None),
        )
        st.success(f"Recorded thesis #{thesis_id} for {ticker} ({stance}).")
    except Exception as e:
        st.error(f"Failed to record thesis: {e}")
        return

    if also_open:
        if not (shares and entry_price):
            st.warning("Position not opened — shares and entry price are "
                       "required to open a position.")
        else:
            try:
                res = db.add_position(
                    ticker,
                    float(shares),
                    float(entry_price),
                    thesis=(note or None),
                    conviction=conviction,
                    position_type=("LONG" if stance != "SHORT" else "SHORT"),
                )
                st.success(f"Opened position: {shares} {ticker} @ "
                           f"{C.fmt_money(entry_price)}.")
            except Exception as e:
                st.error(f"Thesis recorded, but opening the position failed: {e}")

    # Invalidate cached board reads so the new state shows up immediately.
    try:
        from ui import board as _board
        _board._load_latest_opinions.clear()
        _board._load_positions.clear()
    except Exception:
        pass

    st.rerun()


def _render_thesis_history(db: Any, ticker: str) -> None:
    """Show recorded theses for this name + their outcomes if closed.

    This is the visible end of the loop: recorded theses, their scores, and
    (when closed) realized P&L. The RAG learning loop later indexes the
    high-scoring closed theses; surfacing them here makes that pipeline legible.
    """
    try:
        history = db.get_thesis_history(ticker, limit=10)
    except Exception as e:  # pragma: no cover
        st.caption(f"Could not load thesis history: {e}")
        return

    if not history:
        st.caption("No theses recorded for this name yet. Record one above to "
                   "start the outcome loop.")
        return

    for row in history:
        if not isinstance(row, dict):
            continue
        when = row.get("created_at") or row.get("date") or "—"
        stance = row.get("stance") or "—"
        score = row.get("score")
        pnl = row.get("realized_pnl") or row.get("pnl")
        status_txt = row.get("status") or ("closed" if pnl is not None else "open")

        score_str = ""
        if C._is_num(score):
            # Stored 0..1; display as /10 to match the review scale.
            score_str = f" · score {float(score)*10:.1f}/10"
        pnl_str = ""
        if C._is_num(pnl):
            color = C.OUTLOOK_COLORS["bullish"] if pnl >= 0 else C.OUTLOOK_COLORS["bearish"]
            pnl_str = (f" · <span style='color:{color};'>P&L "
                       f"{pnl:+,.0f}</span>")

        st.markdown(
            f"<div style='font-size:0.88em;padding:4px 0;"
            f"border-bottom:1px solid {C._HAIRLINE};'>"
            f"<b>{str(when)[:10]}</b> · {stance} · <i>{status_txt}</i>"
            f"{score_str}{pnl_str}</div>",
            unsafe_allow_html=True)


# ======================================================================
# Small data helpers
# ======================================================================
def _current_position(db: Any, ticker: str) -> Optional[dict[str, Any]]:
    """Return the current open position dict for a ticker, or None."""
    for getter in ("get_position", "get_open_position"):
        fn = getattr(db, getter, None)
        if fn is not None:
            try:
                pos = fn(ticker)
                if pos:
                    return pos
            except Exception:
                continue
    # Fallback: scan a positions list.
    fn = getattr(db, "get_positions", None) or getattr(db, "list_positions", None)
    if fn is not None:
        try:
            for p in (fn() or []):
                if isinstance(p, dict) and str(p.get("ticker", "")).upper() == ticker.upper():
                    return p
        except Exception:
            pass
    return None


def _last_price(prices: pd.DataFrame) -> Optional[float]:
    if isinstance(prices, pd.DataFrame) and not prices.empty and "close" in prices.columns:
        try:
            return float(prices["close"].dropna().iloc[-1])
        except Exception:
            return None
    return None


def _realized_vol_annual(prices: pd.DataFrame) -> Optional[float]:
    if not isinstance(prices, pd.DataFrame) or prices.empty or "close" not in prices.columns:
        return None
    s = prices["close"].dropna()
    if len(s) < 20:
        return None
    try:
        rets = np.log(s / s.shift(1)).dropna()
        return float(rets.std() * np.sqrt(252))
    except Exception:
        return None


__all__ = ["render_stage_rail", "render_decision_stage", "Stage"]

# D:\Ary Fund\ui\pipeline_rail.py
