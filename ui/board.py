"""
ui/board.py
===========

Mission Control: a cross-sectional coverage board over the watchlist + book.

This is the at-a-glance monitoring surface the current dashboard completely
lacks — everything today is one-ticker-at-a-time. The board answers "what
needs my attention this morning?" by showing every covered name on one
screen: outlook, conviction, three-axis risk, day change, thesis age, and
sentiment, all in the shared visual encoding.

DATA STRATEGY — why this is cheap
---------------------------------
The naive approach (call ``pipeline.build_universe_context`` for the whole
universe) is a trap: that builder constructs a FULL per-ticker context in a
loop and can trigger SEC / yfinance / FRED backfill per cold ticker. For a
~560-name universe that would take minutes and hammer providers.

Instead the board reads only the CHEAP, ALREADY-PERSISTED layer:

    1. ``agent_opinions`` (portfolio.db): one SQL read returns the latest
       opinion per ticker. ``payload_json`` already carries outlook,
       confidence, risk_flags (levels/reasons), review scores, and
       bias_score — everything the matrix needs, with no LLM call and no
       network.
    2. ``get_portfolio_snapshot`` (portfolio.db): position rows (ticker,
       market_value, weight, unrealized P&L, sector) to flag held names and
       show day-level economics.

Full context (and any quant compute) is built LAZILY, only when a row is
clicked — which loads that single name into the Desk via the existing
``active_ticker`` contract. So the board renders fast and stays cheap no
matter how large the universe.

The board never writes anything except ``st.session_state['active_ticker']``
on a row click — identical to the screener's contract.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

from ui import components as C
from ui import state as S

logger = logging.getLogger("ary_quant.ui.board")


# ======================================================================
# Cheap persisted-layer readers
# ======================================================================
@st.cache_data(ttl=120, show_spinner=False)
def _load_latest_opinions(db_path: str) -> dict[str, dict[str, Any]]:
    """Return ``{ticker: latest_opinion_dict}`` from agent_opinions.

    One query, grouped to the newest row per ticker. Decodes ``payload_json``
    which holds the full merged opinion. Cached briefly so re-renders are
    instant; the cache is cleared when an opinion job completes (the caller
    handles that). Returns {} if the table is missing or the DB unreachable.
    """
    if not db_path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            # Newest row per ticker. id is monotonic (autoincrement), so
            # MAX(id) per ticker gives the latest opinion without a window
            # function (SQLite version-safe).
            rows = conn.execute(
                "SELECT t.ticker, t.payload_json, t.created_at "
                "FROM agent_opinions t "
                "JOIN (SELECT ticker, MAX(id) AS mid FROM agent_opinions "
                "      GROUP BY ticker) latest "
                "ON t.ticker = latest.ticker AND t.id = latest.mid"
            ).fetchall()
    except sqlite3.OperationalError:
        return {}  # table not created yet
    except Exception as e:  # pragma: no cover
        logger.warning("board: opinion read failed: %s", e)
        return {}

    for ticker, payload, created_at in rows:
        try:
            obj = json.loads(payload) or {}
            obj["_created_at"] = created_at
            out[str(ticker).upper()] = obj
        except Exception:
            continue
    return out


@st.cache_data(ttl=120, show_spinner=False)
def _load_positions(db_path: str) -> dict[str, dict[str, Any]]:
    """Return ``{ticker: position_row}`` from the portfolio snapshot.

    Uses PortfolioDB.get_portfolio_snapshot; passes market_data=None so the
    read is cheap (stored entry prices when no live feed). Returns {} on any
    failure so the board renders the opinion-only view.
    """
    if not db_path:
        return {}
    try:
        from data.portfolio_db import PortfolioDB
    except Exception:
        try:
            from portfolio_db import PortfolioDB  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning("board: portfolio_db unavailable: %s", e)
            return {}
    try:
        db = PortfolioDB(db_path=db_path)
        snap = db.get_portfolio_snapshot(market_data=None) or {}
    except Exception as e:  # pragma: no cover
        logger.warning("board: snapshot failed: %s", e)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for pos in (snap.get("positions") or []):
        if isinstance(pos, dict) and pos.get("ticker"):
            out[str(pos["ticker"]).upper()] = pos
    return out


# ======================================================================
# Row assembly
# ======================================================================
def _thesis_age_days(created_at: Any) -> Optional[int]:
    """Days since the opinion was written, or None if unparseable."""
    if not created_at or not isinstance(created_at, str):
        return None
    txt = created_at.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        try:
            dt = datetime.strptime(txt[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _build_rows(tickers: list[str],
                opinions: dict[str, dict[str, Any]],
                positions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble one matrix row per ticker from the cheap persisted layer.

    Every field tolerates absence — a name with no opinion yet still gets a
    row (so the analyst can see coverage gaps), just with em-dashes for the
    agent-derived columns.
    """
    rows: list[dict[str, Any]] = []
    for t in tickers:
        t = t.upper()
        op = opinions.get(t) or {}
        pos = positions.get(t) or {}

        rf = op.get("risk_flags") or {}
        levels = rf.get("levels") or {}
        review = op.get("review") or {}
        review_scores = review.get("scores") or {}

        rows.append({
            "ticker": t,
            "held": bool(pos),
            "weight": pos.get("portfolio_weight"),
            "day_chg_pct": pos.get("unrealized_pct"),  # proxy if no intraday feed
            "outlook": op.get("outlook"),
            "confidence": op.get("confidence"),
            "bias": op.get("bias_score"),
            "risk_combined": levels.get("combined"),
            "risk_fundamental": levels.get("fundamental"),
            "risk_macro": levels.get("macro"),
            "risk_market": levels.get("market"),
            "review_overall": review_scores.get("overall"),
            "thesis_age": _thesis_age_days(op.get("_created_at")),
            "sector": pos.get("sector") or op.get("sector"),
            "_has_opinion": bool(op),
        })
    return rows


# ======================================================================
# Rendering
# ======================================================================
def _portfolio_header(positions: dict[str, dict[str, Any]],
                      db_path: str) -> None:
    """Top tile strip: value, day P&L, combined book risk, cash, names."""
    try:
        from data.portfolio_db import PortfolioDB
    except Exception:
        try:
            from portfolio_db import PortfolioDB  # type: ignore
        except Exception:
            PortfolioDB = None  # type: ignore

    total_value = sum(
        p.get("market_value", 0) or 0 for p in positions.values()
    )
    total_pnl = sum(
        p.get("unrealized_pnl", 0) or 0 for p in positions.values()
    )
    cash = None
    concentration = "unknown"
    if PortfolioDB is not None and db_path:
        try:
            db = PortfolioDB(db_path=db_path)
            cash = float(db.get_cash())
            rm = db.get_risk_metrics(market_data=None) or {}
            concentration = str(rm.get("concentration", "unknown")).lower()
        except Exception:
            pass

    cols = st.columns(5)
    cols[0].metric("Names held", len(positions))
    cols[1].metric("Invested", C.fmt_big(total_value))
    cols[2].metric("Unrealized P&L", C.fmt_big(total_pnl),
                   delta=f"{total_pnl:+,.0f}" if C._is_num(total_pnl) else None)
    with cols[3]:
        st.markdown("<div style='font-size:0.78em;color:#9ca3af;'>BOOK RISK</div>",
                    unsafe_allow_html=True)
        st.markdown(C.badge_risk(concentration), unsafe_allow_html=True)
    cols[4].metric("Cash", C.fmt_big(cash) if cash is not None else "—")


def _macro_band(macro_snapshot: Optional[dict[str, Any]]) -> None:
    """A compact macro regime band under the portfolio tiles."""
    macro = macro_snapshot or {}
    if not macro:
        return
    rates = macro.get("interest_rates") or {}
    financial = macro.get("financial_conditions") or {}
    recession = macro.get("recession_signals") or {}

    rp = recession.get("recession_probability") or macro.get("recession_probability")
    if C._is_num(rp) and rp > 1.0:
        rp = rp / 100.0
    spread = (rates.get("yield_spread_10y2y") or macro.get("term_spread")
              or macro.get("yield_curve_spread"))
    vix = financial.get("vix") or macro.get("vix")
    inverted = rates.get("yield_curve_inverted")
    regime = ("Inverted" if inverted else "Normal") if isinstance(inverted, bool) \
        else (macro.get("regime") or macro.get("market_regime"))

    cols = st.columns(4)
    cols[0].metric("Recession Prob.", C.fmt_pct(rp) if C._is_num(rp) else "—")
    cols[1].metric("10Y-2Y", f"{spread:+.2f}%" if C._is_num(spread) else "—")
    cols[2].metric("VIX", f"{vix:.1f}" if C._is_num(vix) else "—")
    cols[3].metric("Curve", str(regime) if regime else "—")


def _coverage_matrix(rows: list[dict[str, Any]]) -> Optional[str]:
    """Render the sortable coverage matrix; return a clicked ticker or None.

    Uses st.dataframe with selection so a row click hands off to the Desk —
    the same mechanism the screener uses. Glyph/encoding columns are rendered
    as compact text (st.dataframe can't host HTML), so we translate levels
    and meters into terse symbols here: risk as letter+level, conviction as
    a 0-5 pip count, outlook as an arrow.
    """
    if not rows:
        st.info("No names to display. Add tickers to the watchlist or run the "
                "agent chain to populate coverage.")
        return None

    def _pip_str(conf: Any) -> str:
        if not C._is_num(conf):
            return "·····"
        n = int(round(max(0.0, min(1.0, float(conf))) * 5))
        return "●" * n + "·" * (5 - n)

    def _outlook_arrow(o: Any) -> str:
        return {"bullish": "▲ bull", "bearish": "▼ bear",
                "neutral": "■ neutral"}.get(str(o or "").lower(), "—")

    def _risk_str(level: Any) -> str:
        lv = str(level or "").upper()
        return lv if lv in ("LOW", "MEDIUM", "HIGH") else "—"

    display = []
    for r in rows:
        display.append({
            "Ticker": r["ticker"],
            "Held": "✓" if r["held"] else "",
            "Wt %": round(r["weight"], 1) if C._is_num(r["weight"]) else None,
            "Outlook": _outlook_arrow(r["outlook"]),
            "Conviction": _pip_str(r["confidence"]),
            "Bias": round(r["bias"], 2) if C._is_num(r["bias"]) else None,
            "Risk": _risk_str(r["risk_combined"]),
            "F": _risk_str(r["risk_fundamental"])[:1] if r["risk_fundamental"] else "",
            "M": _risk_str(r["risk_macro"])[:1] if r["risk_macro"] else "",
            "Mk": _risk_str(r["risk_market"])[:1] if r["risk_market"] else "",
            "Memo": round(r["review_overall"], 1) if C._is_num(r["review_overall"]) else None,
            "Age (d)": r["thesis_age"],
            "Sector": r["sector"] or "",
        })

    df = pd.DataFrame(display)

    col_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small", pinned=True),
        "Held": st.column_config.TextColumn("Held", width="small"),
        "Wt %": st.column_config.NumberColumn("Wt %", width="small", format="%.1f"),
        "Outlook": st.column_config.TextColumn("Outlook", width="small"),
        "Conviction": st.column_config.TextColumn("Conviction", width="small"),
        "Bias": st.column_config.NumberColumn("Bias", width="small", format="%.2f"),
        "Risk": st.column_config.TextColumn("Risk", width="small"),
        "F": st.column_config.TextColumn("F", width="small",
                                         help="Fundamental risk"),
        "M": st.column_config.TextColumn("M", width="small", help="Macro risk"),
        "Mk": st.column_config.TextColumn("Mk", width="small", help="Market risk"),
        "Memo": st.column_config.NumberColumn("Memo", width="small", format="%.1f",
                                              help="Self-review score /10"),
        "Age (d)": st.column_config.NumberColumn("Age (d)", width="small",
                                                 help="Days since opinion written"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
    }

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        height=min(640, 80 + 36 * len(df)),
    )

    # Resolve a row selection to a ticker.
    try:
        sel = event.selection.rows  # type: ignore[attr-defined]
        if sel:
            idx = sel[0]
            return str(df.iloc[idx]["Ticker"])
    except Exception:
        pass
    return None


# ======================================================================
# Filters
# ======================================================================
def _apply_filters(rows: list[dict[str, Any]],
                   *, only_held: bool, only_high_risk: bool,
                   only_covered: bool, sort_by: str) -> list[dict[str, Any]]:
    out = list(rows)
    if only_held:
        out = [r for r in out if r["held"]]
    if only_high_risk:
        out = [r for r in out if str(r["risk_combined"] or "").upper() == "HIGH"]
    if only_covered:
        out = [r for r in out if r["_has_opinion"]]

    # Sorting. None values sort last.
    def _key_conf(r):
        v = r["confidence"]
        return (v is None, -(v or 0))

    def _key_risk(r):
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return order.get(str(r["risk_combined"] or "").upper(), 3)

    def _key_age(r):
        v = r["thesis_age"]
        return (v is None, -(v or 0))

    def _key_memo(r):
        v = r["review_overall"]
        return (v is None, (v or 0))  # lowest memo score first (needs work)

    if sort_by == "Conviction":
        out.sort(key=_key_conf)
    elif sort_by == "Combined risk":
        out.sort(key=_key_risk)
    elif sort_by == "Thesis age":
        out.sort(key=_key_age)
    elif sort_by == "Memo score":
        out.sort(key=_key_memo)
    else:  # Ticker
        out.sort(key=lambda r: r["ticker"])
    return out


# ======================================================================
# PUBLIC ENTRY POINT
# ======================================================================
def render_board(
    *,
    db_path: str,
    watchlist: list[str],
    macro_snapshot: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Render Mission Control. Returns a ticker if a row was clicked, else None.

    Parameters
    ----------
    db_path:
        Portfolio SQLite path (config.PORTFOLIO_DB_PATH).
    watchlist:
        Tickers to show. Typically held names ∪ config.WATCHLIST ∪ any covered
        names found in agent_opinions. The caller assembles this; the board
        just renders whatever it's given (plus any covered names it finds).
    macro_snapshot:
        Optional global macro dashboard dict for the regime band.

    On a row click the board calls ``ui.state.set_active_ticker`` and returns
    the ticker so the orchestrator can switch to the Desk destination. The
    caller is responsible for the rerun + view switch.
    """
    opinions = _load_latest_opinions(db_path)
    positions = _load_positions(db_path)

    # Universe = given watchlist ∪ held ∪ covered (so coverage gaps AND
    # names you've analyzed but don't hold both show up).
    universe = set(t.upper() for t in (watchlist or []))
    universe |= set(positions.keys())
    universe |= set(opinions.keys())
    tickers = sorted(universe)

    # --- Header tiles + macro band --------------------------------------
    st.markdown("### Mission Control")
    _portfolio_header(positions, db_path)
    _macro_band(macro_snapshot)
    st.markdown("---")

    # --- Filters --------------------------------------------------------
    fcols = st.columns([1, 1, 1, 1.5, 1])
    only_held = fcols[0].toggle("Held only", value=False, key="board_held")
    only_high = fcols[1].toggle("High risk", value=False, key="board_highrisk")
    only_cov = fcols[2].toggle("Covered only", value=False, key="board_covered")
    sort_by = fcols[3].selectbox(
        "Sort by",
        ["Ticker", "Conviction", "Combined risk", "Thesis age", "Memo score"],
        key="board_sort")
    if fcols[4].button("🔄 Refresh", use_container_width=True, key="board_refresh"):
        _load_latest_opinions.clear()
        _load_positions.clear()
        st.rerun()

    rows = _build_rows(tickers, opinions, positions)
    rows = _apply_filters(rows, only_held=only_held, only_high_risk=only_high,
                          only_covered=only_cov, sort_by=sort_by)

    st.caption(
        f"{len(rows)} name(s) · {sum(1 for r in rows if r['_has_opinion'])} "
        f"with analysis · {sum(1 for r in rows if r['held'])} held. "
        "Click a row to open it in the Desk."
    )

    clicked = _coverage_matrix(rows)
    if clicked:
        S.set_active_ticker(clicked)
        return clicked
    return None


__all__ = ["render_board"]

# D:\Ary Fund\ui\board.py
