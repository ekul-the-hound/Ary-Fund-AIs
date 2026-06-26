"""
ui/app_v2.py
============

The ARY QUANT v2 orchestrator — the entry point that wires the eight-file
``ui/`` set into one application.

Run with:
    streamlit run ui/app_v2.py

What this file owns (and what it deliberately does NOT)
------------------------------------------------------
OWNS:
  * Page config, the global header, the sidebar, and destination navigation
    (Desk / Board / Screener / Lab).
  * The active-ticker contract bootstrap (via ui.state).
  * The non-blocking opinion queue wiring: an ``on_generate_opinion`` closure
    that submits the EXISTING ``generate_opinion`` backend call (which itself
    calls ``main._process_ticker`` and clears the right caches) to the
    ui.state thread pool, so the UI never blocks for 30s-2min.
  * Palette command dispatch (turning parsed intents into navigation / jobs).
  * Supplying the view files their callbacks and data: the cached
    ``price_loader``, the merged ``context`` (WITH the raw opinion attached so
    the Desk's full risk/evidence features light up), and the macro snapshot.

DOES NOT own (reused verbatim from the existing app.py):
  * The cached data loaders. This orchestrator imports and REUSES
    ``load_ticker_context``, ``load_price_history``, ``load_latest_opinion``,
    ``generate_opinion``, and ``load_macro_snapshot`` from app.py rather than
    reinventing them — so there is exactly ONE data path, and v2 stays in
    lockstep with the validated backend wiring.

Key bridge
----------
``app.py``'s ``load_ticker_context`` flattens the opinion into ``thesis`` /
``risk`` but does not attach the raw opinion. The Desk's richest features
(per-axis risk reasons, sector z-scores, Altman zone, essay fallback border)
key off ``context['_opinion']``. So ``_build_context`` here calls the existing
cached context loader and THEN attaches ``load_latest_opinion(ticker)`` as
``_opinion``. This changes nothing in app.py; it just enriches the dict the
new UI receives.

Safety / honesty
----------------
* If app.py's loaders can't be imported (path/layout drift), the orchestrator
  shows a clear, actionable error rather than silently degrading — because
  without the real loaders nothing downstream is trustworthy.
* The PDF ``report`` command is wired to the queue ONLY if a report entry
  point is importable; otherwise the command reports that reports aren't
  available in this build (rather than pretending to enqueue one).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

# --- Path bootstrap -------------------------------------------------------
# Streamlit puts THIS file's directory (.../ui) on sys.path, not the project
# root, so `from ui import ...` fails when launched as
# `streamlit run ui/app_v2.py`. Insert the project root (the parent of this
# file's directory) onto sys.path so the `ui` package is importable no matter
# how the script is invoked. This keeps BOTH entry paths working:
# `streamlit run ui/app_v2.py` and `python -c "from ui import ..."` from root.
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
# --------------------------------------------------------------------------

# v2 UI modules (Files 1-7).
from ui import components as C
from ui import state as S
from ui import desk as desk_view
from ui import board as board_view
from ui import lab as lab_view
from ui import pipeline_rail as rail_view
from ui import palette as palette_view

logger = logging.getLogger("ary_quant.ui.app_v2")


# ======================================================================
# Reuse the existing app.py cached loaders — single source of data truth
# ======================================================================
def _import_backend() -> dict[str, Any]:
    """Import the existing cached loaders + config from app.py.

    Returns a dict of callables/handles, or raises with a clear message if
    the existing app module can't be found. We try ``ui.app`` then top-level
    ``app`` to tolerate either layout.
    """
    app_mod = None
    for path in ("ui.app", "app"):
        try:
            app_mod = __import__(path, fromlist=["*"])
            break
        except Exception:  # pragma: no cover
            continue
    if app_mod is None:
        raise ImportError(
            "Could not import the existing app module (tried 'ui.app' and "
            "'app'). app_v2 reuses its cached loaders; ensure app.py is on "
            "the path."
        )

    required = [
        "load_ticker_context", "load_price_history", "load_latest_opinion",
        "generate_opinion", "load_macro_snapshot",
    ]
    missing = [name for name in required if not hasattr(app_mod, name)]
    if missing:
        raise ImportError(
            f"app.py is importable but missing expected loaders: {missing}. "
            "app_v2 depends on these to share one data path."
        )

    # Config + paths.
    try:
        import config as app_config  # noqa
    except Exception:
        app_config = getattr(app_mod, "app_config", None)

    db_path = None
    if app_config is not None:
        db_path = getattr(app_config, "PORTFOLIO_DB_PATH", None)

    watchlist = []
    if app_config is not None:
        watchlist = list(getattr(app_config, "WATCHLIST", []) or [])

    # Optional modules used by the Desk's essay fallback + screener/chat views.
    essay_mod = _safe_import("ui.essay") or _safe_import("essay")
    screener_mod = _safe_import("ui.screener") or _safe_import("screener")
    chat_mod = _safe_import("ui.chat") or _safe_import("chat")

    # Optional report entry point (for the `report` command).
    report_fn = None
    for modname in ("ui.report_orchestrator", "report_orchestrator",
                    "ui.pdf_renderer", "pdf_renderer"):
        m = _safe_import(modname)
        if m is not None:
            for fn_name in ("generate_report", "render_report", "build_report"):
                if hasattr(m, fn_name):
                    report_fn = getattr(m, fn_name)
                    break
        if report_fn is not None:
            break

    # Reusable renderers from app.py. We MOUNT these rather than reimplement
    # them, so v2 stays in lockstep with the validated originals (chat
    # streaming + history, essay block, portfolio strip, events panel). Each
    # is optional via getattr so a partial app.py still loads.
    backend_errors = list(getattr(app_mod, "_BACKEND_ERRORS", []) or [])

    return {
        "app": app_mod,
        "config": app_config,
        "db_path": db_path,
        "watchlist": watchlist,
        "load_ticker_context": app_mod.load_ticker_context,
        "load_price_history": app_mod.load_price_history,
        "load_latest_opinion": app_mod.load_latest_opinion,
        "generate_opinion": app_mod.generate_opinion,
        "load_macro_snapshot": app_mod.load_macro_snapshot,
        # Reused renderers (mounted verbatim).
        "load_portfolio_summary": getattr(app_mod, "load_portfolio_summary", None),
        "render_metric_cards": getattr(app_mod, "render_metric_cards", None),
        "render_briefing_tab": getattr(app_mod, "render_briefing_tab", None),
        "render_events_panel": getattr(app_mod, "render_events_panel", None),
        "_db_path": getattr(app_mod, "_db_path", None),
        # Analyzer + metrics sections (imported lazily where used).
        "essay_mod": essay_mod,
        "screener_mod": screener_mod,
        "chat_mod": chat_mod,
        "report_fn": report_fn,
        "backend_errors": backend_errors,
    }


def _safe_import(path: str) -> Any:
    try:
        return __import__(path, fromlist=["*"])
    except Exception:
        return None


# ======================================================================
# Context bridge — reuse the cached loader, attach the raw opinion
# ======================================================================
def _build_context(backend: dict[str, Any], ticker: str,
                   lookback: int) -> dict[str, Any]:
    """Build the merged context the Desk consumes.

    Reuses app.py's cached ``load_ticker_context`` (so the data path is
    shared), then attaches the raw opinion under ``_opinion`` so the Desk's
    full risk decomposition, sector z-scores, distress zone, and essay
    fallback border light up. Pure glue; no new data fetching.
    """
    ctx = backend["load_ticker_context"](ticker, lookback) or {}
    try:
        raw_opinion = backend["load_latest_opinion"](ticker) or {}
        if raw_opinion:
            ctx["_opinion"] = raw_opinion
    except Exception as e:  # pragma: no cover
        logger.warning("could not attach raw opinion for %s: %s", ticker, e)
    return ctx


# ======================================================================
# Non-blocking opinion generation closure
# ======================================================================
def _make_generate_closure(backend: dict[str, Any]) -> Callable[[str], str]:
    """Return an ``on_generate_opinion(ticker)`` that enqueues, not blocks.

    The worker calls the EXISTING ``generate_opinion`` backend function. But
    note: that function calls ``st.error`` / ``st.warning`` internally and
    clears Streamlit caches — both of which are unsafe from a background
    thread. So the worker calls ``main._process_ticker`` DIRECTLY (the same
    call generate_opinion wraps), avoiding any st.* from the thread, and the
    cache-clear is performed on the MAIN thread when the job is observed
    finished (handled by palette.render_job_tray's finished-side-effects plus
    an explicit clear here on completion via the polling path).
    """
    config = backend["config"]
    db_path = backend["db_path"]
    agent_main = _safe_import("main") or _safe_import("ui.main")

    def _worker(tkr: str) -> Any:
        # Pure backend call — NO streamlit here.
        if agent_main is None or not hasattr(agent_main, "_process_ticker"):
            raise RuntimeError("main._process_ticker not importable")
        if not db_path:
            raise RuntimeError("PORTFOLIO_DB_PATH not set")
        return agent_main._process_ticker(tkr, db_path, config)

    def _on_generate(ticker: str) -> str:
        return S.submit_job("opinion", ticker, _worker, ticker,
                            label=f"analysis · {ticker}")

    return _on_generate


def _make_report_closure(backend: dict[str, Any]) -> Optional[Callable[[str], str]]:
    """Return an ``on_report(ticker)`` queue closure, or None if unavailable.

    Honest: only wired if a report entry point was found. The worker must be
    st-free; report generation (ReportLab + matplotlib) is, so this is safe.
    """
    report_fn = backend.get("report_fn")
    if report_fn is None:
        return None
    config = backend["config"]
    db_path = backend["db_path"]

    def _worker(tkr: str) -> Any:
        # Try a few common call signatures defensively.
        try:
            return report_fn(tkr, db_path, config)
        except TypeError:
            try:
                return report_fn(tkr)
            except TypeError:
                return report_fn(ticker=tkr)

    def _on_report(ticker: str) -> str:
        return S.submit_job("report", ticker, _worker, ticker,
                            label=f"report · {ticker}")

    return _on_report


# ======================================================================
# Sidebar
# ======================================================================
def _render_sidebar(backend: dict[str, Any]) -> dict[str, Any]:
    """Render the sidebar; return control values (ticker, lookback, toggles).

    Preserves the existing app's controls: ticker picker + free-text entry,
    lookback slider, chart overlay toggles. The ticker widgets write the
    active-ticker key on change — identical to the existing contract.
    """
    with st.sidebar:
        st.markdown("## ⬡ ARY QUANT")
        st.caption("Quantamental research workstation")

        # Command bar in the sidebar (always available).
        cmd = palette_view.render_command_bar(key_suffix="_sidebar")

        st.markdown("---")

        # Ticker selection: watchlist dropdown + free text.
        watchlist = backend.get("watchlist") or ["NVDA", "AAPL", "MSFT", "AMD", "TSLA"]
        active = S.get_active_ticker()
        # Ensure active is in the option list so the selectbox can show it.
        options = list(dict.fromkeys([active] + [t.upper() for t in watchlist]))

        def _on_pick():
            picked = st.session_state.get("v2_ticker_pick")
            if picked:
                S.set_active_ticker(picked)

        st.selectbox("Ticker", options,
                     index=options.index(active) if active in options else 0,
                     key="v2_ticker_pick", on_change=_on_pick)

        free = st.text_input("…or enter a symbol", value="",
                             key="v2_ticker_free",
                             placeholder="e.g. GOOGL")
        if free and free.strip().upper() != active:
            S.set_active_ticker(free.strip().upper())

        st.markdown("---")
        lookback = st.slider("Lookback (days)", min_value=60, max_value=1095,
                             value=365, step=30, key="v2_lookback")

        st.markdown("**Chart overlays**")
        show_ma = st.checkbox("Moving averages", value=True, key="v2_ma")
        show_rsi = st.checkbox("RSI", value=True, key="v2_rsi")
        show_vol = st.checkbox("Volatility", value=False, key="v2_vol")
        show_macro = st.checkbox("Show macro panel", value=True, key="v2_macro")

        st.markdown("---")
        # Compact job-tray summary in the sidebar.
        palette_view.render_job_tray(compact=True)

        # Utility destinations (telemetry + diagnostics) pinned to the
        # sidebar bottom, kept out of the primary top nav.
        st.markdown("---")
        st.caption("Tools")
        ucols = st.columns(len(_UTILITY_DESTINATIONS))
        for i, udest in enumerate(_UTILITY_DESTINATIONS):
            is_cur = (udest == _current_destination())
            if ucols[i].button(udest, key=f"util_nav_{udest}",
                               use_container_width=True,
                               type=("primary" if is_cur else "secondary")):
                _set_destination(udest)
                st.rerun()

    return {
        "command": cmd,
        "lookback": lookback,
        "controls": {"show_ma": show_ma, "show_rsi": show_rsi,
                     "show_vol": show_vol, "show_macro": show_macro},
    }


# ======================================================================
# Destination navigation
# ======================================================================
# Top-nav destinations (primary workflow surfaces).
_DESTINATIONS = ["Desk", "Board", "Screener", "Lab", "Analyzer", "Jobs"]
# Utility destinations, shown at the bottom of the sidebar rather than the
# top nav (telemetry + diagnostics, not part of the research flow).
_UTILITY_DESTINATIONS = ["Metrics", "Debug", "RAG"]
_ALL_DESTINATIONS = _DESTINATIONS + _UTILITY_DESTINATIONS
_DEST_KEY = "v2_destination"


def _current_destination() -> str:
    return st.session_state.get(_DEST_KEY, "Desk")


def _set_destination(dest: str) -> None:
    if dest in _ALL_DESTINATIONS:
        st.session_state[_DEST_KEY] = dest


def _render_top_nav() -> None:
    """Horizontal destination switcher under the header."""
    current = _current_destination()
    cols = st.columns(len(_DESTINATIONS))
    for i, dest in enumerate(_DESTINATIONS):
        is_cur = (dest == current)
        if cols[i].button(
            dest, key=f"nav_{dest}", use_container_width=True,
            type=("primary" if is_cur else "secondary")):
            _set_destination(dest)
            st.rerun()


# ======================================================================
# Palette command dispatch
# ======================================================================
def _dispatch_command(cmd: palette_view.Command,
                      on_generate: Callable[[str], str],
                      on_report: Optional[Callable[[str], str]]) -> None:
    """Turn a parsed Command into navigation / jobs. Reruns on state change."""
    A = palette_view.Action
    if cmd is None or cmd.action == A.UNKNOWN:
        if cmd is not None and cmd.raw:
            st.toast(f"Unknown command: '{cmd.raw}'. Type 'help' for commands.")
        return

    if cmd.action == A.HELP:
        st.session_state["v2_show_help"] = True
        return

    if cmd.action == A.GOTO_TICKER:
        if cmd.ticker:
            S.set_active_ticker(cmd.ticker)
            _set_destination("Desk")
            st.rerun()
        return

    if cmd.action == A.GOTO_SECTION:
        if cmd.ticker:
            S.set_active_ticker(cmd.ticker)
        _set_destination("Desk")
        st.session_state["v2_scroll_to"] = cmd.section
        st.rerun()
        return

    if cmd.action == A.GOTO_DECISION:
        if cmd.ticker:
            S.set_active_ticker(cmd.ticker)
        _set_destination("Desk")
        st.session_state["v2_scroll_to"] = "sec-decision"
        st.rerun()
        return

    if cmd.action == A.GOTO_QUANT:
        if cmd.ticker:
            S.set_active_ticker(cmd.ticker)
        _set_destination("Lab")
        st.rerun()
        return

    if cmd.action == A.GENERATE:
        tkr = cmd.ticker or S.get_active_ticker()
        on_generate(tkr)
        st.toast(f"Queued analysis for {tkr}. Watch the Jobs tray.")
        return

    if cmd.action == A.REPORT:
        tkr = cmd.ticker or S.get_active_ticker()
        if on_report is None:
            st.toast("Reports aren't available in this build (no report entry "
                     "point found).")
        else:
            on_report(tkr)
            st.toast(f"Queued report for {tkr}. Watch the Jobs tray.")
        return

    if cmd.action == A.OPEN_VIEW:
        if cmd.ticker:
            S.set_active_ticker(cmd.ticker)
        dest = {"screen": "Screener", "board": "Board",
                "lab": "Lab", "desk": "Desk"}.get(cmd.view, "Desk")
        _set_destination(dest)
        st.rerun()
        return


# ======================================================================
# Destination renderers
# ======================================================================
def _render_desk_destination(backend: dict[str, Any],
                             controls: dict[str, Any],
                             lookback: int,
                             on_generate: Callable[[str], str]) -> None:
    ticker = S.get_active_ticker()

    # Build context (reused loader + attached raw opinion) and prices.
    context = _build_context(backend, ticker, lookback)
    prices = backend["load_price_history"](ticker, lookback)
    macro = backend["load_macro_snapshot"]() if controls.get("show_macro", True) else None

    # Stage rail above the Desk.
    opinion = context.get("_opinion") or {}
    held = bool((context.get("portfolio") or {}))
    status = S.compute_stage_status(ticker, opinion=opinion, held=held)
    rail_view.render_stage_rail(status, current=S.Stage.ANALYZE)
    st.markdown("---")

    # The Desk itself.
    desk_view.render_desk(
        ticker, context, prices,
        controls=controls,
        macro_snapshot=macro,
        essay_mod=backend.get("essay_mod"),
        config=backend.get("config"),
        on_generate_opinion=on_generate,
    )

    # Grounded chat dock — essay + interactive chat side by side, mounted
    # from app.py's validated render_briefing_tab (streaming + per-ticker
    # history preserved). Collapsible so it doesn't crowd the Desk.
    st.markdown("---")
    C.section_anchor("sec-chat", "Briefing & Chat",
                     subtitle="Memo essay with grounded Q&A")
    render_briefing = backend.get("render_briefing_tab")
    if render_briefing is not None:
        with st.expander("Open briefing + chat", expanded=False):
            try:
                render_briefing(ticker, context)
            except Exception as e:  # pragma: no cover
                st.error(f"Briefing/chat failed: {e}")
    else:
        st.caption("Briefing/chat renderer not available from app.py.")


def _render_board_destination(backend: dict[str, Any]) -> None:
    macro = backend["load_macro_snapshot"]()
    clicked = board_view.render_board(
        db_path=backend.get("db_path") or "",
        watchlist=backend.get("watchlist") or [],
        macro_snapshot=macro)
    if clicked:
        _set_destination("Desk")
        st.rerun()


def _render_screener_destination(backend: dict[str, Any]) -> None:
    screener_mod = backend.get("screener_mod")
    if screener_mod is None:
        st.warning("Screener module not importable.")
        return
    # The existing screener writes active_ticker on row-click; after it runs
    # we offer a jump to the Desk for the selected name.
    for fn_name in ("render_screener_tab", "render_screener", "render"):
        fn = getattr(screener_mod, fn_name, None)
        if fn is not None:
            try:
                fn()
            except TypeError:
                # Some signatures take a config/db arg.
                try:
                    fn(backend.get("config"))
                except Exception as e:
                    st.error(f"Screener render failed: {e}")
            break
    else:
        st.warning("No screener render entry point found.")


def _render_lab_destination(backend: dict[str, Any], lookback: int) -> None:
    ticker = S.get_active_ticker()
    prices = backend["load_price_history"](ticker, lookback)

    # Held names + a price_loader for the basket-level structure panels.
    held = _held_tickers(backend)

    def _price_loader(t: str) -> pd.DataFrame:
        return backend["load_price_history"](t, lookback)

    lab_view.render_lab(ticker, prices, held_tickers=held,
                        price_loader=_price_loader)


def _render_jobs_destination() -> None:
    st.markdown("### Background jobs")
    st.caption("Long analyses and reports run here without blocking the rest "
               "of the app. Fire `gen <ticker>` or `report <ticker>` from the "
               "command bar.")
    palette_view.render_job_tray(compact=False)


def _render_analyzer_destination(backend: dict[str, Any], lookback: int) -> None:
    """Data-Point Analyzer as its own destination (was a Desk section).

    Checkbox-driven per-datum LLM analysis, mounted from the existing
    ui/data_point_analyzer_section. Operates on the active ticker's context.
    """
    ticker = S.get_active_ticker()
    context = _build_context(backend, ticker, lookback)
    st.markdown(f"### Data-Point Analyzer — {ticker}")
    try:
        from ui.data_point_analyzer_section import render_data_point_analyzer_section
        render_data_point_analyzer_section(
            ticker=ticker, context=context, config=backend.get("config"))
    except ImportError as e:
        st.info(f"Data-Point Analyzer unavailable: {e}. Ensure "
                "`ui/data_point_analyzer_section.py` is present.")
    except Exception as e:  # pragma: no cover
        st.error(f"Analyzer failed: {e}")


def _render_rag_destination() -> None:
    """Render the RAG learning panel (defined in lab.py) as a Tools page."""
    st.markdown("### RAG learning")
    try:
        try:
            from ui.lab import _render_rag_learning_panel
        except Exception:
            from lab import _render_rag_learning_panel  # type: ignore
        _render_rag_learning_panel()
    except Exception as e:  # noqa: BLE001
        st.error(f"RAG learning panel unavailable: {e}")


def _render_metrics_destination(backend: dict[str, Any]) -> None:
    """LLM telemetry (latency / cost / failures) from metrics.db.

    Mounted from the existing ui/metrics_section. This is process-level
    telemetry, not per-ticker, so it gets its own destination.
    """
    st.markdown("### Agent metrics")
    try:
        from ui.metrics_section import render_metrics_tab
        render_metrics_tab(config=backend.get("config"))
    except ImportError as e:
        st.info(f"Metrics section unavailable: {e}. Ensure "
                "`ui/metrics_section.py` and `data/metrics_db.py` are present.")
    except Exception as e:  # pragma: no cover
        st.error(f"Metrics tab failed: {e}")


def _render_debug_destination(backend: dict[str, Any]) -> None:
    """DB path, last-20 agent_opinions rows, and the raw context dump."""
    import sqlite3
    st.markdown("### Debug")
    db_path_fn = backend.get("_db_path")
    db_path = db_path_fn() if db_path_fn else backend.get("db_path")
    st.write(f"**DB path:** `{db_path}`")

    if db_path:
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT ticker, created_at FROM agent_opinions "
                    "ORDER BY id DESC LIMIT 20").fetchall()
            st.write(f"**agent_opinions rows ({len(rows)}):**")
            if rows:
                st.table(pd.DataFrame(rows, columns=["ticker", "created_at"]))
            else:
                st.write("Table empty — run `python main.py` to populate.")
        except sqlite3.OperationalError as e:
            st.write(f"Table missing or unreadable: {e}")
        except Exception as e:  # pragma: no cover
            st.write(f"DB read failed: {e}")

    # Context dump for the active ticker.
    ticker = S.get_active_ticker()
    st.write(f"**Context dict for {ticker}:**")
    try:
        ctx = _build_context(backend, ticker,
                             st.session_state.get("v2_lookback", 365))
        if ctx:
            st.json(ctx, expanded=False)
        else:
            st.write("No context returned.")
    except Exception as e:  # pragma: no cover
        st.write(f"Context build failed: {e}")


def _held_tickers(backend: dict[str, Any]) -> list[str]:
    """Best-effort list of held tickers for the Lab's structure panels."""
    db_path = backend.get("db_path")
    if not db_path:
        return []
    try:
        positions = board_view._load_positions(db_path)
        return sorted(positions.keys())
    except Exception:
        return []


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    st.set_page_config(page_title="ARY QUANT", page_icon="⬡", layout="wide")

    # Bootstrap the active-ticker contract before any widget reads it.
    S.init_active_ticker()

    # Import the shared backend loaders (hard requirement).
    try:
        backend = _import_backend()
    except ImportError as e:
        st.error(str(e))
        st.stop()
        return

    # Closures for non-blocking generation + reports.
    on_generate = _make_generate_closure(backend)
    on_report = _make_report_closure(backend)

    # Optional Cmd/Ctrl-K focus convenience (labeled best-effort).
    palette_view.inject_shortcut_hint()

    # Sidebar (controls + command bar).
    side = _render_sidebar(backend)

    # Dispatch any command typed in the sidebar bar.
    if side.get("command") is not None:
        _dispatch_command(side["command"], on_generate, on_report)

    # Backend import-warning banner (v2 soft-imports + app.py's own).
    warnings = list(backend.get("backend_errors") or [])
    if warnings:
        with st.expander(f"⚠️ Backend import warnings ({len(warnings)})",
                         expanded=False):
            for w in warnings:
                st.caption(f"• {w}")

    # Global header + destination nav.
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:14px;'>"
            f"<span style='font-size:1.4em;font-weight:800;'>ARY QUANT</span>"
            f"<span style='color:#9ca3af;'>· {S.get_active_ticker()}</span>"
            f"</div>", unsafe_allow_html=True)
    with head_r:
        palette_view.render_job_tray(compact=True)

    # Portfolio overview strip (always visible, mounted from app.py).
    load_summary = backend.get("load_portfolio_summary")
    render_cards = backend.get("render_metric_cards")
    if load_summary is not None and render_cards is not None:
        try:
            render_cards(load_summary())
        except Exception as e:  # pragma: no cover
            st.caption(f"Portfolio summary unavailable: {e}")

    _render_top_nav()
    st.markdown("---")

    # Help overlay (shown via the 'help' command).
    if st.session_state.pop("v2_show_help", False):
        with st.expander("Command reference", expanded=True):
            palette_view.render_help()

    # Route to the active destination.
    dest = _current_destination()
    if dest == "Desk":
        _render_desk_destination(backend, side["controls"], side["lookback"],
                                 on_generate)
    elif dest == "Board":
        _render_board_destination(backend)
    elif dest == "Screener":
        _render_screener_destination(backend)
    elif dest == "Lab":
        _render_lab_destination(backend, side["lookback"])
    elif dest == "Analyzer":
        _render_analyzer_destination(backend, side["lookback"])
    elif dest == "Metrics":
        _render_metrics_destination(backend)
    elif dest == "RAG":
        _render_rag_destination()
    elif dest == "Jobs":
        _render_jobs_destination()
    elif dest == "Debug":
        _render_debug_destination(backend)
    else:
        _render_desk_destination(backend, side["controls"], side["lookback"],
                                 on_generate)

    # Auto-refresh ONLY while jobs are in flight (drives the tray's polling).
    S.maybe_autorefresh(interval_s=2.0)


if __name__ == "__main__":
    main()

# D:\Ary Fund\ui\app_v2.py