"""
ui/lab.py
=========

The Quant Lab: a home for the project's quantitative model suite.

Design decision — MOUNT, don't rebuild
---------------------------------------
The repo already contains a fully-built, validated quant surface in
``playground.py``: ``render_playground_tab(ticker, prices)`` exposes four
polished per-ticker models (HMM regime, Hurst exponent, Ornstein-Uhlenbeck
mean reversion, GBM Monte Carlo), each anchored on real price data with its
own parameter widgets, spinners, charts, and a ``_QUANT_ERRORS`` soft-import
guard. It was simply never wired into the old app's tab list — exactly the
gap the project notes flagged.

So this module does NOT reimplement those renderers (that would duplicate
working code and invite divergence). Instead it:

    1. MOUNTS ``playground.render_playground_tab`` as the per-ticker bench,
       passing the active ticker + price window so it stays in sync with the
       rest of the dashboard.
    2. Adds a MODEL CATALOG that names the model families and is honest about
       which modules are surfaced in the UI today vs. which exist in the
       codebase but aren't yet wired (clearly labeled "available · not yet in
       UI" — never presented as if they were live).
    3. Adds three genuinely NEW PORTFOLIO-LEVEL panels that the per-ticker
       playground structurally cannot provide, because they operate on a
       BASKET of names rather than one: RMT correlation filtering, the MST
       correlation network, and HRP allocation. These use
       ``quant.rmt.rmt_from_prices``, ``quant.mst.compute_mst``, and
       ``quant.hrp.hrp_from_prices`` — each verified to take a wide price
       DataFrame (assets in columns) and return the defensive
       ``{available, reason, ...}`` contract requiring >= 4 assets.

What's a future enhancement (labeled as such in the UI)
-------------------------------------------------------
Surfacing the remaining exotic modules (sandpile, omori, lyapunov,
ergodicity, wavelet_regimes, lempel_ziv, rmt single-name spectrum, etc.) as
first-class per-ticker panels is future work. The catalog lists them so the
analyst knows the math exists, but does not fake a UI for them.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

from ui import components as C
from ui import state as S

logger = logging.getLogger("ary_quant.ui.lab")


# ======================================================================
# Soft imports — the Lab renders even if a module or playground is mid-edit
# ======================================================================
def _try(fn: Callable[[], Any], label: str) -> Any:
    try:
        return fn()
    except Exception as e:  # pragma: no cover - defensive UI boundary
        logger.warning("lab: %s unavailable: %s", label, e)
        return None


_render_playground = _try(
    lambda: __import__("ui.playground", fromlist=["render_playground_tab"]).render_playground_tab,
    "ui.playground.render_playground_tab")
# Fall back to a top-level playground import path if not under ui/.
if _render_playground is None:
    _render_playground = _try(
        lambda: __import__("playground", fromlist=["render_playground_tab"]).render_playground_tab,
        "playground.render_playground_tab")

_rmt_from_prices = _try(
    lambda: __import__("quant.rmt", fromlist=["rmt_from_prices"]).rmt_from_prices,
    "quant.rmt.rmt_from_prices")
if _rmt_from_prices is None:
    _rmt_from_prices = _try(
        lambda: __import__("rmt", fromlist=["rmt_from_prices"]).rmt_from_prices, "rmt")

_compute_mst = _try(
    lambda: __import__("quant.mst", fromlist=["compute_mst"]).compute_mst,
    "quant.mst.compute_mst")
if _compute_mst is None:
    _compute_mst = _try(
        lambda: __import__("mst", fromlist=["compute_mst"]).compute_mst, "mst")

_hrp_from_prices = _try(
    lambda: __import__("quant.hrp", fromlist=["hrp_from_prices"]).hrp_from_prices,
    "quant.hrp.hrp_from_prices")
if _hrp_from_prices is None:
    _hrp_from_prices = _try(
        lambda: __import__("hrp", fromlist=["hrp_from_prices"]).hrp_from_prices, "hrp")

# Extended-models panels (the previously-grey 'available' modules).
_render_extended = _try(
    lambda: __import__("ui.lab_models", fromlist=["render_extended_models"]).render_extended_models,
    "ui.lab_models.render_extended_models")
if _render_extended is None:
    _render_extended = _try(
        lambda: __import__("lab_models", fromlist=["render_extended_models"]).render_extended_models,
        "lab_models.render_extended_models")


# ======================================================================
# Model catalog — honest about wired vs. available
# ======================================================================
# Each entry: (display name, one-line "what it tells you", status).
# status in {"ticker" (live per-ticker), "portfolio" (live basket-level),
#            "available" (exists in repo, not yet surfaced)}.
_CATALOG: dict[str, list[tuple[str, str, str]]] = {
    "Regime": [
        ("HMM Regime", "Hidden bull/neutral/bear states + transition matrix", "ticker"),
        ("Wavelet Regimes", "Multi-scale regime decomposition", "ticker"),
        ("RMT Spectrum", "Eigenvalue structure vs. random benchmark", "portfolio"),
    ],
    "Path / Pricing": [
        ("GBM Monte Carlo", "Simulated price paths + terminal VaR/ES", "ticker"),
        ("Black-Scholes", "European option pricing & greeks", "ticker"),
        ("SABR", "Stochastic-vol smile calibration", "ticker"),
        ("Longstaff-Schwartz", "American option LSM pricing", "ticker"),
    ],
    "Mean Reversion / Memory": [
        ("Ornstein-Uhlenbeck", "Mean-reversion speed, half-life, bands", "ticker"),
        ("Hurst Exponent", "Trending vs. mean-reverting vs. random walk", "ticker"),
        ("Lyapunov", "Sensitivity to initial conditions (chaos)", "ticker"),
        ("Ergodicity", "Time- vs. ensemble-average divergence", "ticker"),
        ("Lempel-Ziv", "Complexity / compressibility of the series", "ticker"),
    ],
    "Risk / Sizing": [
        ("VaR / ES", "Tail loss at a confidence level", "ticker"),
        ("Realized Volatility", "Annualized realized & GARCH-forecast vol", "ticker"),
        ("Kelly", "Growth-optimal position fraction", "ticker"),
        ("Poisson jumps", "Jump detection + arrival intensity", "ticker"),
    ],
    "Structure (portfolio)": [
        ("MST Network", "Minimum spanning tree of correlations", "portfolio"),
        ("HRP Allocation", "Hierarchical risk-parity weights", "portfolio"),
        ("RMT Filter", "De-noised correlation matrix", "portfolio"),
    ],
    "Self-organized criticality": [
        ("Sandpile", "Avalanche-size power law", "ticker"),
        ("Omori", "Aftershock decay after large moves", "ticker"),
    ],
}

_STATUS_BADGE = {
    "ticker": ("#16a34a", "per-ticker · live"),
    "portfolio": ("#3b82f6", "portfolio · live"),
    "available": ("#6b7280", "available · not yet in UI"),
}


def _render_catalog() -> None:
    """The left-rail model catalog. Read-only; communicates capability."""
    st.markdown("#### Model catalog")
    st.caption("What the platform can compute. Green/blue = wired into this "
               "Lab; grey = implemented in the codebase, not yet surfaced.")
    for family, models in _CATALOG.items():
        st.markdown(f"**{family}**")
        for name, blurb, status in models:
            color, label = _STATUS_BADGE.get(status, _STATUS_BADGE["available"])
            st.markdown(
                f"<div style='margin:2px 0 8px 0;'>"
                f"<span style='font-size:0.9em;font-weight:600;'>{name}</span> "
                f"<span style='background:{color};color:white;font-size:0.66em;"
                f"padding:1px 6px;border-radius:8px;vertical-align:middle;'>{label}</span>"
                f"<div style='font-size:0.78em;color:#9ca3af;'>{blurb}</div></div>",
                unsafe_allow_html=True)


# ======================================================================
# Portfolio-basket panels (NEW — not in the per-ticker playground)
# ======================================================================
def _wide_prices_for_basket(
    tickers: list[str],
    price_loader: Optional[Callable[[str], pd.DataFrame]],
) -> tuple[Optional[pd.DataFrame], list[str]]:
    """Build a wide close-price DataFrame (assets in columns) for a basket.

    ``price_loader(ticker) -> OHLCV DataFrame`` is supplied by the caller
    (the app already has a cached price loader). We pull each name's close,
    align on the common date index, and drop names with no data. Returns
    (wide_df_or_None, missing_tickers). Requires the loader; without it we
    can't fetch a basket and return (None, all).
    """
    if price_loader is None:
        return None, list(tickers)
    series: dict[str, pd.Series] = {}
    missing: list[str] = []
    for t in tickers:
        try:
            df = price_loader(t)
            if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
                series[t.upper()] = df["close"].dropna()
            else:
                missing.append(t)
        except Exception:
            missing.append(t)
    if len(series) < 2:
        return None, missing
    wide = pd.DataFrame(series).dropna(how="any")
    return (wide if not wide.empty else None), missing


def _render_structure_panel(
    held_tickers: list[str],
    price_loader: Optional[Callable[[str], pd.DataFrame]],
) -> None:
    """RMT + MST + HRP over the held basket — genuinely portfolio-level."""
    st.markdown("### Portfolio structure")
    st.caption(
        "Correlation structure and risk-parity allocation across your held "
        "names. These operate on the whole basket, so they live here rather "
        "than on the single-name Desk. Requires at least 4 held names with "
        "overlapping price history."
    )

    if len(held_tickers) < 4:
        st.info(
            f"Portfolio-structure models need ≥ 4 held names; you have "
            f"{len(held_tickers)}. Add positions to unlock RMT / MST / HRP."
        )
        return

    if price_loader is None:
        st.warning("No price loader provided — cannot assemble the basket "
                   "price matrix in this context.")
        return

    with st.spinner("Loading basket prices…"):
        wide, missing = _wide_prices_for_basket(held_tickers, price_loader)

    if wide is None or wide.shape[1] < 4 or len(wide) < 31:
        st.warning(
            "Not enough overlapping price history across the basket "
            f"(got {0 if wide is None else wide.shape[1]} aligned names, "
            f"{0 if wide is None else len(wide)} common days; need ≥ 4 names "
            "and > N+30 days)."
        )
        return

    if missing:
        st.caption(f"Excluded (no/short data): {', '.join(missing)}")

    tab_hrp, tab_mst, tab_rmt = st.tabs(
        ["HRP allocation", "MST network", "RMT filter"])

    # --- HRP -----------------------------------------------------------
    with tab_hrp:
        if _hrp_from_prices is None:
            st.warning("HRP module unavailable.")
        else:
            res = _hrp_from_prices(wide)
            if not res.get("available", True):
                st.warning(res.get("reason", "HRP unavailable."))
            else:
                weights = res.get("weights")
                cols = st.columns(3)
                pr = res.get("portfolio_return")
                pv = res.get("portfolio_vol")
                sh = res.get("sharpe")
                cols[0].metric("Ann. return", C.fmt_pct(pr) if C._is_num(pr) else "—")
                cols[1].metric("Ann. vol", C.fmt_pct(pv) if C._is_num(pv) else "—")
                cols[2].metric("Sharpe", f"{sh:.2f}" if C._is_num(sh) else "—")
                if weights is not None:
                    try:
                        wdf = weights.sort_values(ascending=False).rename("weight")
                        wdf_disp = (wdf * 100).round(2)
                        st.bar_chart(wdf_disp)
                        st.dataframe(
                            wdf_disp.reset_index().rename(
                                columns={"index": "Ticker", "weight": "Weight %"}),
                            use_container_width=True, hide_index=True)
                    except Exception as e:  # pragma: no cover
                        st.caption(f"Could not render weights: {e}")
                st.caption("Hierarchical risk parity allocates by correlation "
                           "clusters, avoiding the instability of mean-variance "
                           "inversion. Compare these to your actual weights.")

    # --- MST -----------------------------------------------------------
    with tab_mst:
        if _compute_mst is None:
            st.warning("MST module unavailable.")
        else:
            returns = wide.pct_change().dropna()
            res = _compute_mst(returns)
            if not res.get("available", True):
                st.warning(res.get("reason", "MST unavailable."))
            else:
                _plot_mst(res)
                st.caption("The minimum spanning tree keeps only the strongest "
                           "correlation links. Central nodes drive co-movement; "
                           "peripheral nodes diversify.")

    # --- RMT -----------------------------------------------------------
    with tab_rmt:
        if _rmt_from_prices is None:
            st.warning("RMT module unavailable.")
        else:
            res = _rmt_from_prices(wide)
            if not res.get("available", True):
                st.warning(res.get("reason", "RMT unavailable."))
            else:
                n_factors = res.get("n_significant_factors") or res.get("n_factors")
                var_explained = res.get("variance_explained")
                cols = st.columns(2)
                if C._is_num(n_factors):
                    cols[0].metric("Significant factors", int(n_factors),
                                   help="Eigenvalues above the Marchenko-Pastur "
                                        "noise band — genuine common factors.")
                if C._is_num(var_explained):
                    cols[1].metric("Variance explained",
                                   C.fmt_pct(var_explained) if var_explained <= 1
                                   else f"{var_explained:.1f}%")
                st.caption("Random Matrix Theory separates real correlation "
                           "structure from noise. Eigenvalues inside the "
                           "Marchenko-Pastur band are statistically "
                           "indistinguishable from random.")


def _plot_mst(res: dict[str, Any]) -> None:
    """Best-effort MST network plot from a compute_mst result.

    The result shape varies by implementation; we look for an edge list and
    node positions, falling back to a textual edge table if we can't draw a
    network. Never raises.
    """
    edges = res.get("edges") or res.get("mst_edges")
    if not edges:
        # Show whatever summary stats are present.
        length = res.get("total_length") or res.get("mst_length") or res.get("length")
        if C._is_num(length):
            st.metric("MST total length", f"{length:.3f}")
        st.caption("Edge list not exposed by this MST implementation; showing "
                   "summary only.")
        return
    try:
        import plotly.graph_objects as go
        import networkx as nx  # optional

        g = nx.Graph()
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                a, b = e[0], e[1]
                w = e[2] if len(e) > 2 else 1.0
                g.add_edge(str(a), str(b), weight=float(w) if C._is_num(w) else 1.0)
            elif isinstance(e, dict):
                g.add_edge(str(e.get("source")), str(e.get("target")),
                           weight=float(e.get("weight", 1.0)))
        pos = nx.spring_layout(g, seed=42)
        edge_x, edge_y = [], []
        for a, b in g.edges():
            edge_x += [pos[a][0], pos[b][0], None]
            edge_y += [pos[a][1], pos[b][1], None]
        node_x = [pos[n][0] for n in g.nodes()]
        node_y = [pos[n][1] for n in g.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=1, color="#475569"),
                                 hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=list(g.nodes()), textposition="top center",
            marker=dict(size=14, color="#3b82f6"), hoverinfo="text"))
        fig.update_layout(showlegend=False, height=420,
                          margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:  # pragma: no cover
        logger.warning("MST plot fallback: %s", e)
        st.caption("Network plot unavailable (networkx not installed); "
                   "showing edge list.")
        st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)


# ======================================================================
# PUBLIC ENTRY POINT
# ======================================================================
def _rag_job_worker(mode: str, db_path: str) -> dict:
    """Background-job worker: run the RAG learning loop via the scheduler.

    Reuses RefreshScheduler's already-wired loop (curator/auditor/indexer).
    Touches no Streamlit — safe for the job queue. ``mode`` is "learning" or
    "audit". Returns the scheduler method's result dict (or an error dict).
    """
    try:
        try:
            from data.refresh_scheduler import RefreshScheduler
        except Exception:
            from refresh_scheduler import RefreshScheduler  # type: ignore
        sched = RefreshScheduler(db_path=db_path)
        if mode == "audit":
            return sched._run_rag_audit()
        return sched._run_rag_learning()
    except Exception as e:  # noqa: BLE001
        return {"rows": 0, "note": f"error: {type(e).__name__}: {e}"}


def _render_rag_learning_panel() -> None:
    """Buttons to run the RAG learning loop / audit as background jobs."""
    st.markdown("#### RAG learning loop")
    st.caption(
        "Runs the self-indexing pipeline (auditor → scorer → curator → "
        "indexer) over recently-closed theses. P&L-weighted: winners get "
        "indexed into the vector store, losers get demoted."
    )

    # Job-queue handles (optional — fall back to blocking if unavailable).
    try:
        from ui import state as S
    except Exception:
        try:
            import state as S  # type: ignore
        except Exception:
            S = None

    _db = "data/hedgefund.db"

    st.info(
        "Note: this needs the 768-dim embedder. If ARY_EMBED_BACKEND=ollama "
        "isn't set (with Ollama running), indexing degrades. A result like "
        "`{'rows': 0, 'note': 'no_recently_closed_hook'}` means there were no "
        "closed theses to learn from — informative, not an error.",
        icon="ℹ️",
    )

    col_a, col_b = st.columns(2)
    run_learning = col_a.button("Run learning cycle", key="rag_run_learning",
                                use_container_width=True)
    run_audit = col_b.button("Run audit", key="rag_run_audit",
                             use_container_width=True)

    if S is None:
        # No job queue — run inline (blocks the UI briefly).
        if run_learning or run_audit:
            mode = "audit" if run_audit else "learning"
            with st.spinner(f"Running RAG {mode}… (this can take a minute)"):
                result = _rag_job_worker(mode, _db)
            st.write(result)
        return

    job_key = "rag_job_id"
    result_key = "rag_job_result"

    if run_learning:
        job_id = S.submit_job("rag", "LEARNING", _rag_job_worker, "learning",
                              _db, label="RAG learning cycle")
        st.session_state[job_key] = job_id
        st.session_state.pop(result_key, None)
    if run_audit:
        job_id = S.submit_job("rag", "AUDIT", _rag_job_worker, "audit",
                              _db, label="RAG audit")
        st.session_state[job_key] = job_id
        st.session_state.pop(result_key, None)

    # Poll the active job (same pattern as the analyzer section).
    if st.session_state.get(job_key):
        job = S.get_job(st.session_state[job_key])
        if job is not None:
            if job.state in (S.JobState.QUEUED, S.JobState.RUNNING):
                st.warning("RAG job running in the background — this panel "
                           "updates as it progresses.")
                S.maybe_autorefresh()
            elif job.state == S.JobState.DONE:
                st.session_state[result_key] = job.result
                st.session_state.pop(job_key, None)
            elif job.state == S.JobState.ERROR:
                st.error(f"RAG job failed: "
                         f"{getattr(job, 'error', 'unknown error')}")
                st.session_state.pop(job_key, None)

    if st.session_state.get(result_key) is not None:
        st.success("RAG job complete:")
        st.write(st.session_state[result_key])
    elif not st.session_state.get(job_key):
        st.caption("No RAG job has been run yet this session.")


def render_lab(
    ticker: str,
    prices: pd.DataFrame,
    *,
    held_tickers: Optional[list[str]] = None,
    price_loader: Optional[Callable[[str], pd.DataFrame]] = None,
) -> None:
    """Render the Quant Lab.

    Parameters
    ----------
    ticker:
        Active ticker (drives the per-ticker bench).
    prices:
        OHLCV DataFrame for the active ticker (same frame the Desk uses).
    held_tickers:
        The book's held names, for the portfolio-structure panels. Optional;
        without ≥ 4, those panels show an unlock hint.
    price_loader:
        ``(ticker) -> OHLCV DataFrame`` callable (the app's cached loader),
        used to assemble the basket price matrix for RMT/MST/HRP. Optional;
        without it, the structure panels explain they can't fetch the basket.

    Layout: a left catalog rail + a main area with two sub-views — the
    per-ticker bench (mounted playground) and the portfolio-structure panel.
    """
    held_tickers = held_tickers or []

    st.markdown(f"## 🧪 Quant Lab — {ticker}")
    st.caption(
        "Probabilistic, stochastic, and complexity models on your real price "
        "data. Per-ticker models are mounted from the validated playground; "
        "portfolio-structure models operate across the whole book."
    )

    rail, main = st.columns([1, 3])
    with rail:
        _render_catalog()

    with main:
        view = st.radio(
            "View",
            ["Per-ticker bench", "Extended models", "Portfolio structure"],
            horizontal=True,
            key="lab_view")
        st.markdown("---")

        if view == "Per-ticker bench":
            if _render_playground is None:
                st.error(
                    "The quant playground module could not be imported. "
                    "Expected ``playground.render_playground_tab`` (top-level) "
                    "or ``ui/playground.py``. Check that playground.py is on "
                    "the path."
                )
            else:
                # Mount the existing, validated 4-model bench verbatim.
                _render_playground(ticker, prices)
        elif view == "Extended models":
            if _render_extended is None:
                st.error(
                    "Extended models module could not be imported. Expected "
                    "``ui/lab_models.py`` with render_extended_models."
                )
            else:
                _render_extended(ticker, prices)
        else:
            _render_structure_panel(held_tickers, price_loader)


__all__ = ["render_lab"]

# D:\Ary Fund\ui\lab.py
