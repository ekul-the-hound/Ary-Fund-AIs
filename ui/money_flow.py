"""
ui/money_flow.py
================
Streamlit destination for the SEC **Money Flow Graph**.

Renders the D3 network defined in ``ui/money_flow_template.html`` with real
data assembled by :func:`data.money_graph.build_money_graph`. Money flows
``source -> target`` (buyer / capital-holder -> supplier / holding).

Wiring
------
``ui/app_v2.py`` calls :func:`render_money_flow_destination(backend)` (added by
``fix_add_flow_destination.py``). ``backend`` is the shared loader dict; we only
need the DB paths + config from it.

The graph build is cached (``st.cache_data``) keyed on scope + edge sources +
market-fetch toggle, so panning/zooming in the iframe never re-queries. Hit
**Rebuild** to force a fresh pull after ingesting new filings.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger("ary_quant.ui.money_flow")

_TEMPLATE_PATH = Path(__file__).with_name("money_flow_template.html")
_INJECT_TOKEN = "__ARY_GRAPH_JSON__"
_DEFAULT_SOURCES = ["ownership", "f13f", "news", "usaspending"]
_SOURCE_LABELS = {
    "ownership": "13D / 13G ownership",
    "f13f": "13F institutional holdings",
    "news": "News (suspected)",
    "usaspending": "Federal contracts",
}


# ---------------------------------------------------------------------------
# Import shims — tolerate both `data.x` and top-level `x` layouts
# ---------------------------------------------------------------------------
def _import_builder():
    try:
        from data.money_graph import build_money_graph  # type: ignore
        return build_money_graph
    except Exception:
        from money_graph import build_money_graph  # type: ignore
        return build_money_graph


def _universe_tickers() -> list[str]:
    for path in ("data.money_graph", "money_graph"):
        try:
            mod = __import__(path, fromlist=["universe_tickers"])
            return list(mod.universe_tickers())
        except Exception:  # noqa: BLE001
            continue
    return []


def _make_market_data(market_db: str):
    """Best-effort MarketData handle for node enrichment. None if unavailable."""
    for path in ("data.market_data", "market_data"):
        try:
            mod = __import__(path, fromlist=["MarketData"])
            return mod.MarketData(db_path=market_db)
        except Exception:  # noqa: BLE001
            continue
    return None


def _active_ticker() -> Optional[str]:
    for path in ("ui.state", "state"):
        try:
            mod = __import__(path, fromlist=["get_active_ticker"])
            return mod.get_active_ticker()
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def _resolve_db_paths(backend: dict[str, Any]) -> tuple[str, str]:
    cfg = backend.get("config")
    portfolio_db = (backend.get("db_path")
                    or getattr(cfg, "PORTFOLIO_DB_PATH", None)
                    or "data/portfolio.db")
    market_db = (getattr(cfg, "MARKET_DB_PATH", None)
                 or getattr(cfg, "HEDGEFUND_DB_PATH", None)
                 or getattr(cfg, "MARKET_DATA_DB_PATH", None)
                 or "data/hedgefund.db")
    return str(portfolio_db), str(market_db)


# ---------------------------------------------------------------------------
# Cached build (returns plain dict -> cache-friendly)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def _build_graph(scope_tickers: Optional[tuple[str, ...]],
                 sources: tuple[str, ...],
                 fetch_market: bool,
                 portfolio_db: str,
                 market_db: str,
                 drop_isolated: bool = False,
                 max_nodes: Optional[int] = None) -> dict:
    build_money_graph = _import_builder()
    md = _make_market_data(market_db) if fetch_market else None
    return build_money_graph(
        tickers=list(scope_tickers) if scope_tickers else None,
        portfolio_db_path=portfolio_db,
        market_db_path=market_db,
        market_data=md,
        edge_sources=sources,
        drop_isolated=drop_isolated,
        max_nodes=max_nodes,
    )


@st.cache_data(show_spinner=False)
def _load_template() -> Optional[str]:
    try:
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.error("money_flow | template read failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def render_money_flow_destination(backend: dict[str, Any]) -> None:
    st.markdown("### Money Flow Graph")
    st.caption("Company-to-company capital & supply relationships from SEC "
               "ownership filings, institutional holdings, and news. "
               "Money flows **source → target** (buyer/holder → supplier/holding).")

    portfolio_db, market_db = _resolve_db_paths(backend)

    # ---- controls -----------------------------------------------------
    c1, c2, c3 = st.columns([1.1, 1.4, 0.9])
    with c1:
        scope_mode = st.selectbox(
            "Scope",
            ["Portfolio + analyzed", "Active ticker neighborhood",
             "Full universe (~600)", "Custom list"],
            key="mf_scope_mode")
    with c2:
        sources = st.multiselect(
            "Edge sources",
            options=_DEFAULT_SOURCES,
            default=_DEFAULT_SOURCES,
            format_func=lambda s: _SOURCE_LABELS.get(s, s),
            key="mf_sources")
    with c3:
        fetch_market = st.checkbox("Fetch market caps", value=True,
                                   key="mf_fetch_market",
                                   help="Use yfinance (24h cached) to size nodes "
                                        "by market cap. For large scopes this "
                                        "only fetches the companies that survive "
                                        "the connected/cap filter, not all 600.")

    scope_tickers: Optional[tuple[str, ...]] = None
    is_broad = scope_mode == "Full universe (~600)"

    if scope_mode == "Active ticker neighborhood":
        tk = _active_ticker()
        if tk:
            scope_tickers = (tk.upper(),)
            st.caption(f"Neighborhood of **{tk.upper()}** — its institutional "
                       "holders and any in-scope counterparties.")
        else:
            st.info("No active ticker set; showing portfolio scope.")
    elif is_broad:
        uni = _universe_tickers()
        scope_tickers = tuple(uni) or None
        if not uni:
            st.warning("Could not load the universe list "
                       "(`data/universe.py`); falling back to portfolio scope.")
        else:
            st.caption(f"Scanning the full **{len(uni)}-name** universe. A "
                       "money-flow graph is about relationships, so only "
                       "companies with actual flows are drawn — expect far "
                       "fewer than 600 rendered.")
    elif scope_mode == "Custom list":
        raw = st.text_input("Tickers (comma-separated)", value="",
                            placeholder="AAPL, NVDA, MSFT, BRK.B",
                            key="mf_custom")
        toks = tuple(t.strip().upper() for t in raw.split(",") if t.strip())
        scope_tickers = toks or None

    # Legibility guards — default ON for broad scopes.
    g1, g2, _ = st.columns([1.0, 1.0, 1.0])
    with g1:
        connected_only = st.checkbox(
            "Connected only", value=is_broad, key="mf_connected",
            help="Hide companies with no money flows. Strongly recommended for "
                 "broad scopes — otherwise the graph is mostly isolated dots.")
    with g2:
        max_nodes = st.slider(
            "Max nodes", min_value=50, max_value=600,
            value=300 if is_broad else 400, step=25, key="mf_maxnodes",
            help="Hard cap for browser performance. Keeps the highest-degree, "
                 "then largest-cap entities. A force layout gets unusable past "
                 "a few hundred nodes.")

    if st.button("Rebuild graph", key="mf_rebuild"):
        _build_graph.clear()

    if not sources:
        st.warning("Select at least one edge source.")
        return

    # ---- build --------------------------------------------------------
    with st.spinner("Assembling money-flow graph…"):
        try:
            graph = _build_graph(scope_tickers, tuple(sources), bool(fetch_market),
                                 portfolio_db, market_db,
                                 bool(connected_only), int(max_nodes))
        except Exception as e:  # noqa: BLE001
            logger.exception("money_flow | build failed")
            st.error(f"Graph build failed: {e}")
            return

    meta = graph.get("meta", {})
    n_nodes, n_edges = len(graph.get("nodes", [])), len(graph.get("edges", []))

    # ---- provenance line ---------------------------------------------
    if meta.get("source") == "demo":
        st.info("Showing **demo data** — no structured relationships found for "
                "this scope. Populate ownership/13F via "
                "`SECFetcher.refresh_ticker_filings(<ticker>)`, or widen scope, "
                "then **Rebuild**.")
    else:
        prov = meta.get("provenance", {})
        parts = ", ".join(f"{k}: {v}" for k, v in prov.items() if v)
        st.caption(f"**{n_nodes}** entities · **{n_edges}** flows · sources — "
                   f"{parts or 'none matched'}")

    # ---- render -------------------------------------------------------
    template = _load_template()
    if template is None:
        st.error(f"Visualization template missing: expected "
                 f"`{_TEMPLATE_PATH}`. Ensure `ui/money_flow_template.html` "
                 f"is present.")
        return
    if _INJECT_TOKEN not in template:
        st.error("Template is missing the data injection token; re-copy "
                 "`ui/money_flow_template.html`.")
        return

    payload = json.dumps(graph, default=str).replace("</", "<\\/")
    html = template.replace(_INJECT_TOKEN, payload)
    components.html(html, height=860, scrolling=False)

    with st.expander("Data sources & how to enrich this graph", expanded=False):
        st.markdown(
            "**Live edges** come from tables the SEC fetcher already populates in "
            "`data/hedgefund.db`:\n"
            "- `ownership_filings` (13D/13G) → *filer → company* capital edges\n"
            "- `f13f_holdings` (13F-HR) → *institution → holding* edges\n"
            "- Form 4 net insider buying is folded into each node's tooltip.\n\n"
            "To populate them: run "
            "`SECFetcher().refresh_ticker_filings('NVDA')` for each ticker "
            "(this runs XBRL + Form 4 + 13D/13G ingestion), then **Rebuild**.\n\n"
            "**Suspected** (dashed) and **federal-contract** edges are wired as "
            "hooks — see the docstrings in `data/money_graph.py` "
            "(`_edges_from_news`, `_edges_from_usaspending`). They activate "
            "automatically once a headline/awards pair-table exists."
        )


# Back-compat alias.
render = render_money_flow_destination

# D:\Ary Fund\ui\money_flow.py