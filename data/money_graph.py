"""
data/money_graph.py
===================
Build the SEC **money-flow graph** consumed by ``ui/money_flow.py`` (and the
standalone ``ui/money_flow_template.html`` D3 visualization).

The graph is ``{"nodes": [...], "edges": [...], "meta": {...}}`` where money
flows ``source -> target`` (buyer / capital-holder -> supplier / holding).

Data sources (all already present in this project)
--------------------------------------------------
COMPANY NODES
  * ``positions``        (portfolio.db)   -> tracked tickers + sector
  * ``agent_opinions``   (portfolio.db)   -> risk_score, confidence, thesis
  * ``MarketData.get_fundamentals``       -> market_cap (node size), sector, revenue
  * ``sec_filings`` / ``xbrl_facts`` (hedgefund.db) -> filings_count, latest form, revenue fallback

REAL EDGES (structured, free, already ingested by sec_fetcher)
  * ``ownership_filings`` (13D / 13G)  -> filer_name -> ticker      [type=inferred]
  * ``f13f_holdings``     (13F-HR)      -> institution -> issuer     [type=inferred]
  * ``insider_transactions`` (Form 4)  -> folded into node metadata (net insider $)

BEST-EFFORT EDGES (documented hooks — see ``_edges_from_news`` /
``_edges_from_usaspending``). These sources ARE ingested by ``sentiment_news``
and ``geo_supply``, but they currently land in the registry / ``data_points``
as single-entity signals, not company-pair rows. Until a pair-extraction step
writes a table, these contribute 0 and say so in ``meta.provenance``.

Design
------
* Pure CPU + SQLite reads. The only optional network is ``MarketData`` (yfinance,
  24 h cached); pass ``market_data=None`` to stay fully offline / for unit tests.
* Every source is best-effort: a missing table or empty result contributes
  nothing and never raises.
* If the live graph has **no edges** (SEC tables not yet populated) and
  ``demo_fallback=True``, a small bundled demo graph is returned with
  ``meta.source == "demo"`` so the UI is never blank.

Usage
-----
    from data.money_graph import build_money_graph
    g = build_money_graph(market_db_path="data/hedgefund.db",
                          portfolio_db_path="data/portfolio.db")
    # g["nodes"], g["edges"], g["meta"]
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Sector -> the palette the template keys off. Anything not here renders grey.
_KNOWN_SECTORS = {
    "Technology", "Financials", "Healthcare", "Consumer",
    "Energy", "Industrials", "Communication",
}
# yfinance sector strings -> our palette buckets.
_SECTOR_ALIASES = {
    "Financial Services": "Financials",
    "Communication Services": "Communication",
    "Consumer Cyclical": "Consumer",
    "Consumer Defensive": "Consumer",
    "Basic Materials": "Industrials",
    "Utilities": "Energy",
    "Real Estate": "Financials",
}

_RISK_LEVEL_SCORE = {"HIGH": 0.8, "MEDIUM": 0.5, "MODERATE": 0.5, "LOW": 0.25}

# filer_name values that are actually document/form labels, not institutions.
# The subject-company submissions feed often puts the form type in the field
# the ingest reads, so these leak in as fake "institution" nodes. Skip them.
_JUNK_FILER = re.compile(
    r"^(none|n/?a|null|form\s+sc\s+13[dg](/a)?|sc\s+13[dg](/a)?|"
    r"sec\s+schedule\s+13[dg](/a)?|schedule\s+13[dg](/a)?|13[dg](/a)?|"
    r"ms\s+initial|initial|amendment|amended)$", re.I)


def _is_junk_filer(name: str) -> bool:
    s = (name or "").strip()
    return (not s) or (_JUNK_FILER.match(s) is not None)


# Optional institution-name canonicalizer (collapses filer spelling variants
# and resolves 13F CIKs to the same node). Graph still works if it's absent.
try:
    from data.filer_canonical import canonical_name as _fc_name, name_for_cik as _fc_cik
except Exception:  # noqa: BLE001
    try:
        from filer_canonical import canonical_name as _fc_name, name_for_cik as _fc_cik  # type: ignore
    except Exception:  # noqa: BLE001
        _fc_name = None
        _fc_cik = None


def _canonicalize(name: str) -> str:
    if _fc_name is not None:
        try:
            out = _fc_name(name)
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    return " ".join(str(name or "").split()).title()

# Fallback company names used ONLY when MarketData didn't supply one (offline /
# no yfinance). Lets 13F-issuer and news co-mention matching work for common
# mega-caps without a network call. Real names from get_fundamentals win.
_TICKER_NAME_HINTS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "GOOG": "Alphabet", "AMZN": "Amazon", "META": "Meta Platforms",
    "TSLA": "Tesla", "AVGO": "Broadcom", "AMD": "Advanced Micro Devices",
    "QCOM": "Qualcomm", "INTC": "Intel", "MU": "Micron", "TSM": "Taiwan Semiconductor",
    "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe", "NFLX": "Netflix",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "BRK.B": "Berkshire Hathaway",
    "V": "Visa", "MA": "Mastercard", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "PFE": "Pfizer", "JNJ": "Johnson & Johnson", "MRK": "Merck", "ABBV": "AbbVie",
    "XOM": "Exxon Mobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "General Electric",
    "HON": "Honeywell", "WMT": "Walmart", "COST": "Costco", "HD": "Home Depot",
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo", "DIS": "Disney",
}


# =====================================================================
# Public entry point
# =====================================================================
def build_money_graph(
    *,
    tickers: Optional[Iterable[str]] = None,
    portfolio_db_path: str = "data/portfolio.db",
    market_db_path: str = "data/hedgefund.db",
    market_data: Any = None,
    edge_sources: Iterable[str] = ("ownership", "f13f", "news", "usaspending"),
    max_institutions: int = 50,
    drop_isolated: bool = False,
    max_nodes: Optional[int] = None,
    demo_fallback: bool = True,
) -> dict:
    """Assemble the money-flow graph. See module docstring for sources.

    Parameters
    ----------
    tickers:
        Explicit company scope. If ``None``, derived from ``positions`` +
        distinct ``agent_opinions`` tickers. Pass ``universe_tickers()`` for
        the full ~600-name S&P + large/mid-cap universe.
    market_data:
        Optional ``market_data.MarketData`` instance used to enrich nodes
        (market cap / sector / revenue). Nodes are built cheaply first; market
        data is fetched ONLY for the pruned survivor set, so a 600-name scope
        does not trigger 600 network calls. Pass ``None`` for offline.
    drop_isolated:
        If True, drop company nodes with no edges. Essential for large scopes:
        a money-flow graph is about relationships, so edgeless companies are
        noise. Recommended True whenever ``tickers`` is the full universe.
    max_nodes:
        Optional hard cap on rendered nodes (keeps highest-degree, then
        largest-cap). Guards browser performance — a force layout gets
        unusable past a few hundred nodes.
    edge_sources:
        Which edge builders to run. ``ownership`` and ``f13f`` are the real,
        structured sources; ``news`` / ``usaspending`` are best-effort hooks.
    """
    edge_sources = set(edge_sources)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    prov: dict[str, int] = {}

    scope = _resolve_scope(tickers, portfolio_db_path, market_db_path)

    # ---- company nodes (BARE — no network, cheap even for ~600 tickers) --
    for tk in scope:
        node = _build_company_node(
            tk, portfolio_db_path, market_db_path, None)
        nodes[node["id"]] = node

    # ---- edges ---------------------------------------------------------
    # For large scopes, don't let ownership pricing make a network call per
    # ticker; the pct-scaled proxy carries stake size fine. Small scopes still
    # get real share*price amounts.
    edge_md = market_data if len(scope) <= 60 else None
    if "ownership" in edge_sources:
        prov["13D/13G"] = _edges_from_ownership(
            market_db_path, nodes, edges, edge_md, scope)
    if "f13f" in edge_sources:
        prov["13F"] = _edges_from_13f(market_db_path, nodes, edges, scope,
                                      max_institutions)
    if "news" in edge_sources:
        prov["news"] = _edges_from_news(market_db_path, nodes, edges, scope)
    if "usaspending" in edge_sources:
        prov["contracts"] = _edges_from_usaspending(
            market_db_path, nodes, edges, scope)

    scope_size = len(scope)

    # ---- fallback so the UI is never blank -----------------------------
    if not edges and demo_fallback:
        demo = _demo_graph()
        demo["meta"]["source"] = "demo"
        demo["meta"]["note"] = (
            "No structured relationships found. Populate ownership/13F via "
            "SECFetcher.refresh_ticker_filings(<ticker>), or widen scope."
        )
        return demo

    # ---- prune (connected-only + cap) BEFORE enrichment ----------------
    nodes, edges = _prune_graph(nodes, edges, drop_isolated, max_nodes)

    # ---- enrich ONLY survivors with market caps (bounded network) ------
    if market_data is not None:
        for n in list(nodes.values()):
            _enrich_node_market(n, market_data)

    meta = {
        "source": "live",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope_size": scope_size,
        "rendered_nodes": len(nodes),
        "rendered_edges": len(edges),
        "provenance": prov,
    }
    return {"nodes": list(nodes.values()), "edges": edges, "meta": meta}


# =====================================================================
# Universe
# =====================================================================
def universe_tickers() -> list[str]:
    """The curated ~600-name US universe (S&P 500 + large/mid caps), matching
    the screener. Empty list if the module can't be imported."""
    for path in ("data.universe", "universe"):
        try:
            mod = __import__(path, fromlist=["US_UNIVERSE"])
            return [str(t).upper() for t in getattr(mod, "US_UNIVERSE", ())]
        except Exception:  # noqa: BLE001
            continue
    return []


# =====================================================================
# Pruning + market enrichment
# =====================================================================
def _prune_graph(nodes: dict, edges: list, drop_isolated: bool,
                 max_nodes: Optional[int]) -> tuple[dict, list]:
    """Keep the graph legible: optionally drop edgeless nodes, then cap total
    node count by (degree, size). Institutions ride along via their edges."""
    def _degree(edge_list):
        d: dict[str, int] = {}
        for e in edge_list:
            d[e["source"]] = d.get(e["source"], 0) + 1
            d[e["target"]] = d.get(e["target"], 0) + 1
        return d

    deg = _degree(edges)
    keep = {nid for nid in nodes if deg.get(nid, 0) > 0} if drop_isolated \
        else set(nodes)

    if max_nodes and len(keep) > max_nodes:
        keep = set(sorted(
            keep, key=lambda nid: (deg.get(nid, 0), nodes[nid].get("size", 0)),
            reverse=True)[:max_nodes])

    nodes2 = {nid: n for nid, n in nodes.items() if nid in keep}
    edges2 = [e for e in edges
              if e["source"] in keep and e["target"] in keep]

    # capping may have orphaned nodes whose only edges were pruned
    if drop_isolated and edges2:
        deg2 = _degree(edges2)
        nodes2 = {nid: n for nid, n in nodes2.items() if deg2.get(nid, 0) > 0}
        edges2 = [e for e in edges2
                  if e["source"] in nodes2 and e["target"] in nodes2]
    return nodes2, edges2


def _enrich_node_market(node: dict, market_data) -> None:
    """Fetch market cap / name / sector for a single (already-surviving) node.
    Skips institutions. Best-effort; failures leave the bare node intact."""
    if market_data is None:
        return
    if (node.get("metadata") or {}).get("entity_kind") == "institution":
        return
    tk = node.get("ticker")
    if not tk:
        return
    try:
        f = market_data.get_fundamentals(tk) or {}
    except Exception as e:  # noqa: BLE001
        logger.debug("money_graph | enrich failed %s: %s", tk, e)
        return
    if f.get("name"):
        node["name"] = f["name"]
    sec = _norm_sector(f.get("sector"))
    if sec:
        node["sector"] = sec
    mc = _usd_to_b((f.get("overview") or {}).get("market_cap"))
    rev = _usd_to_b((f.get("financials") or {}).get("revenue"))
    if mc:
        node["metadata"]["market_cap"] = round(mc, 2)
        node["size"] = round(mc, 3)
    if rev:
        node["metadata"]["revenue"] = round(rev, 2)


# =====================================================================
# Scope
# =====================================================================
def _resolve_scope(tickers, portfolio_db_path, market_db_path) -> list[str]:
    if tickers:
        return sorted({str(t).upper().strip() for t in tickers if t})
    found: set[str] = set()
    # positions
    for sql, db in (
        ("SELECT DISTINCT ticker FROM positions", portfolio_db_path),
        ("SELECT DISTINCT ticker FROM agent_opinions", portfolio_db_path),
    ):
        for row in _safe_query(db, sql):
            if row and row[0]:
                found.add(str(row[0]).upper().strip())
    return sorted(found)


# =====================================================================
# Company nodes
# =====================================================================
def _build_company_node(ticker, portfolio_db_path, market_db_path,
                        market_data) -> dict:
    ticker = ticker.upper().strip()
    name, sector, market_cap_b, revenue_b = ticker, "", None, None

    # yfinance fundamentals (24 h cached inside MarketData) — optional.
    if market_data is not None:
        try:
            f = market_data.get_fundamentals(ticker) or {}
            name = f.get("name") or name
            sector = _norm_sector(f.get("sector"))
            mc = (f.get("overview") or {}).get("market_cap")
            rev = (f.get("financials") or {}).get("revenue")
            market_cap_b = _usd_to_b(mc)
            revenue_b = _usd_to_b(rev)
        except Exception as e:  # noqa: BLE001
            logger.debug("money_graph | fundamentals failed %s: %s", ticker, e)

    # sector fallback from positions table
    if not sector:
        row = _safe_query(
            portfolio_db_path,
            "SELECT sector FROM positions WHERE ticker=? LIMIT 1", (ticker,))
        if row and row[0] and row[0][0]:
            sector = _norm_sector(row[0][0])

    # name fallback (offline / no yfinance) so 13F + news matching still works
    if name == ticker and ticker in _TICKER_NAME_HINTS:
        name = _TICKER_NAME_HINTS[ticker]

    # revenue fallback from XBRL facts ($B)
    if revenue_b is None:
        row = _safe_query(
            market_db_path,
            "SELECT value FROM xbrl_facts WHERE ticker=? AND "
            "(concept LIKE '%Revenue%' OR concept LIKE '%Sales%') "
            "ORDER BY period_end DESC LIMIT 1", (ticker,))
        if row and row[0] and row[0][0]:
            revenue_b = _usd_to_b(row[0][0])

    # node size: market cap -> revenue -> default
    size = market_cap_b or revenue_b or 50.0

    # opinion-derived risk + provenance confidence
    risk_score, confidence, thesis = _opinion_signals(portfolio_db_path, ticker)

    # filings metadata
    fcount, form_type, latest = _filings_meta(market_db_path, ticker)

    # net insider $ (Form 4) folded into metadata
    insider_net = _insider_net_usd(market_db_path, ticker)

    return {
        "id": ticker,
        "name": name,
        "ticker": ticker,
        "size": round(float(size), 3),
        "sector": sector or "Technology",
        "confidence": round(float(confidence), 3),
        "risk_score": round(float(risk_score), 3),
        "metadata": {
            "market_cap": round(market_cap_b, 2) if market_cap_b else None,
            "revenue": round(revenue_b, 2) if revenue_b else None,
            "form_type": form_type,
            "filings_count": fcount,
            "latest_filing_date": latest,
            "insider_net_usd": insider_net,
            "thesis": (thesis or "")[:240] or None,
        },
    }


def _opinion_signals(portfolio_db_path, ticker) -> tuple[float, float, str]:
    """Parse the latest agent_opinions payload for risk + confidence."""
    row = _safe_query(
        portfolio_db_path,
        "SELECT payload_json FROM agent_opinions WHERE ticker=? "
        "ORDER BY id DESC LIMIT 1", (ticker,))
    if not row or not row[0] or not row[0][0]:
        return 0.3, 0.6, ""
    try:
        p = json.loads(row[0][0])
    except (TypeError, ValueError):
        return 0.3, 0.6, ""

    # risk: try several shapes seen across the pipeline.
    risk_score = 0.3
    risk = p.get("risk") if isinstance(p.get("risk"), dict) else {}
    lvl = (p.get("risk_level") or risk.get("overall") or risk.get("level")
           or p.get("overall_risk"))
    if isinstance(lvl, str):
        risk_score = _RISK_LEVEL_SCORE.get(lvl.upper(), 0.3)
    elif isinstance(lvl, (int, float)):
        risk_score = max(0.0, min(1.0, float(lvl)))

    # confidence / provenance quality.
    conf = (p.get("confidence")
            or (p.get("provenance") or {}).get("quality")
            or (p.get("data_quality") or {}).get("score")
            or p.get("data_quality_score"))
    try:
        confidence = max(0.05, min(1.0, float(conf))) if conf is not None else 0.6
    except (TypeError, ValueError):
        confidence = 0.6

    thesis = ""
    t = p.get("thesis")
    if isinstance(t, str):
        thesis = t
    elif isinstance(t, dict):
        thesis = t.get("summary") or t.get("text") or ""
    return risk_score, confidence, thesis


def _filings_meta(market_db_path, ticker) -> tuple[int, Optional[str], Optional[str]]:
    cnt = _safe_query(
        market_db_path,
        "SELECT COUNT(*) FROM sec_filings WHERE ticker=?", (ticker,))
    fcount = int(cnt[0][0]) if cnt and cnt[0] else 0
    latest = _safe_query(
        market_db_path,
        "SELECT filing_type, filed_date FROM sec_filings WHERE ticker=? "
        "ORDER BY filed_date DESC LIMIT 1", (ticker,))
    if latest and latest[0]:
        return fcount, latest[0][0], latest[0][1]
    return fcount, None, None


def _insider_net_usd(market_db_path, ticker) -> Optional[float]:
    rows = _safe_query(
        market_db_path,
        "SELECT direction, COALESCE(value_usd,0) FROM insider_transactions "
        "WHERE ticker=? AND transaction_date >= date('now','-365 day')",
        (ticker,))
    if not rows:
        return None
    net = 0.0
    for direction, val in rows:
        try:
            v = float(val or 0)
        except (TypeError, ValueError):
            continue
        net += v if direction == "BUY" else (-v if direction == "SELL" else 0)
    return round(net, 0)


# =====================================================================
# Edges — REAL structured sources
# =====================================================================
def _edges_from_ownership(market_db_path, nodes, edges, market_data,
                          scope) -> int:
    """13D / 13G filings: ``filer_name`` holds ``ticker`` -> capital edge."""
    if not scope:
        return 0
    placeholders = ",".join("?" * len(scope))
    rows = _safe_query(
        market_db_path,
        f"SELECT ticker, filer_name, pct_owned, shares_owned, filed_date, "
        f"filing_type, accession_number FROM ownership_filings "
        f"WHERE ticker IN ({placeholders}) ORDER BY filed_date DESC",
        tuple(scope))
    if not rows:
        return 0

    seen: set[tuple[str, str]] = set()
    added = 0
    for ticker, filer, pct, shares, filed, ftype, accession in rows:
        if not filer or not ticker or _is_junk_filer(filer):
            continue
        ticker = ticker.upper()
        inst_id = _inst_id(filer)
        key = (inst_id, ticker)
        if key in seen:            # keep only most-recent filing per pair
            continue
        seen.add(key)

        _ensure_institution_node(nodes, inst_id, filer)

        # amount ($B): shares*price -> pct*marketcap -> stake-scaled proxy
        amount = None
        price = _latest_price(market_data, ticker)
        if shares and price:
            amount = (float(shares) * float(price)) / 1e9
        if amount is None and pct and nodes.get(ticker):
            mc = nodes[ticker]["metadata"].get("market_cap")
            if mc:
                amount = float(pct) / 100.0 * float(mc)
        if amount is None:
            # No price/market-cap (offline): a relative proxy that scales with
            # the reported stake so bigger holders read as thicker edges.
            try:
                pct_p = float(pct) if pct else 3.0
            except (TypeError, ValueError):
                pct_p = 3.0
            amount = 0.1 + min(pct_p, 20.0) / 20.0 * 3.0
        amount = max(0.05, round(amount, 3))

        # confidence scales with stake size (bigger = more clearly material)
        try:
            pct_f = float(pct or 5.0)
        except (TypeError, ValueError):
            pct_f = 5.0
        conf = max(0.4, min(0.95, 0.4 + min(pct_f, 20.0) / 40.0))

        edges.append({
            "source": inst_id,
            "target": ticker,
            "amount": amount,
            "type": "inferred",
            "date": _iso_date(filed),
            "confidence": round(conf, 3),
            "source_doc": _edgar_url(accession),
            "description": f"{ftype or '13D/G'} ownership: {filer} reports "
                           f"{_pct_str(pct)} of {ticker}.",
        })
        added += 1
    return added


def _edges_from_13f(market_db_path, nodes, edges, scope,
                    max_institutions) -> int:
    """13F holdings: institution -> issuer, matched to in-scope tickers by name.

    We connect a 13F holding only when its ``issuer_name`` matches a company
    already in scope (by ticker or fuzzy name), so no CUSIP->ticker table is
    required. Rows whose issuer isn't in scope are skipped.
    """
    if not scope:
        return 0
    name_index = _company_name_index(nodes)
    rows = _safe_query(
        market_db_path,
        "SELECT filer_cik, issuer_name, value_usd, shares, period_of_report, "
        "accession_number FROM f13f_holdings ORDER BY value_usd DESC "
        "LIMIT 40000")
    if not rows:
        return 0

    seen: set[tuple[str, str]] = set()
    added = 0
    inst_count: dict[str, int] = {}
    for filer_cik, issuer, value_usd, shares, period, accession in rows:
        target = _match_company(issuer, name_index)
        if not target:
            continue
        # resolve the CIK to a real institution name so 13F filers merge with
        # the 13D/13G hubs (both route through _inst_id -> canonical_name).
        disp = None
        if filer_cik and _fc_cik is not None:
            try:
                disp = _fc_cik(filer_cik)
            except Exception:  # noqa: BLE001
                disp = None
        disp = disp or (f"CIK {filer_cik}" if filer_cik else None)
        inst_id = _inst_id(disp) if disp else None
        if not inst_id:
            continue
        # cap fan-out per institution to keep the graph legible
        if inst_count.get(inst_id, 0) >= 12:
            continue
        key = (inst_id, target)
        if key in seen:
            continue
        seen.add(key)
        inst_count[inst_id] = inst_count.get(inst_id, 0) + 1
        if len(inst_count) > max_institutions and inst_id not in inst_count:
            continue

        _ensure_institution_node(nodes, inst_id, disp)
        amount = max(0.05, round(_usd_to_b(value_usd) or 0.1, 3))
        edges.append({
            "source": inst_id,
            "target": target,
            "amount": amount,
            "type": "inferred",
            "date": _iso_date(period),
            "confidence": 0.9,
            "source_doc": _edgar_url(accession),
            "description": f"13F-HR position: filer {filer_cik} holds "
                           f"{issuer} (${amount:.2f}B reported).",
        })
        added += 1
    return added


# =====================================================================
# Edges — best-effort hooks (see module docstring)
# =====================================================================
def _edges_from_news(market_db_path, nodes, edges, scope) -> int:
    """Suspected (dashed) edges from the ``news_edges`` company-pair table.

    Populated by ``backfill_news_edges.py``, which scans ``news_articles``
    headlines for co-mentioned in-universe companies with partnership / M&A /
    supply language. Reading a purpose-built pair table keeps graph builds fast
    and pushes the NLP to the extractor. Absent the table -> returns 0.

    Expected schema (best-effort; missing columns tolerated):
        news_edges(src_ticker, dst_ticker, headline, url, published_at,
                   confidence, event_type)
    """
    if not scope:
        return 0
    rows = _safe_query(
        market_db_path,
        "SELECT src_ticker, dst_ticker, headline, url, published_at, "
        "confidence, event_type FROM news_edges ORDER BY published_at DESC")
    if not rows:
        return 0
    seen: set[tuple[str, str]] = set()
    added = 0
    for src, dst, headline, url, pub, conf, etype in rows:
        if not src or not dst:
            continue
        src, dst = str(src).upper(), str(dst).upper()
        # only draw edges between companies already in the graph scope
        if src not in nodes or dst not in nodes or src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            continue
        seen.add(key)
        try:
            c = float(conf) if conf is not None else 0.4
        except (TypeError, ValueError):
            c = 0.4
        edges.append({
            "source": src, "target": dst, "amount": 1.0,
            "type": "suspected",
            "date": _iso_date(pub),
            "confidence": round(max(0.05, min(0.9, c)), 3),
            "source_doc": url or "news://co-mention",
            "description": (headline or f"{etype or 'news'} co-mention")[:200],
        })
        added += 1
    return added


def _edges_from_usaspending(market_db_path, nodes, edges, scope) -> int:
    """Confirmed gov->company money flow from USASpending contract awards.

    HOOK: ``geo_supply.fetch_usaspending_for_recipient`` pulls awards but writes
    them to the registry / ``data_points`` as recipient signals. If a table
    ``contract_awards`` / ``usaspending_awards`` with (recipient/ticker,
    amount, date) exists, this adds a synthetic ``US Government`` node and a
    confirmed edge per recipient. Absent that table -> returns 0.
    """
    if not scope:
        return 0
    table, cols = _find_table(
        market_db_path,
        candidates=["contract_awards", "usaspending_awards", "gov_contracts"],
        need_any=[("ticker", "recipient", "recipient_name"),
                  ("amount", "amount_usd", "obligated")])
    if not table:
        return 0
    tcol = cols["ticker"]
    acol = cols["amount"]
    name_index = _company_name_index(nodes)
    gov_id = "US Government"
    added = 0
    agg: dict[str, float] = {}
    dates: dict[str, str] = {}
    rows = _safe_query(
        market_db_path,
        f"SELECT {tcol}, {acol} FROM {table}")
    for recipient, amount in rows or []:
        target = _match_company(recipient, name_index) or (
            recipient.upper() if recipient and recipient.upper() in nodes else None)
        if not target:
            continue
        try:
            agg[target] = agg.get(target, 0.0) + float(amount or 0)
        except (TypeError, ValueError):
            continue
    if agg:
        nodes.setdefault(gov_id, {
            "id": gov_id, "name": "U.S. Federal Government", "ticker": "GOV",
            "size": 500.0, "sector": "Financials", "confidence": 0.99,
            "risk_score": 0.0,
            "metadata": {"market_cap": None, "revenue": None,
                         "form_type": "USASpending", "filings_count": 0,
                         "latest_filing_date": None, "insider_net_usd": None,
                         "thesis": None},
        })
    for target, total in agg.items():
        amount = max(0.05, round(total / 1e9, 3))
        edges.append({
            "source": gov_id, "target": target, "amount": amount,
            "type": "confirmed",
            "date": dates.get(target, datetime.now(timezone.utc).date().isoformat()),
            "confidence": 0.9,
            "source_doc": "https://www.usaspending.gov/",
            "description": f"Federal contract obligations to {target} "
                           f"(${amount:.2f}B, USASpending.gov).",
        })
        added += 1
    return added


# =====================================================================
# Helpers
# =====================================================================
def _safe_query(db_path: str, sql: str, params: tuple = ()) -> list:
    """Run a read query; return [] on any error (missing table/db/column)."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    except Exception as e:  # noqa: BLE001
        logger.debug("money_graph | query failed (%s): %s", sql[:60], e)
        return []


def _find_table(db_path, candidates, need_any) -> tuple[Optional[str], dict]:
    """Return (table, {logical: real_col}) for the first candidate table that
    exists and has at least one column from each ``need_any`` group. Lets the
    news/USASpending hooks bind to whatever schema is present without guessing.
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            existing = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            for cand in candidates:
                if cand not in existing:
                    continue
                cols = {r[1].lower(): r[1]
                        for r in conn.execute(f"PRAGMA table_info({cand})")}
                logical = {}
                ok = True
                for group in need_any:
                    match = next((cols[c] for c in group if c in cols), None)
                    if match is None:
                        ok = False
                        break
                    logical[group[0] if group[0] in ("ticker", "amount",
                            "headline") else group[0]] = match
                # normalize logical keys
                if ok:
                    keymap = {}
                    for group in need_any:
                        m = next((cols[c] for c in group if c in cols), None)
                        keymap[_logical_key(group)] = m
                    return cand, keymap
    except Exception as e:  # noqa: BLE001
        logger.debug("money_graph | _find_table failed: %s", e)
    return None, {}


def _logical_key(group: tuple) -> str:
    first = group[0]
    if first in ("ticker", "symbol", "recipient", "recipient_name"):
        return "ticker"
    if first in ("amount", "amount_usd", "obligated"):
        return "amount"
    if first in ("headline", "title", "text"):
        return "headline"
    return first


def _inst_id(name: str) -> str:
    return _canonicalize(name)[:60]


def _ensure_institution_node(nodes, inst_id, display_name) -> None:
    if inst_id in nodes:
        return
    nodes[inst_id] = {
        "id": inst_id,
        "name": _inst_id(display_name),
        "ticker": None,
        "size": 120.0,          # modest fixed size; institutions aren't sized by cap
        "sector": "Financials",
        "confidence": 0.9,
        "risk_score": 0.0,
        "metadata": {
            "market_cap": None, "revenue": None, "form_type": "13F/13D",
            "filings_count": None, "latest_filing_date": None,
            "insider_net_usd": None, "thesis": None,
            "entity_kind": "institution",
        },
    }


def _company_name_index(nodes) -> list[tuple[str, str]]:
    """[(needle_lower, node_id)] for company (non-institution) nodes, longest
    first so 'Advanced Micro Devices' matches before 'AMD'."""
    idx: list[tuple[str, str]] = []
    for nid, n in nodes.items():
        if (n.get("metadata") or {}).get("entity_kind") == "institution":
            continue
        idx.append((nid.lower(), nid))
        nm = (n.get("name") or "").strip()
        if nm and nm.lower() != nid.lower():
            # also index the name without common suffixes
            base = nm
            for suf in (" inc.", " inc", " corp.", " corp", " co.", " company",
                        " ltd.", " ltd", " plc", " group", " holdings", ","):
                base = base.replace(suf, "").replace(suf.upper(), "")
            idx.append((base.strip().lower(), nid))
    idx.sort(key=lambda kv: len(kv[0]), reverse=True)
    return idx


def _match_company(text, name_index, exclude=None) -> Optional[str]:
    if not text:
        return None
    t = str(text).lower()
    for needle, nid in name_index:
        if len(needle) < 3:
            continue
        if exclude and nid == exclude:
            continue
        # word-boundary-ish match for short tickers, substring for names
        if len(needle) <= 5:
            import re as _re
            if _re.search(rf"\b{_re.escape(needle)}\b", t):
                return nid
        elif needle in t:
            return nid
    return None


def _latest_price(market_data, ticker) -> Optional[float]:
    if market_data is None:
        return None
    try:
        p = market_data.get_latest_price(ticker) or {}
        return float(p.get("price")) if p.get("price") else None
    except Exception:  # noqa: BLE001
        return None


def _norm_sector(s) -> str:
    if not s:
        return ""
    s = str(s).strip()
    if s in _KNOWN_SECTORS:
        return s
    return _SECTOR_ALIASES.get(s, s if s in _KNOWN_SECTORS else "")


def _usd_to_b(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f <= 0:
        return None
    return f / 1e9


def _iso_date(v) -> str:
    if not v:
        return datetime.now(timezone.utc).date().isoformat()
    s = str(v)
    # accept 'YYYY-MM-DD', 'YYYYMMDD', 'YYYY-Qn', full ISO
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return datetime.now(timezone.utc).date().isoformat()


def _edgar_url(accession) -> str:
    if not accession:
        return "https://www.sec.gov/cgi-bin/browse-edgar"
    acc = str(accession).replace("-", "")
    return f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={acc}"


def _pct_str(pct) -> str:
    try:
        return f"{float(pct):.1f}%"
    except (TypeError, ValueError):
        return "a position"


# =====================================================================
# Bundled demo graph (fallback so the UI is never blank)
# =====================================================================
def _demo_graph() -> dict:
    def node(id, name, size, sector, conf, risk):
        return {"id": id, "name": name, "ticker": id, "size": size,
                "sector": sector, "confidence": conf, "risk_score": risk,
                "metadata": {"market_cap": size, "revenue": round(size * 0.15, 1),
                             "form_type": "10-K", "filings_count": 150,
                             "latest_filing_date": "2025-02-01",
                             "insider_net_usd": None, "thesis": None}}

    def edge(s, t, amt, typ, date, conf, doc, desc):
        return {"source": s, "target": t, "amount": amt, "type": typ,
                "date": date, "confidence": conf, "source_doc": doc,
                "description": desc}

    nodes = [
        node("AAPL", "Apple Inc.", 3200, "Technology", 0.98, 0.22),
        node("NVDA", "NVIDIA Corp.", 2900, "Technology", 0.97, 0.31),
        node("MSFT", "Microsoft Corp.", 3100, "Technology", 0.98, 0.18),
        node("TSM", "Taiwan Semiconductor", 900, "Technology", 0.94, 0.44),
        node("AMD", "Advanced Micro Devices", 260, "Technology", 0.90, 0.35),
        node("AVGO", "Broadcom Inc.", 780, "Technology", 0.92, 0.28),
        node("GOOGL", "Alphabet Inc.", 2100, "Communication", 0.97, 0.24),
        node("AMZN", "Amazon.com Inc.", 1900, "Consumer", 0.96, 0.29),
        node("TSLA", "Tesla Inc.", 800, "Consumer", 0.90, 0.64),
        node("JPM", "JPMorgan Chase", 620, "Financials", 0.95, 0.27),
        node("BRK.B", "Berkshire Hathaway", 900, "Financials", 0.99, 0.12),
        node("UNH", "UnitedHealth Group", 480, "Healthcare", 0.94, 0.41),
        node("LLY", "Eli Lilly & Co.", 780, "Healthcare", 0.94, 0.26),
        node("BA", "Boeing Co.", 130, "Industrials", 0.88, 0.79),
    ]
    edges = [
        edge("AAPL", "TSM", 42, "confirmed", "2024-11-01", 0.95,
             "https://sec.gov/AAPL/10-K", "Primary logic-chip fabrication."),
        edge("NVDA", "TSM", 30, "confirmed", "2024-10-28", 0.93,
             "https://sec.gov/NVDA/10-K", "Sole-source advanced-GPU foundry."),
        edge("AMD", "TSM", 12, "confirmed", "2024-09-30", 0.90,
             "https://sec.gov/AMD/10-K", "Leading-edge node capacity."),
        edge("AAPL", "AVGO", 15, "confirmed", "2024-08-15", 0.90,
             "https://sec.gov/AVGO/10-K", "Wireless/RF; Apple ~20% of AVGO rev."),
        edge("MSFT", "NVDA", 22, "confirmed", "2025-01-29", 0.90,
             "https://sec.gov/MSFT/8-K", "Azure AI datacenter GPU procurement."),
        edge("AMZN", "NVDA", 18, "confirmed", "2024-12-10", 0.87,
             "https://sec.gov/AMZN/10-K", "AWS accelerated-compute buildout."),
        edge("GOOGL", "NVDA", 10, "confirmed", "2025-02-04", 0.82,
             "https://sec.gov/GOOGL/10-Q", "Cloud GPU capacity alongside TPUs."),
        edge("TSLA", "NVDA", 4, "confirmed", "2024-11-20", 0.80,
             "https://sec.gov/TSLA/10-K", "Autopilot training cluster GPUs."),
        edge("AMZN", "AMD", 6, "confirmed", "2024-10-15", 0.80,
             "https://sec.gov/AMD/10-Q", "EPYC server CPUs across AWS."),
        edge("MSFT", "AMD", 5, "confirmed", "2024-11-05", 0.78,
             "https://sec.gov/AMD/10-Q", "MI300X as second-source to Nvidia."),
        edge("UNH", "LLY", 8, "confirmed", "2024-08-22", 0.72,
             "https://sec.gov/UNH/10-K", "OptumRx pharmacy purchases."),
        edge("BRK.B", "AAPL", 75, "inferred", "2024-06-30", 0.99,
             "https://sec.gov/BRK/13F-HR", "Berkshire 13F position in Apple."),
        edge("BRK.B", "BA", 1.5, "inferred", "2024-06-30", 0.60,
             "https://sec.gov/BRK/13F-HR", "Modeled industrial exposure."),
        edge("AAPL", "GOOGL", 2, "suspected", "2025-06-11", 0.50,
             "news://bloomberg", "Reported Gemini-for-Siri licensing talks."),
        edge("MSFT", "AMD", 3, "suspected", "2025-06-02", 0.42,
             "news://reuters", "Rumored expanded MI-series order."),
        edge("TSLA", "AMD", 1, "suspected", "2025-05-19", 0.40,
             "news://theinformation", "Speculated infotainment SoC deal."),
        edge("LLY", "NVDA", 1.5, "suspected", "2025-05-27", 0.45,
             "news://stat", "AI drug-discovery supercomputing collaboration."),
    ]
    return {"nodes": nodes, "edges": edges,
            "meta": {"source": "demo", "provenance": {"demo": len(edges)}}}


# =====================================================================
# CLI: quick check / dump graph.json
# =====================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Build the money-flow graph.")
    ap.add_argument("--portfolio-db", default="data/portfolio.db")
    ap.add_argument("--market-db", default="data/hedgefund.db")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--no-market", action="store_true",
                    help="skip yfinance enrichment (offline)")
    ap.add_argument("--out", default="graph.json")
    args = ap.parse_args()

    md = None
    if not args.no_market:
        try:
            from data.market_data import MarketData  # type: ignore
            md = MarketData(db_path=args.market_db)
        except Exception:
            try:
                from market_data import MarketData  # type: ignore
                md = MarketData(db_path=args.market_db)
            except Exception as e:  # noqa: BLE001
                logger.warning("MarketData unavailable (%s) — nodes unsized", e)

    g = build_money_graph(tickers=args.tickers,
                          portfolio_db_path=args.portfolio_db,
                          market_db_path=args.market_db, market_data=md)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(g, fh, indent=2, default=str)
    print(f"nodes={len(g['nodes'])} edges={len(g['edges'])} "
          f"source={g['meta'].get('source')} provenance={g['meta'].get('provenance')}")
    print(f"wrote {args.out}")

# D:\Ary Fund\data\money_graph.py