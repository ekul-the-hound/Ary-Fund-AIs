"""
tests/test_money_graph.py
=========================
Unit tests for the SEC money-flow graph builder (``data/money_graph.py``) and
the filer canonicalizer (``data/filer_canonical.py``).

All tests run fully offline: synthetic SQLite DBs under tmp_path and
``market_data=None`` so no network is touched. Follows the suite's graceful-skip
convention — if the modules can't be imported the whole file skips rather than
erroring.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest


# --- graceful import (support both data.x and top-level x layouts) --------
def _import(name):
    for path in (f"data.{name}", name):
        try:
            return __import__(path, fromlist=["*"])
        except Exception:  # noqa: BLE001
            continue
    return None


mg = _import("money_graph")
fc = _import("filer_canonical")
pytestmark = pytest.mark.skipif(mg is None, reason="money_graph not importable")


# --- synthetic DB builder -------------------------------------------------
def _market_db(path, *, ownership=(), f13f=(), news=(), contracts=(),
               xbrl=(), insiders=(), filings=()):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE sec_filings(ticker TEXT,filing_type TEXT,filed_date TEXT,accession_number TEXT);
        CREATE TABLE xbrl_facts(ticker TEXT,concept TEXT,period_end TEXT,value REAL);
        CREATE TABLE ownership_filings(ticker TEXT,cik TEXT,filing_type TEXT,filed_date TEXT,
            accession_number TEXT,filer_name TEXT,pct_owned REAL,shares_owned REAL);
        CREATE TABLE insider_transactions(ticker TEXT,direction TEXT,value_usd REAL,transaction_date TEXT);
        CREATE TABLE f13f_holdings(filer_cik TEXT,accession_number TEXT,period_of_report TEXT,
            cusip TEXT,issuer_name TEXT,value_usd REAL,shares REAL);
        CREATE TABLE news_edges(src_ticker TEXT,dst_ticker TEXT,headline TEXT,url TEXT,
            published_at TEXT,confidence REAL,event_type TEXT);
        CREATE TABLE contract_awards(ticker TEXT,recipient_name TEXT,amount_usd REAL,
            award_count INTEGER,last_award_date TEXT,window_days INTEGER);
    """)
    c.executemany("INSERT INTO ownership_filings VALUES(?,?,?,?,?,?,?,?)", ownership)
    c.executemany("INSERT INTO f13f_holdings VALUES(?,?,?,?,?,?,?)", f13f)
    c.executemany("INSERT INTO news_edges VALUES(?,?,?,?,?,?,?)", news)
    c.executemany("INSERT INTO contract_awards VALUES(?,?,?,?,?,?)", contracts)
    c.executemany("INSERT INTO xbrl_facts VALUES(?,?,?,?)", xbrl)
    c.executemany("INSERT INTO insider_transactions VALUES(?,?,?,?)", insiders)
    c.executemany("INSERT INTO sec_filings VALUES(?,?,?,?)", filings)
    c.commit()
    c.close()
    return str(path)


def _portfolio_db(path, positions=(), opinions=()):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE positions(ticker TEXT,sector TEXT);
        CREATE TABLE agent_opinions(id INTEGER PRIMARY KEY,ticker TEXT,created_at TEXT,payload_json TEXT);
    """)
    c.executemany("INSERT INTO positions VALUES(?,?)", positions)
    c.executemany("INSERT INTO agent_opinions(ticker,created_at,payload_json) VALUES(?,?,?)",
                  opinions)
    c.commit()
    c.close()
    return str(path)


def _insts(graph):
    return sorted(n["id"] for n in graph["nodes"]
                  if (n.get("metadata") or {}).get("entity_kind") == "institution")


def _companies(graph):
    return sorted(n["id"] for n in graph["nodes"]
                  if (n.get("metadata") or {}).get("entity_kind") != "institution")


# --- fixtures -------------------------------------------------------------
@pytest.fixture
def dbs(tmp_path):
    """A representative populated pair of DBs."""
    d10 = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    d5 = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    m = _market_db(
        tmp_path / "h.db",
        ownership=[
            ("AAPL", "320", "SC 13G", "2025-02-10", "o1", "VANGUARD GROUP INC", 8.0, 1e9),
            ("AAPL", "320", "SC 13G", "2024-02-10", "o0", "VANGUARD GROUP INC", 7.9, 9e8),  # older dup
            ("AAPL", "320", "SC 13G", "2025-02-11", "o2", "SC 13G", None, None),            # junk
            ("NVDA", "104", "SC 13G", "2025-02-12", "o3", "BlackRock Inc.", 6.0, 4e8),
        ],
        # 13F: Vanguard (CIK 102909) holds NVIDIA -> merges with the 13D/G
        # Vanguard->AAPL edge onto ONE "Vanguard Group" node.
        f13f=[("0000102909", "f1", "2024-12-31", "67066G104", "NVIDIA Corp", 2.4e11, 1e8)],
        news=[("AAPL", "GOOGL", "Apple and Google in AI partnership talks", "u1",
               "2025-06-11", 0.5, "partnership")],
        contracts=[("BA", "THE BOEING COMPANY", 8.0e9, 40, "2025-05-01", 365)],
        xbrl=[("NVDA", "Revenues", "2025-01-31", 130e9)],
        insiders=[("AAPL", "BUY", 5e6, d10), ("AAPL", "SELL", 2e6, d5)],
        filings=[("AAPL", "10-K", "2024-11-01", "k1"), ("AAPL", "8-K", "2025-01-15", "k2")],
    )
    p = _portfolio_db(
        tmp_path / "p.db",
        positions=[("AAPL", "Technology"), ("NVDA", "Technology"),
                   ("GOOGL", "Communication"), ("BA", "Industrials")],
        opinions=[("AAPL", "2025-06-01",
                   json.dumps({"risk_level": "MEDIUM", "confidence": 0.82})),
                  ("NVDA", "2025-06-01",
                   json.dumps({"risk": {"overall": "HIGH"}}))],
    )
    return m, p


# =========================================================================
# builder tests
# =========================================================================
def test_company_nodes_and_opinion_signals(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m,
                             market_data=None)
    aapl = next(n for n in g["nodes"] if n["id"] == "AAPL")
    nvda = next(n for n in g["nodes"] if n["id"] == "NVDA")
    assert aapl["sector"] == "Technology"
    assert aapl["risk_score"] == 0.5           # MEDIUM
    assert aapl["confidence"] == 0.82
    assert nvda["risk_score"] == 0.8           # HIGH via nested risk.overall
    assert aapl["metadata"]["filings_count"] == 2
    assert aapl["metadata"]["latest_filing_date"] == "2025-01-15"
    assert aapl["metadata"]["insider_net_usd"] == 3e6  # 5M buy - 2M sell


def test_ownership_edges_and_dedup(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None)
    own = [e for e in g["edges"] if e["type"] == "inferred"
           and e["source"] == "Vanguard Group" and e["target"] == "AAPL"]
    assert len(own) == 1                       # older duplicate collapsed
    assert own[0]["date"] == "2025-02-10"      # kept the most recent


def test_junk_filer_excluded(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None)
    insts = _insts(g)
    assert "Sc 13G" not in insts and "SC 13G" not in insts
    assert "Vanguard Group" in insts and "BlackRock" in insts


def test_13f_and_canonical_merge(dbs):
    """13D/13G 'VANGUARD GROUP INC' + 13F CIK 102909 must be ONE node."""
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None,
                             edge_sources=("ownership", "f13f"))
    insts = _insts(g)
    assert insts.count("Vanguard Group") == 1
    # Vanguard should now have BOTH an AAPL (13D/G) and NVDA (13F) edge
    v_targets = sorted(e["target"] for e in g["edges"] if e["source"] == "Vanguard Group")
    assert v_targets == ["AAPL", "NVDA"]


def test_news_edges(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "GOOGL"], portfolio_db_path=p,
                             market_db_path=m, market_data=None,
                             edge_sources=("news",), demo_fallback=False)
    sus = [e for e in g["edges"] if e["type"] == "suspected"]
    assert len(sus) == 1
    assert {sus[0]["source"], sus[0]["target"]} == {"AAPL", "GOOGL"}
    assert sus[0]["confidence"] == 0.5


def test_usaspending_edges_and_gov_node(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["BA"], portfolio_db_path=p, market_db_path=m,
                             market_data=None, edge_sources=("usaspending",),
                             demo_fallback=False)
    gov = [e for e in g["edges"] if e["type"] == "confirmed"
           and e["target"] == "BA"]
    assert len(gov) == 1
    assert gov[0]["amount"] == pytest.approx(8.0, abs=0.01)   # $8B -> 8.0
    assert any(n["id"] == "US Government" for n in g["nodes"])


def test_drop_isolated(dbs):
    m, p = dbs
    # GOOGL/BA have no ownership/13F edges; with news+usaspending off they drop
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None,
                             edge_sources=("ownership", "f13f"), drop_isolated=True)
    comps = _companies(g)
    assert "GOOGL" not in comps and "BA" not in comps
    assert "AAPL" in comps and "NVDA" in comps


def test_max_nodes_cap(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None,
                             drop_isolated=True, max_nodes=2)
    assert len(g["nodes"]) <= 2


def test_demo_fallback_on_empty(tmp_path):
    empty = _market_db(tmp_path / "e.db")
    pe = _portfolio_db(tmp_path / "pe.db")
    g = mg.build_money_graph(tickers=["AAPL"], portfolio_db_path=pe,
                             market_db_path=empty, market_data=None,
                             demo_fallback=True)
    assert g["meta"]["source"] == "demo"
    assert len(g["nodes"]) > 0 and len(g["edges"]) > 0


def test_output_is_json_serializable(dbs):
    m, p = dbs
    g = mg.build_money_graph(tickers=["AAPL", "NVDA", "GOOGL", "BA"],
                             portfolio_db_path=p, market_db_path=m, market_data=None)
    s = json.dumps(g)                          # must not raise
    assert '"nodes"' in s and '"edges"' in s
    # every node has the required shape
    for n in g["nodes"]:
        assert {"id", "name", "size", "sector", "confidence",
                "risk_score", "metadata"} <= set(n)
    for e in g["edges"]:
        assert {"source", "target", "amount", "type", "date",
                "confidence"} <= set(e)


def test_missing_tables_are_tolerated(tmp_path):
    """A market DB with no relationship tables must not raise."""
    c = sqlite3.connect(tmp_path / "bare.db")
    c.execute("CREATE TABLE unrelated(x INTEGER)")
    c.commit(); c.close()
    p = _portfolio_db(tmp_path / "pb.db", positions=[("AAPL", "Technology")])
    g = mg.build_money_graph(tickers=["AAPL"], portfolio_db_path=p,
                             market_db_path=str(tmp_path / "bare.db"),
                             market_data=None, demo_fallback=True)
    # no live edges -> demo fallback, but no exception
    assert "nodes" in g and "meta" in g


# =========================================================================
# canonicalizer tests
# =========================================================================
@pytest.mark.skipif(fc is None, reason="filer_canonical not importable")
class TestCanonicalizer:
    @pytest.mark.parametrize("raw,expected", [
        ("VANGUARD GROUP INC", "Vanguard Group"),
        ("The Vanguard Group", "Vanguard Group"),
        ("Vanguard Group Inc.", "Vanguard Group"),
        ("BLACKROCK INC.", "BlackRock"),
        ("BlackRock Fund Advisors", "BlackRock"),
        ("STATE STREET CORP", "State Street"),
        ("BERKSHIRE HATHAWAY INC /DE/", "Berkshire Hathaway"),
        ("FMR LLC", "Fidelity (FMR)"),
        ("PRICE T ROWE ASSOCIATES INC /MD/", "T. Rowe Price"),
    ])
    def test_known_aliases_collapse(self, raw, expected):
        assert fc.canonical_name(raw) == expected

    def test_long_tail_variants_collapse(self):
        a = fc.canonical_name("ACME CAPITAL PARTNERS LLC")
        b = fc.canonical_name("ACME CAPITAL PARTNERS LP")
        assert a == b and a  # same, non-empty

    def test_name_for_cik_padded_and_unpadded(self):
        assert fc.name_for_cik("1067983") == "Berkshire Hathaway"
        assert fc.name_for_cik("0000102909") == "Vanguard Group"
        assert fc.name_for_cik("999999999") is None

    def test_empty_is_none(self):
        assert fc.canonical_name("") is None
        assert fc.canonical_name(None) is None
