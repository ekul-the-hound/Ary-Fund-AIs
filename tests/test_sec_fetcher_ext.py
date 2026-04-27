"""
test_sec_fetcher_ext.py
=======================
Coverage for the extended SEC fetcher: Form 4 XML parsing, insider aggregate
recomputation, corporate-action recording, and registry integration.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from data.sec_fetcher import (
    SECFetcher,
    SRC_FORM4,
    SRC_XBRL,
    SRC_FILING,
    XBRL_CONCEPT_MAP,
)


# ---------------------------------------------------------------------------
# Sample Form 4 XML (drastically reduced from a real filing but with all
# tags the parser cares about).
# ---------------------------------------------------------------------------

SAMPLE_FORM4_BUY = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>COOK TIMOTHY D</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
      <isDirector>1</isDirector>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-04-20</value></transactionDate>
      <transactionCoding>
        <transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>185.50</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


SAMPLE_FORM4_SELL_10B5_1 = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>MAESTRI LUCA</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>CFO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-04-22</value></transactionDate>
      <transactionCoding>
        <transactionCode>S</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>187.20</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <footnotes>
    <footnote>This sale was effected pursuant to a 10b5-1 plan.</footnote>
  </footnotes>
</ownershipDocument>
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fetcher(tmp_path: Path) -> SECFetcher:
    db = tmp_path / "sec_test.db"
    f = SECFetcher(
        db_path=str(db),
        cache_dir=str(tmp_path / "cache"),
        agent_name="test", agent_email="x@y.com",
    )
    return f


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_extended_tables_created(fetcher: SECFetcher):
    with sqlite3.connect(fetcher.db_path) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    needed = {"sec_filings", "insider_transactions", "ownership_filings",
              "f13f_holdings", "xbrl_facts", "corporate_actions"}
    assert needed.issubset(tables)


def test_sources_registered(fetcher: SECFetcher):
    with sqlite3.connect(fetcher.db_path) as conn:
        srcs = {r[0] for r in conn.execute("SELECT source_id FROM data_sources")}
    assert {"sec_edgar", "sec_xbrl", "sec_form4", "sec_13d", "sec_13g", "sec_13f"}.issubset(srcs)


# ---------------------------------------------------------------------------
# Form 4 parsing — patch _get and EDGAR index lookup
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, json_data: Any = None, text: str = ""):
        self._json = json_data
        self.text = text
        self.status_code = 200
        self.headers = {}
    def json(self): return self._json
    def raise_for_status(self): pass


def _patch_get_for_form4(fetcher: SECFetcher, xml_text: str):
    """Replace fetcher._get with a stub that returns a stub index json then
    the supplied XML for the form4 file."""
    calls = {"index_seen": False}
    def fake_get(url, **kwargs):
        if url.endswith("index.json"):
            calls["index_seen"] = True
            return _FakeResponse(json_data={
                "directory": {"item": [{"name": "form4.xml"}]}
            })
        # Anything ending in form4.xml -> return the xml
        if url.endswith("form4.xml") or url.endswith(".xml"):
            return _FakeResponse(text=xml_text)
        raise AssertionError(f"unexpected URL in test: {url}")
    fetcher._get = fake_get  # type: ignore[assignment]
    return calls


def test_parse_form4_buy(fetcher: SECFetcher):
    _patch_get_for_form4(fetcher, SAMPLE_FORM4_BUY)
    filing = {
        "ticker": "AAPL", "cik": "0000320193",
        "filed_date": "2026-04-20",
        "accession_number": "0000320193-26-000001",
        "primary_doc_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/x.htm",
    }
    txns = fetcher._parse_form4(filing)
    assert len(txns) == 1
    t = txns[0]
    assert t["insider_name"] == "COOK TIMOTHY D"
    assert "CEO" in (t["insider_title"] or "")
    assert "Director" in (t["insider_title"] or "")
    assert t["transaction_code"] == "P"
    assert t["direction"] == "BUY"
    assert t["shares"] == 1000
    assert t["price"] == 185.50
    assert t["value_usd"] == 185_500
    assert t["is_10b5_1"] is False


def test_parse_form4_sell_with_10b5_1(fetcher: SECFetcher):
    _patch_get_for_form4(fetcher, SAMPLE_FORM4_SELL_10B5_1)
    filing = {
        "ticker": "AAPL", "cik": "0000320193",
        "filed_date": "2026-04-22",
        "accession_number": "0000320193-26-000002",
        "primary_doc_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019326000002/x.htm",
    }
    txns = fetcher._parse_form4(filing)
    assert len(txns) == 1
    t = txns[0]
    assert t["insider_name"] == "MAESTRI LUCA"
    assert t["direction"] == "SELL"
    assert t["transaction_code"] == "S"
    assert t["shares"] == 5000
    assert t["is_10b5_1"] is True


# ---------------------------------------------------------------------------
# Insider aggregate refresh: synthesize raw rows, then recompute
# ---------------------------------------------------------------------------


def test_insider_aggregates_use_recent_30d_only(fetcher: SECFetcher):
    """Stuff the raw table with a mix of recent + old txns and verify
    the registry receives only the 30-day rolling counts."""
    from datetime import datetime, timedelta
    today = datetime.now().date()
    fresh = (today - timedelta(days=10)).isoformat()
    stale = (today - timedelta(days=120)).isoformat()
    with sqlite3.connect(fetcher.db_path) as conn:
        conn.executemany(
            """INSERT INTO insider_transactions
                (ticker, cik, accession_number, insider_name, insider_title,
                 transaction_date, transaction_code, direction, shares, price, value_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("AAPL","320193","a1","X","CEO", fresh,"P","BUY",  100, 100, 10_000),
                ("AAPL","320193","a2","Y","CFO", fresh,"S","SELL", 200, 100, 20_000),
                ("AAPL","320193","a3","Z","Dir", fresh,"P","BUY",  300, 100, 30_000),
                ("AAPL","320193","a4","Q","Dir", stale,"S","SELL", 500, 100, 50_000),  # excluded
            ],
        )

    fetcher._refresh_insider_aggregates("AAPL")

    reg = fetcher.registry
    buys = reg.latest_value("AAPL", "ticker.ownership.insider_buys_30d")
    sells = reg.latest_value("AAPL", "ticker.ownership.insider_sells_30d")
    net = reg.latest_value("AAPL", "ticker.ownership.insider_net_usd_30d")
    assert buys == 2.0
    assert sells == 1.0
    # net = 10k + 30k (buys) - 20k (sells) = +20k
    assert net == 20_000.0


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------


def test_record_corp_action_emits_event(fetcher: SECFetcher):
    n = fetcher._record_corp_action(
        "AAPL", "0000320193", "share_repurchase_announce",
        "2026-04-25", "0000000-26-001",
        amount_usd=1_000_000_000,
        detail={"desc": "Board authorized $1B buyback"},
    )
    assert n == 1
    events = fetcher.registry.recent_events(entity_id="AAPL")
    assert any(e["event_type"] == "share_repurchase_announce" for e in events)
    payload = next(e for e in events if e["event_type"] == "share_repurchase_announce")["payload_json"]
    assert payload["amount_usd"] == 1_000_000_000


def test_record_corp_action_idempotent(fetcher: SECFetcher):
    for _ in range(3):
        fetcher._record_corp_action(
            "AAPL", "0000320193", "officer_director_change",
            "2026-04-25", "X-001", detail={"desc": "CFO retired"},
        )
    with sqlite3.connect(fetcher.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM corporate_actions WHERE accession_number='X-001'"
        ).fetchone()[0]
    assert n == 1


# ---------------------------------------------------------------------------
# XBRL concept map sanity — mainly to detect typos when editing the dict
# ---------------------------------------------------------------------------


def test_xbrl_concept_map_canonical_fields_present():
    from data.data_registry import CANONICAL_FIELDS
    for canonical_field in XBRL_CONCEPT_MAP:
        assert canonical_field in CANONICAL_FIELDS, \
            f"{canonical_field} maps an XBRL concept but isn't in CANONICAL_FIELDS"


def test_xbrl_concept_map_lists_non_empty():
    for f, lst in XBRL_CONCEPT_MAP.items():
        assert isinstance(lst, list) and lst, f"empty XBRL candidates for {f}"
