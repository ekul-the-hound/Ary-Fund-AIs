"""
tests/test_etf_holdings_loader.py
=================================
Tests for ``data.etf_holdings_loader`` and ``data.etf_providers``.

Conventions:
* All network I/O is patched. Tests are offline & deterministic.
* Each test gets a tmp DB via the ``tmp_path`` fixture.
* Real iShares / SPDR file structures are used in the fixtures (banner
  rows + footers), not stripped-down ideal CSVs — the parsers need to
  survive the real-world junk.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from data.etf_holdings_loader import (
    CanonicalHolding,
    ETFHoldingsLoader,
    TickerResolver,
)
from data.etf_providers import (
    GenericIssuerConfig,
    GenericIssuerProvider,
    IsharesProvider,
    RawHolding,
    SpdrProvider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Real-shaped iShares CSV: 5 banner rows + 2 blank rows + header + data + footer.
ISHARES_CSV = b'''Fund Holdings as of: ,"Apr 30, 2024",,,,,,,,
Inception Date: ,"Dec 18, 2009",,,,,,,,
Shares Outstanding: ,"123,456,789",,,,,,,,
Stock,-,,,,,,,,
,,,,,,,,,
Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Notional Value,Shares,CUSIP,ISIN
AAPL,APPLE INC,Information Technology,Equity,"$58,234,567,890.00",7.25,"$58,234,567,890.00","348,432,000",037833100,US0378331005
MSFT,MICROSOFT CORP,Information Technology,Equity,"$54,000,000,000.00",6.80,"$54,000,000,000.00","200,000,000",594918104,US5949181045
NVDA,NVIDIA CORP,Information Technology,Equity,"$30,000,000,000.00",3.75,"$30,000,000,000.00","20,000,000",67066G104,US67066G1040
-,USD CASH,-,Cash,"$1,200,000.00",0.05,,1200000,,
The Trust has Cash and/or Derivatives that may be reported here.
'''

# SPDR-style CSV (XLSX is the canonical format on the wire, but the provider
# falls back to CSV parsing when the bytes aren't a zip).
SPDR_CSV = b'''State Street Global Advisors,,,,,,,
,,,,,,,
Fund Name:,Technology Select Sector SPDR Fund,,,,,,
Date:,4/30/2024,,,,,,
,,,,,,,
Name,Ticker,Identifier,SEDOL,Weight,Sector,Shares Held,Local Currency
APPLE INC,AAPL,037833100,2046251,21.85,Information Technology,"168,432,000",USD
MICROSOFT CORP,MSFT,594918104,2588173,18.50,Information Technology,"100,000,000",USD
NVIDIA CORP,NVDA,67066G104,2379504,8.30,Information Technology,"40,000,000",USD
'''


@pytest.fixture
def loader(tmp_path: Path, monkeypatch) -> ETFHoldingsLoader:
    """Loader pointed at a tmp DB, with the real MarketData class but a
    private DB so production data isn't touched."""
    db = tmp_path / "etf_test.db"
    # Pre-create the etf_holdings table that the real MarketData would
    # create — avoids hitting yfinance during MarketData.__init__.
    _bootstrap_market_tables(str(db))
    # Patch MarketData so the loader gets the minimal stub below
    monkeypatch.setattr(
        "data.etf_holdings_loader.MarketData",
        _StubMarketData,
    )
    return ETFHoldingsLoader(
        db_path=str(db),
        provider_config={
            "IVV": {
                "provider": "ishares",
                "product_id": "239726",
                "slug": "ishares-core-sp-500-etf",
            },
            "XLK": {"provider": "spdr"},
        },
    )


def _bootstrap_market_tables(db_path: str) -> None:
    """Create the etf_holdings table without invoking real MarketData
    (which would hit network / yfinance on init)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS etf_holdings (
                etf_ticker TEXT NOT NULL,
                holding_ticker TEXT NOT NULL,
                weight REAL,
                shares REAL,
                market_value REAL,
                as_of TEXT NOT NULL,
                source TEXT,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (etf_ticker, holding_ticker, as_of)
            )"""
        )
        conn.commit()


class _StubMarketData:
    """In-test replacement for MarketData with set_etf_holdings only.

    Identical SQL to the real one (mirrored from market_data.py).
    Lives here rather than under tests/conftest.py so the test file is
    self-contained.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        _bootstrap_market_tables(db_path)
        self.calls: list[dict] = []  # spy

    def set_etf_holdings(
        self, etf_ticker: str, holdings: list[dict],
        as_of=None, source: str = "issuer_csv",
    ) -> int:
        from datetime import datetime as _dt
        etf_ticker = etf_ticker.upper()
        as_of = as_of or _dt.now().strftime("%Y-%m-%d")
        self.calls.append({
            "etf_ticker": etf_ticker,
            "holdings": holdings,
            "as_of": as_of,
            "source": source,
        })
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for h in holdings:
                tk = (h.get("ticker") or "").upper()
                if not tk:
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO etf_holdings
                        (etf_ticker, holding_ticker, weight, shares,
                         market_value, as_of, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (etf_ticker, tk, h.get("weight"), h.get("shares"),
                     h.get("market_value"), as_of, source),
                )
                n += 1
            conn.commit()
        return n


# ===========================================================================
# Test 1: iShares-style file
# ===========================================================================


class TestIsharesIngest:
    def test_parser_extracts_holdings(self):
        p = IsharesProvider(product_id="239726", slug="ishares-core-sp-500-etf")
        result = p.parse_bytes(ISHARES_CSV, "IVV", "http://test/ivv.csv")
        assert result.ok, f"errors: {result.errors}"
        assert result.as_of_date == "2024-04-30"
        # 3 equities + 1 cash row preserved
        assert len(result.holdings) == 4
        assert result.holdings[0].raw_ticker == "AAPL"
        assert result.holdings[0].name == "APPLE INC"

    def test_weight_is_decimal_not_percent(self):
        p = IsharesProvider(product_id="0", slug="x")
        result = p.parse_bytes(ISHARES_CSV, "IVV", "")
        # 7.25% → 0.0725
        assert abs(result.holdings[0].weight - 0.0725) < 1e-9
        # 6.80% → 0.068
        assert abs(result.holdings[1].weight - 0.068) < 1e-9

    def test_cusip_captured(self):
        p = IsharesProvider(product_id="0", slug="x")
        result = p.parse_bytes(ISHARES_CSV, "IVV", "")
        aapl = result.holdings[0]
        assert aapl.raw_identifier == "037833100"
        assert aapl.identifier_type == "cusip"

    def test_footer_stops_parsing(self):
        """The 'The Trust has Cash...' footer must not become a holding row."""
        p = IsharesProvider(product_id="0", slug="x")
        result = p.parse_bytes(ISHARES_CSV, "IVV", "")
        for h in result.holdings:
            joined = " ".join(filter(None, [h.raw_ticker, h.name or ""]))
            assert "trust" not in joined.lower()

    def test_full_ingest_writes_through_api(self, loader):
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=ISHARES_CSV,
        ):
            r = loader.run_single("IVV")
        assert r.status == "success", f"error: {r.error}"
        assert r.as_of_date == "2024-04-30"
        # 4 raw rows in, 4 written (deduper keeps distinct tickers)
        assert r.rows_loaded == 4

        # Verify the set_etf_holdings call shape
        market = loader.market  # the stub
        assert len(market.calls) == 1
        call = market.calls[0]
        assert call["etf_ticker"] == "IVV"
        assert call["as_of"] == "2024-04-30"
        assert call["source"] == "ishares_csv"

        # Payload must be the (ticker, weight, shares, market_value) shape
        api_payload = call["holdings"]
        tickers = {h["ticker"] for h in api_payload}
        assert {"AAPL", "MSFT", "NVDA", "CASH"} == tickers

    def test_data_in_etf_holdings_table(self, loader):
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=ISHARES_CSV,
        ):
            loader.run_single("IVV")

        with sqlite3.connect(loader.db_path) as conn:
            rows = conn.execute(
                """SELECT holding_ticker, weight, as_of, source
                   FROM etf_holdings WHERE etf_ticker = 'IVV'
                   ORDER BY weight DESC NULLS LAST"""
            ).fetchall()
        assert rows[0][0] == "AAPL"
        assert abs(rows[0][1] - 0.0725) < 1e-9
        assert rows[0][2] == "2024-04-30"
        assert rows[0][3] == "ishares_csv"


# ===========================================================================
# Test 2: SPDR-style file
# ===========================================================================


class TestSpdrIngest:
    def test_parser_handles_spdr_banner(self):
        p = SpdrProvider()
        result = p.parse_bytes(SPDR_CSV, "XLK", "http://test/xlk.csv")
        assert result.ok, f"errors: {result.errors}"
        assert result.as_of_date == "2024-04-30"
        assert len(result.holdings) == 3

    def test_identifier_column_treated_as_cusip(self):
        """SPDR labels CUSIPs as 'Identifier' rather than 'CUSIP' —
        the column map needs to handle that alias."""
        p = SpdrProvider()
        result = p.parse_bytes(SPDR_CSV, "XLK", "")
        aapl = result.holdings[0]
        assert aapl.raw_identifier == "037833100"
        assert aapl.identifier_type == "cusip"

    def test_weight_decimal(self):
        p = SpdrProvider()
        result = p.parse_bytes(SPDR_CSV, "XLK", "")
        # 21.85% → 0.2185
        assert abs(result.holdings[0].weight - 0.2185) < 1e-9

    def test_full_ingest(self, loader):
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=SPDR_CSV,
        ):
            r = loader.run_single("XLK")
        assert r.status == "success", r.error
        assert r.rows_loaded == 3
        assert r.issuer == "spdr"


# ===========================================================================
# Test 3: Unmapped identifiers are preserved
# ===========================================================================


class TestTickerResolution:
    def test_unmapped_cusip_kept_with_prefix(self, tmp_path):
        """A holding with only a CUSIP and no resolver entry must come
        through as ``CUSIP:...`` — never silently dropped."""
        resolver = TickerResolver(db_path=str(tmp_path / "t.db"))
        raw = RawHolding(
            raw_ticker=None,
            raw_identifier="123456789",
            identifier_type="cusip",
            name="Mystery Bond",
            weight=0.01,
            shares=None,
            market_value=None,
        )
        assert resolver.resolve(raw) == "CUSIP:123456789"

    def test_seeded_cusip_resolves_to_ticker(self, tmp_path):
        resolver = TickerResolver(db_path=str(tmp_path / "t.db"))
        resolver.add_mapping("037833100", "cusip", "AAPL")
        raw = RawHolding(
            raw_ticker=None,
            raw_identifier="037833100",
            identifier_type="cusip",
            name="APPLE INC",
            weight=0.07,
            shares=None,
            market_value=None,
        )
        assert resolver.resolve(raw) == "AAPL"

    def test_cash_row_normalized(self, tmp_path):
        resolver = TickerResolver(db_path=str(tmp_path / "t.db"))
        raw = RawHolding(
            raw_ticker="-", raw_identifier="-", identifier_type="ticker",
            name="USD CASH", weight=0.001, shares=None, market_value=None,
        )
        assert resolver.resolve(raw) == "CASH"

    def test_currency_normalized(self, tmp_path):
        resolver = TickerResolver(db_path=str(tmp_path / "t.db"))
        raw = RawHolding(
            raw_ticker="USD", raw_identifier=None, identifier_type=None,
            name=None, weight=None, shares=None, market_value=None,
        )
        assert resolver.resolve(raw) == "CASH:USD"

    def test_unmapped_holdings_reach_etf_holdings_table(self, tmp_path, loader):
        """End-to-end: an issuer file with only CUSIP identifiers must
        still produce rows in etf_holdings (with CUSIP-prefixed
        identifiers)."""
        cusip_only_csv = b'''Fund Holdings as of: ,"May 1, 2024",,
,,,,
Name,CUSIP,Weight (%),Shares
ACME GLOBAL,99988877X,3.45,10000
WIDGET INC,11122233Z,1.10,5000
'''
        # Add a generic CUSIP-only provider to the loader's config
        loader.provider_config["BOND"] = {
            "provider": "generic",
            "config": {
                "issuer": "synthetic",
                "url_template": "http://test/{ticker_lower}.csv",
                "column_map": {
                    "name": ["name"],
                    "cusip": ["cusip"],
                    "weight": ["weight (%)", "weight"],
                    "shares": ["shares"],
                },
                "source_tag": "synthetic_csv",
                "header_hints": {"name", "weight"},
            },
        }
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=cusip_only_csv,
        ):
            r = loader.run_single("BOND")
        assert r.status == "success", r.error

        with sqlite3.connect(loader.db_path) as conn:
            tickers = sorted(
                row[0] for row in conn.execute(
                    "SELECT holding_ticker FROM etf_holdings WHERE etf_ticker='BOND'"
                ).fetchall()
            )
        # Both rows present, both prefixed
        assert "CUSIP:99988877X" in tickers
        assert "CUSIP:11122233Z" in tickers


# ===========================================================================
# Test 4: Duplicates + malformed data
# ===========================================================================


class TestDataQuality:
    def test_duplicate_tickers_deduped(self, loader):
        dup_csv = b'''Fund Holdings as of: ,"May 1, 2024",,,
,,,,
Ticker,Name,Weight (%),Shares,CUSIP
AAPL,APPLE INC,7.25,1000,037833100
AAPL,APPLE INC CLASS A,7.25,1000,037833100
MSFT,MICROSOFT,6.80,500,594918104
'''
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=dup_csv,
        ):
            r = loader.run_single("IVV")
        assert r.status == "success", r.error
        # Two unique tickers, not three
        assert r.rows_loaded == 2

    def test_malformed_file_fails_cleanly(self, loader):
        """A junk file should fail with a structured error, not corrupt
        the table or crash the run."""
        junk = b"this is not csv\nat all\njust some bytes"
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=junk,
        ):
            r = loader.run_single("IVV")
        assert r.status == "failed"
        assert r.error  # populated
        assert r.rows_loaded == 0

        # Critically: the etf_holdings table must not have any IVV rows
        with sqlite3.connect(loader.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM etf_holdings WHERE etf_ticker='IVV'"
            ).fetchone()[0]
        assert count == 0, "failed parse must not write partial rows"

    def test_historical_data_preserved_across_runs(self, loader):
        """Two runs on different as_of dates must produce two snapshots,
        not overwrite each other."""
        csv_apr = b'''Fund Holdings as of: ,"Apr 30, 2024",,,
,,,,
Ticker,Name,Weight (%),Shares,CUSIP
AAPL,APPLE INC,7.25,1000,037833100
'''
        csv_may = b'''Fund Holdings as of: ,"May 31, 2024",,,
,,,,
Ticker,Name,Weight (%),Shares,CUSIP
AAPL,APPLE INC,7.50,1100,037833100
'''
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[csv_apr, csv_may],
        ):
            loader.run_single("IVV")
            loader.run_single("IVV")

        with sqlite3.connect(loader.db_path) as conn:
            rows = conn.execute(
                """SELECT as_of, weight FROM etf_holdings
                   WHERE etf_ticker='IVV' AND holding_ticker='AAPL'
                   ORDER BY as_of"""
            ).fetchall()
        assert len(rows) == 2, "both snapshots should be preserved"
        assert rows[0][0] == "2024-04-30"
        assert rows[1][0] == "2024-05-31"
        assert abs(rows[0][1] - 0.0725) < 1e-9
        assert abs(rows[1][1] - 0.075) < 1e-9

    def test_idempotent_same_date_run(self, loader):
        """Re-running the loader with the SAME file should not duplicate
        rows. INSERT OR REPLACE on (etf, holding, as_of) handles this."""
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            return_value=ISHARES_CSV,
        ):
            loader.run_single("IVV")
            with sqlite3.connect(loader.db_path) as conn:
                n1 = conn.execute(
                    "SELECT COUNT(*) FROM etf_holdings WHERE etf_ticker='IVV'"
                ).fetchone()[0]
            loader.run_single("IVV")
            with sqlite3.connect(loader.db_path) as conn:
                n2 = conn.execute(
                    "SELECT COUNT(*) FROM etf_holdings WHERE etf_ticker='IVV'"
                ).fetchone()[0]
        assert n1 == n2, f"idempotency broken: {n1} → {n2}"


# ===========================================================================
# Test 5: Batch ingestion + partial failures
# ===========================================================================


class TestBatchIngestion:
    def test_batch_processes_multiple_etfs(self, loader):
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[ISHARES_CSV, SPDR_CSV],
        ):
            summary = loader.run_batch(["IVV", "XLK"])
        assert summary.success_count == 2
        assert summary.failure_count == 0
        assert {r.etf_ticker for r in summary.results} == {"IVV", "XLK"}

    def test_one_failure_does_not_kill_the_batch(self, loader):
        """If one ETF's file is broken, the others must still complete."""
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[b"junk junk", SPDR_CSV],
        ):
            summary = loader.run_batch(["IVV", "XLK"])

        ivv = next(r for r in summary.results if r.etf_ticker == "IVV")
        xlk = next(r for r in summary.results if r.etf_ticker == "XLK")
        assert ivv.status == "failed"
        assert xlk.status == "success"
        assert summary.success_count == 1
        assert summary.failure_count == 1

    def test_network_failure_does_not_break_batch(self, loader):
        """A connection error on one ETF should fail that ETF only."""
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[RuntimeError("DNS failure"), SPDR_CSV],
        ):
            # Need to also blank out the cache fallback so the test is
            # deterministic
            with patch.object(
                ETFHoldingsLoader, "_load_cached", return_value=None,
            ):
                summary = loader.run_batch(["IVV", "XLK"])

        assert summary.success_count == 1
        assert summary.failure_count == 1
        ivv = next(r for r in summary.results if r.etf_ticker == "IVV")
        assert ivv.status == "failed"
        assert "DNS" in (ivv.error or "")

    def test_unknown_etf_is_skipped_not_failed(self, loader):
        r = loader.run_single("UNKNOWN_TICKER")
        assert r.status == "skipped"
        assert "no provider configured" in (r.error or "")

    def test_coverage_log_populated(self, loader):
        """Every ingest attempt must leave a row in etf_holdings_ingestion_log."""
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[ISHARES_CSV, b"broken"],
        ):
            loader.run_batch(["IVV", "XLK"])

        with sqlite3.connect(loader.db_path) as conn:
            rows = conn.execute(
                """SELECT etf_ticker, status, rows_loaded
                   FROM etf_holdings_ingestion_log
                   ORDER BY etf_ticker"""
            ).fetchall()
        statuses = {r[0]: r[1] for r in rows}
        assert statuses["IVV"] == "success"
        assert statuses["XLK"] == "failed"

    def test_run_id_groups_a_batch(self, loader):
        """All rows from one batch must share a run_id."""
        with patch.object(
            ETFHoldingsLoader, "_download_with_retry",
            side_effect=[ISHARES_CSV, SPDR_CSV],
        ):
            summary = loader.run_batch(["IVV", "XLK"])

        with sqlite3.connect(loader.db_path) as conn:
            run_ids = {
                r[0] for r in conn.execute(
                    "SELECT DISTINCT run_id FROM etf_holdings_ingestion_log"
                ).fetchall()
            }
        assert run_ids == {summary.run_id}


# ===========================================================================
# Bonus: GenericIssuerProvider is genuinely drop-in
# ===========================================================================


class TestGenericProvider:
    def test_arbitrary_csv_layout(self):
        """No subclass needed: just a config dict."""
        weird_csv = b'''Some Issuer Holdings
As of: 5/15/2024

Symbol,Name,Pct,Qty
ABC,Acme Corp,12.50,1000
DEF,Delta Energy,7.00,500
'''
        cfg = GenericIssuerConfig(
            issuer="custom",
            url_template="http://test/{ticker}.csv",
            column_map={
                "ticker": ["symbol"],
                "name":   ["name"],
                "weight": ["pct"],
                "shares": ["qty"],
            },
            header_hints={"symbol", "pct"},
        )
        p = GenericIssuerProvider(cfg)
        result = p.parse_bytes(weird_csv, "TEST", "")
        assert result.ok, f"errors: {result.errors}"
        assert result.as_of_date == "2024-05-15"
        assert len(result.holdings) == 2
        assert result.holdings[0].raw_ticker == "ABC"
        assert abs(result.holdings[0].weight - 0.125) < 1e-9


# ===========================================================================
# Bonus: validation rejects implausible weights
# ===========================================================================


class TestValidation:
    def test_implausible_weight_rejected(self, tmp_path):
        bad = [CanonicalHolding(
            etf_ticker="X", underlying_ticker="AAPL", security_id=None,
            identifier_type=None, security_name=None,
            weight=42.0,  # 4200% — clearly something went wrong
            shares=None, market_value=None,
            as_of_date="2024-01-01", issuer="x", source_url="", source_file="",
            source_type="csv", ingested_at="x",
        )]
        err = ETFHoldingsLoader._validate_canonical(bad)
        assert err is not None and "weight" in err

    def test_empty_canonical_rejected(self):
        err = ETFHoldingsLoader._validate_canonical([])
        assert err is not None
