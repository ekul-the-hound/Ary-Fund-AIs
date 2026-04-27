"""
test_data_registry.py
=====================
Pytest coverage for data_registry.DataRegistry — schema bootstrap, idempotent
upserts, priority resolution, time series, events, refresh log, snapshot.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from data.data_registry import (
    DataRegistry,
    SOURCE_PRIORITY,
    CANONICAL_FIELDS,
    get_default_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reg(tmp_path: Path) -> DataRegistry:
    db = tmp_path / "registry_test.db"
    r = DataRegistry(str(db))
    r.register_source("yfinance",  "price",  "hourly",    base_priority=2)
    r.register_source("sec_xbrl",  "filing", "quarterly", base_priority=1)
    r.register_source("finnhub",   "analyst","daily",     base_priority=3)
    r.register_source("fred",      "macro",  "daily",     base_priority=1)
    r.register_source("sec_form4", "filing", "daily",     base_priority=1)
    return r


# =============================================================================
# Schema
# =============================================================================


def test_schema_creates_required_tables(reg: DataRegistry):
    with reg._conn() as conn:
        rows = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    for needed in {"data_sources", "data_points", "events",
                   "data_quality_log", "refresh_log", "schema_migrations"}:
        assert needed in rows, f"missing table {needed}"


def test_migration_is_idempotent(tmp_path: Path):
    db = tmp_path / "x.db"
    DataRegistry(str(db))
    DataRegistry(str(db))  # second construct must not crash
    r = DataRegistry(str(db))
    with r._conn() as conn:
        v = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert v == 1


# =============================================================================
# Upserts
# =============================================================================


def test_upsert_point_idempotent(reg: DataRegistry):
    rid1 = reg.upsert_point(
        "AAPL", "ticker", "ticker.price.adj_close",
        as_of="2026-04-25", source_id="yfinance",
        value_num=187.43,
    )
    rid2 = reg.upsert_point(
        "AAPL", "ticker", "ticker.price.adj_close",
        as_of="2026-04-25", source_id="yfinance",
        value_num=187.50,
    )
    # Same UNIQUE key -> ON CONFLICT update; should not insert a 2nd row
    with reg._conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM data_points WHERE entity_id='AAPL'"
        ).fetchone()[0]
    assert n == 1
    # Updated value should win
    assert reg.latest_value("AAPL", "ticker.price.adj_close") == 187.50


def test_upsert_clamps_confidence(reg: DataRegistry):
    reg.upsert_point("AAPL", "ticker", "ticker.price.close",
                     as_of="2026-04-25", source_id="yfinance",
                     value_num=100.0, confidence=2.5)
    row = reg.latest("AAPL", "ticker.price.close")
    assert row["confidence"] == 1.0

    reg.upsert_point("AAPL", "ticker", "ticker.price.close",
                     as_of="2026-04-26", source_id="yfinance",
                     value_num=99.0, confidence=-3)
    row = reg.latest("AAPL", "ticker.price.close")
    assert row["confidence"] == 0.0


def test_bulk_upsert(reg: DataRegistry):
    rows = [
        {
            "entity_id": "AAPL", "entity_type": "ticker",
            "field": "ticker.price.adj_close",
            "as_of": f"2026-04-{d:02d}", "source_id": "yfinance",
            "value_num": 100.0 + d,
        }
        for d in range(1, 11)
    ]
    n = reg.upsert_points_bulk(rows)
    assert n == 10
    ts = reg.time_series("AAPL", "ticker.price.adj_close")
    assert len(ts) == 10
    assert ts[0]["as_of"] == "2026-04-01"
    assert ts[-1]["as_of"] == "2026-04-10"


def test_json_value_roundtrip(reg: DataRegistry):
    payload = {"US": 0.6, "China": 0.25, "EU": 0.15}
    reg.upsert_point(
        "AAPL", "ticker", "ticker.disclosure.geographic_revenue",
        as_of="2025-12-31", source_id="sec_xbrl",
        value_json=payload,
    )
    got = reg.latest("AAPL", "ticker.disclosure.geographic_revenue")
    assert got["value_json"] == payload


# =============================================================================
# Priority resolution
# =============================================================================


def test_priority_picks_higher_authority_source(reg: DataRegistry):
    # eps_diluted_ttm priority: ['sec_xbrl', 'finnhub', 'yfinance']
    reg.upsert_point("AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
                     as_of="2026-03-31", source_id="yfinance",
                     value_num=6.79, confidence=0.9)
    reg.upsert_point("AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
                     as_of="2026-03-31", source_id="finnhub",
                     value_num=6.81, confidence=0.85)
    reg.upsert_point("AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
                     as_of="2026-03-31", source_id="sec_xbrl",
                     value_num=6.84, confidence=1.0)
    assert (
        reg.latest_value("AAPL", "ticker.fundamental.eps_diluted_ttm")
        == 6.84
    )


def test_priority_ignores_freshness_when_authoritative_present(reg: DataRegistry):
    # Even if yfinance is fresher, sec_xbrl should win.
    reg.upsert_point("AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
                     as_of="2026-03-31", source_id="sec_xbrl",
                     value_num=6.84, confidence=1.0)
    reg.upsert_point("AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
                     as_of="2026-04-25", source_id="yfinance",
                     value_num=6.95, confidence=0.9)
    assert (
        reg.latest_value("AAPL", "ticker.fundamental.eps_diluted_ttm")
        == 6.84
    )


def test_no_priority_fallback_uses_confidence(reg: DataRegistry):
    # `ticker.signal.atr_14` is not in SOURCE_PRIORITY -> fallback rule.
    reg.upsert_point("AAPL", "ticker", "ticker.signal.atr_14",
                     as_of="2026-04-25", source_id="derived",
                     value_num=2.5, confidence=0.5)
    reg.upsert_point("AAPL", "ticker", "ticker.signal.atr_14",
                     as_of="2026-04-25", source_id="yfinance",
                     value_num=2.6, confidence=0.9)
    assert reg.latest_value("AAPL", "ticker.signal.atr_14") == 2.6


def test_latest_respects_as_of_max(reg: DataRegistry):
    for d in [1, 5, 10, 20]:
        reg.upsert_point(
            "AAPL", "ticker", "ticker.price.adj_close",
            as_of=f"2026-04-{d:02d}", source_id="yfinance",
            value_num=100.0 + d,
        )
    assert reg.latest_value("AAPL", "ticker.price.adj_close",
                            as_of_max="2026-04-08") == 105.0


def test_latest_returns_none_when_missing(reg: DataRegistry):
    assert reg.latest_value("XYZ", "ticker.price.adj_close") is None


# =============================================================================
# Events
# =============================================================================


def test_upsert_event_idempotent(reg: DataRegistry):
    reg.upsert_event(
        event_type="form4_buy",
        occurred_at="2026-04-25T16:00:00",
        source_id="sec_form4",
        entity_id="AAPL", entity_type="ticker",
        severity=0.7,
        payload={"insider": "Cook, Tim", "shares": 1000, "value": 187_430},
    )
    reg.upsert_event(
        event_type="form4_buy",
        occurred_at="2026-04-25T16:00:00",
        source_id="sec_form4",
        entity_id="AAPL", entity_type="ticker",
        severity=0.8,
        payload={"insider": "Cook, Tim", "shares": 1000, "value": 187_500},
    )
    events = reg.recent_events(entity_id="AAPL")
    assert len(events) == 1
    assert events[0]["severity"] == 0.8
    assert events[0]["payload_json"]["value"] == 187_500


def test_recent_events_filters(reg: DataRegistry):
    reg.upsert_event("form4_buy",  "2026-04-25T16:00:00", "sec_form4",
                     entity_id="AAPL", entity_type="ticker", severity=0.5)
    reg.upsert_event("form4_sell", "2026-04-25T16:00:00", "sec_form4",
                     entity_id="AAPL", entity_type="ticker", severity=0.3)
    reg.upsert_event("form4_buy",  "2026-04-25T16:00:00", "sec_form4",
                     entity_id="MSFT", entity_type="ticker", severity=0.6)

    aapl = reg.recent_events(entity_id="AAPL")
    assert len(aapl) == 2
    buys = reg.recent_events(event_type="form4_buy")
    assert len(buys) == 2


# =============================================================================
# Refresh log
# =============================================================================


def test_refresh_run_logs_success(reg: DataRegistry):
    with reg.refresh_run("hourly_prices") as run:
        run["rows_written"] = 42
    assert reg.last_run("hourly_prices") is not None
    with reg._conn() as conn:
        row = conn.execute(
            "SELECT status, rows_written FROM refresh_log WHERE task='hourly_prices'"
        ).fetchone()
    assert row[0] == "ok"
    assert row[1] == 42


def test_refresh_run_logs_failures(reg: DataRegistry):
    with pytest.raises(RuntimeError):
        with reg.refresh_run("daily_filings"):
            raise RuntimeError("EDGAR returned 503")
    with reg._conn() as conn:
        row = conn.execute(
            "SELECT status, error FROM refresh_log WHERE task='daily_filings'"
        ).fetchone()
    assert row[0] == "error"
    assert "EDGAR" in row[1]
    # last_run() should NOT report this failed run
    assert reg.last_run("daily_filings") is None


def test_is_due_respects_interval(reg: DataRegistry):
    # First call: due (no history)
    assert reg.is_due("freight_index", 3600)
    with reg.refresh_run("freight_index") as run:
        run["rows_written"] = 1
    # Just ran -> not due
    assert reg.is_due("freight_index", 3600) is False
    # Tiny interval -> due again immediately
    time.sleep(0.05)
    assert reg.is_due("freight_index", 0)


# =============================================================================
# Source registration
# =============================================================================


def test_register_source_idempotent(reg: DataRegistry):
    reg.register_source("test_src", "price", "hourly", base_priority=5, notes="v1")
    reg.register_source("test_src", "price", "daily",  base_priority=2, notes="v2")
    with reg._conn() as conn:
        row = conn.execute(
            "SELECT category, refresh_cadence, base_priority, notes FROM data_sources WHERE source_id='test_src'"
        ).fetchone()
    assert row == ("price", "daily", 2, "v2")


def test_mark_source_run_success_and_failure(reg: DataRegistry):
    reg.register_source("foo", "price", "hourly")
    reg.mark_source_run("foo", success=True)
    reg.mark_source_run("foo", success=False, error="timeout")
    with reg._conn() as conn:
        row = conn.execute(
            "SELECT last_success, last_error FROM data_sources WHERE source_id='foo'"
        ).fetchone()
    assert row[0] is not None
    assert row[1] == "timeout"


# =============================================================================
# Snapshot helper
# =============================================================================


def test_snapshot_returns_stable_shape(reg: DataRegistry):
    reg.upsert_point("AAPL", "ticker", "ticker.price.adj_close",
                     as_of="2026-04-25", source_id="yfinance", value_num=187.43)
    snap = reg.snapshot(
        "AAPL",
        ["ticker.price.adj_close",
         "ticker.fundamental.eps_diluted_ttm",
         "ticker.signal.rsi_14"],
    )
    assert snap["ticker.price.adj_close"] == 187.43
    assert snap["ticker.fundamental.eps_diluted_ttm"] is None
    assert snap["ticker.signal.rsi_14"] is None


# =============================================================================
# Canonical schema sanity
# =============================================================================


def test_priority_keys_subset_of_canonical_fields():
    """Every field in SOURCE_PRIORITY should also appear in CANONICAL_FIELDS."""
    missing = sorted(set(SOURCE_PRIORITY) - set(CANONICAL_FIELDS))
    assert not missing, f"priority lists fields not in canonical schema: {missing}"
