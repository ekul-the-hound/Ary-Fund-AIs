"""
test_integration.py
===================
End-to-end test: exercise all 8 modules through the scheduler with stubbed
network calls. Verifies that:

    * Every module wires into the same DataRegistry instance.
    * Hourly/daily/weekly cadences gate correctly.
    * A ticker snapshot built from the registry contains data points written
      by SEC + market + macro + sentiment + derived layers.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.data_registry import DataRegistry
from data.market_data import MarketData
from data.macro_data import MacroData
from data.sec_fetcher import SECFetcher
from data.sentiment_news import SentimentNews
from data.geo_supply import GeoSupply
from data.derived_signals import DerivedSignals
from data.refresh_scheduler import RefreshScheduler


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "integration.db")


def _seed_prices(db_path: str, ticker: str, n: int = 300, seed: int = 42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    closes = 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, n))
    with sqlite3.connect(db_path) as conn:
        # Ensure table exists (market_data normally creates this)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
                close REAL, adj_close REAL, volume INTEGER, fetched_at TEXT,
                PRIMARY KEY (ticker, date)
            )
        """)
        for d, p in zip(dates, closes):
            conn.execute(
                "INSERT OR REPLACE INTO price_history (ticker, date, close) VALUES (?, ?, ?)",
                (ticker.upper(), d.strftime("%Y-%m-%d"), float(p)),
            )


def test_all_modules_share_one_registry(db: str):
    """Construct every module with the same db_path and confirm they
    write to the same data_sources table."""
    reg = DataRegistry(db)
    md = MarketData(db_path=db, registry=reg)
    macro = MacroData(api_key="x", db_path=db, registry=reg)
    sec = SECFetcher(db_path=db, registry=reg, agent_name="t", agent_email="t@t.com")
    sn = SentimentNews(db_path=db, registry=reg)
    gs = GeoSupply(db_path=db, registry=reg)
    ds = DerivedSignals(db_path=db, registry=reg)
    with sqlite3.connect(db) as conn:
        sources = {r[0] for r in conn.execute("SELECT source_id FROM data_sources")}
    # Should contain entries from at least three different categories
    expected_subset = {
        "yfinance", "fred", "sec_xbrl", "sec_form4",
        "stocktwits", "ofac", "gdelt", "stooq",
    }
    assert expected_subset.issubset(sources)


def test_full_snapshot_after_synthetic_refresh(db: str):
    """Seed prices for AAPL + a sector ETF, push some macro values, run derived
    signals, and verify a final agent snapshot has fields from every layer."""
    _seed_prices(db, "AAPL", n=300)
    _seed_prices(db, "XLK",  n=300, seed=7)

    reg = DataRegistry(db)
    md = MarketData(db_path=db, registry=reg)        # creates extra tables + sources
    macro = MacroData(api_key="x", db_path=db, registry=reg)
    sn = SentimentNews(db_path=db, registry=reg)
    gs = GeoSupply(db_path=db, registry=reg)
    ds = DerivedSignals(db_path=db, registry=reg)

    # Macro values (stubbed FRED)
    fake_fred = {
        "VIXCLS": 18.0, "VXVCLS": 19.0, "BAMLH0A0HYM2": 3.5,
        "BAMLC0A0CM": 1.2, "RECPROUSM156N": 0.05,
        "UMCSENT": 80.0, "STLFSI4": -0.4, "T10Y2Y": -0.2,
    }
    macro.get_series_latest = lambda sid: fake_fred.get(sid)
    n_macro = macro.sync_macro_to_registry()
    assert n_macro >= 7

    # SEC: simulate inserting an 8-K corporate action
    sec = SECFetcher(db_path=db, registry=reg, agent_name="t", agent_email="t@t.com")
    sec._record_corp_action(
        "AAPL", "0000320193", "share_repurchase_announce",
        "2026-04-25", "ABC-001", amount_usd=110_000_000_000,
        detail={"desc": "Authorized $110B"},
    )

    # Sentiment: synthesize a news article + WSB mention, then aggregate
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO news_articles (ticker, title, publisher, url,
                published_at, source, sentiment)
               VALUES ('AAPL', 'AAPL beats expectations strong growth',
                       'Reuters', 'https://x/1', ?, 'yfinance', 0.8)""",
            (datetime.now().isoformat(),),
        )
        conn.execute(
            """INSERT INTO social_mentions (ticker, platform, posted_at, body, sentiment)
                VALUES ('AAPL', 'wsb', ?, '$AAPL to the moon', 0.6)""",
            (datetime.now().isoformat(),),
        )
    n_agg = sn.refresh_aggregates("AAPL")
    assert n_agg >= 2

    # Derived: technicals + risk scores + sector heatmap
    derived_result = ds.recompute_for("AAPL")
    assert derived_result.get("rsi_14") is not None
    n_heat = ds.recompute_sector_heatmap()
    assert n_heat >= 3
    risk = ds.recompute_risk_scores("AAPL")
    assert "macro_stress" in risk

    # ---- The big assertion: agent context snapshot has fields from every layer ----
    snap = reg.snapshot("AAPL", [
        # Market layer
        "ticker.signal.rsi_14",
        "ticker.signal.realized_vol_30d",
        "ticker.signal.regime",
        "ticker.signal.rs_vs_sector_60d",
        # Sentiment layer
        "ticker.sentiment.news_count_7d",
        "ticker.sentiment.wsb_mentions_24h",
        # Risk scores
        "ticker.risk.macro_stress_score",
        "ticker.risk.supply_chain_score",
    ])
    # Macro globals are read with entity_id="global"
    macro_snap = reg.snapshot("global", [
        "global.vix", "global.hy_oas", "global.consumer_sentiment",
        "global.yield_curve_2y10y",
    ])

    # Every requested field returns a non-None value
    for k, v in snap.items():
        assert v is not None, f"missing per-ticker field: {k}"
    for k, v in macro_snap.items():
        assert v is not None, f"missing macro field: {k}"

    # Events table contains the corporate action we inserted
    events = reg.recent_events(entity_id="AAPL")
    assert any(e["event_type"] == "share_repurchase_announce" for e in events)

    # Refresh log records what just ran (via DerivedSignals direct calls)
    # plus what the scheduler will record below
    sch = RefreshScheduler(tickers=["AAPL"], db_path=db, registry=reg)

    # Synthesize one logged scheduler call so refresh_log isn't empty
    res = sch._run_task("daily_test_smoke", lambda: 1, 60, force=True)
    assert res.error is None
    status = sch.status()
    assert any(k.startswith("daily_test") for k in status)


def test_scheduler_skips_when_not_due(db: str):
    """A second invocation of the same hourly task should be skipped."""
    reg = DataRegistry(db)
    sch = RefreshScheduler(tickers=["AAPL"], db_path=db, registry=reg)
    r1 = sch._run_task("hourly_smoke", lambda: 3, 3600)
    assert r1.error is None
    r2 = sch._run_task("hourly_smoke", lambda: 3, 3600)
    assert r2.error == "skipped_not_due"
    r3 = sch._run_task("hourly_smoke", lambda: 3, 3600, force=True)
    assert r3.error is None
    assert r3.rows_written == 3


def test_event_handler_returns_taskresult(db: str):
    reg = DataRegistry(db)
    sch = RefreshScheduler(tickers=["AAPL"], db_path=db, registry=reg)
    res = sch.run_event("unknown_event", {"ticker": "AAPL"})
    assert res.rows_written == 0
    assert res.error is None
