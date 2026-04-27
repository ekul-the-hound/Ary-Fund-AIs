"""
test_market_data_ext.py
=======================
Tests the new market_data methods that don't require a live network call:
schema, source registration, options aggregate math, borrow/ETF writes,
technicals -> registry sync.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.market_data import MarketData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def md(tmp_path: Path) -> MarketData:
    db = tmp_path / "md_test.db"
    return MarketData(db_path=str(db))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_extended_tables_created(md: MarketData):
    with sqlite3.connect(md.db_path) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"price_history", "fundamentals_cache",
            "options_chain", "short_interest", "borrow_data",
            "etf_holdings"}.issubset(tables)


def test_sources_registered(md: MarketData):
    with sqlite3.connect(md.db_path) as conn:
        srcs = {r[0] for r in conn.execute("SELECT source_id FROM data_sources")}
    assert {"yfinance", "finra", "ibkr_scrape", "issuer_csv"}.issubset(srcs)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_atm_iv_picks_closest_strike():
    df = pd.DataFrame({
        "strike": [100, 105, 110, 115, 120],
        "impliedVolatility": [0.30, 0.28, 0.25, 0.27, 0.32],
    })
    assert MarketData._atm_iv(df, spot=110) == 0.25
    assert MarketData._atm_iv(df, spot=107) == 0.28  # 105 closer than 110
    assert MarketData._atm_iv(df, spot=None) is None
    assert MarketData._atm_iv(pd.DataFrame(), spot=100) is None


def test_unusual_activity_score():
    df = pd.DataFrame({
        "volume":       [50, 200, 1500, 100, 80],
        "openInterest": [200, 400, 300,  200, 100],
    })
    # Max ratio = 1500 / 300 = 5.0
    assert MarketData._unusual_activity_score(df) == 5.0


def test_unusual_activity_handles_zero_oi():
    df = pd.DataFrame({"volume": [10, 20], "openInterest": [0, 0]})
    assert MarketData._unusual_activity_score(df) is None


# ---------------------------------------------------------------------------
# Options aggregates -> registry
# ---------------------------------------------------------------------------


def test_write_options_aggregates(md: MarketData):
    """Synthesize a chain dict shaped like get_options_chain output and
    confirm the registry receives the expected canonical fields."""
    from datetime import date, timedelta
    today = date.today()
    chain = {
        "ticker": "AAPL", "spot": 187.0,
        "expiries": {
            (today + timedelta(days=29)).isoformat(): {
                "atm_iv": 0.30, "put_volume": 100, "call_volume": 200,
                "put_call_ratio": 0.5, "unusual_activity": 1.5,
            },
            (today + timedelta(days=58)).isoformat(): {
                "atm_iv": 0.32, "put_volume": 50, "call_volume": 80,
                "put_call_ratio": 0.625, "unusual_activity": 2.5,
            },
            (today + timedelta(days=89)).isoformat(): {
                "atm_iv": 0.34, "put_volume": 30, "call_volume": 40,
                "put_call_ratio": 0.75, "unusual_activity": 3.2,
            },
        },
    }
    md._write_options_aggregates("AAPL", chain)
    reg = md.registry
    assert reg.latest_value("AAPL", "ticker.options.iv_30d") == 0.30
    assert reg.latest_value("AAPL", "ticker.options.iv_60d") == 0.32
    assert reg.latest_value("AAPL", "ticker.options.iv_90d") == 0.34
    # IV slope = 90d - 30d = 0.04
    slope = reg.latest_value("AAPL", "ticker.options.iv_term_slope")
    assert abs(slope - 0.04) < 1e-9
    # P/C ratio: 180 puts / 320 calls = 0.5625
    pc = reg.latest_value("AAPL", "ticker.options.put_call_ratio")
    assert abs(pc - (180/320)) < 1e-9
    # UA = max
    assert reg.latest_value("AAPL", "ticker.options.unusual_activity") == 3.2


# ---------------------------------------------------------------------------
# Borrow / ETF holdings
# ---------------------------------------------------------------------------


def test_set_borrow_data_writes_raw_and_registry(md: MarketData):
    md.set_borrow_data("GME", borrow_fee_bps=350, shares_available=10_000)
    with sqlite3.connect(md.db_path) as conn:
        row = conn.execute(
            "SELECT borrow_fee_bps, shares_available FROM borrow_data WHERE ticker='GME'"
        ).fetchone()
    assert row[0] == 350
    assert row[1] == 10_000
    assert md.registry.latest_value("GME", "ticker.short.borrow_fee_bps") == 350
    assert md.registry.latest_value("GME", "ticker.short.borrow_available") == 10_000


def test_set_etf_holdings_and_overlap(md: MarketData):
    md.set_etf_holdings("XLK", [
        {"ticker": "AAPL", "weight": 0.15, "market_value": 50e9},
        {"ticker": "MSFT", "weight": 0.14, "market_value": 48e9},
        {"ticker": "NVDA", "weight": 0.10, "market_value": 35e9},
    ])
    md.set_etf_holdings("QQQ", [
        {"ticker": "AAPL", "weight": 0.12, "market_value": 40e9},
    ])
    overlap = md.get_etf_overlap_for_ticker("AAPL")
    etfs = {r["etf_ticker"] for r in overlap}
    assert etfs == {"XLK", "QQQ"}
    # Heaviest first
    assert overlap[0]["etf_ticker"] == "XLK"


# ---------------------------------------------------------------------------
# Technicals -> registry sync via stubbed get_technicals
# ---------------------------------------------------------------------------


def test_sync_technicals_to_registry(md: MarketData, monkeypatch):
    fake_tech = {
        "ticker": "AAPL",
        "rsi_14": 62.5, "atr_14": 2.1,
        "sma_50": 180.0, "sma_200": 175.0,
        "macd": {"macd_line": 0.4, "signal_line": 0.3, "histogram": 0.1},
        "signal": {"overall": "BULLISH", "score": 1.5, "signals": []},
    }
    monkeypatch.setattr(md, "get_technicals", lambda ticker, period="1y": fake_tech)
    n = md.sync_technicals_to_registry("AAPL")
    assert n >= 5
    reg = md.registry
    assert reg.latest_value("AAPL", "ticker.signal.rsi_14") == 62.5
    assert reg.latest_value("AAPL", "ticker.signal.atr_14") == 2.1
    assert reg.latest_value("AAPL", "ticker.signal.sma_50") == 180.0
    assert reg.latest_value("AAPL", "ticker.signal.sma_200") == 175.0
    assert reg.latest_value("AAPL", "ticker.signal.macd_hist") == 0.1
    assert reg.latest_value("AAPL", "ticker.signal.regime") == "BULL"


def test_sync_technicals_handles_empty(md: MarketData, monkeypatch):
    monkeypatch.setattr(md, "get_technicals", lambda ticker, period="1y": {})
    assert md.sync_technicals_to_registry("XYZ") == 0


def test_sync_prices_to_registry(md: MarketData, monkeypatch):
    """Synthesize a small price DataFrame and verify bulk-write to registry."""
    idx = pd.date_range("2026-04-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open":   [100, 101, 102, 103, 104],
        "High":   [101, 102, 103, 104, 105],
        "Low":    [ 99, 100, 101, 102, 103],
        "Close":  [100.5, 101.5, 102.5, 103.5, 104.5],
        "Volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
    }, index=idx)
    monkeypatch.setattr(md, "get_prices", lambda *a, **kw: df)
    n = md.sync_prices_to_registry("AAPL")
    # 5 days * 6 fields each
    assert n == 30
    assert md.registry.latest_value("AAPL", "ticker.price.close") == 104.5
    assert md.registry.latest_value("AAPL", "ticker.price.volume") == 1_400_000
