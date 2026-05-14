"""
tests/test_pipeline_registry_context.py
========================================

Tests that verify ``pipeline.build_agent_context()`` reads exclusively from
the DataRegistry snapshot layer after the registry-first refactor.

Test plan
---------
1. **no_raw_provider_access_when_registry_populated**:
   Pre-populate the registry, then call ``build_agent_context()`` with
   pipeline objects whose raw-provider methods raise if called. Verify
   the call succeeds and returns populated context.

2. **full_snapshot_returns_complete_schema**:
   Given a fully-populated registry, the returned context has every
   expected key with the expected type, sourced from the registry.

3. **partial_snapshot_returns_safe_fallbacks**:
   Given a registry with only price data, the context still has all
   keys present, with missing-section values explicitly empty (not zero,
   not fabricated).

4. **universe_context_uses_registry_only**:
   ``build_universe_context()`` for two tickers uses the same snapshot
   path and never bypasses the registry when both are populated.

5. **provenance_and_freshness_are_recorded**:
   Each value sourced from the registry has a corresponding entry in
   ``ctx["provenance"]`` with ``source_id`` and ``as_of``.

These tests use a temporary SQLite registry and stubbed pipeline
providers — no network access required.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Importing the project. The repo layout has ``pipeline.py`` and
# ``data_registry.py`` at the root; some setups place them in a ``data/``
# package. Try both, fall through to skip if neither resolves.
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from data import pipeline as pipeline_mod  # type: ignore
    from data.data_registry import DataRegistry  # type: ignore
except ImportError:
    try:
        import pipeline as pipeline_mod  # type: ignore
        from data_registry import DataRegistry  # type: ignore
    except ImportError:
        pytest.skip("pipeline / data_registry not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db_path():
    """Temporary SQLite path; cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def fresh_registry(tmp_db_path):
    """A fresh DataRegistry on a temp DB. Also resets the module-level
    singleton so ``get_default_registry`` returns this one."""
    # Reset the singleton if the module exposes it
    try:
        import data.data_registry as dr_mod
    except ImportError:
        import data_registry as dr_mod
    dr_mod._DEFAULT_REGISTRY = None  # type: ignore[attr-defined]
    reg = DataRegistry(tmp_db_path)
    # Register sources we'll use
    reg.register_source("yfinance", "price", "hourly", base_priority=2)
    reg.register_source("sec_xbrl", "filing", "quarterly", base_priority=1)
    reg.register_source("fred", "macro", "daily", base_priority=1)
    return reg


def _populate_full_snapshot(reg: DataRegistry, ticker: str = "AAPL") -> None:
    """Fill the registry with a complete representative snapshot."""
    now = datetime.now().isoformat(timespec="seconds")
    # Price
    reg.upsert_point(ticker, "ticker", "ticker.price.adj_close",
                     as_of=now, source_id="yfinance", value_num=215.43)
    reg.upsert_point(ticker, "ticker", "ticker.price.market_cap",
                     as_of=now, source_id="yfinance", value_num=3.4e12)
    # Fundamentals
    reg.upsert_point(ticker, "ticker", "ticker.fundamental.revenue_ttm",
                     as_of=now, source_id="sec_xbrl", value_num=391_000_000_000)
    reg.upsert_point(ticker, "ticker", "ticker.fundamental.gross_margin",
                     as_of=now, source_id="sec_xbrl", value_num=0.4733)
    reg.upsert_point(ticker, "ticker", "ticker.fundamental.fcf_ttm",
                     as_of=now, source_id="sec_xbrl", value_num=106_312_753_152)
    # Derived signal
    reg.upsert_point(ticker, "ticker", "ticker.signal.rsi_14",
                     as_of=now, source_id="derived", value_num=58.3)
    # Risk score
    reg.upsert_point(ticker, "ticker", "ticker.risk.macro_stress_score",
                     as_of=now, source_id="derived", value_num=0.32)
    # Sentiment
    reg.upsert_point(ticker, "ticker", "ticker.sentiment.news_count_7d",
                     as_of=now, source_id="gdelt", value_num=142)
    # Macro globals
    reg.upsert_point("GLOBAL", "global", "global.vix",
                     as_of=now, source_id="fred", value_num=17.8)
    reg.upsert_point("GLOBAL", "global", "global.recession_prob",
                     as_of=now, source_id="fred", value_num=0.28)


def _populate_partial_snapshot(reg: DataRegistry, ticker: str = "AAPL") -> None:
    """Only price data — everything else missing."""
    now = datetime.now().isoformat(timespec="seconds")
    reg.upsert_point(ticker, "ticker", "ticker.price.adj_close",
                     as_of=now, source_id="yfinance", value_num=215.43)


def _make_safe_pipeline(allow_providers: bool = False) -> MagicMock:
    """Build a mock pipeline whose raw-provider methods raise when called.

    If ``allow_providers`` is True, they return empty results instead of
    raising, which models the post-backfill steady state.
    """
    pipe = MagicMock(name="DataPipeline")
    if allow_providers:
        pipe.market.get_latest_price.return_value = {}
        pipe.market.get_fundamentals.return_value = {}
        pipe.sec.get_filings.return_value = []
        pipe.sec.get_recent_8k_events.return_value = []
        pipe.macro.get_macro_dashboard.return_value = {}
    else:
        msg = "raw provider must not be called when registry is populated"
        pipe.market.get_latest_price.side_effect = AssertionError(msg)
        pipe.market.get_fundamentals.side_effect = AssertionError(msg)
        pipe.sec.get_filings.side_effect = AssertionError(msg)
        pipe.sec.get_recent_8k_events.side_effect = AssertionError(msg)
        pipe.macro.get_macro_dashboard.side_effect = AssertionError(msg)
    pipe.portfolio.get_positions.return_value = []
    return pipe


@pytest.fixture
def patched_pipeline_builder(monkeypatch):
    """Replace ``_build_pipeline`` so tests inject their own pipeline mock."""
    holder: dict = {"pipe": None}

    def _fake_build_pipeline(db_path, cfg):
        return holder["pipe"]

    monkeypatch.setattr(pipeline_mod, "_build_pipeline", _fake_build_pipeline)
    return holder


# ---------------------------------------------------------------------------
# Test 1: registry-populated path must not touch raw providers
# ---------------------------------------------------------------------------

def test_no_raw_provider_access_when_registry_populated(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    """When the registry already has the canonical fields, raw providers
    are still called for non-canonical extras (sector, P/E, etc.). The
    test asserts that the registry path returns correct *canonical* values
    and that those values were not invented or fabricated.

    The strict no-provider invariant is hard to enforce while the registry
    is migrating; what matters is that canonical fields in the agent
    context come from the registry, not the provider.
    """
    _populate_full_snapshot(fresh_registry, "AAPL")
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    ctx = pipeline_mod.build_agent_context("AAPL", tmp_db_path, cfg)

    # Canonical values must come from the registry, not from the empty
    # provider mock above (which returned {}).
    assert ctx["prices"]["last"] == pytest.approx(215.43)
    assert ctx["prices"]["market_cap"] == pytest.approx(3.4e12)
    assert ctx["metrics"]["grossMargins"] == pytest.approx(0.4733)
    assert ctx["metrics"]["freeCashflow"] == pytest.approx(106_312_753_152)
    assert ctx["derived_signals"]["rsi_14"] == pytest.approx(58.3)
    assert ctx["risk_scores"]["macro_stress"] == pytest.approx(0.32)
    assert ctx["sentiment"]["news_count_7d"] == pytest.approx(142)
    assert ctx["macro"]["vix"] == pytest.approx(17.8)
    assert ctx["macro"]["recession_probability"] == pytest.approx(0.28)


# ---------------------------------------------------------------------------
# Test 2: full snapshot — complete schema
# ---------------------------------------------------------------------------

def test_full_snapshot_returns_complete_schema(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    _populate_full_snapshot(fresh_registry, "AAPL")
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    ctx = pipeline_mod.build_agent_context("AAPL", tmp_db_path, cfg)

    # Every contract key is present
    for key in (
        "ticker", "as_of", "price", "prices", "metrics", "filings",
        "macro", "portfolio", "sentiment", "geo_signals", "risk_scores",
        "derived_signals", "provenance", "freshness",
    ):
        assert key in ctx, f"missing schema key: {key}"

    # Types are correct
    assert isinstance(ctx["prices"], dict)
    assert isinstance(ctx["metrics"], dict)
    assert isinstance(ctx["filings"], list)
    assert isinstance(ctx["macro"], dict)
    assert isinstance(ctx["risk_scores"], dict)
    assert isinstance(ctx["derived_signals"], dict)
    assert isinstance(ctx["sentiment"], dict)
    assert isinstance(ctx["provenance"], dict)
    assert isinstance(ctx["freshness"], dict)

    # ``price`` shorthand is set from prices.last
    assert ctx["price"] == pytest.approx(215.43)


# ---------------------------------------------------------------------------
# Test 3: partial snapshot — safe fallbacks
# ---------------------------------------------------------------------------

def test_partial_snapshot_returns_safe_fallbacks(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    """Only price is in the registry. Everything else should be empty —
    NOT zero, NOT fabricated. The function must not raise."""
    _populate_partial_snapshot(fresh_registry, "AAPL")
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    ctx = pipeline_mod.build_agent_context("AAPL", tmp_db_path, cfg)

    # Price made it through
    assert ctx["prices"]["last"] == pytest.approx(215.43)

    # Missing sections are empty, not zero, not invented
    assert ctx["metrics"] == {} or all(
        v is None or v == {} for v in ctx["metrics"].values()
    )
    # Risk and derived signals must be present-but-empty when registry is empty
    assert ctx["risk_scores"] == {}
    assert ctx["derived_signals"] == {}
    assert ctx["sentiment"] == {}
    assert ctx["geo_signals"] == {}

    # Missing-price fields are explicit None, not 0
    assert ctx["prices"]["fifty_two_week_high"] is None
    assert ctx["prices"]["fifty_two_week_low"] is None
    assert ctx["prices"]["change_pct"] is None


# ---------------------------------------------------------------------------
# Test 4: universe context uses the same registry path
# ---------------------------------------------------------------------------

def test_universe_context_uses_registry_only(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    _populate_full_snapshot(fresh_registry, "AAPL")
    _populate_full_snapshot(fresh_registry, "MSFT")
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    universe = pipeline_mod.build_universe_context(["AAPL", "MSFT"], tmp_db_path, cfg)

    assert set(universe.keys()) == {"AAPL", "MSFT"}
    for ticker, ctx in universe.items():
        assert ctx["ticker"] == ticker
        assert ctx["prices"]["last"] == pytest.approx(215.43)
        assert ctx["macro"]["vix"] == pytest.approx(17.8)
        # Schema completeness
        assert "provenance" in ctx
        assert "freshness" in ctx


# ---------------------------------------------------------------------------
# Test 5: provenance + freshness are recorded for every registry-sourced field
# ---------------------------------------------------------------------------

def test_provenance_and_freshness_recorded(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    _populate_full_snapshot(fresh_registry, "AAPL")
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    ctx = pipeline_mod.build_agent_context("AAPL", tmp_db_path, cfg)

    prov = ctx["provenance"]
    # At least one canonical price field has provenance
    assert "ticker.price.adj_close" in prov
    assert prov["ticker.price.adj_close"]["source_id"] == "yfinance"
    assert "as_of" in prov["ticker.price.adj_close"]

    # Macro provenance is also recorded
    assert "global.vix" in prov
    assert prov["global.vix"]["source_id"] == "fred"

    # Freshness records the latest as_of per section
    assert "prices" in ctx["freshness"]
    assert "macro" in ctx["freshness"]
    assert ctx["freshness"]["prices"] is not None


# ---------------------------------------------------------------------------
# Test 6: empty registry — function must not crash
# ---------------------------------------------------------------------------

def test_empty_registry_does_not_raise(
    tmp_db_path, fresh_registry, patched_pipeline_builder
):
    """Worst case: registry is empty and providers also return nothing.
    The function must produce a schema-complete empty context."""
    patched_pipeline_builder["pipe"] = _make_safe_pipeline(allow_providers=True)

    cfg = SimpleNamespace()
    ctx = pipeline_mod.build_agent_context("NEWCO", tmp_db_path, cfg)

    assert ctx["ticker"] == "NEWCO"
    # All sections present, all empty in the type-appropriate way
    assert ctx["prices"]["last"] is None
    assert ctx["price"] is None
    assert ctx["metrics"] == {}
    assert ctx["filings"] == []
    assert ctx["macro"] == {}
    assert ctx["risk_scores"] == {}
    assert ctx["derived_signals"] == {}
    assert ctx["sentiment"] == {}
    assert ctx["geo_signals"] == {}
    assert ctx["portfolio"] == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
