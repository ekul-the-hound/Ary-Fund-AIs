"""
tests/test_global_risk_pulse.py
================================
Tests for ``data.global_risk_pulse.recompute_global_risk_pulse``.

The spec required six scenario tests (A–F). All are present here, plus
additional coverage of the configuration, weighting, and aggregation
helpers. Every test is offline and deterministic — the only inputs are
SQLite tables seeded inline.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from data.global_risk_pulse import (
    PulseConfig,
    _aggregate,
    _compute_concentration,
    _compute_confidence,
    global_pulse_score,
    recompute_global_risk_pulse,
)


# ===========================================================================
# Fixtures: tmp DB + synthetic universe builder
# ===========================================================================


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS price_history (
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
    close REAL, adj_close REAL, volume INTEGER,
    fetched_at TEXT,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE,
    shares REAL,
    avg_entry_price REAL,
    sector TEXT,
    thesis TEXT,
    conviction TEXT,
    position_type TEXT,
    opened_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE,
    target_entry REAL, target_exit REAL, stop_loss REAL,
    thesis TEXT, priority TEXT,
    added_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS risk_scores (
    ticker TEXT, as_of TEXT, macro_stress REAL,
    supply_chain REAL, sanctions_pressure REAL,
    commodity_sensitivity REAL, energy_crisis REAL,
    PRIMARY KEY (ticker, as_of)
);
CREATE TABLE IF NOT EXISTS data_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT, entity_type TEXT, field TEXT,
    value_num REAL, value_text TEXT, value_json TEXT,
    as_of TEXT,
    fetched_at TEXT DEFAULT (datetime('now')),
    source_id TEXT,
    confidence REAL DEFAULT 1.0,
    UNIQUE(entity_id, field, as_of, source_id)
);
"""


def _seed_db(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def _seed_macro(db_path: str, **fields: float) -> None:
    """Drop given macro fields into the registry at as_of=today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        for k, v in fields.items():
            conn.execute(
                """INSERT OR REPLACE INTO data_points
                   (entity_id, entity_type, field, value_num, as_of, source_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("global", "global", k, float(v), today, "fred"),
            )


def _seed_universe(
    db_path: str,
    n_tickers: int = 20,
    n_days: int = 500,
    end_date: datetime = datetime(2026, 5, 13),
    vol_multiplier: float = 1.0,
    bear_fraction: float = 0.0,
    market_cap_concentration: float = 0.0,
    sectors: Optional[list[str]] = None,
    seed: int = 42,
) -> list[str]:
    """Build a synthetic price universe and seed positions + market caps.

    Parameters
    ----------
    vol_multiplier :
        Multiplier on idiosyncratic noise (1.0 = baseline; 2.0 = vol-spike scenario).
    bear_fraction :
        Fraction of tickers given a strongly negative last-week return
        (used for the breadth-collapse scenario).
    market_cap_concentration :
        If > 0, top-3 tickers get market caps multiplied by this factor
        (e.g., 100x) to create concentration.
    """
    np.random.seed(seed)
    tickers = [f"TK{i:02d}" for i in range(n_tickers)]
    dates = pd.date_range(end=end_date, periods=n_days, freq="D")
    if sectors is None:
        sectors = (
            ["Tech"] * 5 + ["Financials"] * 3 + ["Healthcare"] * 3
            + ["Energy"] * 2 + ["Industrials"] * 2 + ["Consumer"] * 3
            + ["Utilities"] * 1 + ["Materials"] * 1
        )[:n_tickers]

    common = np.random.normal(0, 0.01, n_days)

    with sqlite3.connect(db_path) as conn:
        for i, t in enumerate(tickers):
            beta = 0.5 + 0.3 * (i % 3)
            idio = np.random.normal(0, 0.015 * vol_multiplier, n_days)
            rets = beta * common + idio
            # Bear-fraction: drag down the last 5 days for a subset
            if bear_fraction > 0 and i < int(n_tickers * bear_fraction):
                rets[-5:] -= 0.02
            prices = 100 * np.cumprod(1 + rets)
            volumes = np.random.randint(1_000_000, 10_000_000, n_days)
            rows = [
                (t, d.strftime("%Y-%m-%d"), float(p), float(p), int(v))
                for d, p, v in zip(dates, prices, volumes)
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO price_history
                   (ticker, date, close, adj_close, volume)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            conn.execute(
                """INSERT OR REPLACE INTO positions
                   (ticker, shares, avg_entry_price, sector, conviction, position_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (t, 100.0, float(prices[-1]), sectors[i], "MEDIUM", "LONG"),
            )
            # Market cap — base × (i+1), with optional concentration on top-3
            mcap = float(prices[-1] * 1e9 * (i + 1))
            if market_cap_concentration > 0 and i < 3:
                mcap *= market_cap_concentration
            conn.execute(
                """INSERT OR REPLACE INTO data_points
                   (entity_id, entity_type, field, value_num, as_of, source_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (t, "ticker", "ticker.price.market_cap",
                 mcap, end_date.strftime("%Y-%m-%d"), "yfinance"),
            )
            # Per-ticker risk score
            rscore = float(0.3 + 0.3 * np.random.random())
            conn.execute(
                """INSERT OR REPLACE INTO risk_scores
                   (ticker, as_of, macro_stress) VALUES (?, ?, ?)""",
                (t, end_date.strftime("%Y-%m-%d"), rscore),
            )
        conn.commit()
    return tickers


@pytest.fixture
def db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "pulse_test.db")
    _seed_db(db_path)
    return db_path


@pytest.fixture
def base_universe(db: str) -> list[str]:
    return _seed_universe(db)


# ===========================================================================
# Test A: never delegates to __GLOBAL__
# ===========================================================================


class TestNoGlobalMisuse:
    """Spec: the new function must NEVER treat '__GLOBAL__' as a ticker."""

    def test_global_sentinel_not_in_returned_tickers(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        # Coverage should never list a __GLOBAL__ sentinel as included
        excluded = out["coverage"]["excluded_reasons"]
        assert "__GLOBAL__" not in excluded
        assert "_GLOBAL_" not in excluded

    def test_no_global_sentinel_in_price_panel(self, db, base_universe):
        """If __GLOBAL__ were treated as a ticker, the price-load step
        would attempt to look it up. Seed a row with that name and
        confirm the loader doesn't pick it up unless explicitly asked."""
        with sqlite3.connect(db) as conn:
            conn.execute(
                """INSERT INTO price_history (ticker, date, close) VALUES (?, ?, ?)""",
                ("__GLOBAL__", "2026-05-13", 100.0),
            )
            conn.commit()
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        # The universe was explicit; no __GLOBAL__ row should appear
        diag = out.get("diagnostics", {})
        top_pos = diag.get("top_contributors_positive", [])
        top_neg = diag.get("top_contributors_negative", [])
        all_diag_tickers = {x["ticker"] for x in top_pos + top_neg}
        assert "__GLOBAL__" not in all_diag_tickers

    def test_dispersion_excludes_global_sentinel_rows(self, db, base_universe):
        """Even if a stray __GLOBAL__ row landed in risk_scores, the
        dispersion subcomponent must not include it in its std-dev."""
        with sqlite3.connect(db) as conn:
            conn.execute(
                """INSERT INTO risk_scores (ticker, as_of, macro_stress)
                   VALUES (?, ?, ?)""",
                ("__GLOBAL__", "2026-05-13", 0.99),  # outlier
            )
            conn.commit()
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        # If the outlier sneaked in, σ would jump; we just verify the
        # dispersion score is still reasonable for the synthetic data.
        disp = out["subcomponents"]["dispersion"]
        assert disp["coverage"] == len(base_universe), (
            "dispersion should count only universe tickers, not __GLOBAL__"
        )


# ===========================================================================
# Test B: coverage guard
# ===========================================================================


class TestCoverageGuard:
    def test_low_coverage_reduces_confidence(self, db):
        """Spec: when only ~30% of universe has fresh data, confidence
        should fall below 0.5 and the result must be flagged partial."""
        tickers = _seed_universe(db, n_tickers=20)
        # Backdate the last 15 tickers' data well past the staleness cutoff.
        # Default max_staleness_days=7; we shift their max date to 60 days ago.
        with sqlite3.connect(db) as conn:
            for t in tickers[5:]:  # only 5 tickers stay fresh = 25% coverage
                conn.execute(
                    "DELETE FROM price_history WHERE ticker = ? AND date > ?",
                    (t, "2026-03-13"),  # 2 months before fixture end
                )
            conn.commit()

        out = recompute_global_risk_pulse(
            universe=tickers,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=db,
            persist=False,
        )
        assert out["coverage"]["coverage_pct"] < 0.5
        assert out["confidence"] < 0.5
        # Excluded reasons must be populated
        assert len(out["coverage"]["excluded_reasons"]) >= 10

    def test_empty_universe_returns_null_pulse(self, db):
        out = recompute_global_risk_pulse(
            universe=[], db_path=db, persist=False,
        )
        assert out["pulse_score"] is None
        assert out["confidence"] == 0.0
        assert "diagnostics" in out

    def test_all_stale_returns_null(self, db):
        tickers = _seed_universe(db, n_tickers=10)
        # Wipe all recent data
        with sqlite3.connect(db) as conn:
            conn.execute(
                "DELETE FROM price_history WHERE date > ?",
                ("2026-01-01",),
            )
            conn.commit()
        out = recompute_global_risk_pulse(
            universe=tickers,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=db,
            persist=False,
        )
        assert out["pulse_score"] is None
        assert out["confidence"] == 0.0


# ===========================================================================
# Test C: volatility spike
# ===========================================================================


class TestVolatilitySpike:
    def test_volatility_subcomponent_rises_with_spike(self, db, tmp_path):
        """A universe with 2x volatility should produce a higher
        volatility subcomponent than a baseline universe."""
        # Baseline
        baseline_db = db
        base_t = _seed_universe(baseline_db, n_tickers=15, seed=1)
        out_base = recompute_global_risk_pulse(
            universe=base_t,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=baseline_db,
            persist=False,
        )

        # Spike universe — fresh DB with 2x vol multiplier
        spike_path = str(tmp_path / "spike.db")
        _seed_db(spike_path)
        spike_t = _seed_universe(
            spike_path, n_tickers=15, vol_multiplier=2.0, seed=1,
        )
        out_spike = recompute_global_risk_pulse(
            universe=spike_t,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=spike_path,
            persist=False,
        )

        v_base = out_base["subcomponents"]["volatility"]["score"]
        v_spike = out_spike["subcomponents"]["volatility"]["score"]
        assert v_base is not None and v_spike is not None
        assert v_spike > v_base, (
            f"vol spike must raise the volatility subcomponent "
            f"(base={v_base:.3f}, spike={v_spike:.3f})"
        )


# ===========================================================================
# Test D: breadth collapse
# ===========================================================================


class TestBreadthCollapse:
    def test_breadth_collapse_pushes_pulse_risk_off(self, db, tmp_path):
        baseline_db = db
        base_t = _seed_universe(baseline_db, n_tickers=20, seed=2)
        out_base = recompute_global_risk_pulse(
            universe=base_t,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=baseline_db,
            persist=False,
        )

        # 90% of tickers go negative over last week
        collapse_path = str(tmp_path / "collapse.db")
        _seed_db(collapse_path)
        collapse_t = _seed_universe(
            collapse_path, n_tickers=20, bear_fraction=0.9, seed=2,
        )
        out_coll = recompute_global_risk_pulse(
            universe=collapse_t,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=collapse_path,
            persist=False,
        )

        b_base = out_base["subcomponents"]["breadth"]["score"]
        b_coll = out_coll["subcomponents"]["breadth"]["score"]
        assert b_base is not None and b_coll is not None
        assert b_coll > b_base, (
            f"breadth collapse must raise the breadth subcomponent "
            f"(base={b_base:.3f}, collapse={b_coll:.3f})"
        )
        # And the pulse should also move risk-off
        assert out_coll["pulse_score"] > out_base["pulse_score"]


# ===========================================================================
# Test E: concentration sensitivity (HHI)
# ===========================================================================


class TestConcentration:
    def test_concentration_helper(self):
        """Unit test for the pure helper, independent of the DB path."""
        # 20 tickers, equal weight → HHI = 0.05 = 1/N → score ≈ -1
        eq = pd.Series([1.0 / 20] * 20, index=[f"T{i}" for i in range(20)])
        sub = _compute_concentration(eq)
        assert sub["score"] is not None
        assert sub["score"] <= -0.95

        # Same 20, but 80% in one ticker
        conc = pd.Series([0.8] + [0.2 / 19] * 19,
                         index=[f"T{i}" for i in range(20)])
        sub2 = _compute_concentration(conc)
        assert sub2["score"] is not None
        assert sub2["score"] > sub["score"]
        assert sub2["score"] >= 0.5

    def test_market_cap_concentration_raises_hhi(self, db, tmp_path):
        """A universe with one mega-cap dominating market_cap weight
        should see its concentration subcomponent increase compared to
        a more uniform universe."""
        base_db = db
        _seed_universe(base_db, n_tickers=15, seed=3)
        out_base = recompute_global_risk_pulse(
            universe=[f"TK{i:02d}" for i in range(15)],
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=base_db,
            config={"weighting": "market_cap"},
            persist=False,
        )

        conc_path = str(tmp_path / "conc.db")
        _seed_db(conc_path)
        _seed_universe(
            conc_path, n_tickers=15, market_cap_concentration=100.0, seed=3,
        )
        out_conc = recompute_global_risk_pulse(
            universe=[f"TK{i:02d}" for i in range(15)],
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=conc_path,
            config={"weighting": "market_cap"},
            persist=False,
        )

        c_base = out_base["subcomponents"]["concentration"]["score"]
        c_conc = out_conc["subcomponents"]["concentration"]["score"]
        assert c_conc > c_base, (
            f"100x mega-cap should raise concentration "
            f"(base={c_base:.3f}, conc={c_conc:.3f})"
        )


# ===========================================================================
# Test F: end-to-end sanity
# ===========================================================================


class TestEndToEndSanity:
    def test_returns_full_schema(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe,
            as_of=datetime(2026, 5, 13, tzinfo=timezone.utc),
            db_path=db,
            persist=False,
        )
        required_keys = {
            "pulse_score", "scale", "subcomponents", "weights", "coverage",
            "confidence", "timestamp_utc", "provenance", "thresholds",
            "diagnostics",
        }
        assert required_keys.issubset(out.keys())

        # Required subcomponents
        sub_required = {
            "volatility", "breadth", "correlation",
            "concentration", "dispersion", "macro_regime",
        }
        assert sub_required.issubset(out["subcomponents"].keys())
        # Each subcomponent has the standard shape
        for name, sub in out["subcomponents"].items():
            assert set(sub.keys()) >= {"score", "coverage", "notes"}
            if sub["score"] is not None:
                assert -1.0 <= sub["score"] <= 1.0

    def test_pulse_score_in_bounds(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        assert out["pulse_score"] is not None
        assert -1.0 <= out["pulse_score"] <= 1.0

    def test_coverage_block_correct(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        c = out["coverage"]
        assert c["ticker_count"] == len(base_universe)
        assert c["included_tickers"] + c["excluded_tickers"] >= len(base_universe) - 1
        assert 0.0 <= c["coverage_pct"] <= 1.0

    def test_persistence_creates_history_row(self, db, base_universe):
        recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=True,
        )
        with sqlite3.connect(db) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM global_risk_pulse_history"
            ).fetchone()[0]
        assert n >= 1

    def test_delta_vs_previous_populated(self, db, base_universe):
        """Two persisted runs should produce a delta diagnostic."""
        recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=True,
        )
        # Mutate macro to nudge the pulse
        _seed_macro(db, **{"global.vix": 35.0})
        out2 = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=True,
        )
        assert out2["diagnostics"]["delta_vs_previous"] is not None


# ===========================================================================
# Bonus: weighting methods + macro regime + compat wrapper
# ===========================================================================


class TestWeightingMethods:
    def test_equal_weighting(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db,
            config={"weighting": "equal"}, persist=False,
        )
        assert out["weights"]["universe_weighting"]["method"] == "equal"

    def test_market_cap_weighting(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db,
            config={"weighting": "market_cap"}, persist=False,
        )
        assert out["weights"]["universe_weighting"]["method"] == "market_cap"

    def test_hybrid_weighting(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db,
            config={"weighting": "hybrid"}, persist=False,
        )
        meta = out["weights"]["universe_weighting"]
        assert meta["method"] == "hybrid"
        assert meta["hybrid"]["mcap"] + meta["hybrid"]["liquidity"] == pytest.approx(1.0)

    def test_sector_balanced_weighting(self, db, base_universe):
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db,
            config={"weighting": "sector_balanced"}, persist=False,
        )
        assert out["weights"]["universe_weighting"]["method"] == "sector_balanced"
        assert out["weights"]["universe_weighting"]["sector_count"] >= 3


class TestMacroRegime:
    def test_crisis_macro_pushes_score_positive(self, db, base_universe):
        _seed_macro(
            db,
            **{
                "global.vix": 45.0,
                "global.yield_curve_2y10y": -0.5,
                "global.recession_prob": 50.0,
                "global.financial_stress": 2.5,
                "global.hy_oas": 9.5,
            },
        )
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        macro = out["subcomponents"]["macro_regime"]["score"]
        assert macro is not None
        assert macro > 0.7  # all indicators near saturated stress

    def test_calm_macro_pushes_score_negative(self, db, base_universe):
        _seed_macro(
            db,
            **{
                "global.vix": 11.0,
                "global.yield_curve_2y10y": 1.5,
                "global.recession_prob": 5.0,
                "global.financial_stress": -1.5,
                "global.hy_oas": 2.5,
            },
        )
        out = recompute_global_risk_pulse(
            universe=base_universe, db_path=db, persist=False,
        )
        macro = out["subcomponents"]["macro_regime"]["score"]
        assert macro is not None
        assert macro < -0.5


class TestCompatibilityWrapper:
    def test_global_pulse_score_returns_float(self, db, base_universe):
        score = global_pulse_score(
            universe=base_universe, db_path=db, persist=False,
        )
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_global_pulse_score_returns_none_on_empty(self, db):
        score = global_pulse_score(universe=[], db_path=db, persist=False)
        assert score is None


# ===========================================================================
# Bonus: aggregation helper handles missing subcomponents
# ===========================================================================


class TestAggregation:
    def test_missing_subs_redistribute_weight(self):
        subs = {
            "volatility":    {"score": 0.5,  "coverage": 10, "notes": ""},
            "breadth":       {"score": None, "coverage": 0,  "notes": "missing"},
            "correlation":   {"score": 0.0,  "coverage": 10, "notes": ""},
            "concentration": {"score": -0.5, "coverage": 10, "notes": ""},
            "dispersion":    {"score": None, "coverage": 0,  "notes": "missing"},
            "macro_regime":  {"score": 0.2,  "coverage": 4,  "notes": ""},
        }
        weights = {
            "volatility": 0.25, "breadth": 0.20, "correlation": 0.15,
            "concentration": 0.10, "dispersion": 0.10, "macro_regime": 0.20,
        }
        pulse, applied = _aggregate(subs, weights)
        assert pulse is not None
        # Weights of present subs should sum to 1
        present = {k for k, v in subs.items() if v["score"] is not None}
        s = sum(applied[k] for k in present)
        assert s == pytest.approx(1.0, abs=1e-4)
        # Missing subs should have weight 0
        assert applied["breadth"] == 0.0
        assert applied["dispersion"] == 0.0

    def test_all_missing_returns_none(self):
        subs = {
            "volatility":    {"score": None, "coverage": 0, "notes": ""},
            "breadth":       {"score": None, "coverage": 0, "notes": ""},
            "correlation":   {"score": None, "coverage": 0, "notes": ""},
            "concentration": {"score": None, "coverage": 0, "notes": ""},
            "dispersion":    {"score": None, "coverage": 0, "notes": ""},
            "macro_regime":  {"score": None, "coverage": 0, "notes": ""},
        }
        weights = dict.fromkeys(subs.keys(), 0.2)
        pulse, _applied = _aggregate(subs, weights)
        assert pulse is None


class TestConfidence:
    def test_confidence_grows_with_coverage(self):
        good_cov = {"coverage_pct": 1.0}
        bad_cov = {"coverage_pct": 0.1}
        all_subs = {k: {"score": 0.0} for k in
                    ("volatility", "breadth", "correlation",
                     "concentration", "dispersion", "macro_regime")}
        c_good = _compute_confidence(good_cov, all_subs, 100)
        c_bad = _compute_confidence(bad_cov, all_subs, 100)
        assert c_good > c_bad

    def test_confidence_penalized_for_missing_subs(self):
        cov = {"coverage_pct": 1.0}
        all_subs = {k: {"score": 0.0} for k in
                    ("volatility", "breadth", "correlation",
                     "concentration", "dispersion", "macro_regime")}
        few_subs = {k: ({"score": 0.0} if k == "volatility" else {"score": None})
                    for k in all_subs}
        c_full = _compute_confidence(cov, all_subs, 50)
        c_few = _compute_confidence(cov, few_subs, 50)
        assert c_full > c_few
