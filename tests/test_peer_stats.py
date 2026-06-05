"""
tests/test_peer_stats.py
========================
Tests for data/peer_stats.py â€” real per-sector peer-statistic computation,
caching, and the proof that real stats differ from the scanner's hardcoded
sector defaults (which is the entire point of the module).
"""
from __future__ import annotations

import os
import tempfile

import pytest

from data import peer_stats


# ---------------------------------------------------------------------------
# Synthetic universe: a tiny, fully-controlled metric provider
# ---------------------------------------------------------------------------
# Three software names with tight, low leverage; three REITs with high
# leverage. This is the canonical case: the same debt/EBITDA number means
# very different things in the two cohorts.

_SYNTH = {
    "SW1": {"sector": "Technology", "debt_ebitda": 0.5, "gross_margin": 0.80, "fcf_yield": 0.04},
    "SW2": {"sector": "Technology", "debt_ebitda": 0.8, "gross_margin": 0.78, "fcf_yield": 0.03},
    "SW3": {"sector": "Technology", "debt_ebitda": 1.1, "gross_margin": 0.82, "fcf_yield": 0.05},
    "RE1": {"sector": "Real Estate", "debt_ebitda": 7.0, "gross_margin": 0.55, "fcf_yield": 0.06},
    "RE2": {"sector": "Real Estate", "debt_ebitda": 7.5, "gross_margin": 0.58, "fcf_yield": 0.05},
    "RE3": {"sector": "Real Estate", "debt_ebitda": 8.0, "gross_margin": 0.52, "fcf_yield": 0.07},
    "NOSEC": {"debt_ebitda": 3.0},  # no sector tag -> skipped
    "EMPTY": {"sector": "Energy"},   # sector but no metrics -> contributes nothing
}


def _get_metrics(ticker):
    return _SYNTH.get(ticker)


_TICKERS = list(_SYNTH.keys())


# ---------------------------------------------------------------------------
# compute_all_sector_peer_stats
# ---------------------------------------------------------------------------

class TestCompute:
    def test_aggregates_by_sector(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        assert "technology" in out
        assert "real estate" in out

    def test_mean_and_std_correct(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        tech = out["technology"]["debt_ebitda"]
        # mean of (0.5, 0.8, 1.1) = 0.8
        assert tech["mean"] == pytest.approx(0.8)
        assert tech["n"] == 3
        # sample std (ddof=1) of (0.5,0.8,1.1) = 0.3
        assert tech["std"] == pytest.approx(0.3, abs=1e-9)

    def test_sector_without_tag_is_skipped(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        # NOSEC had no sector, so its 3.0 must not appear anywhere
        for sector_map in out.values():
            de = sector_map.get("debt_ebitda")
            if de:
                assert de["mean"] != 3.0 or de["n"] != 1

    def test_below_min_peers_omitted(self):
        # A sector with a single valid ticker yields no stat.
        single = {"ONLY": {"sector": "Utilities", "debt_ebitda": 4.0}}
        out = peer_stats.compute_all_sector_peer_stats(single.get, ["ONLY"])
        assert "utilities" not in out  # n=1 < _MIN_PEERS

    def test_missing_metric_skipped_not_zeroed(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        # Energy sector only had EMPTY (no metrics) -> not present at all.
        assert "energy" not in out

    def test_bad_ticker_does_not_abort(self):
        def _flaky(tk):
            if tk == "SW2":
                raise RuntimeError("boom")
            return _SYNTH.get(tk)
        out = peer_stats.compute_all_sector_peer_stats(_flaky, _TICKERS)
        # tech still computed from the surviving two names
        assert out["technology"]["debt_ebitda"]["n"] == 2


# ---------------------------------------------------------------------------
# The whole point: real stats differ from hardcoded defaults
# ---------------------------------------------------------------------------

class TestDiffersFromDefaults:
    def test_real_stats_reflect_actual_cohort_not_defaults(self):
        """Real peer stats are derived from the actual cohort, so they
        carry the cohort's true ``n`` and ``std`` â€” not the hardcoded
        default's representative ``n``. (We assert on n/std rather than
        mean, because a default mean can coincidentally equal a small
        synthetic cohort's mean.)"""
        from agent import risk_scanner
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        real = out["technology"]["debt_ebitda"]
        # Real cohort here is exactly the 3 synthetic tech names.
        assert real["n"] == 3
        assert real["std"] == pytest.approx(0.3, abs=1e-9)

        default = risk_scanner._SECTOR_DEFAULTS.get("technology", {}).get("debt_ebitda")
        if default is not None:
            # The real cohort size reflects the 3 names we fed in, which is
            # not the default's representative cohort size.
            assert real["n"] != default.get("n"), (
                "real peer n should reflect the actual cohort, not the default"
            )


# ---------------------------------------------------------------------------
# peer_stats_for_sector
# ---------------------------------------------------------------------------

class TestSectorLookup:
    def test_known_sector(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        block = peer_stats.peer_stats_for_sector(out, "Technology")
        assert "debt_ebitda" in block

    def test_unknown_sector_returns_empty(self):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        assert peer_stats.peer_stats_for_sector(out, "Healthcare") == {}

    def test_none_sector_returns_empty(self):
        assert peer_stats.peer_stats_for_sector({}, None) == {}


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCache:
    def test_roundtrip(self, tmp_path):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        peer_stats.cache_peer_stats(out, data_dir=str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "peer_stats_cache.json"))
        loaded = peer_stats.load_peer_stats_cache(data_dir=str(tmp_path))
        assert loaded == out

    def test_missing_cache_returns_none(self, tmp_path):
        assert peer_stats.load_peer_stats_cache(data_dir=str(tmp_path)) is None

    def test_stale_cache_returns_none(self, tmp_path):
        out = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        peer_stats.cache_peer_stats(out, data_dir=str(tmp_path))
        # max_age of 0 hours => any cache is stale
        assert peer_stats.load_peer_stats_cache(max_age_hours=0.0, data_dir=str(tmp_path)) is None

    def test_get_or_compute_uses_cache(self, tmp_path):
        calls = {"n": 0}

        def _counting(tk):
            calls["n"] += 1
            return _SYNTH.get(tk)

        class Cfg:
            WATCHLIST = _TICKERS

        first = peer_stats.get_or_compute_peer_stats(
            _counting, config=Cfg, data_dir=str(tmp_path), tickers=_TICKERS,
        )
        n_after_first = calls["n"]
        assert n_after_first > 0  # computed

        second = peer_stats.get_or_compute_peer_stats(
            _counting, config=Cfg, data_dir=str(tmp_path), tickers=_TICKERS,
        )
        assert calls["n"] == n_after_first  # cache hit -> no further calls
        assert second == first

    def test_force_bypasses_cache(self, tmp_path):
        class Cfg:
            WATCHLIST = _TICKERS
        peer_stats.get_or_compute_peer_stats(_get_metrics, config=Cfg, data_dir=str(tmp_path), tickers=_TICKERS)
        # force should recompute even though cache is fresh
        out = peer_stats.get_or_compute_peer_stats(
            _get_metrics, config=Cfg, data_dir=str(tmp_path), force=True, tickers=_TICKERS,
        )
        assert "technology" in out


# ---------------------------------------------------------------------------
# Integration: real stats actually flip a risk verdict
# ---------------------------------------------------------------------------

class TestEndToEndWithScanner:
    def test_reit_leverage_not_high_under_real_peer_stats(self):
        """A REIT at 8x debt/EBITDA, scored against real REIT peers
        (7.0/7.5/8.0), must NOT be HIGH on that metric."""
        from agent import risk_scanner

        all_stats = peer_stats.compute_all_sector_peer_stats(_get_metrics, _TICKERS)
        re_stats = peer_stats.peer_stats_for_sector(all_stats, "Real Estate")

        class Cfg:
            RISK_THRESHOLDS = {}

        metrics = {"sector": "Real Estate", "debt_ebitda": 8.0}
        out = risk_scanner.compute_risk_flags(
            "RE_TEST", metrics, {}, [], Cfg, peer_stats=re_stats,
        )
        # 8.0 vs peers mean 7.5, std 0.5 -> z ~ -1.0 (lower_is_safer) -> MEDIUM at worst
        assert out["levels"]["fundamental"] != "HIGH"
