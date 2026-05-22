"""
tests/test_risk_scanner_sector.py
=================================

Tests for the sector-aware risk-scoring upgrade in
:mod:`agent.risk_scanner`. Covers:

1. Legacy behaviour regression (existing thresholds still trigger).
2. Sector z-scoring: REIT vs software case.
3. Distress models: Altman / Piotroski / Beneish unit tests + integration.
4. Public-API stability (return shape unchanged, optional kwarg).

These supplement (not replace) the existing ``tests/test_risk_scanner.py``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from agent.risk_scanner import (
    compute_altman_z,
    compute_beneish_m,
    compute_piotroski_f,
    compute_risk_flags,
    zscore_risk_signed,
    ztier,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}


@pytest.fixture
def config() -> SimpleNamespace:
    """Minimal config carrying the default RISK_THRESHOLDS."""
    return SimpleNamespace(RISK_THRESHOLDS={
        "debt_ebitda_high": 3.0,
        "debt_ebitda_medium": 2.0,
        "interest_coverage_low": 2.0,
        "fcf_yield_low": 0.02,
        "recession_prob_high": 0.6,
        "recession_prob_medium": 0.35,
        "vix_high": 28.0,
        "vix_medium": 20.0,
        "drawdown_high": 0.25,
        "drawdown_medium": 0.15,
        "realized_vol_high": 0.45,
        "realized_vol_medium": 0.30,
    })


@pytest.fixture
def benign_metrics() -> Dict[str, Any]:
    """Healthy fundamentals, no sector tag, no distress inputs."""
    return {
        "ticker": "GOOD",
        "price": 150.0,
        "debt_ebitda": 1.5,
        "debt_to_ebitda": 1.5,
        "interest_coverage": 12.0,
        "fcf_yield": 0.06,
        "free_cash_flow": 5_000_000_000.0,
        "cash_conversion": 1.05,
        "roic": 0.18,
        "market_cap": 100_000_000_000.0,
    }


@pytest.fixture
def benign_macro() -> Dict[str, Any]:
    return {
        "recession_probability": 0.10,
        "vix": 15.0,
        "yield_curve_inverted": False,
        "yield_curve_spread": 0.02,
    }


def _all_levels(out: Dict[str, Any]) -> list:
    return [v for v in out["levels"].values()]


# ---------------------------------------------------------------------------
# Z-score helper unit tests
# ---------------------------------------------------------------------------
class TestZScoreHelpers:
    def test_higher_is_safer_sign(self):
        # x above the mean of a higher-is-safer metric -> positive z (safe)
        z = zscore_risk_signed(15.0, mean=10.0, std=5.0,
                               direction="higher_is_safer", n=30)
        assert z == pytest.approx(1.0)

    def test_lower_is_safer_sign(self):
        # x above the mean of a lower-is-safer metric -> negative z (risky)
        z = zscore_risk_signed(15.0, mean=10.0, std=5.0,
                               direction="lower_is_safer", n=30)
        assert z == pytest.approx(-1.0)

    def test_shrinkage_kicks_in_below_n_floor(self):
        # n=1 vs n=30 with identical mean/std/value -> shrunken z smaller in magnitude
        full = zscore_risk_signed(0, 10, 5, "lower_is_safer", n=30)
        small = zscore_risk_signed(0, 10, 5, "lower_is_safer", n=1)
        assert abs(small) < abs(full)

    def test_zero_std_returns_neutral(self):
        assert zscore_risk_signed(10, 10, 0, "higher_is_safer") == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            zscore_risk_signed(1, 1, 1, "sideways")

    @pytest.mark.parametrize("z,tier", [
        (-2.0, "HIGH"),
        (-1.5, "MEDIUM"),    # boundary: z == _Z_TIER_HIGH falls to MEDIUM
        (-1.0, "MEDIUM"),
        (-0.5, "LOW"),       # boundary: z == _Z_TIER_MEDIUM falls to LOW
        (0.0, "LOW"),
        (2.0, "LOW"),
    ])
    def test_ztier_mapping(self, z, tier):
        assert ztier(z) == tier


# ---------------------------------------------------------------------------
# Sector-aware: REIT vs software (the canonical mis-fire case)
# ---------------------------------------------------------------------------
class TestSectorRelativeScoring:
    def test_reit_with_high_debt_relative_to_peers_is_not_high(
        self, config, benign_macro
    ):
        """REIT at 8x debt/EBITDA with REIT peers at 7.5x±1.8 should NOT
        be HIGH risk. Under the old absolute rule, 8x > 3x -> HIGH.
        Under the new sector-z rule, z = (7.5 - 8) / 1.8 = -0.28 -> LOW.
        """
        metrics = {
            "sector": "Real Estate",
            "debt_ebitda": 8.0,
            "debt_to_ebitda": 8.0,
            "interest_coverage": 2.8,
            "fcf_yield": 0.06,
        }
        out = compute_risk_flags("REIT_X", metrics, benign_macro, [], config)
        assert out["levels"]["fundamental"] in {"LOW", "MEDIUM"}, (
            f"REIT at peer-typical leverage should not be HIGH; got "
            f"{out['levels']['fundamental']} | reasons={out['reasons']['fundamental']}"
        )
        # And the combined level should not be HIGH either, since macro/market are clean.
        assert out["levels"]["combined"] in {"LOW", "MEDIUM"}

    def test_software_with_same_debt_ratio_is_flagged_high(
        self, config, benign_macro
    ):
        """Same 8x debt/EBITDA but sector=Technology where peers average
        0.8x±0.6 should be HIGH (z roughly -12).
        """
        metrics = {
            "sector": "Technology",
            "debt_ebitda": 8.0,
            "debt_to_ebitda": 8.0,
            "interest_coverage": 5.0,
            "fcf_yield": 0.03,
        }
        out = compute_risk_flags("SOFT_X", metrics, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH", (
            f"Software at REIT-level debt should be HIGH; got "
            f"{out['levels']['fundamental']} | reasons={out['reasons']['fundamental']}"
        )

    def test_explicit_peer_stats_override_sector_defaults(
        self, config, benign_macro
    ):
        """User-supplied peer_stats kwarg overrides sector defaults."""
        metrics = {
            "sector": "Real Estate",     # default would say LOW at 8x
            "debt_ebitda": 8.0,
            "debt_to_ebitda": 8.0,
            "interest_coverage": 5.0,
            "fcf_yield": 0.05,
        }
        # Tight peer cohort: mean 2x, std 0.5 -> z = -12, HIGH.
        peer_stats = {
            "debt_ebitda": {"mean": 2.0, "std": 0.5, "n": 20},
        }
        out = compute_risk_flags(
            "X", metrics, benign_macro, [], config,
            peer_stats=peer_stats,
        )
        assert out["levels"]["fundamental"] == "HIGH"

    def test_no_sector_no_peers_uses_absolute_threshold(
        self, config, benign_macro, benign_metrics
    ):
        """Backwards-compat: snapshot with no sector / no peers should
        behave like the old absolute-threshold scanner.
        """
        # Stress debt/EBITDA -> should hit absolute HIGH at 8.5x > 3.0x.
        metrics = dict(benign_metrics, debt_ebitda=8.5, debt_to_ebitda=8.5)
        out = compute_risk_flags("X", metrics, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH"


# ---------------------------------------------------------------------------
# Distress models: unit tests for each
# ---------------------------------------------------------------------------
class TestAltmanZ:
    def test_safe_firm_returns_low_distress(self):
        # Strong balance sheet: lots of equity, positive earnings.
        metrics = {
            "sector": "Industrials",
            "total_assets": 1000.0,
            "total_liabilities": 300.0,
            "working_capital": 200.0,
            "retained_earnings": 400.0,
            "ebit": 150.0,
            "revenue": 800.0,
            "market_cap": 2000.0,
        }
        out = compute_altman_z(metrics)
        assert out is not None
        assert out["variant"] == "original"
        assert out["zone"] == "safe"
        assert out["distress"] == pytest.approx(0.0)
        assert out["z"] > 2.99

    def test_distressed_firm_returns_high_distress(self):
        # Negative retained earnings, weak EBIT, debt-heavy.
        metrics = {
            "sector": "Industrials",
            "total_assets": 1000.0,
            "total_liabilities": 900.0,
            "working_capital": -50.0,
            "retained_earnings": -300.0,
            "ebit": 5.0,
            "revenue": 400.0,
            "market_cap": 50.0,
        }
        out = compute_altman_z(metrics)
        assert out is not None
        assert out["zone"] == "distress"
        assert out["distress"] >= 0.8
        assert out["z"] < 1.81

    def test_picks_zdoubleprime_for_financials(self):
        metrics = {
            "sector": "Financial Services",
            "total_assets": 1000.0,
            "total_liabilities": 800.0,
            "working_capital": 100.0,
            "retained_earnings": 100.0,
            "ebit": 20.0,
            # Note: no revenue, no market_cap — Z'' doesn't need them.
        }
        out = compute_altman_z(metrics)
        assert out is not None
        assert out["variant"] == "zdoubleprime"

    def test_returns_none_when_inputs_missing(self):
        assert compute_altman_z({"sector": "Industrials"}) is None


class TestPiotroski:
    def test_strong_firm_high_f_score(self):
        metrics = {
            "net_income": 100.0,
            "net_income_prev": 80.0,
            "operating_cash_flow": 120.0,  # > NI -> accrual quality OK
            "total_assets": 1000.0,
            "total_assets_prev": 950.0,
            "long_term_debt": 200.0,
            "long_term_debt_prev": 250.0,
            "current_ratio": 2.2,
            "current_ratio_prev": 2.0,
            "shares_outstanding": 1000.0,
            "shares_outstanding_prev": 1000.0,
            "gross_margin": 0.48,
            "gross_margin_prev": 0.45,
            "revenue": 500.0,
            "revenue_prev": 420.0,
        }
        out = compute_piotroski_f(metrics)
        assert out is not None
        assert out["f"] >= 7  # all 9 should pass for this snapshot
        assert out["distress"] < 0.5

    def test_weak_firm_low_f_score(self):
        metrics = {
            "net_income": -50.0,           # fail #1
            "net_income_prev": -30.0,      # ROA worsening
            "operating_cash_flow": -20.0,  # fail #2
            "total_assets": 1000.0,
            "total_assets_prev": 950.0,
            "long_term_debt": 400.0,       # leverage up
            "long_term_debt_prev": 300.0,
            "current_ratio": 0.9,          # liquidity down
            "current_ratio_prev": 1.4,
            "shares_outstanding": 1200.0,  # dilution
            "shares_outstanding_prev": 1000.0,
            "gross_margin": 0.30,          # GM down
            "gross_margin_prev": 0.40,
            "revenue": 400.0,              # turnover roughly flat
            "revenue_prev": 420.0,
        }
        out = compute_piotroski_f(metrics)
        assert out is not None
        assert out["f"] <= 2
        assert out["distress"] >= 0.75

    def test_returns_none_when_too_few_tests_evaluable(self):
        # Only NI and OCF available — 2 of 9 tests; below the 5-test floor.
        out = compute_piotroski_f({
            "net_income": 100.0,
            "operating_cash_flow": 80.0,
        })
        assert out is None


class TestBeneishM:
    def test_clean_firm_low_m_score(self):
        # A boring boring company: all ratios near 1.0, near-zero accruals.
        metrics = {
            "revenue": 1000.0, "revenue_prev": 950.0,
            "accounts_receivable": 100.0, "accounts_receivable_prev": 95.0,
            "gross_margin": 0.40, "gross_margin_prev": 0.40,
            "current_assets": 300.0, "current_assets_prev": 290.0,
            "ppe": 400.0, "ppe_prev": 390.0,
            "total_assets": 1000.0, "total_assets_prev": 970.0,
            "depreciation": 40.0, "depreciation_prev": 38.0,
            "sga": 100.0, "sga_prev": 95.0,
            "net_income": 80.0, "operating_cash_flow": 85.0,  # NI <= OCF
            "long_term_debt": 200.0, "long_term_debt_prev": 195.0,
            "current_liabilities": 150.0, "current_liabilities_prev": 145.0,
        }
        out = compute_beneish_m(metrics)
        assert out is not None
        assert out["m"] < -1.78  # below manipulator threshold
        assert out["distress"] < 0.5

    def test_manipulator_high_m_score(self):
        # Receivables ballooning, sales surging, accruals huge.
        metrics = {
            "revenue": 2000.0, "revenue_prev": 1000.0,        # SGI 2.0
            "accounts_receivable": 600.0, "accounts_receivable_prev": 100.0,  # AR/Sales jump
            "gross_margin": 0.30, "gross_margin_prev": 0.45,  # GM collapsed
            "current_assets": 500.0, "current_assets_prev": 400.0,
            "ppe": 300.0, "ppe_prev": 400.0,
            "total_assets": 1500.0, "total_assets_prev": 1000.0,
            "depreciation": 20.0, "depreciation_prev": 40.0,  # dep rate collapsed
            "sga": 250.0, "sga_prev": 100.0,
            "net_income": 200.0, "operating_cash_flow": 30.0, # huge accruals
            "long_term_debt": 600.0, "long_term_debt_prev": 200.0,
            "current_liabilities": 300.0, "current_liabilities_prev": 150.0,
        }
        out = compute_beneish_m(metrics)
        assert out is not None
        assert out["m"] > -1.78
        assert out["distress"] >= 0.8

    def test_returns_none_when_too_few_ratios(self):
        out = compute_beneish_m({
            "revenue": 1000.0, "revenue_prev": 950.0,
        })
        assert out is None


# ---------------------------------------------------------------------------
# Integration: distress signals escalate (but never demote)
# ---------------------------------------------------------------------------
class TestDistressEscalation:
    def test_canonical_distressed_firm_triggers_high(self, config, benign_macro):
        """A firm whose ratios look OK in isolation but Altman flags as
        distressed should still surface HIGH risk via escalation.
        """
        # Per-metric values look unremarkable individually (debt 2.5x is
        # MEDIUM under absolute; FCF yield 3% is LOW), so without distress
        # signals we'd land MEDIUM. Altman should escalate to HIGH.
        metrics = {
            # Per-metric: moderate
            "debt_ebitda": 2.5, "debt_to_ebitda": 2.5,
            "interest_coverage": 3.0,
            "fcf_yield": 0.03,
            # Altman inputs: clearly distressed
            "sector": "Industrials",
            "total_assets": 1000.0,
            "total_liabilities": 950.0,
            "working_capital": -100.0,
            "retained_earnings": -400.0,
            "ebit": -20.0,
            "revenue": 300.0,
            "market_cap": 30.0,
        }
        out = compute_risk_flags("DISTRESS", metrics, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH"
        reasons_blob = " ".join(out["reasons"]["fundamental"]).lower()
        assert "altman" in reasons_blob, (
            f"expected Altman to appear in reasons; got {out['reasons']['fundamental']}"
        )

    def test_distress_does_not_demote_existing_high(self, config, benign_macro):
        """Escalation-only: if per-metric is already HIGH and distress
        signals show 'safe', the level stays HIGH.
        """
        metrics = {
            # Per-metric: HIGH via 8.5x debt and 1.5x coverage
            "debt_ebitda": 8.5, "debt_to_ebitda": 8.5,
            "interest_coverage": 1.5,
            "fcf_yield": 0.01,
            # Altman inputs: clearly safe
            "sector": "Industrials",
            "total_assets": 1000.0,
            "total_liabilities": 200.0,
            "working_capital": 300.0,
            "retained_earnings": 500.0,
            "ebit": 200.0,
            "revenue": 1000.0,
            "market_cap": 3000.0,
        }
        out = compute_risk_flags("MIXED", metrics, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH"

    def test_no_distress_inputs_no_escalation(self, config, benign_macro,
                                              benign_metrics):
        """Without Altman/Piotroski/Beneish inputs, no escalation occurs."""
        out = compute_risk_flags("X", benign_metrics, benign_macro, [], config)
        # Reason list should not mention Altman/Piotroski/Beneish.
        blob = " ".join(out["reasons"]["fundamental"]).lower()
        assert "altman" not in blob
        assert "piotroski" not in blob
        assert "beneish" not in blob


# ---------------------------------------------------------------------------
# Public API contract regression
# ---------------------------------------------------------------------------
class TestApiContract:
    def test_shape_unchanged(self, config, benign_metrics, benign_macro):
        out = compute_risk_flags("X", benign_metrics, benign_macro, [], config)
        assert set(out.keys()) == {"levels", "reasons"}
        assert set(out["levels"].keys()) == {
            "fundamental", "macro", "market", "agent", "combined",
        }
        for lvl in out["levels"].values():
            assert lvl in VALID_LEVELS
        assert set(out["reasons"].keys()) == {
            "fundamental", "macro", "market", "agent",
        }
        for v in out["reasons"].values():
            assert isinstance(v, list)
            assert all(isinstance(x, str) for x in v)

    def test_existing_call_signature_still_positional(
        self, config, benign_metrics, benign_macro
    ):
        """The existing 5-positional-arg call form must still work."""
        out = compute_risk_flags(
            "X", benign_metrics, benign_macro, [], config
        )
        assert isinstance(out, dict) and "levels" in out

    def test_peer_stats_is_keyword_only(self, config, benign_metrics,
                                       benign_macro):
        """peer_stats is keyword-only — positional passing must fail."""
        with pytest.raises(TypeError):
            compute_risk_flags("X", benign_metrics, benign_macro, [], config,
                               {"debt_ebitda": {"mean": 1, "std": 1, "n": 5}})

    def test_benign_inputs_are_low(self, config, benign_metrics, benign_macro):
        out = compute_risk_flags("X", benign_metrics, benign_macro, [], config)
        # Per-metric LOW, no distress, no macro/market stress, no agent risks.
        assert out["levels"]["fundamental"] == "LOW"
        assert out["levels"]["macro"] == "LOW"
        assert out["levels"]["market"] == "LOW"
        assert out["levels"]["agent"] == "LOW"
        assert out["levels"]["combined"] == "LOW"

    def test_deterministic(self, config, benign_metrics, benign_macro):
        out1 = compute_risk_flags("X", benign_metrics, benign_macro, [], config)
        out2 = compute_risk_flags("X", benign_metrics, benign_macro, [], config)
        assert out1 == out2


# ---------------------------------------------------------------------------
# Legacy regression: rules that fired before must still fire
# ---------------------------------------------------------------------------
class TestLegacyRegression:
    def test_high_debt_ebitda_no_sector(self, config, benign_metrics,
                                        benign_macro):
        m = dict(benign_metrics, debt_to_ebitda=8.5, debt_ebitda=8.5)
        out = compute_risk_flags("X", m, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH"
        assert any("debt" in r.lower() or "ebitda" in r.lower()
                   for r in out["reasons"]["fundamental"])

    def test_three_year_negative_fcf_streak(self, config, benign_metrics,
                                            benign_macro):
        m = dict(
            benign_metrics,
            free_cash_flow=-500_000_000.0,
            free_cash_flow_3y=[-100e6, -300e6, -500e6],
        )
        out = compute_risk_flags("X", m, benign_macro, [], config)
        assert out["levels"]["fundamental"] == "HIGH"

    def test_macro_stress_still_works(self, config, benign_metrics):
        stressed = {
            "recession_probability": 0.70,
            "vix": 32.0,
            "yield_curve_inverted": True,
            "yield_curve_spread": -0.005,
        }
        out = compute_risk_flags("X", benign_metrics, stressed, [], config)
        assert out["levels"]["macro"] == "HIGH"

    def test_many_agent_risks_escalate(self, config, benign_metrics,
                                       benign_macro):
        risks = ["a", "b", "c", "d", "e"]
        out = compute_risk_flags("X", benign_metrics, benign_macro, risks,
                                 config)
        assert out["levels"]["agent"] == "HIGH"
