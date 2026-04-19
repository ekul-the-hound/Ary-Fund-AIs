"""
Unit tests for agent.risk_scanner.compute_risk_flags.

These tests exercise *real* risk logic (we do not mock the scanner);
we only vary inputs to drive each branch.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def risk_scanner(require_module):
    return require_module("agent.risk_scanner", "compute_risk_flags")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def _all_level_values(levels: dict) -> list[str]:
    """Flatten risk levels (which may be nested or flat) to a list of strings."""
    out = []
    for v in levels.values():
        if isinstance(v, dict):
            out.extend(str(x) for x in v.values())
        else:
            out.append(str(v))
    return out


# ---------------------------------------------------------------------------
# Structure contract
# ---------------------------------------------------------------------------
class TestStructure:
    def test_returns_levels_and_reasons(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        out = risk_scanner.compute_risk_flags(
            "TEST", sample_metrics, sample_macro, [], sample_config
        )
        assert isinstance(out, dict)
        assert "levels" in out, f"missing 'levels' key. Got: {list(out.keys())}"
        assert "reasons" in out, f"missing 'reasons' key. Got: {list(out.keys())}"
        assert isinstance(out["levels"], dict)
        assert isinstance(out["reasons"], (list, dict))

    def test_all_levels_are_valid(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        out = risk_scanner.compute_risk_flags(
            "TEST", sample_metrics, sample_macro, [], sample_config
        )
        for lvl in _all_level_values(out["levels"]):
            assert lvl in VALID_LEVELS, (
                f"invalid risk level {lvl!r}; must be one of {VALID_LEVELS}"
            )


# ---------------------------------------------------------------------------
# Rule triggers
# ---------------------------------------------------------------------------
class TestFundamentalRiskRules:
    def test_high_debt_to_ebitda_triggers_high_fundamental_risk(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        m = dict(sample_metrics, debt_to_ebitda=8.5)
        out = risk_scanner.compute_risk_flags(
            "TEST", m, sample_macro, [], sample_config
        )
        levels = _all_level_values(out["levels"])
        assert "HIGH" in levels, (
            f"expected some 'HIGH' level after debt/EBITDA spike; got {out['levels']}"
        )
        # reason text should reference leverage somewhere
        reasons_blob = str(out["reasons"]).lower()
        assert any(
            kw in reasons_blob for kw in ("debt", "leverage", "ebitda")
        ), f"expected leverage mention in reasons; got {out['reasons']}"

    def test_three_years_negative_fcf_triggers_high_fundamental_risk(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        m = dict(
            sample_metrics,
            free_cash_flow=-500_000_000.0,
            free_cash_flow_3y=[-100_000_000.0, -300_000_000.0, -500_000_000.0],
        )
        out = risk_scanner.compute_risk_flags(
            "TEST", m, sample_macro, [], sample_config
        )
        levels = _all_level_values(out["levels"])
        assert "HIGH" in levels, (
            "sustained negative FCF should surface a 'HIGH' risk level"
        )


class TestMacroRiskRules:
    def test_high_recession_probability_triggers_high_macro_risk(
        self, risk_scanner, sample_metrics, stressed_macro, sample_config
    ):
        out = risk_scanner.compute_risk_flags(
            "TEST", sample_metrics, stressed_macro, [], sample_config
        )
        levels = _all_level_values(out["levels"])
        assert "HIGH" in levels
        reasons_blob = str(out["reasons"]).lower()
        assert any(
            kw in reasons_blob
            for kw in ("recession", "macro", "vix", "yield")
        )


class TestAgentRiskAggregation:
    def test_many_agent_risks_raise_combined_risk(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        few = risk_scanner.compute_risk_flags(
            "TEST", sample_metrics, sample_macro, ["minor concern"], sample_config
        )
        many = risk_scanner.compute_risk_flags(
            "TEST",
            sample_metrics,
            sample_macro,
            [
                "regulatory investigation",
                "customer concentration",
                "cyber incident",
                "pending litigation",
                "management turnover",
                "supply-chain shock",
            ],
            sample_config,
        )
        severity = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

        def _score(flags):
            return max(severity.get(l, 0) for l in _all_level_values(flags["levels"]))

        assert _score(many) >= _score(few), (
            "more agent-identified risks should not lower overall risk score"
        )


class TestLowRiskBaseline:
    def test_benign_inputs_are_not_all_high(
        self, risk_scanner, sample_metrics, sample_macro, sample_config
    ):
        out = risk_scanner.compute_risk_flags(
            "TEST", sample_metrics, sample_macro, [], sample_config
        )
        levels = _all_level_values(out["levels"])
        assert not all(l == "HIGH" for l in levels), (
            "benign inputs should not yield uniformly 'HIGH' risk; "
            f"got {out['levels']}"
        )