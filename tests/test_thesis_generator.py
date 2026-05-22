"""
tests/test_thesis_generator.py
==============================
Coverage for the upgraded ``agent.thesis_generator.generate_thesis``.

Three test layers:

1. **Existing contract preserved** — every assertion from the legacy
   block in ``test_all.py`` (TestThesisContract, TestDirectionalBias,
   TestDeterminism) is replicated here against the upgraded module.
   Backward compatibility is non-negotiable: the learning loop,
   ``portfolio_db.save_agent_opinion``, and ``main._assemble_final_opinion``
   all depend on the original field shape.

2. **New fields** — ``short_rationale``, ``thesis_markdown``,
   ``valuation`` are present, well-formed, and deterministic. The
   Markdown contains every section we promised, the YAML frontmatter
   parses, and the valuation math is self-consistent.

3. **Determinism is strict** — two back-to-back calls with the same
   inputs produce byte-identical ``thesis_markdown``. This rules out
   any wall-clock leakage (which is the most likely silent regression
   when adding date fields).
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from agent.thesis_generator import generate_thesis


# ======================================================================
# Fixtures (mirror what conftest.py provides in the real project)
# ======================================================================
@pytest.fixture
def benign_filings_summary() -> Dict[str, Any]:
    return {
        "management_tone": "confident",
        "red_flags": [],
        "risk_factors": [],
        "by_year": {
            "2024": {
                "10-K": "Revenue grew 12% Y/Y; margins expanded.",
                "10-Q": "Sustained momentum; FCF positive.",
                "8-K": "CFO transition announced; guidance reaffirmed.",
            },
            "2023": {"10-K": "Prior-year strength with 15% revenue growth."},
        },
    }


@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    return {
        "ticker": "TEST",
        "price": 150.0,
        "revenue_growth_yoy": 0.10,
        "revenue_growth_3y": 0.12,
        "revenue_growth_5y": 0.11,
        "gross_margin": 0.55,
        "operating_margin": 0.25,
        "profit_margin": 0.18,
        "margin_trend": "expanding",
        "debt_to_ebitda": 1.5,
        "net_debt_to_ebitda": 1.0,
        "interest_coverage": 12.0,
        "free_cash_flow": 5_000_000_000.0,
        "cash_conversion": 1.05,
        "p_e": 25.0,
        "pe_ratio": 25.0,
        "forward_pe": 22.0,
        "ev_ebitda": 18.0,
        "fcf_yield": 0.06,
        "market_cap": 100_000_000_000.0,
        "roic": 0.18,
    }


@pytest.fixture
def risky_metrics(sample_metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(sample_metrics)
    out.update({
        "revenue_growth_yoy": -0.05,
        "revenue_growth_3y": -0.08,
        "gross_margin": 0.30,
        "operating_margin": 0.02,
        "profit_margin": -0.01,
        "margin_trend": "compressing",
        "debt_ebitda": 6.0,
        "debt_to_ebitda": 6.0,
        "interest_coverage": 1.2,
        "free_cash_flow": -500_000_000.0,
        "cash_flow_negative_3_years": True,
        "cash_conversion": 0.5,
        "fcf_yield": -0.02,
        "p_e": 80.0,
        "roic": -0.05,
    })
    return out


@pytest.fixture
def sample_macro() -> Dict[str, Any]:
    return {
        "recession_probability": 0.10,
        "vix": 15.0,
        "yield_curve_inverted": False,
        "yield_curve_spread": 0.02,
        "as_of": "2026-05-21",
    }


@pytest.fixture
def stressed_macro() -> Dict[str, Any]:
    return {
        "recession_probability": 0.70,
        "vix": 32.0,
        "yield_curve_inverted": True,
        "yield_curve_spread": -0.005,
        "as_of": "2026-05-21",
    }


@pytest.fixture
def favorable_flags() -> Dict[str, Any]:
    return {
        "levels": {
            "fundamental": "low",
            "macro": "low",
            "combined": "LOW",
        },
        "reasons": [],
    }


@pytest.fixture
def risky_flags() -> Dict[str, Any]:
    return {
        "levels": {
            "fundamental": "high",
            "macro": "high",
            "combined": "HIGH",
        },
        "reasons": ["leverage", "recession probability elevated"],
    }


# ======================================================================
# Layer 1: Backward-compatible contract (verbatim from test_all.py)
# ======================================================================
class TestLegacyContract:

    def test_required_keys_present(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        required = {
            "outlook", "time_horizon", "price_direction",
            "confidence", "key_risks", "key_opportunities",
        }
        missing = required - set(out.keys())
        assert not missing, f"thesis missing required keys: {missing}"

    def test_time_horizon_is_1y(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert out["time_horizon"] == "1Y"

    def test_confidence_in_unit_interval(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        c = out["confidence"]
        assert isinstance(c, (int, float))
        assert 0.0 <= float(c) <= 1.0

    def test_outlook_and_direction_from_allowed_set(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert out["outlook"] in {"bullish", "neutral", "bearish"}
        assert out["price_direction"] in {
            "strong_up", "moderate_up", "flat",
            "moderate_down", "strong_down",
        }

    def test_lists_are_lists(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert isinstance(out["key_risks"], list)
        assert isinstance(out["key_opportunities"], list)

    def test_rationale_field_preserved(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        # Legacy field — main._assemble_final_opinion reads thesis["rationale"]
        # and stores it under the same key in the persisted opinion.
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "rationale" in out
        assert isinstance(out["rationale"], str)
        assert len(out["rationale"]) > 0

    def test_bias_score_field_preserved(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "bias_score" in out
        assert -1.0 <= float(out["bias_score"]) <= 1.0


# ======================================================================
# Layer 1b: Directional bias preserved
# ======================================================================
class TestDirectionalBias:

    def test_risky_inputs_tilt_bearish_or_down(
        self, benign_filings_summary, risky_metrics, stressed_macro,
        risky_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, risky_metrics, stressed_macro,
            risky_flags,
        )
        assert out["outlook"] in {"bearish", "neutral"}
        assert out["price_direction"] in {"moderate_down", "strong_down", "flat"}

    def test_favorable_inputs_tilt_bullish_or_up(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert out["outlook"] in {"bullish", "neutral"}
        assert out["price_direction"] in {"moderate_up", "strong_up", "flat"}


# ======================================================================
# Layer 1c: Determinism (the critical one)
# ======================================================================
class TestDeterminism:

    def test_same_inputs_produce_identical_dict(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        a = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        b = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert a == b

    def test_thesis_markdown_byte_identical(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        # Strongest determinism check — if a wall-clock leaked into a
        # date field, the markdown would diverge at sub-second resolution.
        a = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        b = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert a["thesis_markdown"] == b["thesis_markdown"]
        assert a["thesis_markdown"].encode() == b["thesis_markdown"].encode()

    def test_determinism_without_macro_as_of(
        self, benign_filings_summary, sample_metrics, favorable_flags,
    ):
        # When no as_of date is in macro, the placeholder must be the
        # deterministic literal 'unknown', not datetime.now().
        macro_no_date = {"recession_probability": 0.10, "vix": 15.0}
        a = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, macro_no_date,
            favorable_flags,
        )
        b = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, macro_no_date,
            favorable_flags,
        )
        assert a == b
        assert "created: unknown" in a["thesis_markdown"]


# ======================================================================
# Layer 2a: New field — short_rationale
# ======================================================================
class TestShortRationale:

    def test_short_rationale_present_and_string(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "short_rationale" in out
        assert isinstance(out["short_rationale"], str)
        assert len(out["short_rationale"]) > 0

    def test_short_rationale_includes_ticker_and_verdict(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "NVDA", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        sr = out["short_rationale"]
        assert "NVDA" in sr
        assert out["outlook"] in sr
        assert out["price_direction"] in sr

    def test_short_rationale_is_single_line(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "\n" not in out["short_rationale"]


# ======================================================================
# Layer 2b: New field — thesis_markdown
# ======================================================================
class TestThesisMarkdown:

    @pytest.fixture
    def md_output(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ) -> str:
        return generate_thesis(
            "NVDA", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )["thesis_markdown"]

    def test_thesis_markdown_present_and_string(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "thesis_markdown" in out
        assert isinstance(out["thesis_markdown"], str)
        assert len(out["thesis_markdown"]) > 100

    def test_yaml_frontmatter_present(self, md_output: str):
        lines = md_output.split("\n")
        assert lines[0] == "---"
        close_idx = next(
            (i for i, ln in enumerate(lines[1:], start=1) if ln == "---"),
            None,
        )
        assert close_idx is not None
        assert close_idx > 1

    def test_yaml_contains_required_fields(self, md_output: str):
        for key in (
            "type: thesis_note",
            "ticker: NVDA",
            "sector:",
            "status: open",
            "created:",
            "outlook:",
            "confidence:",
            "bias_score:",
            "combined_risk:",
        ):
            assert key in md_output, f"YAML missing '{key}'"

    def test_all_section_headings_present(self, md_output: str):
        for heading in (
            "# NVDA — 1Y Thesis",
            "## Snapshot",
            "## Bull case",
            "## Bear case",
            "## Catalysts",
            "## Risks",
            "## Valuation summary",
            "## Monitoring plan",
        ):
            assert heading in md_output, f"missing heading: {heading!r}"

    def test_sections_appear_in_correct_order(self, md_output: str):
        order = [
            "## Snapshot", "## Bull case", "## Bear case",
            "## Catalysts", "## Risks",
            "## Valuation summary", "## Monitoring plan",
        ]
        positions = [md_output.index(h) for h in order]
        assert positions == sorted(positions), (
            f"sections out of order: positions={positions}"
        )

    def test_bull_section_has_bullets_for_favorable_inputs(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "AAPL", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        md = out["thesis_markdown"]
        bull_section = md.split("## Bull case")[1].split("## Bear case")[0]
        bear_section = md.split("## Bear case")[1].split("## Catalysts")[0]
        n_bull = bull_section.count("\n- ")
        n_bear = bear_section.count("\n- ")
        assert n_bull >= 3, f"expected >=3 bull bullets, got {n_bull}"
        assert n_bull > n_bear, (
            f"favorable inputs should yield more bull ({n_bull}) "
            f"than bear ({n_bear}) bullets"
        )

    def test_bear_section_dominates_for_risky_inputs(
        self, benign_filings_summary, risky_metrics, stressed_macro,
        risky_flags,
    ):
        out = generate_thesis(
            "BAD", benign_filings_summary, risky_metrics, stressed_macro,
            risky_flags,
        )
        md = out["thesis_markdown"]
        bull_section = md.split("## Bull case")[1].split("## Bear case")[0]
        bear_section = md.split("## Bear case")[1].split("## Catalysts")[0]
        n_bull = bull_section.count("\n- ")
        n_bear = bear_section.count("\n- ")
        assert n_bear >= 3, f"expected >=3 bear bullets, got {n_bear}"
        assert n_bear > n_bull, (
            f"risky inputs should yield more bear ({n_bear}) "
            f"than bull ({n_bull}) bullets"
        )

    def test_verdict_line_matches_machine_fields(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        md = out["thesis_markdown"]
        assert out["outlook"] in md
        assert out["price_direction"] in md


# ======================================================================
# Layer 2c: New field — valuation
# ======================================================================
class TestValuation:

    def test_valuation_present_and_dict(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        assert "valuation" in out
        assert isinstance(out["valuation"], dict)

    def test_valuation_shape_complete(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        v = out["valuation"]
        required = {
            "current_price", "trailing_pe", "forward_pe",
            "implied_eps", "target_multiple",
            "fair_value_low", "fair_value_mid", "fair_value_high",
            "upside_to_mid_pct", "basis", "missing_inputs",
        }
        assert set(v.keys()) == required

    def test_valuation_math_self_consistent(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        v = out["valuation"]
        assert v["current_price"] == pytest.approx(150.0)
        assert v["forward_pe"] == pytest.approx(22.0)
        assert v["implied_eps"] == pytest.approx(150.0 / 22.0, rel=1e-3)
        # mid = EPS * target_multiple
        assert v["fair_value_mid"] == pytest.approx(
            v["implied_eps"] * v["target_multiple"], rel=1e-3
        )
        # Range = mid * (1 ± 0.15)
        assert v["fair_value_low"] == pytest.approx(
            v["fair_value_mid"] * 0.85, rel=1e-3
        )
        assert v["fair_value_high"] == pytest.approx(
            v["fair_value_mid"] * 1.15, rel=1e-3
        )

    def test_valuation_missing_price_explicit(
        self, benign_filings_summary, sample_macro, favorable_flags,
    ):
        metrics_no_price = {"forward_pe": 22.0, "revenue_growth_3y": 0.10}
        out = generate_thesis(
            "TEST", benign_filings_summary, metrics_no_price, sample_macro,
            favorable_flags,
        )
        v = out["valuation"]
        assert v["current_price"] is None
        assert v["implied_eps"] is None
        assert v["fair_value_mid"] is None
        assert v["fair_value_low"] is None
        assert v["fair_value_high"] is None
        assert v["upside_to_mid_pct"] is None
        assert "current_price" in v["missing_inputs"]
        assert "missing" in v["basis"].lower()

    def test_valuation_missing_inputs_shown_in_markdown(
        self, benign_filings_summary, sample_macro, favorable_flags,
    ):
        metrics_no_price = {"forward_pe": 22.0, "revenue_growth_3y": 0.10}
        out = generate_thesis(
            "TEST", benign_filings_summary, metrics_no_price, sample_macro,
            favorable_flags,
        )
        md = out["thesis_markdown"]
        val_section = md.split("## Valuation summary")[1].split(
            "## Monitoring plan"
        )[0]
        assert (
            "unavailable" in val_section.lower()
            or "missing" in val_section.lower()
        )
        assert "Fair value (mid) | $0" not in val_section

    def test_valuation_target_multiple_grows_with_growth(
        self, benign_filings_summary, sample_macro, favorable_flags,
    ):
        low_growth = {
            "price": 150.0, "forward_pe": 22.0, "p_e": 25.0,
            "revenue_growth_3y": 0.02,
        }
        high_growth = {
            "price": 150.0, "forward_pe": 22.0, "p_e": 25.0,
            "revenue_growth_3y": 0.50,
        }
        v_low = generate_thesis(
            "TEST", benign_filings_summary, low_growth, sample_macro,
            favorable_flags,
        )["valuation"]
        v_high = generate_thesis(
            "TEST", benign_filings_summary, high_growth, sample_macro,
            favorable_flags,
        )["valuation"]
        assert v_high["target_multiple"] > v_low["target_multiple"]

    def test_valuation_target_multiple_capped(
        self, benign_filings_summary, sample_macro, favorable_flags,
    ):
        extreme = {
            "price": 150.0, "forward_pe": 22.0, "p_e": 25.0,
            "revenue_growth_3y": 5.0,
        }
        v = generate_thesis(
            "TEST", benign_filings_summary, extreme, sample_macro,
            favorable_flags,
        )["valuation"]
        assert v["target_multiple"] <= 40.0 + 1e-6


# ======================================================================
# Layer 2d: Monitoring plan
# ======================================================================
class TestMonitoringPlan:
    """Monitoring plan is now rendered into the Markdown body. We test
    both the rendered output and that the predicates we promised are
    present."""

    def test_monitoring_section_has_all_four_triggers(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        md = out["thesis_markdown"]
        section = md.split("## Monitoring plan")[1]
        # All four triggers labeled
        for label in ("Price move", "Earnings", "Risk escalation", "Fundamentals"):
            assert label in section, f"monitoring missing trigger: {label}"

    def test_monitoring_includes_price_move_threshold(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        # Spec: ±15% price-move trigger
        assert "0.15" in out["thesis_markdown"]

    def test_monitoring_includes_fundamentals_predicate(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        section = out["thesis_markdown"].split("## Monitoring plan")[1]
        # The fundamentals deceleration predicate
        assert "revenue_growth_yoy" in section
        assert "deceleration" in section.lower()


# ======================================================================
# Layer 2e: Downstream-contract acceptance
# ======================================================================
class TestDownstreamContract:
    """The original return shape (read by main._assemble_final_opinion,
    save_agent_opinion, the learning loop) must still be intact."""

    def test_all_original_keys_still_present(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        original = {
            "ticker", "outlook", "time_horizon", "price_direction",
            "confidence", "key_risks", "key_opportunities",
            "rationale", "bias_score",
        }
        missing = original - set(out.keys())
        assert not missing, f"original contract broken — missing: {missing}"

    def test_only_three_new_keys_added(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        expected = {
            "ticker", "outlook", "time_horizon", "price_direction",
            "confidence", "key_risks", "key_opportunities",
            "rationale", "bias_score",
            "short_rationale", "thesis_markdown", "valuation",
        }
        unexpected = set(out.keys()) - expected
        assert not unexpected, (
            f"unexpected new keys (update contract test if intentional): "
            f"{unexpected}"
        )

    def test_thesis_markdown_obsidian_safe(
        self, benign_filings_summary, sample_metrics, sample_macro,
        favorable_flags,
    ):
        # Obsidian (and most YAML parsers) require frontmatter to open
        # at byte 0 and close with a '---' line before any content.
        out = generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro,
            favorable_flags,
        )
        md = out["thesis_markdown"]
        assert md.startswith("---\n"), "frontmatter must start at byte 0"
        assert "\n---\n\n#" in md, "frontmatter must close before the title"
