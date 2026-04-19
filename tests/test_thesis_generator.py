"""
Unit tests for agent.thesis_generator.generate_thesis.

Real logic is exercised; only inputs vary.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def thesis_generator(require_module):
    return require_module("agent.thesis_generator", "generate_thesis")


@pytest.fixture
def benign_filings_summary(sample_filings):
    """A plausible filings summary in the shape the thesis generator expects."""
    return {
        "2024": {
            "10-K": "Revenue grew 12% Y/Y; margins expanded.",
            "10-Q": "Sustained momentum; FCF positive.",
            "8-K": "CFO transition announced; guidance reaffirmed.",
        },
        "2023": {"10-K": "Prior-year strength with 15% revenue growth."},
    }


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
class TestThesisContract:
    def test_required_keys_present(
        self,
        thesis_generator,
        benign_filings_summary,
        sample_metrics,
        sample_macro,
        thesis_contract,
    ):
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []},
        )
        missing = thesis_contract["required_keys"] - set(out.keys())
        assert not missing, f"thesis missing required keys: {missing}"

    def test_time_horizon_is_1y(
        self, thesis_generator, benign_filings_summary, sample_metrics, sample_macro
    ):
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []},
        )
        assert out["time_horizon"] == "1Y", (
            f"time_horizon must be exactly '1Y'; got {out['time_horizon']!r}"
        )

    def test_confidence_in_unit_interval(
        self, thesis_generator, benign_filings_summary, sample_metrics, sample_macro
    ):
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []},
        )
        c = out["confidence"]
        assert isinstance(c, (int, float))
        assert 0.0 <= float(c) <= 1.0

    def test_outlook_and_direction_from_allowed_set(
        self,
        thesis_generator,
        benign_filings_summary,
        sample_metrics,
        sample_macro,
        thesis_contract,
    ):
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []},
        )
        assert out["outlook"] in thesis_contract["valid_outlooks"]
        assert out["price_direction"] in thesis_contract["valid_price_directions"]

    def test_lists_are_lists(
        self, thesis_generator, benign_filings_summary, sample_metrics, sample_macro
    ):
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []},
        )
        assert isinstance(out["key_risks"], list)
        assert isinstance(out["key_opportunities"], list)


# ---------------------------------------------------------------------------
# Directional bias
# ---------------------------------------------------------------------------
class TestDirectionalBias:
    def test_risky_inputs_tilt_bearish_or_down(
        self,
        thesis_generator,
        benign_filings_summary,
        risky_metrics,
        stressed_macro,
    ):
        risky_flags = {
            "levels": {"fundamental": "high", "macro": "high"},
            "reasons": ["leverage", "recession probability elevated"],
        }
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            risky_metrics,
            stressed_macro,
            risky_flags,
        )
        assert out["outlook"] in {"bearish", "neutral"}, (
            f"risky inputs should not produce bullish outlook; got {out['outlook']}"
        )
        assert out["price_direction"] in {
            "moderate_down",
            "strong_down",
            "flat",
        }, f"risky inputs should not produce upward direction; got {out['price_direction']}"

    def test_favorable_inputs_tilt_bullish_or_up(
        self,
        thesis_generator,
        benign_filings_summary,
        sample_metrics,
        sample_macro,
    ):
        favorable_flags = {
            "levels": {"fundamental": "low", "macro": "low"},
            "reasons": [],
        }
        out = thesis_generator.generate_thesis(
            "TEST",
            benign_filings_summary,
            sample_metrics,
            sample_macro,
            favorable_flags,
        )
        assert out["outlook"] in {"bullish", "neutral"}, (
            f"favorable inputs should not produce bearish outlook; got {out['outlook']}"
        )
        assert out["price_direction"] in {
            "moderate_up",
            "strong_up",
            "flat",
        }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_same_inputs_produce_same_output(
        self,
        thesis_generator,
        benign_filings_summary,
        sample_metrics,
        sample_macro,
    ):
        flags = {"levels": {"fundamental": "low", "macro": "low"}, "reasons": []}
        a = thesis_generator.generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro, flags
        )
        b = thesis_generator.generate_thesis(
            "TEST", benign_filings_summary, sample_metrics, sample_macro, flags
        )
        assert a == b, "generate_thesis must be deterministic for identical inputs"
