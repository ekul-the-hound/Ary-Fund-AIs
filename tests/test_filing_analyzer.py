"""
Unit tests for agent.filing_analyzer.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def filing_analyzer(require_module):
    return require_module(
        "agent.filing_analyzer",
        "summarize_filings_by_year",
        "extract_key_metrics_for_agent",
    )


# ---------------------------------------------------------------------------
# summarize_filings_by_year
# ---------------------------------------------------------------------------
class TestSummarizeFilings:
    def test_returns_a_dict(self, filing_analyzer, sample_filings):
        out = filing_analyzer.summarize_filings_by_year("TEST", sample_filings)
        assert isinstance(out, dict), "summary must be a dict"

    def test_summary_is_non_empty_for_valid_input(
        self, filing_analyzer, sample_filings
    ):
        out = filing_analyzer.summarize_filings_by_year("TEST", sample_filings)
        assert len(out) > 0

    def test_summary_references_filing_types_or_years(
        self, filing_analyzer, sample_filings
    ):
        """
        We don't pin the schema too tightly, but the summary should
        at minimum reference filing form types or filing years.
        """
        out = filing_analyzer.summarize_filings_by_year("TEST", sample_filings)
        flat = str(out).lower()
        hits_form = any(
            ft.lower() in flat for ft in ("10-k", "10-q", "8-k")
        )
        hits_year = any(y in flat for y in ("2023", "2024"))
        assert hits_form or hits_year, (
            "summary should group or reference form types / years; got: "
            f"{out!r}"
        )

    def test_empty_filings_list_is_handled_gracefully(self, filing_analyzer):
        out = filing_analyzer.summarize_filings_by_year("TEST", [])
        assert isinstance(out, dict), "must return a dict even on empty input"

        # Empty contract: either the dict is literally empty, OR every value
        # is itself empty/None, OR the output clearly signals emptiness via
        # a sentinel string ("empty", "no filings", etc.). No real filing
        # data — dates, form types, metric values — should leak through.
        if out == {}:
            return

        values_are_empty = all(
            v in (None, "", [], {}, 0) or v in ("empty", "no filings", "none")
            for v in out.values()
        )
        flat = str(out).lower()
        signals_empty = "empty" in flat or "no filings" in flat or "no data" in flat

        # Belt-and-suspenders: none of the sample_filings dates/forms should
        # appear, since we passed []. This catches accidental fixture bleed.
        assert "2024" not in flat and "2023" not in flat, (
            f"empty input must not reference filing years; got {out!r}"
        )
        assert "10-k" not in flat and "10-q" not in flat and "8-k" not in flat, (
            f"empty input must not reference form types; got {out!r}"
        )

        assert values_are_empty or signals_empty, (
            f"empty-filings output must be empty, contain only empty values, "
            f"or clearly signal 'empty'/'no filings'; got {out!r}"
        )

    def test_respects_max_filings_cap(self, filing_analyzer, sample_filings):
        """
        With max_filings=2, the analyzer must not process more than two
        filings. We detect this by counting how many filing dates / form
        types end up referenced anywhere in the output.
        """
        out = filing_analyzer.summarize_filings_by_year(
            "TEST", sample_filings, max_filings=2
        )
        flat = str(out)

        # Count how many of our 5 distinct filing dates appear in the output.
        all_dates = [f["filing_date"] for f in sample_filings]
        present = sum(1 for d in all_dates if d in flat)
        assert present <= 2, (
            f"expected <= 2 filings referenced, saw {present}. Output: {out!r}"
        )


# ---------------------------------------------------------------------------
# extract_key_metrics_for_agent
# ---------------------------------------------------------------------------
class TestExtractMetrics:
    CRITICAL_METRIC_KEYS = {
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "free_cash_flow",
        "debt_to_ebitda",
        "pe_ratio",
    }

    def test_returns_dict_with_critical_keys(
        self, filing_analyzer, sample_metrics
    ):
        out = filing_analyzer.extract_key_metrics_for_agent(
            "TEST", sample_metrics, price=sample_metrics["price"]
        )
        assert isinstance(out, dict)
        present = self.CRITICAL_METRIC_KEYS & set(out.keys())
        # At least half the critical keys must survive normalization.
        assert len(present) >= len(self.CRITICAL_METRIC_KEYS) // 2, (
            f"normalized metrics missing too many critical keys. "
            f"Got: {sorted(out.keys())}"
        )

    def test_handles_missing_values_without_crashing(self, filing_analyzer):
        """
        Only a partial metrics dict should still produce output, not raise.
        """
        partial = {
            "revenue_growth_yoy": 0.10,
            "gross_margin": None,          # explicitly missing
            # operating_margin absent entirely
            "debt_to_ebitda": 2.1,
        }
        out = filing_analyzer.extract_key_metrics_for_agent(
            "TEST", partial, price=100.0
        )
        assert isinstance(out, dict)

    def test_price_is_propagated_or_preserved(
        self, filing_analyzer, sample_metrics
    ):
        out = filing_analyzer.extract_key_metrics_for_agent(
            "TEST", sample_metrics, price=123.45
        )
        # accept either a top-level `price` or embedded in any value
        assert "price" in out or 123.45 in (
            v for v in out.values() if isinstance(v, (int, float))
        ) or "123.45" in str(out)