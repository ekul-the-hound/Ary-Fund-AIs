"""
tests/test_filing_analyzer_tone.py
==================================
Regression guard for management-tone and red-flag calibration in
``agent.filing_analyzer``, validated end-to-end through the downstream
``agent.thesis_generator._score_filings_bias``.

Background
----------
Two bugs were fixed here, in sequence:

1. Filing **text was never hydrated** into the analyzer, so tone was
   permanently ``neutral`` and the filings bias permanently 0.00.
2. Once text flowed in, tone became permanently ``defensive`` because
   severe phrases (``material weakness``, ``restatement``, ``going
   concern``) were matched as bare substrings — and those appear in
   DEFINITIONAL / HYPOTHETICAL form in essentially every 10-K
   ("a material weakness exists when ..."; "if we were to restate ...").
   The same false-positive inflated red-flag counts.

The fix requires AFFIRMATIVE DECLARATION ("we identified a material
weakness", "we restated ...", "substantial doubt about ... going
concern") before a severe condition counts. These tests pin that
behaviour in both directions so a future lexicon tweak can't silently
regress it.

The distressed/clean fixtures mirror real filing language; the targeted
cases isolate the exact phrasings that previously produced false
positives.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agent.filing_analyzer import (
    summarize_filings_by_year,
    _infer_tone,
    _find_red_flags,
)
from agent.thesis_generator import _score_filings_bias


# ======================================================================
# Fixtures
# ======================================================================
def _filing(text: str) -> List[Dict[str, Any]]:
    return [{
        "accession_number": "TEST-0001",
        "filing_type": "10-K",
        "filed_date": "2025-07-30",
        "text": text,
    }]


DISTRESSED = """
ITEM 1A. RISK FACTORS
Our recent operating losses and negative cash flows raise substantial doubt
about our ability to continue as a going concern. Our auditors have included
an explanatory paragraph in their report expressing this substantial doubt.
During fiscal 2025, management identified a material weakness in our internal
control over financial reporting related to revenue recognition. As a result,
we concluded that our disclosure controls and procedures were not effective.
We have restated our previously issued consolidated financial statements for
fiscal 2023 and 2024. We recorded an impairment of goodwill of $412 million.
We were notified by our lenders of a covenant breach under our senior credit
facility.
"""

# Clean filing that INCLUDES the definitional material-weakness boilerplate
# every 10-K carries — this is the exact language that used to break things.
CLEAN = """
ITEM 1A. RISK FACTORS
We face intense competition across all markets for our products and services.
A material weakness is a deficiency, or combination of deficiencies, such that
there is a reasonable possibility that a material misstatement will not be
prevented or detected. A material weakness exists when such a deficiency is
present. Based on management's assessment, our internal control over financial
reporting was effective, and no material weakness was identified.
We delivered record revenue with strong demand, robust growth, expanding
operating margins, and continued market leadership. From time to time we are
subject to litigation and regulatory investigation in the ordinary course of
business.
"""


# ======================================================================
# Tone classification — both directions
# ======================================================================
def test_distressed_text_reads_defensive():
    assert _infer_tone(DISTRESSED.lower()) == "defensive"


def test_clean_text_reads_confident_not_defensive():
    tone = _infer_tone(CLEAN.lower())
    assert tone != "defensive", f"clean filing misread as {tone}"
    assert tone in {"confident", "neutral"}


def test_empty_text_is_neutral():
    assert _infer_tone("") == "neutral"


# ======================================================================
# Severe-term affirmative vs hypothetical/definitional (the core fix)
# ======================================================================
@pytest.mark.parametrize("text", [
    # definitional (every 10-K)
    "a material weakness exists when such a deficiency is present",
    # negated conclusion
    "no material weakness was identified during our assessment",
    "we did not identify any material weakness in our controls",
    # hypothetical risk factor
    "if we identify a material weakness, investors could lose confidence",
    "a restatement of our financial statements could harm our reputation",
])
def test_hypothetical_severe_language_not_defensive(text):
    assert _infer_tone(text) != "defensive"


@pytest.mark.parametrize("text", [
    "during 2025 management identified a material weakness in revenue recognition",
    "we identified a material weakness related to our tax provision",
    "we restated our previously issued financial statements",
    "the company restated its consolidated financial statements",
    "substantial doubt about our ability to continue as a going concern",
])
def test_declared_severe_language_is_defensive(text):
    assert _infer_tone(text) == "defensive"


# ======================================================================
# Red flags — fire on real trouble, silent on boilerplate
# ======================================================================
def test_clean_filing_has_no_red_flags():
    assert _find_red_flags(CLEAN) == []


def test_distressed_filing_has_red_flags():
    flags = _find_red_flags(DISTRESSED)
    assert len(flags) >= 3


# ======================================================================
# End-to-end through the bias scorer
# ======================================================================
def test_distressed_bias_strongly_negative():
    summary = summarize_filings_by_year("TEST", _filing(DISTRESSED))
    assert summary["management_tone"] == "defensive"
    assert _score_filings_bias(summary) <= -0.8


def test_clean_bias_non_negative():
    summary = summarize_filings_by_year("TEST", _filing(CLEAN))
    assert summary["management_tone"] in {"confident", "neutral"}
    assert summary["red_flags"] == []
    assert _score_filings_bias(summary) >= 0.0
