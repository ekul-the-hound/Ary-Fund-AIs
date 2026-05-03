"""
Unit tests for agent.data_point_analyzer parser.

Scope
-----
- ``_normalize_header`` recognizes plain, colon, markdown (#/##/###),
  bold, and combined formats.
- Lines that aren't headers (prose, values, sentence-punctuated text)
  are rejected so they stay in the body.
- ``_split_text`` correctly assembles overview + per-point paragraphs
  for each accepted heading style.
- A genuinely bad output (no recognizable structure) still produces an
  empty parse result, so the existing ``has_structure`` gate in
  ``analyze_data_points`` will trip the fallback as it should.

These tests are imports-only: no Ollama, no streamlit, no network.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def dpa():
    """Import the analyzer module."""
    import agent.data_point_analyzer as mod
    return mod


@pytest.fixture
def selected_keys():
    """A small set of selected keys covering both ``prices`` and ``metrics``."""
    return ["prices.last", "metrics.trailing_pe", "metrics.free_cash_flow"]


# ---------------------------------------------------------------------------
# _normalize_header — accept the formats LLMs actually emit
# ---------------------------------------------------------------------------
class TestNormalizeHeaderAccepts:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Plain
            ("Overview",                      "overview"),
            ("Trailing P/E",                  "trailing p/e"),
            # Colon-terminated (the original "canonical" format)
            ("Overview:",                     "overview"),
            ("Trailing P/E:",                 "trailing p/e"),
            # Markdown headers
            ("# Overview",                    "overview"),
            ("## Overview",                   "overview"),
            ("### Overview",                  "overview"),
            ("#### Overview",                 "overview"),
            ("## Trailing P/E",               "trailing p/e"),
            # Bold
            ("**Overview**",                  "overview"),
            ("**Trailing P/E**",              "trailing p/e"),
            ("__Overview__",                  "overview"),
            # Bold + colon
            ("**Overview:**",                 "overview"),
            ("**Trailing P/E:**",             "trailing p/e"),
            ("**Overview**:",                 "overview"),
            # Combined: markdown + bold
            ("## **Overview**",               "overview"),
            ("### **Trailing P/E**",          "trailing p/e"),
            ("## **Overview:**",              "overview"),
            # Whitespace tolerance
            ("  Overview  ",                  "overview"),
            ("   ##   Overview   ",           "overview"),
            # Case insensitive
            ("OVERVIEW",                      "overview"),
            ("overview:",                     "overview"),
            ("OvErViEw",                      "overview"),
        ],
    )
    def test_recognizes_heading_format(self, dpa, raw, expected):
        assert dpa._normalize_header(raw) == expected


class TestNormalizeHeaderRejects:
    @pytest.mark.parametrize(
        "raw",
        [
            # Empty / whitespace
            "",
            "   ",
            # Prose — too long
            "This is a normal sentence that goes on and on and on "
            "well past the eighty character ceiling and is definitely not a header",
            # Sentence punctuation
            "NVDA looks attractive.",
            "First, we consider the P/E.",
            "Strong buy; clear setup",
            # Embedded values (these are body content, not labels)
            "Trailing P/E is 40.7x",
            "Price = $199.57",
            "+12.20%",
            "$28.5B in free cash flow",
        ],
    )
    def test_rejects_non_header_line(self, dpa, raw):
        assert dpa._normalize_header(raw) is None


# ---------------------------------------------------------------------------
# _split_text end-to-end — every heading style produces a clean parse
# ---------------------------------------------------------------------------
class TestSplitTextEndToEnd:
    """Each test uses the same content but a different heading style."""

    def _assert_full_parse(self, dpa_mod, text, selected):
        overview, paragraphs = dpa_mod._split_text(text, selected)
        assert overview, "overview paragraph should be parsed"
        assert "NVDA looks attractive" in overview
        # Each selected key must produce a paragraph
        for k in selected:
            assert k in paragraphs, f"missing paragraph for {k}"
            assert paragraphs[k].strip(), f"empty paragraph for {k}"

    def test_colon_headings(self, dpa, selected_keys):
        text = (
            "Overview:\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "Last Price:\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "Trailing P/E:\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "Free Cash Flow:\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_plain_headings(self, dpa, selected_keys):
        text = (
            "Overview\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "Last Price\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "Trailing P/E\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "Free Cash Flow\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_markdown_h2_headings(self, dpa, selected_keys):
        text = (
            "## Overview\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "## Last Price\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "## Trailing P/E\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "## Free Cash Flow\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_markdown_h3_headings(self, dpa, selected_keys):
        text = (
            "### Overview\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "### Last Price\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "### Trailing P/E\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "### Free Cash Flow\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_bold_headings(self, dpa, selected_keys):
        text = (
            "**Overview**\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "**Last Price**\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "**Trailing P/E**\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "**Free Cash Flow**\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_markdown_plus_bold_headings(self, dpa, selected_keys):
        """The phi3 case from the production failure."""
        text = (
            "## **Overview**\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "### **Last Price**\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "### **Trailing P/E**\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "### **Free Cash Flow**\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)

    def test_mixed_styles_in_one_response(self, dpa, selected_keys):
        """Models often drift between formats mid-output. All should still parse."""
        text = (
            "## Overview\n"
            "NVDA looks attractive on growth metrics.\n\n"
            "Last Price:\n"
            "At $199.57, NVDA is below recent highs.\n\n"
            "**Trailing P/E**\n"
            "At 40.7x, premium but supported by growth.\n\n"
            "### **Free Cash Flow:**\n"
            "$28.5B trailing FCF supports buying.\n"
        )
        self._assert_full_parse(dpa, text, selected_keys)


# ---------------------------------------------------------------------------
# Genuinely-bad output: parser must NOT manufacture structure
# ---------------------------------------------------------------------------
class TestSplitTextGenuinelyBadOutput:
    """The validation gate in analyze_data_points checks
    ``has_structure = bool(overview) or len(paragraphs) >= 1``.
    These outputs must produce an empty parse so that gate trips and
    the fallback runs — which is the correct behavior for unusable
    output.
    """

    def test_empty_string_has_no_structure(self, dpa, selected_keys):
        overview, paragraphs = dpa._split_text("", selected_keys)
        assert overview == ""
        assert paragraphs == {}

    def test_refusal_response_has_no_structure(self, dpa, selected_keys):
        text = "I cannot help with that request."
        overview, paragraphs = dpa._split_text(text, selected_keys)
        assert overview == ""
        assert paragraphs == {}

    def test_pure_prose_no_headers_has_no_structure(self, dpa, selected_keys):
        # No recognisable headers — every line is sentence-punctuated prose.
        text = (
            "NVDA is a semiconductor company. The stock trades at "
            "premium multiples. Free cash flow is strong. Analysts rate "
            "it a strong buy."
        )
        overview, paragraphs = dpa._split_text(text, selected_keys)
        assert overview == ""
        assert paragraphs == {}

    def test_unknown_headers_dont_create_paragraphs(self, dpa, selected_keys):
        # Headers exist but don't match any selected display name.
        text = (
            "## Bonus Section\n"
            "This is some content.\n\n"
            "## Random Topic\n"
            "More content.\n"
        )
        overview, paragraphs = dpa._split_text(text, selected_keys)
        assert overview == ""
        assert paragraphs == {}


# ---------------------------------------------------------------------------
# Regression: the original phi3 production failure
# ---------------------------------------------------------------------------
class TestRegressionPhi3MarkdownFailure:
    """Repro of the production failure that motivated this fix.

    Pre-fix behavior: phi3 emitted ``## Overview`` / ``### Last Price`` /
    etc. The parser only matched colon-terminated headers, so it
    returned ``("", {})``. The validator at line 387 then raised
    ``RuntimeError("Output unusable: ... overview=no, paragraphs_parsed=0")``
    and the user got a deterministic fallback even though the LLM
    response was perfectly fine.

    Post-fix behavior: parser recognises markdown headers, validator
    sees real structure, the live LLM output is returned to the UI.
    """

    def test_phi3_style_markdown_response_is_accepted(self, dpa, selected_keys):
        # A response shaped like phi3 actually emits.
        text = (
            "## **Overview**\n"
            "NVDA's recent metrics suggest a HOLD with constructive bias. "
            "Trailing P/E of 40.7x sits well above the S&P 500 average "
            "but is supported by 67% earnings growth and $28.5B FCF.\n\n"
            "### **Last Price**\n"
            "Last price of $199.57 is roughly 7% below the 52-week high "
            "of $215. The pullback creates a more reasonable entry; "
            "supports buying on technical grounds.\n\n"
            "### **Trailing P/E**\n"
            "At 40.7x trailing earnings, NVDA trades at a meaningful "
            "premium to the S&P 500's ~17x. Justified by growth profile; "
            "is neutral.\n\n"
            "### **Free Cash Flow**\n"
            "Trailing $28.5B in free cash flow is exceptional for a "
            "semiconductor company. Strong capital return capacity; "
            "supports buying.\n"
        )

        # Parse must succeed with full structure
        overview, paragraphs = dpa._split_text(text, selected_keys)
        assert overview, "overview must parse"
        assert len(paragraphs) == 3, f"expected 3 paragraphs, got {len(paragraphs)}"

        # And the validator's has_structure check must now pass
        has_structure = bool(overview) or len(paragraphs) >= 1
        assert has_structure, "structure check must pass for valid markdown response"

        # Word count is realistic for a phi3 response — would have
        # comfortably cleared the 100-word floor too.
        word_count = len(text.split())
        assert word_count >= 100
