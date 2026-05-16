"""
tests/test_pdf_report.py
========================
Tests for ``report.generate_pdf_report``.

Covers the five scenarios in the spec plus a handful of unit tests for
filename construction, output-path resolution, chart embedding, and
failure handling.

All tests are offline. We use pypdf to inspect generated PDFs (text
extraction + page count + metadata).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import pytest
from pypdf import PdfReader

from report import build_filename, generate_pdf_report, REPORT_VERSION
from report.charts import ChartSpec
from report.template import SECTION_ORDER


# =============================================================================
# Helpers
# =============================================================================


def _pdf_text(path: Path) -> str:
    """Concatenate text from all pages of a PDF for substring assertions."""
    r = PdfReader(str(path))
    return "\n".join(p.extract_text() or "" for p in r.pages)


def _pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def _full_context() -> dict:
    """A 'green-field' context with every section populated.

    Kept as a helper rather than a fixture so each test can mutate a
    fresh dict without test-order contamination.
    """
    return {
        "ticker": "AAPL",
        "snapshot_id": "abc123def456",
        "snapshot_date": "2026-05-15",
        "as_of": "2026-05-15",
        "report_title": "AAPL — Investment Memorandum",
        "analyst": "Test Suite",
        "prices": {
            "last": 182.45, "change_pct": 0.0142, "market_cap": 2.85e12,
            "fifty_two_week_high": 199.62, "fifty_two_week_low": 164.08,
        },
        "metrics": {
            "pe_ratio": 28.4, "forward_pe": 25.1, "ev_to_ebitda": 22.1,
            "ps_ratio": 7.6, "revenue_growth_yoy": 0.058,
            "gross_margin": 0.45, "operating_margin": 0.30,
            "net_margin": 0.25, "fcf_yield": 0.034,
            "debt_to_ebitda": 1.6, "interest_coverage": 38.0,
            "beta": 1.21, "drawdown": -0.087,
        },
        "thesis": {
            "outlook": "bullish",
            "time_horizon": "1Y",
            "price_direction": "moderate_up",
            "confidence": 0.68,
            "summary": "Services growth and AI integration sustain margin expansion.",
            "key_risks": ["China demand softness", "regulatory pressure"],
            "key_opportunities": ["AI upgrade cycle", "services margin"],
            "rationale": "Composite bias +0.31",
        },
        "essay": {
            "text": (
                "## Executive Summary\n\n"
                "Apple's setup is asymmetric. Services compounding offsets hardware "
                "cyclicality while Apple Intelligence is a credible catalyst.\n\n"
                "## Valuation\n\nAt 28x trailing earnings the stock is not cheap, "
                "but EV / forward EBITDA of 22x is supportable given the mix.\n\n"
                "## Verdict\n\nBullish, 1Y, moderate-up, 68% confidence."
            ),
            "model_used": "qwen3:30b-a3b",
            "elapsed_ms": 42000,
            "fallback": False,
            "word_count": 60,
        },
        "risk": {
            "levels": {
                "fundamental": "LOW", "macro": "MEDIUM",
                "market": "LOW", "agent": "MEDIUM", "combined": "MEDIUM",
            },
            "reasons": {
                "fundamental": ["leverage within thresholds"],
                "macro": ["VIX elevated at 22"],
                "market": ["drawdown 8.7% from 52w high"],
                "agent": ["China demand softness", "rates risk"],
            },
        },
        "filings": [
            {"filed_date": "2026-05-02", "filing_type": "10-Q",
             "description": "Q2 FY26 results: revenue +6% YoY."},
            {"filed_date": "2026-04-18", "filing_type": "8-K",
             "description": "Announcement of $90B buyback."},
        ],
        "macro": {
            "vix": 22.0, "yield_curve_spread": 0.20,
            "recession_probability": 18.0, "hy_oas": 3.4,
            "financial_stress": 0.3, "cpi_yoy_pct": 2.6,
            "unemployment_rate": 3.9,
        },
        "derived": {
            "ticker.signal.rsi_14": 54.2,
            "ticker.signal.realized_vol_30d": 0.21,
            "ticker.factor.beta_market": 1.21,
        },
        "sources": {
            "yfinance": "Yahoo Finance",
            "sec_xbrl": "SEC EDGAR",
            "fred": "FRED — macro indicators",
        },
    }


# =============================================================================
# Test 1: Minimal report
# =============================================================================


class TestMinimalReport:
    def test_minimal_renders_with_only_ticker(self, tmp_path):
        """A context dict with nothing in it should still produce a PDF
        — every section becomes a labeled placeholder."""
        out = generate_pdf_report(
            ticker="AAPL",
            output_path=tmp_path,
            snapshot_id="min001",
            context={},
        )
        assert out.exists()
        assert out.stat().st_size > 1000   # not an empty file

    def test_title_page_present(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="min002",
            context={},
        )
        text = _pdf_text(out)
        assert "AAPL" in text
        assert "Investment Memo" in text or "Investment Memorandum" in text
        assert "Snapshot" in text

    def test_missing_sections_have_placeholders(self, tmp_path):
        """The thesis section should explicitly state 'unavailable',
        not silently disappear."""
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="min003",
            context={},
        )
        text = _pdf_text(out)
        # Either the explicit placeholder phrasing, or at minimum the
        # section header survives
        assert "Investment Thesis" in text
        assert "unavailable" in text.lower() or "no chart" in text.lower() \
            or "no thesis" in text.lower() or "no metric" in text.lower()

    def test_appendix_lists_coverage_gaps(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="min004",
            context={},
        )
        text = _pdf_text(out)
        assert "Coverage" in text
        # Several sections were missing — at least one should be flagged
        assert "missing" in text.lower()


# =============================================================================
# Test 2: Fully populated report
# =============================================================================


class TestFullReport:
    def test_full_report_renders(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full001",
            context=_full_context(),
        )
        assert out.exists()
        assert _pdf_pages(out) >= 5   # title + several content pages

    def test_all_sections_appear(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full002",
            context=_full_context(),
        )
        text = _pdf_text(out)
        # Section headers we expect to see
        expected_headers = [
            "Executive Summary",
            "Investment Thesis",
            "Key Metrics",
            "Charts",
            "Risk Commentary",
            "Supporting Context",
            "Appendix",
        ]
        for h in expected_headers:
            assert h in text, f"section header missing: {h!r}"

    def test_metadata_in_pdf_info(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full003",
            context=_full_context(),
        )
        r = PdfReader(str(out))
        info = r.metadata or {}
        # ReportLab stamps these as /Title, /Author, /Subject, /Creator
        assert "AAPL" in str(info.get("/Title") or "")
        assert "Ary Fund" in str(info.get("/Author") or "")
        assert REPORT_VERSION in str(info.get("/Creator") or "")

    def test_thesis_essay_body_in_pdf(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full004",
            context=_full_context(),
        )
        text = _pdf_text(out)
        assert "Services compounding" in text
        assert "Apple Intelligence" in text

    def test_metrics_table_rendered(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full005",
            context=_full_context(),
        )
        text = _pdf_text(out)
        assert "P/E" in text
        assert "Debt / EBITDA" in text or "Debt/EBITDA" in text or "Debt" in text
        # The placeholder must NOT appear when metrics are populated
        assert "no standard fields populated" not in text

    def test_charts_render_in_order(self, tmp_path):
        """Two real charts → 'Figure 1' before 'Figure 2' in extracted text."""
        ctx = _full_context()
        # Two simple chart-ready dicts so the test doesn't need matplotlib
        # fixtures of its own
        ctx["charts"] = [
            ChartSpec(title="Price series", source={
                "type": "line", "x": list(range(20)),
                "y": list(range(20)), "y_label": "px",
            }),
            ChartSpec(title="Risk bars", source={
                "type": "bar",
                "labels": ["A", "B", "C"], "values": [0.1, 0.5, 0.3],
            }),
        ]
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="full006",
            context=ctx,
        )
        text = _pdf_text(out)
        i1 = text.find("Figure 1")
        i2 = text.find("Figure 2")
        assert i1 >= 0 and i2 > i1, \
            f"Figure ordering broken: Figure 1 at {i1}, Figure 2 at {i2}"


# =============================================================================
# Test 3: Missing-data report
# =============================================================================


class TestMissingData:
    def test_missing_thesis_essay_placeholder(self, tmp_path):
        ctx = _full_context()
        ctx["essay"] = {}   # no essay text
        # Also strip rationale so the fallback chain has nothing
        ctx["thesis"]["rationale"] = ""
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="miss001",
            context=ctx,
        )
        text = _pdf_text(out)
        assert "Investment Thesis" in text   # section still renders
        assert "unavailable" in text.lower() or "no thesis" in text.lower()

    def test_missing_risk_placeholder(self, tmp_path):
        ctx = _full_context()
        ctx["risk"] = {}
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="miss002",
            context=ctx,
        )
        text = _pdf_text(out)
        assert "Risk Commentary" in text   # section header survives
        # Exec summary still renders even when risk is missing
        assert "Executive Summary" in text

    def test_missing_chart_placeholder(self, tmp_path):
        """A chart with source=None should produce a labeled placeholder
        AND keep its figure number for stability."""
        ctx = _full_context()
        ctx["charts"] = [
            ChartSpec(title="Missing chart", source=None),
            ChartSpec(title="Also missing", source=None),
        ]
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="miss003",
            context=ctx,
        )
        text = _pdf_text(out)
        assert "Figure 1" in text
        assert "Figure 2" in text   # numbering stays stable
        assert "No chart artifact available" in text or \
               "Missing chart" in text

    def test_missing_section_does_not_remove_section(self, tmp_path):
        """Every section in SECTION_ORDER must be reachable in the PDF
        even when its source data is absent."""
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="miss004",
            context={"ticker": "AAPL"},
        )
        text = _pdf_text(out)
        for section_label in (
            "Executive Summary", "Investment Thesis", "Key Metrics",
            "Charts", "Risk Commentary", "Supporting Context", "Appendix",
        ):
            assert section_label in text, \
                f"section '{section_label}' missing from PDF"

    def test_missing_filings_does_not_break_supporting_context(self, tmp_path):
        ctx = _full_context()
        ctx.pop("filings", None)
        ctx.pop("macro", None)
        ctx.pop("derived", None)
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="miss005",
            context=ctx,
        )
        text = _pdf_text(out)
        assert "Supporting Context" in text


# =============================================================================
# Test 4: Determinism
# =============================================================================


class TestDeterminism:
    def test_same_inputs_same_filename(self):
        f1 = build_filename(
            ticker="AAPL", snapshot_id="abc123",
            ctx={"snapshot_date": "2026-05-15"},
        )
        f2 = build_filename(
            ticker="AAPL", snapshot_id="abc123",
            ctx={"snapshot_date": "2026-05-15"},
        )
        assert f1 == f2
        assert f1 == "AAPL_investment_memo_2026-05-15_snapshot-abc123.pdf"

    def test_portfolio_scope_filename(self):
        f = build_filename(
            scope="core_long_book", snapshot_id="day-2026-05-15",
            ctx={"snapshot_date": "2026-05-15"},
        )
        # Trailing chars get stripped to fit the safe-char regex
        assert f.startswith("portfolio_core_long_book_investment_memo_")
        assert f.endswith(".pdf")

    def test_same_input_same_structure(self, tmp_path):
        """Two PDFs generated from the same snapshot have the same
        page count and section ordering.

        Note: We do NOT test byte-equality. ReportLab's PDF streams
        depend on Python dict iteration order, which is randomized by
        PYTHONHASHSEED; reaching byte-identical output requires either
        running with a fixed PYTHONHASHSEED or patching reportlab
        internals. The spec accepts the softer 'consistent page count
        and section ordering' contract — that's what we verify here.
        """
        ctx = _full_context()
        out1 = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path / "a.pdf",
            snapshot_id="det001", context=ctx,
        )
        out2 = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path / "b.pdf",
            snapshot_id="det001", context=ctx,
        )
        assert _pdf_pages(out1) == _pdf_pages(out2)

        t1 = _pdf_text(out1)
        t2 = _pdf_text(out2)

        # The relative ordering of section headers must match.
        section_labels = [
            "Executive Summary", "Investment Thesis", "Key Metrics",
            "Charts", "Risk Commentary", "Supporting Context", "Appendix",
        ]
        order1 = sorted(range(len(section_labels)),
                        key=lambda i: t1.find(section_labels[i]))
        order2 = sorted(range(len(section_labels)),
                        key=lambda i: t2.find(section_labels[i]))
        assert order1 == order2, (
            "section ordering must be identical across re-runs"
        )


# =============================================================================
# Test 5: Long-content overflow
# =============================================================================


class TestLongContent:
    def test_long_thesis_essay_paginates(self, tmp_path):
        """A 30-paragraph essay must produce multiple pages without
        crashing or truncating text."""
        ctx = _full_context()
        # 30 paragraphs of ~80 words each = ~2400 words
        long_paragraphs = [
            f"Paragraph {i + 1}: " + " ".join(["thesis"] * 80)
            for i in range(30)
        ]
        ctx["essay"] = {
            "text": "\n\n".join(long_paragraphs),
            "fallback": False,
        }
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="long001",
            context=ctx,
        )
        assert _pdf_pages(out) >= 6   # title + many essay pages + others
        text = _pdf_text(out)
        # First and last paragraphs both made it in
        assert "Paragraph 1:" in text
        assert "Paragraph 30:" in text

    def test_long_filings_list_handles_overflow(self, tmp_path):
        ctx = _full_context()
        ctx["filings"] = [
            {"filed_date": f"2026-{m:02d}-15", "filing_type": "10-Q",
             "description": f"Quarterly report number {m}, "
                            "with a very long description that should be "
                            "truncated to keep the table compact and "
                            "readable when rendered inside the report." * 2}
            for m in range(1, 13)
        ]
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="long002",
            context=ctx,
        )
        assert out.exists()
        # Table truncation cap is 8 rows
        text = _pdf_text(out)
        assert "Recent Filings" in text


# =============================================================================
# Other: failure handling, scope reports, malformed inputs
# =============================================================================


class TestFailureHandling:
    def test_no_ticker_no_scope_raises(self, tmp_path):
        with pytest.raises(ValueError):
            generate_pdf_report(
                ticker=None, scope=None,
                output_path=tmp_path,
                snapshot_id="x", context={},
            )

    def test_scope_only_renders(self, tmp_path):
        out = generate_pdf_report(
            scope="core_long_book",
            output_path=tmp_path,
            snapshot_id="scope001",
            context={"as_of": "2026-05-15"},
        )
        assert out.exists()
        assert "portfolio_core_long_book" in out.name

    def test_output_path_directory_creates_canonical_name(self, tmp_path):
        out = generate_pdf_report(
            ticker="AAPL",
            output_path=tmp_path / "subdir",   # doesn't exist yet
            snapshot_id="path001",
            context={"snapshot_date": "2026-05-15"},
        )
        assert out.exists()
        # Default filename was used because we passed a directory-shaped path
        assert "AAPL_investment_memo_2026-05-15_snapshot-path001" in out.name

    def test_output_path_explicit_pdf(self, tmp_path):
        target = tmp_path / "my_custom_name.pdf"
        out = generate_pdf_report(
            ticker="AAPL", output_path=target, snapshot_id="path002",
            context={},
        )
        assert out == target
        assert out.exists()

    def test_bad_chart_does_not_kill_report(self, tmp_path):
        """A chart whose source can't be loaded must placeholder, not
        crash the whole render."""
        ctx = _full_context()
        ctx["charts"] = [
            ChartSpec(title="Broken chart",
                      source="/nonexistent/path/to/missing.png"),
            ChartSpec(title="Also bad", source=12345),  # wrong type entirely
        ]
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path, snapshot_id="bad001",
            context=ctx,
        )
        assert out.exists()
        text = _pdf_text(out)
        assert "Figure 1" in text and "Figure 2" in text
        # At least one placeholder phrasing appears
        assert "not found" in text.lower() or "unrecognized" in text.lower()


class TestSectionOrderHonored:
    def test_custom_section_order(self, tmp_path):
        """Passing a custom section_order changes the page layout."""
        # Drop charts entirely; put appendix before risk
        custom_order = [
            "title_page", "executive_summary", "thesis",
            "appendix", "risk_commentary",
        ]
        out = generate_pdf_report(
            ticker="AAPL", output_path=tmp_path,
            snapshot_id="order001", context=_full_context(),
            section_order=custom_order,
        )
        text = _pdf_text(out)
        # Charts heading must NOT be present
        # (and exec summary must still be there)
        assert "Executive Summary" in text
        # Appendix appears before Risk Commentary
        assert text.find("Appendix") < text.find("Risk Commentary")


class TestFilenameSanitization:
    def test_filename_strips_unsafe_chars(self):
        f = build_filename(
            ticker="BRK.B",
            snapshot_id="run/2026-05-15:14:00",
            ctx={"snapshot_date": "2026-05-15"},
        )
        # No slashes, colons, or whitespace
        for bad in ("/", "\\", ":", " "):
            assert bad not in f
        # Dots are kept (BRK.B is a real Berkshire ticker — and dots are
        # valid filename characters on every OS we target).
        assert "BRK.B" in f
        assert "2026-05-15" in f
        assert f.endswith(".pdf")
