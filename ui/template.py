"""
report/template.py
==================
Layout, typography, and section-ordering config for the PDF report.

Splitting these from the renderer means future redesigns (different
branding, larger fonts, alternate section order, A4 vs Letter) are a
single-file edit — the renderer doesn't care about colors or section
ordering, it just iterates this config.

Three things live here:

  * ``DocTheme``     — page size, margins, color palette, font sizes.
  * ``REPORT_VERSION`` and ``RENDERER_VERSION`` — surface in metadata
    so re-rendered PDFs can be traced back to the code that produced them.
  * ``SECTION_ORDER`` — canonical ordering of sections in the memo. Each
    item maps to a builder function in ``content_builder.py``. Re-order
    or omit entries here to change the memo layout without touching
    renderer code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch


# =============================================================================
# Versions (surface in metadata)
# =============================================================================

REPORT_VERSION = "ary-fund-memo:v1"
RENDERER_VERSION = "reportlab-platypus"


# =============================================================================
# Section ordering — change this to reorder the memo
# =============================================================================
#
# Each entry is the canonical name of a section. ``content_builder.py``
# exposes a function ``build_<name>(ctx)`` for each. Sections with no
# data return a labeled placeholder rather than disappearing.
#
# If you want to remove a section permanently, delete it here. If you
# want to add a new section, write ``build_<new>(ctx)`` in
# content_builder.py and add its name here.

SECTION_ORDER = (
    "title_page",
    "executive_summary",
    "thesis",
    "key_metrics",
    "charts",
    "risk_commentary",
    "supporting_context",
    "appendix",
)


# =============================================================================
# Theme
# =============================================================================


@dataclass(frozen=True)
class DocTheme:
    """Typography + layout config. Frozen so a theme can be shared across
    concurrent renderer instances without surprise mutation."""

    # ---- Page geometry ----
    page_size: tuple = LETTER
    margin_left: float = 0.85 * inch
    margin_right: float = 0.85 * inch
    margin_top: float = 0.9 * inch
    margin_bottom: float = 0.9 * inch

    # ---- Colors ----
    color_primary: colors.Color = colors.HexColor("#0B3D91")
    color_secondary: colors.Color = colors.HexColor("#3D4F66")
    color_muted: colors.Color = colors.HexColor("#7A8499")
    color_warn: colors.Color = colors.HexColor("#B6470E")
    color_good: colors.Color = colors.HexColor("#0E7F4D")
    color_rule: colors.Color = colors.HexColor("#CFD6E0")
    color_table_bg_alt: colors.Color = colors.HexColor("#F4F6F9")

    # ---- Fonts ----
    font_body: str = "Helvetica"
    font_body_bold: str = "Helvetica-Bold"
    font_body_italic: str = "Helvetica-Oblique"
    font_mono: str = "Courier"

    # ---- Sizes (points) ----
    size_title: float = 26
    size_subtitle: float = 16
    size_h1: float = 16
    size_h2: float = 13
    size_h3: float = 11
    size_body: float = 10
    size_small: float = 8.5
    size_caption: float = 8

    # ---- Spacing ----
    leading_body: float = 14
    space_after_para: float = 6

    # ---- Chart sizing ----
    chart_max_width: float = 6.5 * inch
    chart_max_height: float = 3.5 * inch

    def styles(self) -> dict[str, ParagraphStyle]:
        """Build a registry of ParagraphStyle objects keyed by role.

        Returned fresh per call so callers can tweak one style without
        leaking changes into other documents.
        """
        base = getSampleStyleSheet()["Normal"]
        body = ParagraphStyle(
            "Body", parent=base,
            fontName=self.font_body, fontSize=self.size_body,
            leading=self.leading_body,
            textColor=colors.HexColor("#1A1F2B"),
            spaceAfter=self.space_after_para,
        )
        return {
            "Title": ParagraphStyle(
                "Title", parent=body,
                fontName=self.font_body_bold, fontSize=self.size_title,
                leading=self.size_title + 4,
                textColor=self.color_primary,
                spaceAfter=12, alignment=0,
            ),
            "Subtitle": ParagraphStyle(
                "Subtitle", parent=body,
                fontName=self.font_body, fontSize=self.size_subtitle,
                leading=self.size_subtitle + 4,
                textColor=self.color_secondary,
                spaceAfter=18,
            ),
            "H1": ParagraphStyle(
                "H1", parent=body,
                fontName=self.font_body_bold, fontSize=self.size_h1,
                leading=self.size_h1 + 3,
                textColor=self.color_primary,
                spaceBefore=14, spaceAfter=8,
                keepWithNext=1,  # prevent orphan headers
            ),
            "H2": ParagraphStyle(
                "H2", parent=body,
                fontName=self.font_body_bold, fontSize=self.size_h2,
                leading=self.size_h2 + 2,
                textColor=self.color_secondary,
                spaceBefore=10, spaceAfter=6,
                keepWithNext=1,
            ),
            "H3": ParagraphStyle(
                "H3", parent=body,
                fontName=self.font_body_bold, fontSize=self.size_h3,
                leading=self.size_h3 + 1,
                textColor=self.color_secondary,
                spaceBefore=8, spaceAfter=4,
                keepWithNext=1,
            ),
            "Body": body,
            "BodyMuted": ParagraphStyle(
                "BodyMuted", parent=body,
                textColor=self.color_muted,
            ),
            "Caption": ParagraphStyle(
                "Caption", parent=body,
                fontName=self.font_body_italic,
                fontSize=self.size_caption,
                leading=self.size_caption + 2,
                textColor=self.color_muted,
                spaceAfter=10, alignment=1,  # centered
            ),
            "Meta": ParagraphStyle(
                "Meta", parent=body,
                fontSize=self.size_small,
                textColor=self.color_muted,
                leading=self.size_small + 2,
            ),
            "Placeholder": ParagraphStyle(
                "Placeholder", parent=body,
                fontName=self.font_body_italic,
                textColor=self.color_warn,
                backColor=colors.HexColor("#FFF4E6"),
                borderColor=self.color_warn,
                borderWidth=0,
                borderPadding=4,
                leftIndent=4, rightIndent=4,
                spaceAfter=10,
            ),
            "Mono": ParagraphStyle(
                "Mono", parent=body,
                fontName=self.font_mono, fontSize=self.size_small,
                leading=self.size_small + 2,
            ),
        }


DEFAULT_THEME = DocTheme()
