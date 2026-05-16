"""
report/pdf_renderer.py
======================
Final stage: take a Story (list of ReportLab flowables built by
``content_builder``) and render it to a PDF on disk.

This module owns:

  * the SimpleDocTemplate construction (page size, margins, metadata)
  * the page template (header rule, footer with ticker + page number)
  * deterministic metadata stamping (so re-renders of the same snapshot
    differ only by ``/CreationDate``)
  * the failure-handling contract: if rendering raises, write nothing
    to ``output_path`` (no partial PDFs left on disk).
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
)

from .template import DEFAULT_THEME, REPORT_VERSION, RENDERER_VERSION, DocTheme

logger = logging.getLogger(__name__)


# =============================================================================
# Public entry
# =============================================================================


def render_pdf(
    story: list[Flowable],
    output_path: Path,
    ticker_or_scope: str,
    snapshot_id: str,
    theme: DocTheme = DEFAULT_THEME,
    title: Optional[str] = None,
) -> Path:
    """Render a Story to a PDF at ``output_path``.

    Parameters
    ----------
    story :
        List of ReportLab flowables produced by content_builder.
    output_path :
        Final destination. Parent directories are created if needed.
    ticker_or_scope :
        Used in the page footer.
    snapshot_id :
        Used in the page footer for traceability.
    theme :
        Layout/style theme. Defaults to ``DEFAULT_THEME``.
    title :
        PDF /Title metadata. Defaults to ``"{ticker} Investment Memo"``.

    Returns
    -------
    Path
        The actual path the PDF was written to.

    Raises
    ------
    Any exception from the renderer is re-raised AFTER cleaning up the
    partial output file, so callers never see a corrupted PDF on disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if title is None:
        title = f"{ticker_or_scope} Investment Memo"

    logger.info(
        "pdf_renderer | writing %s (story=%d flowables, snapshot=%s)",
        output_path, len(story), snapshot_id,
    )

    # Render to a temp file first, then move atomically to the final
    # path. That way a crash midway through never leaves a partially-
    # written PDF where downstream consumers might pick it up.
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".pdf",
        prefix=".pending_",
        dir=str(output_path.parent),
    )
    os.close(tmp_fd)
    tmp_path_p = Path(tmp_path)

    try:
        doc = _build_doc(tmp_path_p, theme, title, ticker_or_scope, snapshot_id)
        doc.build(story)
    except Exception:
        # Clean up before re-raising
        try:
            tmp_path_p.unlink(missing_ok=True)
        except OSError:
            logger.warning("pdf_renderer | failed to clean up %s", tmp_path_p)
        raise

    # Atomic-ish replace. On POSIX, Path.replace is atomic when source
    # and dest are on the same filesystem (which they are — we put the
    # tempfile in the same dir).
    tmp_path_p.replace(output_path)
    logger.info("pdf_renderer | wrote %s", output_path)
    return output_path


# =============================================================================
# Document construction
# =============================================================================


def _build_doc(
    output_path: Path,
    theme: DocTheme,
    title: str,
    ticker_or_scope: str,
    snapshot_id: str,
) -> BaseDocTemplate:
    """Build a BaseDocTemplate with our header/footer page template."""

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=theme.page_size,
        leftMargin=theme.margin_left,
        rightMargin=theme.margin_right,
        topMargin=theme.margin_top,
        bottomMargin=theme.margin_bottom,
        title=title,
        author="Ary Fund Research System",
        subject=f"Investment memo: {ticker_or_scope}",
        creator=f"{REPORT_VERSION} ({RENDERER_VERSION})",
    )

    frame = Frame(
        theme.margin_left,
        theme.margin_bottom,
        theme.page_size[0] - theme.margin_left - theme.margin_right,
        theme.page_size[1] - theme.margin_top - theme.margin_bottom,
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
        id="body",
    )

    page_template = PageTemplate(
        id="default",
        frames=[frame],
        onPage=_make_page_decorator(theme, title, ticker_or_scope, snapshot_id),
    )
    doc.addPageTemplates([page_template])
    return doc


def _make_page_decorator(
    theme: DocTheme,
    title: str,
    ticker_or_scope: str,
    snapshot_id: str,
):
    """Return an ``onPage`` callback that draws header & footer on every page.

    Page 1 (the title page) gets no header/footer — the title page is
    self-contained.
    """
    short_snap = snapshot_id[:12] if snapshot_id else ""

    def _decorate(canvas: Canvas, doc) -> None:
        if doc.page == 1:
            return
        canvas.saveState()
        width, height = theme.page_size

        # Top rule + header text
        canvas.setStrokeColor(theme.color_rule)
        canvas.setLineWidth(0.5)
        canvas.line(
            theme.margin_left,
            height - theme.margin_top + 18,
            width - theme.margin_right,
            height - theme.margin_top + 18,
        )
        canvas.setFont(theme.font_body, theme.size_small)
        canvas.setFillColor(theme.color_muted)
        # Left: ticker / scope.  Right: report title (truncated).
        canvas.drawString(
            theme.margin_left,
            height - theme.margin_top + 22,
            _truncate(str(ticker_or_scope), 80),
        )
        canvas.drawRightString(
            width - theme.margin_right,
            height - theme.margin_top + 22,
            _truncate(title, 80),
        )

        # Footer: snapshot ID + page X of Y (Y not available without
        # two-pass build, so we just show page number).
        canvas.setFont(theme.font_body, theme.size_small)
        canvas.setFillColor(theme.color_muted)
        canvas.drawString(
            theme.margin_left,
            theme.margin_bottom - 18,
            f"Snapshot: {short_snap}" if short_snap else "",
        )
        canvas.drawRightString(
            width - theme.margin_right,
            theme.margin_bottom - 18,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    return _decorate


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"
