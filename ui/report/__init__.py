"""
report
======
PDF investment-memo generator for the Ary Fund system.

Public entry point:

    >>> from report import generate_pdf_report
    >>> path = generate_pdf_report(
    ...     ticker="AAPL",
    ...     scope=None,
    ...     output_path=Path("./reports"),
    ...     snapshot_id="abc123",
    ...     context=ctx,   # see content_builder.py for accepted keys
    ... )

The function:

  1. Resolves the canonical filename (deterministic from ticker /
     scope / snapshot date / snapshot id).
  2. Builds the Story by iterating ``template.SECTION_ORDER`` and
     calling each section builder in ``content_builder``.
  3. Calls ``pdf_renderer.render_pdf`` to write the PDF atomically.
  4. Returns the actual path on disk.

Failure handling:

  * One bad section never kills the whole report — section builders
    catch their own exceptions and emit a placeholder.
  * If the renderer itself raises, no PDF is written to ``output_path``
    (the tempfile path used during rendering is removed).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .content_builder import build_section
from .pdf_renderer import render_pdf
from .template import DEFAULT_THEME, REPORT_VERSION, RENDERER_VERSION, SECTION_ORDER, DocTheme

logger = logging.getLogger(__name__)


__all__ = [
    "generate_pdf_report",
    "build_filename",
    "REPORT_VERSION",
    "RENDERER_VERSION",
]


# =============================================================================
# Public API
# =============================================================================


def generate_pdf_report(
    ticker: Optional[str] = None,
    scope: Optional[str] = None,
    output_path: Optional[Path] = None,
    snapshot_id: str = "",
    context: Optional[dict] = None,
    *,
    theme: DocTheme = DEFAULT_THEME,
    section_order: Iterable[str] = SECTION_ORDER,
) -> Path:
    """Generate a PDF investment memo.

    Parameters
    ----------
    ticker :
        Single-ticker scope. Mutually exclusive with ``scope``.
    scope :
        Portfolio / watchlist / strategy scope label. Used when no
        single ticker applies. Mutually exclusive with ``ticker``.
    output_path :
        Either a target file path (``.pdf``) or a directory. If a
        directory, the canonical filename is used. If omitted, the
        report is written to ``./reports/`` with the canonical filename.
    snapshot_id :
        Snapshot / run identifier used in the filename, footer, and
        metadata block. Caller is responsible for stable IDs.
    context :
        Snapshot dict. See ``content_builder.py`` for the keys each
        section consumes. Any missing key produces a placeholder rather
        than a render failure.
    theme :
        Layout/style theme. Override for custom branding.
    section_order :
        Iterable of section names. Override (with names from
        ``SECTION_ORDER``) to reorder or skip sections without editing
        the template.

    Returns
    -------
    Path
        The actual on-disk path of the generated PDF.

    Raises
    ------
    ValueError
        If neither ``ticker`` nor ``scope`` is given.

    Determinism
    -----------
    Same inputs produce the same filename, page count, and section
    ordering. Byte-for-byte identity is NOT guaranteed: ReportLab's
    compressed page streams depend on Python dict iteration order
    (PYTHONHASHSEED) and stamps a real /CreationDate on each render.
    """
    ctx = dict(context or {})
    if not ticker and not scope:
        ticker = ctx.get("ticker")
        scope = ctx.get("scope")
    if not ticker and not scope:
        raise ValueError(
            "generate_pdf_report needs at least one of `ticker` or `scope`"
        )

    # Stamp ticker / scope into the context so builders can reach them
    ctx.setdefault("ticker", ticker)
    ctx.setdefault("scope", scope)
    ctx.setdefault("snapshot_id", snapshot_id)
    ctx.setdefault("generated_at", _now_iso())

    # Resolve the destination path
    filename = build_filename(ticker=ticker, scope=scope, snapshot_id=snapshot_id, ctx=ctx)
    resolved = _resolve_output_path(output_path, filename)
    ctx["output_filename"] = resolved.name

    # Build the Story by iterating the section order
    story = _build_story(ctx, section_order, theme)

    label = ticker or scope or "memo"
    return render_pdf(
        story=story,
        output_path=resolved,
        ticker_or_scope=label,
        snapshot_id=snapshot_id,
        theme=theme,
        title=ctx.get("report_title") or f"{label} Investment Memorandum",
    )


# =============================================================================
# Filename
# =============================================================================


_FILENAME_BAD_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def build_filename(
    ticker: Optional[str] = None,
    scope: Optional[str] = None,
    snapshot_id: str = "",
    ctx: Optional[dict] = None,
) -> str:
    """Canonical deterministic filename.

    Format
    ------
    ``{LABEL}_investment_memo_{YYYY-MM-DD}_snapshot-{ID}.pdf``

    * ``LABEL`` = ticker.upper() OR ``portfolio_{slug}`` for a scope.
    * Date is taken from ``ctx['snapshot_date']`` / ``ctx['as_of']``
      first; falls back to today's UTC date.
    * Snapshot id is truncated to 16 chars and stripped of separators.

    The same inputs always produce the same filename, which is what the
    spec's determinism test relies on.
    """
    ctx = ctx or {}
    if ticker:
        label = re.sub(_FILENAME_BAD_CHARS, "", ticker).upper() or "UNKNOWN"
    elif scope:
        slug = re.sub(_FILENAME_BAD_CHARS, "_", scope).strip("_").lower() or "default"
        label = f"portfolio_{slug}"
    else:
        label = "memo"

    snap_date = (
        ctx.get("snapshot_date")
        or ctx.get("as_of")
        or _today_iso()
    )
    # Normalize to YYYY-MM-DD even if caller passed a full datetime
    snap_date = str(snap_date)[:10]
    snap_date = re.sub(_FILENAME_BAD_CHARS, "-", snap_date)

    snap_id = re.sub(_FILENAME_BAD_CHARS, "", snapshot_id or "")[:16] or "none"

    return f"{label}_investment_memo_{snap_date}_snapshot-{snap_id}.pdf"


def _resolve_output_path(output_path: Optional[Path], filename: str) -> Path:
    """Return a concrete file path.

    If ``output_path`` is None → ``./reports/<filename>``
    If ``output_path`` is a directory (existing OR ends with /) → join
    If ``output_path`` ends with .pdf → use as-is
    Otherwise treat as a file path.
    """
    if output_path is None:
        return Path("reports") / filename
    p = Path(output_path)
    if str(p).endswith(("/", "\\")):
        return p / filename
    if p.suffix.lower() == ".pdf":
        return p
    if p.exists() and p.is_dir():
        return p / filename
    # Caller passed a path that doesn't exist yet — if it has no .pdf
    # suffix, treat it as a directory; if it has a suffix that's not
    # .pdf, treat it as a file.
    if p.suffix == "":
        return p / filename
    return p


# =============================================================================
# Story assembly
# =============================================================================


def _build_story(
    ctx: dict, section_order: Iterable[str], theme: DocTheme,
) -> list:
    """Iterate section_order and concatenate section flowables.

    Adds a PageBreak after every section except the title page (which
    handles its own page break) and the last section.
    """
    sections = list(section_order)
    story: list = []
    included: list[str] = []
    missing: list[str] = []

    for i, name in enumerate(sections):
        try:
            flowables = build_section(name, ctx, theme)
        except Exception:
            logger.exception("report | section %r failed; skipping", name)
            missing.append(name)
            continue

        if not flowables:
            missing.append(name)
            continue

        included.append(name)
        story.extend(flowables)

        # No PageBreak after the title page (it provides one itself) or
        # after the last section.
        if name == "title_page" or i == len(sections) - 1:
            continue
        story.append(_section_break())

    logger.info(
        "report | sections included=%s missing=%s",
        included, missing,
    )
    return story


def _section_break():
    """A page break between sections. Defined as a function to keep the
    import surface small and to ease future tweaks (e.g. switch to a
    spacer or rule on some sections)."""
    from reportlab.platypus import PageBreak
    return PageBreak()


# =============================================================================
# Helpers
# =============================================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")