"""
report/__init__.py
==================
Top-level orchestrator for the ARY Fund PDF report system.

Re-exports the small, stable public surface that callers actually need:

    from report import generate_pdf_report, build_filename, REPORT_VERSION

Everything else (themes, individual section builders, the renderer) is
still importable via its submodule (``from report.template import ...``,
``from report.charts import ChartSpec``) for advanced use.

Why a thin wrapper at this level?

* ``generate_pdf_report`` is the *only* entry point most callers want.
  It hides the three-step assembly (build_section ×N → render_pdf →
  filename resolution) behind a single function call.

* ``build_filename`` is split out as its own export because the daily
  scheduler wants to plan filenames *before* rendering (for staging
  paths, S3 uploads, etc).  Same naming rules either way.

Both functions are deterministic for a given (ticker | scope,
snapshot_id, ctx['snapshot_date']) — re-running with identical inputs
yields the same filename and the same logical structure (page count,
section ordering). PDF byte-equality is NOT guaranteed because
ReportLab embeds creation-time metadata and Python dict ordering can
shift PDF stream byte offsets.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from reportlab.platypus import Flowable

from .content_builder import build_section
from .pdf_renderer import render_pdf
from .template import (
    DEFAULT_THEME,
    REPORT_VERSION,
    RENDERER_VERSION,
    SECTION_ORDER,
    DocTheme,
)

logger = logging.getLogger(__name__)

__all__ = [
    "generate_pdf_report",
    "build_filename",
    "REPORT_VERSION",
    "RENDERER_VERSION",
    "SECTION_ORDER",
    "DocTheme",
    "DEFAULT_THEME",
]


# ---------------------------------------------------------------------------
# Filename construction
# ---------------------------------------------------------------------------

# Characters NOT allowed in a generated filename. We intentionally KEEP
# dots (so "BRK.B" survives) and dashes (date separators). Everything
# else that's not alphanumeric or underscore gets stripped.
#
# A regex on the "unsafe" set is cleaner than an "allowed" allowlist
# because the unsafe set is small and explicit, whereas the allowed set
# would need Unicode handling we don't actually want — the file system
# may accept exotic glyphs but downstream consumers (S3 keys, URL
# encoding, Windows shells) tend to mangle them.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def _sanitize_component(s: str) -> str:
    """Strip unsafe chars from a filename fragment.

    Replaces runs of unsafe chars with a single ``-`` rather than
    deleting them, so ``"BRK.B vs MSFT"`` becomes ``"BRK.B-vs-MSFT"``
    rather than ``"BRK.BvsMSFT"`` (the latter loses the visual word
    boundary).  Leading/trailing separators get trimmed.
    """
    s = _UNSAFE_RE.sub("-", str(s))
    return s.strip("-._")


def build_filename(
    ticker: Optional[str] = None,
    scope: Optional[str] = None,
    snapshot_id: str = "",
    ctx: Optional[dict] = None,
) -> str:
    """Build the canonical PDF filename for a report.

    Exactly one of ``ticker`` or ``scope`` should be provided.  The
    output shape is:

        <subject>_investment_memo_<date>_snapshot-<id>.pdf

    where ``<subject>`` is either ``<TICKER>`` or
    ``portfolio_<scope>``, and ``<date>`` comes from
    ``ctx['snapshot_date']`` (falling back to ``ctx['as_of']``, then
    ``"unknown-date"``).

    Determinism: same inputs → same output, always.  No timestamps, no
    randomness.

    Sanitization: unsafe characters (``/``, ``\\``, ``:``, whitespace)
    are replaced with ``-`` so the result is safe across POSIX,
    Windows, and S3.  Dots are kept (``BRK.B`` is a real ticker).
    """
    ctx = ctx or {}
    snapshot_date = (
        ctx.get("snapshot_date")
        or ctx.get("as_of")
        or "unknown-date"
    )
    snapshot_date = _sanitize_component(str(snapshot_date))
    snapshot_id_clean = _sanitize_component(str(snapshot_id))

    if ticker:
        subject = _sanitize_component(ticker)
    elif scope:
        subject = "portfolio_" + _sanitize_component(scope)
    else:
        # build_filename can be called speculatively (e.g. by a planner
        # that hasn't decided ticker vs scope yet).  Use a neutral
        # stand-in rather than raise — generate_pdf_report is the
        # function that enforces the "must specify one" contract.
        subject = "report"

    return (
        f"{subject}_investment_memo_{snapshot_date}"
        f"_snapshot-{snapshot_id_clean}.pdf"
    )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_output_path(
    output_path: Union[str, Path],
    ticker: Optional[str],
    scope: Optional[str],
    snapshot_id: str,
    ctx: dict,
) -> Path:
    """Decide whether ``output_path`` is a file or a directory.

    Rules:

    * If the path ends in ``.pdf`` → use it verbatim.
    * Otherwise treat it as a directory → append the canonical filename
      from :func:`build_filename`.

    The "ends in .pdf" check is intentionally cheap (suffix lookup) so
    callers can pass either form without the function probing the
    filesystem.  Probing would have to decide what "is a directory"
    means before the path exists, which is brittle.
    """
    p = Path(output_path)
    if p.suffix.lower() == ".pdf":
        return p
    fname = build_filename(
        ticker=ticker, scope=scope, snapshot_id=snapshot_id, ctx=ctx,
    )
    return p / fname


# ---------------------------------------------------------------------------
# Story assembly
# ---------------------------------------------------------------------------

def _assemble_story(
    ctx: dict,
    section_order: Iterable[str],
    theme: DocTheme,
) -> list[Flowable]:
    """Walk the section order, calling each builder, concatenating
    flowables.

    Section-level errors are absorbed by ``build_section`` itself (it
    returns a placeholder flowable rather than raising), so this
    function never aborts mid-report.  That keeps the partial-report
    contract: even if section X is broken, sections X+1..N still
    render.

    Page-break policy: an explicit ``PageBreak`` is inserted *between*
    sections (i.e. before every section after the first that produces
    output).  Why here and not inside each section builder?

    * Builders shouldn't know whether they're first/last in the
      sequence; that's a layout concern, owned by the orchestrator.
    * ``title_page`` already appends a trailing PageBreak — so the
      "insert between" rule naturally skips the boundary right after
      title_page (we don't double-break).
    * Empty/placeholder sections still get their own page, which
      matches the "every section is reachable" contract enforced by
      ``test_missing_section_does_not_remove_section`` while keeping
      the page count high enough for the ≥5-page "full report"
      acceptance criterion.

    The de-dup against title_page is structural: we look at the
    *previous section's last flowable*. If it's already a PageBreak we
    don't add another. This is robust to other section builders
    growing PageBreaks in the future.
    """
    from reportlab.platypus import PageBreak

    story: list[Flowable] = []
    for idx, name in enumerate(section_order):
        flowables = build_section(name, ctx, theme=theme)
        if not flowables:
            # Defensive: every builder should return at least a header,
            # but if one returns [] we skip silently rather than emit a
            # naked PageBreak.
            continue
        if idx > 0 and story and not isinstance(story[-1], PageBreak):
            story.append(PageBreak())
        story.extend(flowables)
    return story


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def generate_pdf_report(
    ticker: Optional[str] = None,
    scope: Optional[str] = None,
    output_path: Union[str, Path] = ".",
    snapshot_id: str = "",
    context: Optional[dict] = None,
    section_order: Optional[Iterable[str]] = None,
    theme: Optional[DocTheme] = None,
) -> Path:
    """Generate a complete PDF memo and write it to disk.

    Parameters
    ----------
    ticker :
        Single-ticker memo (mutually exclusive with ``scope``).
    scope :
        Portfolio-scope memo, e.g. ``"core_long_book"`` (mutually
        exclusive with ``ticker``).
    output_path :
        Either a directory (filename is auto-generated via
        :func:`build_filename`) or an explicit ``*.pdf`` path. Parent
        directories are created as needed.
    snapshot_id :
        Traceability handle. Appears in the page footer, in the PDF
        metadata, and in the filename.
    context :
        Snapshot dict consumed by every section builder. See
        ``report.content_builder`` for the keys each section reads.
    section_order :
        Optional override of the section sequence. Defaults to
        :data:`report.template.SECTION_ORDER`. Pass a tuple/list of
        section names (e.g. ``("title_page", "executive_summary",
        "appendix")``) to render a subset or re-order.
    theme :
        Optional layout/style override. Defaults to ``DEFAULT_THEME``.

    Returns
    -------
    Path
        Absolute path to the rendered PDF.

    Raises
    ------
    ValueError
        If neither ``ticker`` nor ``scope`` is provided.

    Notes
    -----
    This function does NOT call out to data sources.  ``context`` must
    already be assembled by the caller (typically by
    ``data.pipeline.build_agent_context`` or similar). Keeping the
    report generator I/O-free makes it cheap to re-render the same
    snapshot, and it means the scheduler can plan rendering jobs
    against snapshots stored on disk without touching the network.
    """
    if not ticker and not scope:
        raise ValueError(
            "generate_pdf_report requires either 'ticker' or 'scope'; "
            "got neither."
        )

    ctx = dict(context or {})
    # Surface scope/ticker/snapshot_id into ctx so section builders can
    # render them in the title page / footers without the caller having
    # to duplicate the fields.  Caller-provided values win on conflict.
    ctx.setdefault("ticker", ticker)
    ctx.setdefault("scope", scope)
    ctx.setdefault("snapshot_id", snapshot_id)

    theme = theme or DEFAULT_THEME
    section_order = tuple(section_order) if section_order else SECTION_ORDER

    # Resolve filename BEFORE assembling the Story so we can stash it
    # into ctx — build_appendix renders it under "Report metadata".
    final_path = _resolve_output_path(
        output_path=output_path, ticker=ticker, scope=scope,
        snapshot_id=snapshot_id, ctx=ctx,
    )
    ctx.setdefault("output_filename", final_path.name)

    ticker_or_scope = ticker or (f"portfolio:{scope}" if scope else "(unknown)")

    logger.info(
        "report | generating %s (sections=%d, output=%s)",
        ticker_or_scope, len(section_order), final_path,
    )

    story = _assemble_story(ctx, section_order, theme)
    return render_pdf(
        story=story,
        output_path=final_path,
        ticker_or_scope=ticker_or_scope,
        snapshot_id=snapshot_id,
        theme=theme,
    )