"""
report_orchestrator.py
=====================
The missing glue that turns a ticker into a finished PDF memo.

The report package had all the pieces but no orchestrator:
  * content_builder.py  — section builders (consume a ``ctx`` dict)
  * template.py         — SECTION_ORDER + theme
  * pdf_renderer.py     — render_pdf(story, path, ...) writes the PDF
  * charts.py           — chart flowables

...but nothing assembled a ctx for a ticker, built the story, and called
render_pdf. ``app_v2.py``'s ``report`` command looked for a
``generate_report`` / ``render_report`` / ``build_report`` function and found
none, so it reported "Reports aren't available in this build." This module
supplies ``generate_report`` (the name the resolver checks first).

PIPELINE
--------
    generate_report(ticker, db_path, cfg)
      → build_agent_context(ticker, db_path, cfg)      # registry-backed data
      → _map_agent_ctx_to_report_ctx(...)              # shape into report ctx
      → for name in SECTION_ORDER: build_section(...)  # assemble the Story
      → render_pdf(story, output_path, ...)            # write the PDF
      → returns the output Path

The section builders degrade gracefully (missing input → labeled
placeholder, never a crash), so even a partial ctx yields a valid PDF. Thesis
prose and chart artifacts are best-effort: if they aren't in the agent
context, those sections render placeholders rather than failing.

OUTPUT
------
PDFs are written to ``reports/`` under the project root, named
``{TICKER}_memo_{YYYY-MM-DD}.pdf``. The function returns the Path; the caller
(job queue) can surface it.

IMPORTS
-------
The report modules use intra-package relative imports (``.charts``,
``.template``), so they must be imported as a package. We try a few candidate
package roots (``ui``, ``report``, top-level) so this works regardless of how
the tree is laid out, and fall back to a clear RuntimeError if none resolve.
"""
from __future__ import annotations

import importlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tolerant imports of the report package pieces
# ---------------------------------------------------------------------------
def _import_report_pieces():
    """Return (build_section, render_pdf, SECTION_ORDER, DEFAULT_THEME).

    Tries candidate package prefixes so this works whether the report files
    live in a ``ui`` package, a ``report`` package, or top-level.
    """
    prefixes = ("ui.", "report.", "")
    last_err: Optional[Exception] = None
    for p in prefixes:
        try:
            cb = importlib.import_module(f"{p}content_builder")
            pr = importlib.import_module(f"{p}pdf_renderer")
            tmpl = importlib.import_module(f"{p}template")
            return (
                cb.build_section,
                pr.render_pdf,
                tmpl.SECTION_ORDER,
                getattr(tmpl, "DEFAULT_THEME", None),
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(
        f"report_orchestrator: could not import the report package "
        f"(content_builder/pdf_renderer/template). Last error: {last_err}"
    )


def _import_build_agent_context():
    for p in ("pipeline", "ui.pipeline"):
        try:
            mod = importlib.import_module(p)
            if hasattr(mod, "build_agent_context"):
                return mod.build_agent_context
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Map the agent context onto the report ctx the builders expect
# ---------------------------------------------------------------------------
def _map_agent_ctx_to_report_ctx(
    ticker: str, agent_ctx: dict, snapshot_id: str,
) -> dict:
    """Translate build_agent_context()'s output into the keys the report
    section builders read. Unknown/absent inputs are simply omitted; the
    builders render placeholders for missing sections."""
    agent_ctx = agent_ctx or {}
    as_of = agent_ctx.get("as_of") or datetime.now().date().isoformat()

    report_ctx: dict[str, Any] = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "snapshot_date": as_of,
        "snapshot_id": snapshot_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_title": f"{ticker.upper()} Investment Memo",
        "scope": ticker.upper(),
        # Direct passthroughs (same meaning in both contexts).
        "prices": agent_ctx.get("prices") or {},
        "metrics": agent_ctx.get("metrics") or {},
        "key_metrics": agent_ctx.get("metrics") or {},
        "filings": agent_ctx.get("filings") or [],
        "macro": agent_ctx.get("macro") or {},
        "provenance": agent_ctx.get("provenance") or {},
        # Risk: agent has risk_scores; builders read 'risk' + 'risk_flags'.
        "risk": agent_ctx.get("risk_scores") or {},
        "risk_flags": agent_ctx.get("risk_flags") or [],
        # Derived signals: builders read 'derived' + 'signals'.
        "derived": agent_ctx.get("derived_signals") or {},
        "signals": agent_ctx.get("derived_signals") or {},
        # Retrieved RAG chunks become the supporting 'sources'.
        "sources": agent_ctx.get("retrieved_context") or [],
        # Thesis/essay/charts are best-effort; absent → placeholder sections.
        "thesis": agent_ctx.get("thesis") or agent_ctx.get("thesis_text") or "",
        "essay": agent_ctx.get("essay") or "",
        "charts": agent_ctx.get("charts") or [],
        "analyst": agent_ctx.get("analyst") or {},
        "filings_summary": agent_ctx.get("filings_summary") or "",
    }
    return report_ctx


# ---------------------------------------------------------------------------
# Public entry point — the name app_v2's resolver looks for
# ---------------------------------------------------------------------------
def generate_report(
    ticker: str,
    db_path: str = "data/hedgefund.db",
    cfg: Any = None,
    output_dir: str = "reports",
) -> Path:
    """Build a PDF investment memo for ``ticker`` and return its Path.

    Parameters
    ----------
    ticker :
        Symbol to report on.
    db_path :
        SQLite DB the registry/data layer reads (default data/hedgefund.db).
    cfg :
        App config object passed through to build_agent_context. If None, we
        try to import the project config; the report still renders without it
        (sections degrade to placeholders where data is missing).
    output_dir :
        Directory for the PDF (created if needed). Default ``reports/``.

    Returns
    -------
    Path to the written PDF.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("generate_report: empty ticker.")

    build_section, render_pdf, SECTION_ORDER, DEFAULT_THEME = \
        _import_report_pieces()

    # Resolve cfg if not provided (best-effort; report works without it).
    if cfg is None:
        for p in ("config", "ui.config"):
            try:
                cfg = importlib.import_module(p)
                break
            except Exception:
                continue

    # Gather the registry-backed context for this ticker.
    build_agent_context = _import_build_agent_context()
    agent_ctx: dict = {}
    if build_agent_context is not None:
        try:
            agent_ctx = build_agent_context(ticker, db_path, cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "report_orchestrator: build_agent_context failed for %s (%s); "
                "rendering a skeleton report.", ticker, e,
            )
            agent_ctx = {}
    else:
        logger.warning(
            "report_orchestrator: build_agent_context unavailable; rendering a "
            "skeleton report for %s.", ticker,
        )

    snapshot_id = f"{ticker}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    report_ctx = _map_agent_ctx_to_report_ctx(ticker, agent_ctx, snapshot_id)

    # Assemble the Story by iterating the canonical section order.
    story: list = []
    for name in SECTION_ORDER:
        try:
            if DEFAULT_THEME is not None:
                story.extend(build_section(name, report_ctx, DEFAULT_THEME))
            else:
                story.extend(build_section(name, report_ctx))
        except Exception as e:  # noqa: BLE001 — one section must not kill the doc
            logger.exception(
                "report_orchestrator: section %s raised; continuing.", name)

    if not story:
        raise RuntimeError(
            "report_orchestrator: produced an empty Story — nothing to render."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{ticker}_memo_{datetime.now().date().isoformat()}.pdf"

    written = render_pdf(
        story=story,
        output_path=output_path,
        ticker_or_scope=ticker,
        snapshot_id=snapshot_id,
        title=f"{ticker} Investment Memo",
    )
    logger.info("report_orchestrator: wrote %s", written)
    return Path(written)


__all__ = ["generate_report"]

# D:\Ary Fund\ui\report_orchestrator.py
