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


def _resolve_portfolio_db(passed: str, cfg: Any) -> str:
    """Find the portfolio DB that actually holds agent_opinions.

    The report worker may pass a db_path that resolves to an empty root-level
    portfolio.db (there are two: an empty ./portfolio.db and the real
    data/portfolio.db). So we try, in order:
      1. cfg.PORTFOLIO_DB_PATH (the authoritative config value)
      2. the passed db_path
      3. data/portfolio.db anchored at the project root
    and return the first one whose ``agent_opinions`` table exists and is
    non-empty. Falls back to the config value (or the passed path) if none
    qualify.
    """
    import sqlite3
    candidates = []
    cfg_path = getattr(cfg, "PORTFOLIO_DB_PATH", None) if cfg else None
    if cfg_path:
        candidates.append(str(cfg_path))
    if passed:
        candidates.append(str(passed))
    try:
        _root = Path(__file__).resolve().parent.parent
    except Exception:
        _root = Path.cwd()
    candidates.append(str(_root / "data" / "portfolio.db"))

    for cand in candidates:
        try:
            if not Path(cand).exists():
                continue
            with sqlite3.connect(cand) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='agent_opinions'"
                ).fetchone()
                if row and row[0] > 0:
                    n = conn.execute(
                        "SELECT COUNT(*) FROM agent_opinions"
                    ).fetchone()
                    if n and n[0] > 0:
                        return cand
        except Exception:
            continue
    return cfg_path or passed or str(_root / "data" / "portfolio.db")


def _load_latest_opinion(ticker: str, portfolio_db_path: str) -> dict:
    """st-free read of the latest agent opinion from portfolio.db.

    Mirrors app.load_latest_opinion's body (without the @st.cache_data
    decorator, so it's safe on a worker thread). The opinions table is written
    by main.py via portfolio_db.save_agent_opinion; payload_json holds the full
    merged opinion (outlook, confidence, rationale, key_risks, risk_flags...).
    Returns {} if absent.
    """
    import sqlite3
    import json
    if not portfolio_db_path:
        return {}
    try:
        with sqlite3.connect(portfolio_db_path) as conn:
            row = conn.execute(
                "SELECT payload_json FROM agent_opinions "
                "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchone()
        if not row:
            return {}
        return json.loads(row[0]) or {}
    except Exception:
        return {}


def _merge_opinion_into_ctx(ctx: dict, opinion: dict) -> dict:
    """Layer the agent-opinion fields onto a raw context using the ACTUAL
    payload shape stored in agent_opinions.

    The stored opinion is rich: it has a full markdown ``essay`` string, a
    builder-shaped ``risk_flags`` ({levels:{...combined...}, reasons:{...}}),
    ``key_metrics`` (dict), ``filings_summary`` (dict), plus the scalar thesis
    fields. We pass each through to the keys the report builders read, rather
    than synthesizing or reshaping."""
    if not opinion:
        return ctx
    ctx = dict(ctx or {})

    # Thesis scalars (for the exec-summary badge row + structured fallback).
    ctx["thesis"] = {
        "outlook": opinion.get("outlook"),
        "price_direction": opinion.get("price_direction"),
        "confidence": opinion.get("confidence"),
        "time_horizon": opinion.get("time_horizon"),
        "rationale": opinion.get("rationale"),
        "key_risks": opinion.get("key_risks", []),
        "opportunities": opinion.get("key_opportunities", []),
    }

    # Essay: the payload stores the full markdown narrative as a string.
    # build_thesis reads essay.get("text"), so wrap it.
    essay = opinion.get("essay_revised") or opinion.get("essay")
    if isinstance(essay, str) and essay.strip():
        ctx["essay"] = {"text": essay, "fallback": False}
    elif isinstance(essay, dict):
        ctx["essay"] = essay

    # Risk: the payload's risk_flags is ALREADY in the builder shape
    # ({levels, reasons}) — pass it straight through.
    rf = opinion.get("risk_flags")
    if isinstance(rf, dict) and rf:
        ctx["risk"] = rf
        ctx["risk_flags"] = rf

    # Key metrics + filings summary the builders can render.
    km = opinion.get("key_metrics")
    if isinstance(km, dict) and km:
        ctx["metrics"] = km
    fs = opinion.get("filings_summary")
    if fs:
        ctx["filings_summary"] = fs

    return ctx


# ---------------------------------------------------------------------------
# Map the agent context onto the report ctx the builders expect
# ---------------------------------------------------------------------------
def _map_agent_ctx_to_report_ctx(
    ticker: str, agent_ctx: dict, snapshot_id: str,
) -> dict:
    """Translate a ticker context (from load_ticker_context OR
    build_agent_context) into the keys the report section builders read.

    Handles the load_ticker_context opinion-overlay shape specifically, since
    that's the populated path:
      * thesis: {outlook, direction, confidence, summary, key_risks, ...}
        → builders want thesis.rationale / thesis.outlook / thesis.price_direction,
          plus an essay.text. We remap names and synthesize an essay from the
          summary so the Thesis section renders prose instead of a placeholder.
      * risk: {combined_level, fundamental_risk, macro_risk, market_risk, flags}
        → builders want risk.levels.{combined,fundamental,macro,market} and
          risk.reasons. We reshape into that nested form.
    Unknown/absent inputs are omitted; builders render placeholders for those.
    """
    agent_ctx = agent_ctx or {}
    as_of = agent_ctx.get("as_of") or datetime.now().date().isoformat()

    # --- Thesis: remap load_ticker_context's dict to builder field names ---
    raw_thesis = agent_ctx.get("thesis")
    thesis_out: dict = {}
    essay_out: dict = {}
    if isinstance(raw_thesis, dict) and raw_thesis:
        summary = (raw_thesis.get("summary") or raw_thesis.get("rationale")
                   or "")
        thesis_out = {
            "outlook": raw_thesis.get("outlook"),
            # builder reads price_direction; load_ticker_context uses direction
            "price_direction": (raw_thesis.get("price_direction")
                                or raw_thesis.get("direction")),
            "confidence": raw_thesis.get("confidence"),
            "time_horizon": raw_thesis.get("time_horizon"),
            "rationale": summary,
            "key_risks": raw_thesis.get("key_risks") or [],
            "opportunities": raw_thesis.get("opportunities")
            or raw_thesis.get("key_opportunities") or [],
        }
        # Synthesize an essay from the summary so build_thesis renders prose.
        if summary:
            essay_out = {"text": summary, "fallback": False}
    elif isinstance(raw_thesis, str) and raw_thesis.strip():
        essay_out = {"text": raw_thesis, "fallback": False}

    # Prefer an explicit essay if the context already had one.
    ctx_essay = agent_ctx.get("essay")
    if isinstance(ctx_essay, dict) and ctx_essay.get("text"):
        essay_out = ctx_essay
    elif isinstance(ctx_essay, str) and ctx_essay.strip():
        essay_out = {"text": ctx_essay, "fallback": False}

    # --- Risk: reshape flat load_ticker_context risk into nested levels ---
    raw_risk = agent_ctx.get("risk") or {}
    risk_out: dict = {}
    if isinstance(raw_risk, dict) and raw_risk:
        if "levels" in raw_risk:
            # Already in builder shape.
            risk_out = raw_risk
        else:
            levels = {
                "combined": raw_risk.get("combined_level"),
                "fundamental": raw_risk.get("fundamental_risk"),
                "macro": raw_risk.get("macro_risk"),
                "market": raw_risk.get("market_risk"),
            }
            risk_out = {
                "levels": {k: v for k, v in levels.items() if v},
                "reasons": raw_risk.get("flags") or raw_risk.get("reasons") or [],
            }

    report_ctx: dict[str, Any] = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "snapshot_date": as_of,
        "snapshot_id": snapshot_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_title": f"{ticker.upper()} Investment Memo",
        "scope": ticker.upper(),
        # Passthroughs.
        "prices": agent_ctx.get("prices") or agent_ctx.get("price_summary") or {},
        "metrics": agent_ctx.get("metrics") or {},
        "key_metrics": agent_ctx.get("metrics") or {},
        "filings": agent_ctx.get("filings") or [],
        "macro": agent_ctx.get("macro") or agent_ctx.get("macro_extras") or {},
        "provenance": agent_ctx.get("provenance") or {},
        # Remapped thesis/risk/essay.
        "thesis": thesis_out,
        "essay": essay_out,
        "risk": risk_out,
        "risk_flags": risk_out,
        # Derived signals.
        "derived": agent_ctx.get("derived_signals") or agent_ctx.get("derived") or {},
        "signals": agent_ctx.get("derived_signals") or agent_ctx.get("signals") or {},
        "sources": agent_ctx.get("retrieved_context") or agent_ctx.get("sources") or [],
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
    context: Optional[dict] = None,
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
    context :
        OPTIONAL pre-built context dict. When the caller already has the
        ticker's context (e.g. the Desk's ``load_ticker_context`` result, which
        layers the agent-opinion thesis/risk on top of the raw data), pass it
        here and we render from it directly — skipping our own
        build_agent_context call. This is the path that produces a fully
        populated report, because load_ticker_context includes the LLM opinion
        overlay (thesis summary, risk levels) that bare build_agent_context
        does not.

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

    # Prefer a caller-supplied context (the Desk's load_ticker_context result,
    # which includes the thesis/risk opinion overlay). Only fall back to our
    # own build_agent_context call when no context was passed.
    if context:
        agent_ctx = context
    else:
        build_agent_context = _import_build_agent_context()
        agent_ctx = {}
        if build_agent_context is not None:
            try:
                agent_ctx = build_agent_context(ticker, db_path, cfg)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "report_orchestrator: build_agent_context failed for %s "
                    "(%s); rendering a skeleton report.", ticker, e,
                )
                agent_ctx = {}
        else:
            logger.warning(
                "report_orchestrator: build_agent_context unavailable; "
                "rendering a skeleton report for %s.", ticker,
            )

        # Layer the LLM opinion (thesis + risk) on top — same as the Desk's
        # load_ticker_context. Resolve the portfolio DB robustly: the passed
        # db_path can point at an empty root-level portfolio.db, so we prefer
        # config's PORTFOLIO_DB_PATH / data/portfolio.db where the
        # agent_opinions table actually lives.
        try:
            _pdb = _resolve_portfolio_db(db_path, cfg)
            opinion = _load_latest_opinion(ticker, _pdb)
            if opinion:
                agent_ctx = _merge_opinion_into_ctx(agent_ctx, opinion)
            else:
                logger.info(
                    "report_orchestrator: no agent opinion found for %s in %s "
                    "— thesis/risk sections will be placeholders. Run `gen %s` "
                    "first to produce an opinion.", ticker, _pdb, ticker,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "report_orchestrator: opinion merge failed for %s (%s).",
                ticker, e,
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

    # Resolve output_dir to an ABSOLUTE path anchored at the project root, so
    # the PDF lands in a predictable place regardless of the process CWD (the
    # Streamlit server's CWD isn't guaranteed to be the project root). The root
    # is this file's parent's parent (ui/ -> root); fall back to CWD if that
    # can't be determined.
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        try:
            _root = Path(__file__).resolve().parent.parent
        except Exception:
            _root = Path.cwd()
        out_dir = _root / output_dir
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