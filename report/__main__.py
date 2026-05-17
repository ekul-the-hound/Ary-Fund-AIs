"""
report.__main__
===============
CLI for the report generator. Lets you build a PDF from a saved
snapshot without touching the dashboard.

Examples
--------

Generate a memo for a ticker, pulling its snapshot from a JSON file::

    python -m report --ticker AAPL --snapshot-id abc123 \
        --context ./snapshots/aapl_2026-05-15.json \
        --output ./reports/

Generate a portfolio-scope memo::

    python -m report --scope "core_long_book" --snapshot-id day-2026-05-15 \
        --context ./snapshots/portfolio_2026-05-15.json \
        --output ./reports/

If no ``--context`` is supplied, the CLI tries to build one from
``pipeline.build_agent_context`` (single-ticker path only).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from . import generate_pdf_report


def _load_context(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"context file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_pipeline_context(ticker: str) -> dict:
    """Best-effort: ask the pipeline to build a fresh context."""
    try:
        from data import pipeline as _pipeline   # type: ignore
        import config as _config                  # type: ignore
        # config.py exposes PORTFOLIO_DB_PATH (not DB_PATH); the legacy
        # fallback string is kept as a last-resort default for callers
        # that import this CLI without a real project config.
        db_path = getattr(_config, "PORTFOLIO_DB_PATH", "data/portfolio.db")
        return _pipeline.build_agent_context(ticker, db_path, _config) or {}
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: could not build context via pipeline: {e}",
            file=sys.stderr,
        )
        return {}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m report",
        description="Generate a PDF investment memo from a snapshot.",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="Single-ticker scope")
    g.add_argument("--scope",  help="Portfolio/watchlist scope label")
    parser.add_argument(
        "--snapshot-id", default="",
        help="Snapshot identifier (any string). Drives filename + footer.",
    )
    parser.add_argument(
        "--context",
        help="Path to a JSON file with the context dict. If omitted and "
             "--ticker is given, the CLI tries pipeline.build_agent_context.",
    )
    parser.add_argument(
        "--output", default="./reports/",
        help="Output file or directory. Default: ./reports/",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    ctx = _load_context(args.context)
    if not ctx and args.ticker:
        ctx = _try_pipeline_context(args.ticker)

    try:
        out = generate_pdf_report(
            ticker=args.ticker,
            scope=args.scope,
            output_path=Path(args.output),
            snapshot_id=args.snapshot_id,
            context=ctx,
        )
    except Exception as e:  # noqa: BLE001
        print(f"report generation failed: {e}", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())