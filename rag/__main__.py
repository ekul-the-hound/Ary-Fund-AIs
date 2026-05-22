"""
rag.__main__
============
Minimal CLI dispatcher for the ``rag`` package.

Why this exists
---------------
``rag/__init__.py`` and ``rag/integration_snippets.py`` document
``python -m rag --help`` / ``python -m rag index --tickers ...`` as if
they were a real entrypoint, but no ``__main__.py`` existed to back
that promise. This module fulfils the promise by **dispatching to
entry points that already exist** elsewhere in the package — it
implements no new behaviour of its own.

Subcommands and what they delegate to
-------------------------------------
* ``index``     → ``rag.indexer.Indexer.run_tickers`` (existing)
* ``stats``     → ``rag.indexer.Indexer.stats``       (existing)
* ``benchmark`` → ``rag.eval.benchmark.main``         (existing)

Run::

    python -m rag --help
    python -m rag index --tickers AAPL MSFT
    python -m rag stats
    python -m rag benchmark --eval-set rag/eval/test_queries.json

Design constraints
------------------
This is a **routing layer only**.  It must NOT:

* introduce new business logic
* re-implement anything ``rag.indexer`` or ``rag.benchmark`` already does
* import any external service eagerly (Ollama, ChromaDB) — lazy-import
  inside each subcommand handler so ``--help`` works on a fresh checkout
  with no model server running

The lazy-import discipline is what lets the CI pipeline run
``python -m rag --help`` without needing ``ollama`` installed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional


# ----------------------------------------------------------------------
# Subcommand handlers — each one lazy-imports its dependencies so the
# top-level --help works in environments without Ollama / ChromaDB.
# ----------------------------------------------------------------------
def _cmd_index(args: argparse.Namespace) -> int:
    """Delegate to Indexer.run_tickers — no logic added here."""
    try:
        from rag.indexer import Indexer
    except Exception as e:  # noqa: BLE001
        print(f"rag index: indexer unavailable ({e})", file=sys.stderr)
        return 2

    # Optional dependency wiring. The user supplies them via env / config
    # in normal operation; in CLI usage they're optional and the indexer
    # silently skips missing loaders. We pass None explicitly to keep
    # the routing layer's contract clean — Indexer.run_tickers handles
    # the optional-arg branching, not us.
    sec_fetcher = None
    portfolio_db = None
    try:
        ix = Indexer(tracking_db_path=args.tracking_db)
    except Exception as e:  # noqa: BLE001
        print(f"rag index: could not build Indexer ({e})", file=sys.stderr)
        return 2

    stats = ix.run_tickers(
        tickers=args.tickers,
        sec_fetcher=sec_fetcher,
        portfolio_db=portfolio_db,
        force=args.force,
    )
    # Print the per-loader summary that run_tickers already produces.
    for loader_name, loader_stats in (stats or {}).items():
        print(f"{loader_name}: {loader_stats}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Delegate to Indexer.stats — no logic added here."""
    try:
        from rag.indexer import Indexer
    except Exception as e:  # noqa: BLE001
        print(f"rag stats: indexer unavailable ({e})", file=sys.stderr)
        return 2
    try:
        ix = Indexer(tracking_db_path=args.tracking_db)
    except Exception as e:  # noqa: BLE001
        print(f"rag stats: could not build Indexer ({e})", file=sys.stderr)
        return 2
    out = ix.stats()
    # Print the same dict the method returns — no formatting beyond that.
    for k, v in (out or {}).items():
        print(f"{k}: {v}")
    return 0


def _cmd_benchmark(args: argparse.Namespace, passthrough: list[str]) -> int:
    """Delegate to rag.benchmark.main, passing remaining argv through.

    The benchmark module already owns its argparse configuration; we
    don't re-declare those flags here. ``passthrough`` is the slice of
    argv after the ``benchmark`` subcommand token, which gets handed
    verbatim to ``rag.benchmark.main``.
    """
    try:
        from rag.eval.benchmark import main as benchmark_main
    except Exception as e:  # noqa: BLE001
        print(f"rag benchmark: unavailable ({e})", file=sys.stderr)
        return 2
    return int(benchmark_main(passthrough) or 0)


# ----------------------------------------------------------------------
# Top-level dispatcher
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse tree.

    Help text mirrors what's documented in ``rag/integration_snippets.py``
    so ``python -m rag --help`` and the in-repo docs say the same thing.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description=(
            "rag — Phase 1+2+3+4 RAG package for the hedgefund-ai project. "
            "This CLI is a thin router; every subcommand delegates to an "
            "entry point that already exists elsewhere in the package."
        ),
        epilog=(
            "Examples:\n"
            "  python -m rag index --tickers AAPL MSFT NVDA\n"
            "  python -m rag stats\n"
            "  python -m rag benchmark --eval-set rag/eval/test_queries.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose (DEBUG-level) logging."
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="<command>",
        title="commands",
    )

    # ----- index -----
    p_index = sub.add_parser(
        "index",
        help="Index documents for one or more tickers "
             "(delegates to rag.indexer.Indexer.run_tickers).",
    )
    p_index.add_argument(
        "--tickers", nargs="+", required=True,
        help="One or more ticker symbols to index.",
    )
    p_index.add_argument(
        "--tracking-db", default="data/rag_tracking.db",
        help="Path to the RAG tracking SQLite DB.",
    )
    p_index.add_argument(
        "--force", action="store_true",
        help="Re-index documents even if hash is unchanged.",
    )

    # ----- stats -----
    p_stats = sub.add_parser(
        "stats",
        help="Print descriptive stats about the indexed corpus "
             "(delegates to rag.indexer.Indexer.stats).",
    )
    p_stats.add_argument(
        "--tracking-db", default="data/rag_tracking.db",
        help="Path to the RAG tracking SQLite DB.",
    )

    # ----- benchmark -----
    # The benchmark subparser holds NO flags of its own. Every flag the
    # user types after `benchmark` is collected into REMAINDER and
    # forwarded verbatim to rag.benchmark.main. This avoids drift
    # between the two argparsers — the benchmark module remains the
    # single source of truth for its own flags.
    sub.add_parser(
        "benchmark",
        help="Run the retrieval benchmark "
             "(delegates to rag.eval.benchmark.main; all flags forward).",
        add_help=False,
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Routes the first positional arg to a handler."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # Special-case the benchmark subcommand: anything after the
    # `benchmark` token must NOT be parsed by our argparse (because
    # rag.benchmark owns those flags). We split argv before argparse
    # has a chance to choke on its unknown flags.
    if "benchmark" in argv:
        idx = argv.index("benchmark")
        head = argv[:idx + 1]
        passthrough = argv[idx + 1:]
        parser = _build_parser()
        # Allow `-h`/`--help` after `benchmark` to fall through to the
        # benchmark module's own --help, not ours.
        args = parser.parse_args(head)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return _cmd_benchmark(args, passthrough)

    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "stats":
        return _cmd_stats(args)

    # Defensive — argparse should have rejected any other value
    # against ``choices``-equivalent subcommand registration.
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())