"""
rag/__main__.py
===============
CLI for the RAG package.

Usage
-----
::

    # Backfill or incrementally index every loader
    python -m rag index --tickers AAPL MSFT NVDA

    # Force re-index regardless of content hash
    python -m rag index --tickers AAPL --force

    # Just notes (no tickers needed)
    python -m rag index --notes-only

    # One-off query
    python -m rag query "What are Apple's supply chain risks?" --ticker AAPL

    # Stats: how many docs/chunks in the store right now?
    python -m rag stats

Design note
-----------
The CLI is intentionally thin. Each command delegates to the same
classes you'd use from Python. This keeps "behavior I tested in a
shell" identical to "behavior I'll get inside the agent pipeline."
"""

from __future__ import annotations

import argparse
import logging
import sys

from rag.indexer import Indexer
from rag.retriever import Retriever


def cmd_index(args: argparse.Namespace) -> int:
    indexer = Indexer()

    sec_fetcher = None
    portfolio_db = None
    if not args.notes_only and args.tickers:
        # Lazy-import your data layer so the CLI works even when
        # those modules aren't on the path.
        try:
            from data.sec_fetcher import SECFetcher
            sec_fetcher = SECFetcher()
        except Exception as e:  # noqa: BLE001
            print(f"warning: sec_fetcher unavailable ({e}); skipping filings",
                  file=sys.stderr)
        try:
            from data.portfolio_db import PortfolioDB
            portfolio_db = PortfolioDB()
        except Exception as e:  # noqa: BLE001
            print(f"warning: portfolio_db unavailable ({e}); skipping theses",
                  file=sys.stderr)

    if args.notes_only:
        from rag.document_loaders.notes import NotesLoader
        stats = indexer.index_many(NotesLoader().load_all(), force=args.force)
        print({"notes": stats})
    else:
        stats = indexer.run_tickers(
            tickers=args.tickers or [],
            sec_fetcher=sec_fetcher,
            portfolio_db=portfolio_db,
            force=args.force,
        )
        for source, s in stats.items():
            print(f"{source}: {s}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    retriever = Retriever()
    results = retriever.retrieve(
        query=args.query,
        k=args.k,
        ticker=args.ticker,
        doc_types=args.doc_types,
    )
    if not results:
        print("(no results)")
        return 0
    for i, r in enumerate(results, 1):
        section = r.metadata.get("section") or r.metadata.get("speaker") or ""
        print(f"\n[{i}] score={r.score:.3f}  {r.metadata.get('ticker', '-')} "
              f"{r.metadata.get('doc_type', '-')}  {section}")
        print(f"    doc_id: {r.metadata.get('doc_id')}")
        # Truncate the text for the terminal; full text is in metadata
        snippet = r.text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        print(f"    {snippet}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    indexer = Indexer()
    s = indexer.stats()
    print(f"Total documents:       {s['total_documents']}")
    print(f"Total chunks tracked:  {s['total_chunks_recorded']}")
    print(f"By doc_type:           {s['by_doc_type']}")
    print(f"Vector store:          {s['vector_store_stats']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="RAG ingestion and retrieval CLI for Ary Quant.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable INFO-level logging.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Ingest documents into the vector store.")
    p_index.add_argument("--tickers", nargs="+", default=[],
                         help="Tickers to index filings/theses for.")
    p_index.add_argument("--force", action="store_true",
                         help="Re-index even if content hash is unchanged.")
    p_index.add_argument("--notes-only", action="store_true",
                         help="Skip filings/theses; only index fund notes.")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Run a one-off retrieval query.")
    p_query.add_argument("query", help="Natural-language question.")
    p_query.add_argument("--k", type=int, default=8,
                         help="Number of chunks to return.")
    p_query.add_argument("--ticker", help="Filter to a single ticker.")
    p_query.add_argument("--doc-types", nargs="+",
                         help="Filter to one or more doc types.")
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="Show index size and breakdown.")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
