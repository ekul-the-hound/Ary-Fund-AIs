"""
fill_rag.py — populate the RAG index for specific tickers
==========================================================

WHY: ``python -m rag index --tickers MSFT`` hard-codes sec_fetcher=None
     and portfolio_db=None, so it indexes only global fund notes — never
     a ticker's filings or theses. That leaves research_docs empty and
     every retrieval mismatches/returns nothing. This script wires up the
     REAL SECFetcher and PortfolioDB, then calls the same
     Indexer.run_tickers the CLI uses, so 10-K/10-Q/8-K filing text and
     any saved theses for the requested tickers actually get chunked,
     embedded (via Ollama nomic-embed-text, 768-dim), and written to the
     Chroma vector store.

HOW:
  - Builds SECFetcher and PortfolioDB from config paths
  - Forces the Ollama embedder (ARY_EMBED_BACKEND=ollama) so we never
    silently fall back to 384-dim MiniLM and corrupt the 768-dim store
  - Runs the indexer for the tickers passed on the command line
  - Prints before/after stats so you can see chunks land

RUN:
    python fill_rag.py --tickers MSFT
    python fill_rag.py --tickers MSFT AAPL NVDA --force

Use --force to re-index documents whose content hash hasn't changed
(otherwise already-indexed filings are skipped).
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fill the RAG index for tickers.")
    parser.add_argument(
        "--tickers", nargs="+", required=True,
        help="Ticker symbols to index, e.g. --tickers MSFT AAPL",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-index even if the document content hash is unchanged.",
    )
    args = parser.parse_args(argv)

    # Force the 768-dim Ollama embedder so we never silently fall back to
    # 384-dim MiniLM (which would mismatch the existing vector store).
    os.environ.setdefault("ARY_EMBED_BACKEND", "ollama")

    import config
    from data.sec_fetcher import SECFetcher
    from data.portfolio_db import PortfolioDB
    from rag.indexer import Indexer

    # --- Build the real loaders the CLI omits -----------------------------
    # SECFetcher defaults its own db_path; pass agent identity from config
    # if present. Filings text is cached by the fetcher (your main.py run
    # logged "Returning cached filings for MSFT"), so this reuses that.
    sec_kwargs = {}
    if getattr(config, "SEC_AGENT_NAME", None):
        sec_kwargs["agent_name"] = config.SEC_AGENT_NAME
    if getattr(config, "SEC_AGENT_EMAIL", None):
        sec_kwargs["agent_email"] = config.SEC_AGENT_EMAIL
    sec_fetcher = SECFetcher(**sec_kwargs)

    # ThesesLoader calls portfolio_db.get_thesis_history(ticker), which is
    # an INSTANCE method — so build a PortfolioDB, not the module.
    portfolio_db = PortfolioDB(db_path=config.PORTFOLIO_DB_PATH)

    # --- Build the indexer (defaults read embedder + store from config) ---
    indexer = Indexer(
        tracking_db_path=getattr(config, "RAG_TRACKING_DB", "data/rag_tracking.db"),
        chunk_tokens=getattr(config, "RAG_CHUNK_TOKENS", 500),
        overlap_tokens=getattr(config, "RAG_OVERLAP_TOKENS", 50),
    )

    print(f"Embedder backend = {indexer.embedder.backend_name} "
          f"(dim={indexer.embedder.dimension})")
    if indexer.embedder.dimension != 768:
        print(
            "WARNING: embedder is not 768-dim. The vector store was built "
            "with nomic (768). Indexing now would mix dimensions. Start "
            "Ollama and ensure `ollama pull nomic-embed-text`.",
            file=sys.stderr,
        )
        return 2

    print("\n--- BEFORE ---")
    _print_stats(indexer)

    print(f"\nIndexing tickers: {', '.join(args.tickers)} "
          f"(force={args.force}) ...")
    stats = indexer.run_tickers(
        tickers=args.tickers,
        sec_fetcher=sec_fetcher,
        portfolio_db=portfolio_db,
        force=args.force,
    )
    for loader_name, loader_stats in (stats or {}).items():
        print(f"  {loader_name}: {loader_stats}")

    print("\n--- AFTER ---")
    _print_stats(indexer)
    return 0


def _print_stats(indexer) -> None:
    try:
        s = indexer.stats()
        print(f"  total_documents      = {s.get('total_documents')}")
        print(f"  total_chunks_recorded = {s.get('total_chunks_recorded')}")
        print(f"  by_doc_type          = {s.get('by_doc_type')}")
        print(f"  vector_store_stats   = {s.get('vector_store_stats')}")
    except Exception as e:  # noqa: BLE001
        print(f"  (stats unavailable: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
