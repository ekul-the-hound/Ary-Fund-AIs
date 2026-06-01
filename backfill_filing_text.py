"""
backfill_filing_text.py — download & store full_text for cached filings
========================================================================

WHY: The sec_filings table caches filing METADATA (ticker, type, date,
     accession, primary_doc_url) but full_text is empty (0 chars) for
     every row. The RAG indexer needs the actual document body to chunk
     and embed, so with empty text it indexes nothing.

HOW: SECFetcher.get_filing_text(accession) already knows how to download
     from primary_doc_url, clean the text, and write it back to both the
     on-disk cache and the full_text column. It just has never been called
     for these rows. This script calls it once per empty filing, which
     self-heals the table. EDGAR is rate-limited, so this takes a minute
     or two for a handful of tickers.

RUN:
    python backfill_filing_text.py --tickers MSFT
    python backfill_filing_text.py --tickers MSFT AAPL NVDA
    python backfill_filing_text.py --all          # every ticker in the table

After this succeeds, run:  python fill_rag.py --tickers MSFT
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import config
from data.sec_fetcher import SECFetcher


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backfill full_text for cached filings.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+", help="Tickers to backfill, e.g. --tickers MSFT")
    g.add_argument("--all", action="store_true", help="Backfill every ticker in sec_filings.")
    parser.add_argument(
        "--max-chars", type=int, default=500_000,
        help="Max characters to store per filing (default 500000).",
    )
    args = parser.parse_args(argv)

    db = config.PORTFOLIO_DB_PATH
    sec = SECFetcher(db_path=db)

    # Find rows with empty full_text.
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    if args.all:
        rows = conn.execute(
            "SELECT ticker, filing_type, filed_date, accession_number "
            "FROM sec_filings "
            "WHERE full_text IS NULL OR full_text = '' "
            "ORDER BY ticker, filed_date DESC"
        ).fetchall()
    else:
        tickers = [t.upper() for t in args.tickers]
        placeholders = ",".join("?" for _ in tickers)
        rows = conn.execute(
            f"SELECT ticker, filing_type, filed_date, accession_number "
            f"FROM sec_filings "
            f"WHERE ticker IN ({placeholders}) "
            f"AND (full_text IS NULL OR full_text = '') "
            f"ORDER BY ticker, filed_date DESC",
            tickers,
        ).fetchall()
    conn.close()

    if not rows:
        print("Nothing to backfill — all targeted filings already have text.")
        return 0

    print(f"Backfilling {len(rows)} filing(s) (downloading from EDGAR)...\n")
    ok, failed = 0, 0
    for r in rows:
        acc = r["accession_number"]
        label = f"{r['ticker']:6} {r['filing_type']:8} {r['filed_date']}"
        try:
            text = sec.get_filing_text(acc, max_chars=args.max_chars)
            n = len(text or "")
            if n > 0:
                print(f"  OK   {label} | {n:,} chars")
                ok += 1
            else:
                print(f"  WARN {label} | downloaded but empty")
                failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {label} | {e}")
            failed += 1

    print(f"\nDone. {ok} succeeded, {failed} failed.")
    if ok:
        print("Now run:  python fill_rag.py --tickers " +
              " ".join(sorted({r['ticker'] for r in rows})))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
