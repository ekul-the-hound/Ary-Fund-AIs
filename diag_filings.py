"""
diag_loader.py — test the FilingsLoader directly, bypassing the indexer.

Tells us definitively whether load_for_ticker yields documents and, if
not, exactly where it breaks.

    python diag_loader.py MSFT
"""
import sys
import traceback

import config
from data.sec_fetcher import SECFetcher


def main(ticker: str) -> int:
    ticker = ticker.upper()
    print(f"Testing FilingsLoader for {ticker}\n")

    sec = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

    # 1. Does get_filings (per-type, low count) return rows the way the
    #    patched loader calls it?
    for ft, count in [("10-K", 3), ("10-Q", 2), ("8-K", 10), ("DEF 14A", 2)]:
        try:
            rows = sec.get_filings(ticker, ft, count=count) or []
            print(f"get_filings({ticker!r}, {ft!r}, count={count}) -> {len(rows)} rows")
            for r in rows[:1]:
                # show the keys so we see field names (filed_date vs filing_date)
                print(f"    sample keys: {sorted(r.keys())}")
                print(f"    accession: {r.get('accession_number')}  "
                      f"date: {r.get('filed_date') or r.get('filing_date')}")
        except Exception as e:  # noqa: BLE001
            print(f"get_filings({ft}) FAILED: {e}")
            traceback.print_exc()

    # 2. Now drive the actual loader.
    print("\n--- FilingsLoader.load_for_ticker ---")
    try:
        from rag.document_loaders.filings import FilingsLoader
    except Exception as e:  # noqa: BLE001
        print(f"IMPORT FAILED: {e}")
        traceback.print_exc()
        return 1

    fl = FilingsLoader(sec_fetcher=sec)
    docs = list(fl.load_for_ticker(ticker))
    print(f"load_for_ticker yielded {len(docs)} document(s)")
    for d in docs:
        text_len = len(getattr(d, "text", "") or "")
        print(f"  doc_id={getattr(d, 'doc_id', '?')}  "
              f"type={getattr(d, 'doc_type', '?')}  text={text_len:,} chars")

    if not docs:
        print("\n>>> Loader yielded NOTHING. The per-type get_filings calls "
              "above tell us whether rows came back; if they did but the "
              "loader yielded 0, the break is inside _build_document "
              "(accession/text fetch).")
    return 0


if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    raise SystemExit(main(tk))