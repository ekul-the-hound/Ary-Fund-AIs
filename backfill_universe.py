from __future__ import annotations
import argparse, sqlite3, sys, time
from typing import List, Set
import config
from data.sec_fetcher import SECFetcher
try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE

def _tickers_with_text(db_path: str) -> Set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM sec_filings "
            "WHERE full_text IS NOT NULL AND full_text != ''"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()

def _backfill_one(sec, ticker, kinds, max_10k, max_10q, max_chars):
    hydrated = 0
    for kind in kinds:
        count = max_10k if kind == "10-K" else max_10q
        if count <= 0:
            continue
        try:
            filings = sec.get_filings(ticker, filing_type=kind, count=count)
        except Exception as e:
            print(f"    ! {ticker} {kind}: list fetch failed ({e})")
            continue
        for f in filings:
            acc = f.get("accession_number")
            if not acc:
                continue
            try:
                text = sec.get_filing_text(acc, max_chars=max_chars)
                if text:
                    hydrated += 1
            except Exception as e:
                print(f"    ! {ticker} {kind} {acc}: text fetch failed ({e})")
    return hydrated

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--kinds", nargs="+", default=["10-K", "10-Q"])
    p.add_argument("--max-10k", type=int, default=1)
    p.add_argument("--max-10q", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=500000)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    db_path = config.PORTFOLIO_DB_PATH
    universe = list(args.tickers) if args.tickers else list(US_UNIVERSE)
    done = set() if args.force else _tickers_with_text(db_path)
    pending = [t for t in universe if t not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    print(f"universe={len(universe)}  already_done={len(done)}  pending={total}")
    if not total:
        print("Nothing to do.")
        return 0

    sec = SECFetcher(db_path=db_path)
    t0 = time.time(); ok = 0; failed = 0
    for i, ticker in enumerate(pending, 1):
        try:
            n = _backfill_one(sec, ticker, args.kinds, args.max_10k, args.max_10q, args.max_chars)
            if n: ok += 1
            else: failed += 1
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"[{i}/{total}] {ticker:6} -> {n} filings   (ok={ok} fail={failed}, ETA ~{eta:.0f}m)")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}. Progress saved — re-run to resume.")
            return 130
        except Exception as e:
            failed += 1
            print(f"[{i}/{total}] {ticker:6} -> ERROR ({e})")
    print(f"\nDone. {ok} hydrated, {failed} no-text, in {(time.time()-t0)/60:.1f} min.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
