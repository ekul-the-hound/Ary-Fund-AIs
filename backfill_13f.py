"""
backfill_13f.py
===============
Ingest 13F-HR institutional holdings for the largest managers, so the Money
Flow Graph gains dense who-owns-whom edges (each 13F lists *every* position the
institution holds — hundreds of companies per filer).

Uses the fetcher's existing ``SECFetcher.ingest_13f_filings_by_filer(cik)`` and
the CIK list in ``data/filer_canonical.CIK_TO_NAME`` (single source of truth,
so names/CIKs stay in sync with the graph's node canonicalization).

Rate-limited (the fetcher throttles internally; we add a small pause),
resumable, safe to re-run.

Run from project root (venv active):
    python backfill_13f.py                 # all institutions in the CIK map
    python backfill_13f.py --count 8       # last 8 quarters each
    python backfill_13f.py --cik 1067983 102909   # only these CIKs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from data.sec_fetcher import SECFetcher
except Exception:
    from sec_fetcher import SECFetcher  # type: ignore
try:
    from data.filer_canonical import CIK_TO_NAME, name_for_cik
except Exception:
    from filer_canonical import CIK_TO_NAME, name_for_cik  # type: ignore

_DONE = Path("data") / "f13f_backfill_done.json"
_CORRECTED = Path("data") / "f13f_corrected.json"

try:
    from data import config as _cfg  # type: ignore
except Exception:
    try:
        import config as _cfg        # type: ignore
    except Exception:
        _cfg = None
_MARKET_DB = (getattr(_cfg, "MARKET_DB_PATH", None)
              or getattr(_cfg, "HEDGEFUND_DB_PATH", None)
              or "data/hedgefund.db")


def _load_done() -> set[str]:
    try:
        return set(json.loads(_DONE.read_text()))
    except Exception:
        return set()


def _save_done(done: set[str]) -> None:
    _DONE.parent.mkdir(parents=True, exist_ok=True)
    _DONE.write_text(json.dumps(sorted(done)))


def _correct_post2023_values() -> int:
    """SEC switched 13F <value> from $thousands to whole dollars in 2023-Q1,
    but the fetcher multiplies by 1000 unconditionally — so 2023+ filings come
    out 1000x too large. Divide those back down. Resume-safe: each accession is
    corrected at most once (tracked in f13f_corrected.json)."""
    import sqlite3
    corrected = set()
    try:
        corrected = set(json.loads(_CORRECTED.read_text()))
    except Exception:
        pass
    fixed = 0
    try:
        with sqlite3.connect(_MARKET_DB) as conn:
            accs = [r[0] for r in conn.execute(
                "SELECT DISTINCT accession_number FROM f13f_holdings "
                "WHERE period_of_report >= '2023-01-01'").fetchall()]
            for acc in accs:
                if acc in corrected:
                    continue
                conn.execute(
                    "UPDATE f13f_holdings SET value_usd = value_usd/1000.0 "
                    "WHERE accession_number=? AND period_of_report >= '2023-01-01'",
                    (acc,))
                corrected.add(acc)
                fixed += 1
        _CORRECTED.parent.mkdir(parents=True, exist_ok=True)
        _CORRECTED.write_text(json.dumps(sorted(corrected)))
    except Exception as e:  # noqa: BLE001
        print(f"  (value correction skipped: {e})")
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4,
                    help="quarters of 13F history per institution (default 4)")
    ap.add_argument("--cik", nargs="*", default=None,
                    help="specific CIKs (default: all in filer_canonical)")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    ciks = [str(c) for c in (args.cik or list(CIK_TO_NAME.keys()))]
    done = _load_done()
    todo = [c for c in ciks if c not in done]

    print(f"Institutions: {len(ciks)} | already done: {len(done)} | "
          f"to process: {len(todo)}")

    total_rows = 0
    t0 = time.time()
    if todo:
        f = SECFetcher()
        for i, cik in enumerate(todo, 1):
            name = name_for_cik(cik) or f"CIK {cik}"
            try:
                n = f.ingest_13f_filings_by_filer(cik, count=args.count)
                total_rows += n
                flag = "" if n else "  (0 rows — check CIK / no recent 13F)"
                print(f"[{i:>2}/{len(todo)}] {name:<26} CIK {cik:<9} holdings={n}{flag}")
                done.add(cik)
            except Exception as e:  # noqa: BLE001
                print(f"[{i:>2}/{len(todo)}] {name:<26} CIK {cik:<9} ERROR: {e}")
            _save_done(done)
            time.sleep(args.sleep)
        print(f"\nDone. {total_rows} total holdings rows across {len(todo)} "
              f"institutions in {(time.time()-t0)/60:.1f} min.")
    else:
        print("All filers already ingested — running correction + index only.")

    fixed = _correct_post2023_values()
    print(f"Corrected 2023+ value scaling on {fixed} new filing(s).")

    # index so the builder's "top holdings by value" query stays fast on the
    # now-large table (1M+ rows across the big filers).
    try:
        import sqlite3
        with sqlite3.connect(_MARKET_DB) as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_13f_value "
                         "ON f13f_holdings(value_usd DESC)")
        print("Ensured idx_13f_value index.")
    except Exception as e:  # noqa: BLE001
        print(f"  (index skipped: {e})")
    print("Rebuild the graph:  Scope = Full universe -> Rebuild "
          "(institutions now merge across 13D/G + 13F).")


if __name__ == "__main__":
    main()