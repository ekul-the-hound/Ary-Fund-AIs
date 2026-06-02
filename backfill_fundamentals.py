"""
backfill_fundamentals.py — fetch fundamentals (incl. SECTOR) for the universe.

The filing-text backfill populated sec_filings for ~520 names, but Stage 2
peer-relative scoring buckets by SECTOR, which lives in the separate
fundamentals_cache (sourced from yfinance .info), NOT in sec_filings. The
filing backfill never touched fundamentals, so every backfilled ticker has
filing text but no sector -> peer stats compute "0 sectors".

This fetches fundamentals for each universe ticker via
MarketData.get_fundamentals(), which caches to fundamentals_cache with a
24h TTL. After this, sector is populated and peer-stats bucketing works.

Designed like the filing backfill:
  * RESUMABLE   — get_fundamentals is cache-first (24h), so a re-run skips
                  anything already fetched; nothing is lost on Ctrl-C.
  * FAULT-TOLERANT — one ticker failing never aborts the batch.
  * PROGRESS + ETA.
Lighter than the filing fetch (one yfinance call/ticker, no big docs).

Usage:
    python backfill_fundamentals.py            # whole universe, resume-safe
    python backfill_fundamentals.py --limit 10  # smoke test
    python backfill_fundamentals.py --force     # ignore cache, refetch all
No VRAM used — network + SQLite only.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from typing import Set

import config

try:
    from data.universe import US_UNIVERSE
except Exception:  # noqa: BLE001
    from universe import US_UNIVERSE

try:
    from data.market_data import MarketData
except Exception:  # noqa: BLE001
    from market_data import MarketData


def _tickers_with_sector(db_path: str) -> Set[str]:
    """Tickers whose cached fundamentals already carry a non-empty sector."""
    conn = sqlite3.connect(db_path)
    done: Set[str] = set()
    try:
        rows = conn.execute(
            "SELECT ticker, data_json FROM fundamentals_cache"
        ).fetchall()
    except sqlite3.OperationalError:
        return done
    finally:
        conn.close()
    import json
    for tk, blob in rows:
        try:
            d = json.loads(blob)
            if (d.get("sector") or "").strip():
                done.add(tk)
        except Exception:  # noqa: BLE001
            continue
    return done


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Backfill fundamentals/sector for the universe.")
    p.add_argument("--tickers", nargs="+")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Refetch even if sector already cached.")
    args = p.parse_args(argv)

    db_path = config.PORTFOLIO_DB_PATH
    universe = list(args.tickers) if args.tickers else list(US_UNIVERSE)

    done = set() if args.force else _tickers_with_sector(db_path)
    pending = [t for t in universe if t not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    print(f"universe={len(universe)}  already_have_sector={len(done)}  pending={total}")
    if not total:
        print("Nothing to do — all requested tickers already have a sector.")
        return 0

    md = MarketData(db_path=db_path)
    t0 = time.time()
    ok = 0
    no_sector = 0
    failed = 0

    for i, ticker in enumerate(pending, 1):
        try:
            f = md.get_fundamentals(ticker, use_cache=not args.force)
            sector = (f.get("sector") or "").strip() if isinstance(f, dict) else ""
            if sector:
                ok += 1
                tag = sector
            else:
                no_sector += 1
                tag = "(no sector)"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"[{i}/{total}] {ticker:6} -> {tag:24} "
                  f"(ok={ok} nosec={no_sector} fail={failed}, ETA ~{eta:.0f}m)")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}. Progress saved — re-run to resume.")
            return 130
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{total}] {ticker:6} -> ERROR ({e})")

    print(f"\nDone. {ok} with sector, {no_sector} no-sector, {failed} errors, "
          f"in {(time.time()-t0)/60:.1f} min.")
    print("Next: re-run the pipeline (peer stats will recompute with sectors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
