"""
screener_data.py
================

One command to get the screener's data downloaded and ready.

WHY THIS EXISTS
---------------
The screener (`ui/screener.py`) does not read from a database — it fetches
live from yfinance on each render and only lazy-loads fundamentals for the
top-N symbols by market cap (`_FUNDAMENTALS_LAZY_LIMIT`). So most rows show
"—" for fundamentals until they scroll into view, and the universe carries
some delisted tickers that spam "possibly delisted" warnings.

But the fetch path it uses, `MarketData.get_fundamentals(...)` /
`MarketData.get_prices(...)`, ALREADY caches to `data/hedgefund.db` for 24h.
And the screener calls it with `use_cache=True`. So we don't need a new cache
or any change to the screener: if we pre-warm MarketData's cache for the WHOLE
universe once, every subsequent screener render gets instant cache hits and
fills fundamentals for all names, not just the megacaps.

This script therefore does two things, as subcommands:

    check  — Test every universe ticker against yfinance and report which are
             delisted / returning no data (read-only; changes nothing). Use
             this to find the dead symbols (PXD, SQ, etc.) cluttering the
             universe so you can prune them in universe.py.

    warm   — Fetch prices + fundamentals for the whole universe into the
             MarketData cache (data/hedgefund.db). Progress bar, resumable,
             rate-limited, and skips names already fresh in the cache. After
             this, open the screener and the full universe shows fundamentals.

    all    — check, then warm (warming only the symbols that passed `check`).

DESIGN
------
* Reuses MarketData(db_path="data/hedgefund.db") — the same cache the screener
  reads. No new files, no screener edits.
* Defensive: one bad/delisted ticker never aborts the run; failures are
  collected and reported at the end.
* Rate-limited (small sleep between symbols) to be polite to yfinance and
  avoid throttling on a ~500-name sweep.
* Resumable: `warm` skips symbols whose fundamentals are already cached and
  fresh, so re-running after an interruption is cheap.

USAGE (from project root, venv active):
    python data/screener_data.py check
    python data/screener_data.py warm
    python data/screener_data.py all
    python data/screener_data.py warm --limit 50      # first 50 only (quick test)
    python data/screener_data.py warm --sleep 0.5     # custom inter-symbol delay

NOTE: Run this from the PROJECT ROOT (D:\\Ary Fund), not from inside data/.
The import fallbacks resolve data.* / bare names either way, but running from
root keeps the working directory correct for the hedgefund.db path.

NOTE: this hits live yfinance, so it needs network access and will take a few
minutes for the full universe. Leave it running; the progress line updates in
place.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Working-directory normalization
# ----------------------------------------------------------------------
# MarketData defaults to db_path="data/hedgefund.db", which is relative to the
# CURRENT WORKING DIRECTORY. This script lives in data/ but must behave as if
# launched from the project root so that "data/hedgefund.db" resolves to the
# real DB (not "data/data/hedgefund.db"). We compute the project root as the
# parent of this file's directory and chdir there, so the command works no
# matter where the user invokes it from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    os.chdir(_PROJECT_ROOT)
except Exception:
    pass
# Ensure the root is importable too (so data.* / bare imports both resolve).
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ----------------------------------------------------------------------
# Imports from the project (tolerate package or flat layout)
# ----------------------------------------------------------------------
def _import_universe():
    for path in ("data.universe", "universe"):
        try:
            mod = __import__(path, fromlist=["US_UNIVERSE"])
            return mod
        except Exception:
            continue
    print("ERROR: could not import universe module (tried data.universe and "
          "universe). Run from the project root with the venv active.")
    sys.exit(1)


def _import_market_data():
    for path in ("data.market_data", "market_data"):
        try:
            mod = __import__(path, fromlist=["MarketData"])
            return mod.MarketData
        except Exception:
            continue
    print("ERROR: could not import MarketData (tried data.market_data and "
          "market_data).")
    sys.exit(1)


# ----------------------------------------------------------------------
# check — report delisted / no-data tickers
# ----------------------------------------------------------------------
def cmd_check(symbols: list[str], *, sleep: float) -> list[str]:
    """Probe each symbol; return the list of symbols that returned usable data.

    A symbol "passes" if MarketData returns a non-empty recent price frame.
    Delisted/dead symbols (no data) are collected and printed so they can be
    pruned from universe.py. This is read-only.
    """
    MarketData = _import_market_data()
    md = MarketData()

    good: list[str] = []
    dead: list[str] = []
    total = len(symbols)
    print(f"[check] Probing {total} symbols against yfinance "
          f"(this is read-only)…\n")

    for i, sym in enumerate(symbols, 1):
        ok = False
        try:
            # A short, cheap price pull. We bypass the long-cache so we test
            # the LIVE symbol status, not a stale cache hit.
            df = md.get_prices(sym, period="5d", use_cache=False)
            ok = (df is not None and len(df) > 0)
        except Exception:
            ok = False
        (good if ok else dead).append(sym)

        # In-place progress line.
        print(f"\r[check] {i}/{total}  ok={len(good)}  dead={len(dead)}  "
              f"(last: {sym})        ", end="", flush=True)
        time.sleep(sleep)

    print("\n")
    if dead:
        print(f"[check] {len(dead)} symbol(s) returned NO data (likely "
              f"delisted / renamed):")
        # Print in compact rows of 8.
        for j in range(0, len(dead), 8):
            print("   " + "  ".join(dead[j:j + 8]))
        print()
        print("Consider pruning these from US_UNIVERSE in universe.py "
              "(or remapping renamed tickers, e.g. SQ -> XYZ).")
    else:
        print("[check] All symbols returned data. Universe is clean.")
    print()
    return good


# ----------------------------------------------------------------------
# warm — pre-fetch prices + fundamentals into the MarketData cache
# ----------------------------------------------------------------------
def cmd_warm(symbols: list[str], *, sleep: float,
             limit: Optional[int] = None) -> None:
    """Fill MarketData's 24h cache (data/hedgefund.db) for the universe.

    For each symbol we pull prices (use_cache=True, so already-fresh names are
    skipped cheaply) and fundamentals (use_cache=True). Afterwards the
    screener's `_fetch_fundamentals_one(... use_cache=True)` gets cache hits
    for the whole universe, so fundamentals render for every name, not just
    the lazy top-N.

    Resumable + defensive: failures are collected, not fatal.
    """
    MarketData = _import_market_data()
    md = MarketData()

    if limit:
        symbols = symbols[:limit]
    total = len(symbols)

    warmed = 0
    failed: list[str] = []
    t0 = time.time()
    print(f"[warm] Warming cache for {total} symbols into data/hedgefund.db "
          f"(prices + fundamentals)…\n")

    for i, sym in enumerate(symbols, 1):
        try:
            # Prices first (populates the price cache; cheap if fresh).
            md.get_prices(sym, period="1y", use_cache=True)
            # Fundamentals with use_cache=False: the whole POINT of `warm` is
            # to refresh the cache, so we must BYPASS the 24h TTL read. With
            # use_cache=True, a re-warm within 24h short-circuits on the
            # existing (possibly partial) row and never refetches — which
            # silently leaves stale/partial fundamentals in place. Forcing a
            # fresh fetch here guarantees the cache is rewritten with a
            # complete payload. (get_fundamentals still WRITES to the cache via
            # INSERT OR REPLACE regardless of the read flag.)
            md.get_fundamentals(sym, use_cache=False)
            warmed += 1
        except Exception as e:
            failed.append(f"{sym} ({type(e).__name__})")

        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else 0
        print(f"\r[warm] {i}/{total}  warmed={warmed}  failed={len(failed)}  "
              f"~{eta:5.0f}s left  (last: {sym})        ", end="", flush=True)
        time.sleep(sleep)

    print("\n")
    print(f"[warm] Done. Warmed {warmed}/{total} in {time.time()-t0:.0f}s.")
    if failed:
        print(f"[warm] {len(failed)} failed (delisted or transient):")
        for j in range(0, len(failed), 4):
            print("   " + "   ".join(failed[j:j + 4]))
    print()
    print("Open the Screener — fundamentals should now fill across the whole "
          "universe instead of only the top names. The cache lasts 24h; re-run "
          "`warm` daily (or wire it into refresh_scheduler) to keep it fresh.")
    print()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / cache the screener's universe data.")
    parser.add_argument("command", choices=["check", "warm", "all"],
                        help="check = report delisted; warm = fill cache; "
                             "all = check then warm the survivors")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N symbols (quick test).")
    parser.add_argument("--sleep", type=float, default=0.20,
                        help="Seconds to sleep between symbols (default 0.20).")
    args = parser.parse_args()

    universe = _import_universe()
    symbols = list(getattr(universe, "US_UNIVERSE", ()) or ())
    if not symbols:
        print("ERROR: US_UNIVERSE is empty. Check universe.py.")
        sys.exit(1)

    print(f"Universe size: {len(symbols)} symbols.\n")

    if args.command == "check":
        cmd_check(symbols, sleep=args.sleep)
    elif args.command == "warm":
        cmd_warm(symbols, sleep=args.sleep, limit=args.limit)
    elif args.command == "all":
        good = cmd_check(symbols, sleep=args.sleep)
        print(f"[all] Warming the {len(good)} symbols that passed check…\n")
        cmd_warm(good, sleep=args.sleep, limit=args.limit)


if __name__ == "__main__":
    main()

# D:\Ary Fund\data\screener_data.py