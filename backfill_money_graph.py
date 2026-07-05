"""
backfill_money_graph.py
=======================
One-off: ingest SEC ownership/insider/XBRL data across the whole universe so
the Money Flow Graph has real edges. Rate-limit-aware and resumable.

Run from the project root, inside your venv:
    .\hedgefund_ai\Scripts\Activate.ps1
    python backfill_money_graph.py                # full universe
    python backfill_money_graph.py NVDA AAPL MSFT # just these (quick test)
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

# --- import the fetcher + universe, tolerating both layouts --------------
try:
    from data.sec_fetcher import SECFetcher
except Exception:
    from sec_fetcher import SECFetcher  # type: ignore
try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE  # type: ignore

# progress file so a re-run skips tickers already done
_DONE_FILE = Path("data") / "money_graph_backfill_done.json"


def _load_done() -> set[str]:
    try:
        return set(json.loads(_DONE_FILE.read_text()))
    except Exception:
        return set()


def _save_done(done: set[str]) -> None:
    try:
        _DONE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _DONE_FILE.write_text(json.dumps(sorted(done)))
    except Exception as e:  # noqa: BLE001
        print(f"  (could not save progress: {e})")


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]] or list(US_UNIVERSE)
    done = _load_done()
    todo = [t for t in tickers if t not in done]

    print(f"Universe: {len(tickers)} | already done: {len(done)} | "
          f"to process: {len(todo)}")
    if not todo:
        print("Nothing to do — delete data/money_graph_backfill_done.json to "
              "force a full re-run.")
        return

    f = SECFetcher()
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            counts = f.refresh_ticker_filings(t)  # XBRL + Form 4 + SC 13D/13G
            own = counts.get("SC 13D", 0) + counts.get("SC 13G", 0)
            print(f"[{i:>3}/{len(todo)}] {t:<6} ownership={own} "
                  f"form4={counts.get('form4', 0)} xbrl={counts.get('xbrl_facts', 0)}")
            done.add(t)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>3}/{len(todo)}] {t:<6} ERROR: {e}")
        if i % 20 == 0:
            _save_done(done)
            rate = i / max(1e-6, time.time() - t0)
            eta = (len(todo) - i) / max(1e-6, rate)
            print(f"    …{i} done, ~{rate:.1f}/s, ETA ~{eta/60:.1f} min")

    _save_done(done)
    print(f"\nDone. Processed {len(todo)} tickers in "
          f"{(time.time()-t0)/60:.1f} min.")
    print("Now open the app, choose Scope = 'Full universe', and hit Rebuild.")


if __name__ == "__main__":
    main()
