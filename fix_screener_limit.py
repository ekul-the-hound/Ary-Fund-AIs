"""
fix_screener_limit.py
=====================

Raise the screener's fundamentals lazy-load cap so the warm-cached universe
fills ALL category columns (Valuation, Profitability, Income statement,
Balance sheet, Cash flow, Per share, Technicals), not just the top 60 by
market cap.

CONTEXT
-------
``ui/screener.py`` reads fundamentals via
``MarketData.get_fundamentals(symbol, use_cache=True)`` — i.e. it reads the
same 24h SQLite cache in data/hedgefund.db that ``data/screener_data.py warm``
fills. But it only populates fundamentals for the top
``_FUNDAMENTALS_LAZY_LIMIT`` (= 60) symbols per render, because the cap was
sized for COLD caches where each read is a live yfinance round-trip.

Now that ``warm`` pre-fills the whole universe, those reads are near-instant
cache hits, so the 60 cap is needlessly conservative — it's why every deep
category tab shows ``None`` below the top 60.

THE FIX
-------
Raise ``_FUNDAMENTALS_LAZY_LIMIT`` from 60 to 600 (covers the ~560-name
universe with margin) and update the rationale comment to reflect the
warm-cache workflow.

TRADE-OFF (stated honestly)
---------------------------
At 600, names whose cache entry has EXPIRED (>24h since the last ``warm``)
will trigger live yfinance fetches on render and can slow the first render or
briefly show ``None`` until re-warmed. Keep the cache fresh by re-running
``python data/screener_data.py warm`` daily (or wiring it into
refresh_scheduler). With a fresh warm, the screener fills fully and fast.

SAFETY
------
* Targets ui/screener.py.
* Backs up to ui/screener.py.bak before writing.
* Idempotent: detects the 600 value and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_screener_limit.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"

OLD_BLOCK = """# Cap on lazy fundamentals fetches per render. Each call is a yfinance
# round trip; even with 24h SQLite caching, the *first* render after a
# clean start can hit dozens of network calls for the visible page.
# 60 covers the default Streamlit dataframe height comfortably without
# stalling the UI on cold caches.
_FUNDAMENTALS_LAZY_LIMIT = 60"""

NEW_BLOCK = """# Cap on lazy fundamentals reads per render. Each read goes through
# MarketData.get_fundamentals(use_cache=True), which hits the 24h SQLite
# cache in data/hedgefund.db. With the cache pre-warmed for the whole
# universe (`python data/screener_data.py warm`), these are near-instant
# cache hits, so the cap can cover the full ~560-name universe — this is
# what fills the deep category columns (valuation, profitability, balance
# sheet, etc.) for every row, not just the megacaps. Raised 60 -> 600.
# NOTE: if the cache has expired (>24h since the last warm), names beyond
# the warmed set fall back to live yfinance fetches and can slow the first
# render; re-run `warm` daily (or via refresh_scheduler) to keep it fast.
_FUNDAMENTALS_LAZY_LIMIT = 600"""


def _fail(msg: str) -> None:
    print(f"[fix_screener_limit] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    # Idempotency.
    if "_FUNDAMENTALS_LAZY_LIMIT = 600" in src:
        print("[fix_screener_limit] Already applied — limit is 600. Nothing "
              "to do.")
        return

    # Prefer replacing the whole commented block; fall back to just the line.
    if OLD_BLOCK in src:
        src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    elif "_FUNDAMENTALS_LAZY_LIMIT = 60" in src:
        src = src.replace("_FUNDAMENTALS_LAZY_LIMIT = 60",
                          "_FUNDAMENTALS_LAZY_LIMIT = 600", 1)
        print("[fix_screener_limit] NOTE: comment block differed; bumped the "
              "value only.")
    else:
        _fail("could not find '_FUNDAMENTALS_LAZY_LIMIT = 60'. The file may "
              "have changed; not editing blindly.")

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_limit] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Raised _FUNDAMENTALS_LAZY_LIMIT 60 -> 600")
    print()
    print("Reload the Screener. With the cache warmed, the deep category tabs")
    print("(Valuation PEG/PS/PB, Profitability, Balance sheet, Cash flow, Per")
    print("share, Technicals) should now fill for the whole universe instead")
    print("of only the top 60 names.")
    print()
    print("If some rows still show None, those names' cache entries have")
    print("expired — re-run:  python data/screener_data.py warm")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_limit.py
