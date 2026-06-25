"""
fix_screener_remove_tabs.py
==========================

Remove the **Extended hours** and **Technicals** category tabs from the
screener, and tear out the per-row technicals/performance computation loop
that fed the Technicals tab.

WHY
---
* Extended hours (pre/post-market close/chg/vol): yfinance's daily data
  doesn't carry these, so the columns were always None — a dead tab.
* Technicals (RSI/MA/ATR): these read the price cache per row and only filled
  for a handful of names where the cache was warm enough, giving an
  inconsistent half-populated tab. With the tab gone, the Step 5b loop that
  computed them is pure wasted CPU on every render, so we remove it too.

WHAT THIS DOES
--------------
1. Deletes the ``"Extended hours": [...]`` and ``"Technicals": [...]`` entries
   from ``_CATEGORY_COLUMNS``. Because ``_CATEGORIES`` is derived from the
   dict's keys, both radio tabs disappear automatically.
2. Removes the ``# --- Step 5b ...`` technicals/performance loop from
   ``_build_screener_frame`` (added by fix_screener_technicals.py), restoring
   the direct hand-off to Step 6.

This leaves Performance, Valuation, Dividends, Profitability, Income
statement, Balance sheet, Cash flow, and Per share intact — including the
derived ratios. The _compute_performance helper and _TECHNICALS_LAZY_LIMIT
constant are left in place (harmless, unused) to keep the diff minimal; they
can be removed later if desired.

SAFETY
------
* Targets ui/screener.py.
* Backs up to ui/screener.py.bak before writing.
* Idempotent: if both tabs are already gone, does nothing.
* Verifies ast.parse before saving.
* Tolerant: removes whichever pieces are present (the Step 5b loop may or may
  not exist depending on whether the technicals patch was applied).

Usage (from project root, venv active):
    python fix_screener_remove_tabs.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"

# --- 1. The two category entries to delete (exact text) ------------------
EXTENDED_HOURS = '''    "Extended hours": [
        "symbol", "name", "premarket_close", "premarket_chg_pct",
        "premarket_vol", "postmarket_close", "postmarket_chg_pct",
        "postmarket_vol",
    ],
'''

TECHNICALS_CAT = '''    "Technicals": [
        "symbol", "name", "price", "change_pct", "rsi_14", "ma_50",
        "ma_200", "beta", "atr_14", "volatility_1m",
    ],
'''

# --- 2. The Step 5b loop to remove (added by fix_screener_technicals.py) --
# We replace the whole inserted block back down to the Step 6 marker.
STEP5B_START = "    # --- Step 5b: technicals + performance for the top-N by market cap ---"
STEP6_MARKER = "    # --- Step 6: derive rel_volume from volume / 30-day average ----"


def _fail(msg: str) -> None:
    print(f"[fix_screener_remove_tabs] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")
    original = src
    did = []

    # 1. Remove the two category entries.
    if EXTENDED_HOURS in src:
        src = src.replace(EXTENDED_HOURS, "", 1)
        did.append("Extended hours tab")
    if TECHNICALS_CAT in src:
        src = src.replace(TECHNICALS_CAT, "", 1)
        did.append("Technicals tab")

    # 2. Remove the Step 5b loop if present (collapse to the Step 6 marker).
    s = src.find(STEP5B_START)
    if s != -1:
        e = src.find(STEP6_MARKER, s)
        if e == -1:
            _fail("found the Step 5b start but not the Step 6 marker after it; "
                  "refusing to guess the block boundary.")
        src = src[:s] + src[e:]
        did.append("Step 5b technicals loop")

    if not did:
        print("[fix_screener_remove_tabs] Nothing to do — both tabs already "
              "removed and no Step 5b loop present.")
        return

    if src == original:
        print("[fix_screener_remove_tabs] No changes made.")
        return

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_remove_tabs] SUCCESS")
    print(f"  • Backed up original to {backup}")
    for d in did:
        print(f"  • Removed: {d}")
    print()
    print("Fully restart Streamlit. The Extended hours and Technicals radio")
    print("tabs should be gone; the remaining tabs are unchanged.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_remove_tabs.py
