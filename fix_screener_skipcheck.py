"""
fix_screener_skipcheck.py
========================

Fix for the screener showing `None` in the DEEP fundamentals columns
(PEG, P/S, P/B, gross/op/profit margin, ROE, ROA, FCF, balance-sheet, etc.)
for exactly the big-cap names at the top of the table (NVDA, AAPL, MSFT,
AMZN, META, ...), even though the cache and yfinance both have the data.

ROOT CAUSE
----------
In ``_build_screener_frame``:

  * Step 4 merges ``_OFFLINE_FALLBACK_STOCKS`` into the frame. Each fallback
    row provides ONLY shallow fields: market_cap, pe, sector (plus price etc.)
    — it does NOT include deep fields (margins, ROE, PEG, P/S, P/B, FCF...).

  * Step 5's lazy-fetch loop has an "already have fundamentals?" skip-check:

        already = (pd.notna(market_cap) and pd.notna(pe)
                   and sector not in (None, "", "-"))
        if already: continue

    For every fallback name, Step 4 already filled market_cap + pe + sector,
    so this check is True → the loop SKIPS the real fundamentals fetch for
    those names. They keep only the 3 shallow offline values, and the deep
    columns stay None forever.

So the most prominent names (the fallback list = top megacaps) are precisely
the ones that never get deep fundamentals. That's the bug you see.

THE FIX
-------
Make the skip-check key on a DEEP field the offline fallback never provides
(gross_margin) instead of the shallow pe/sector it does provide. Then a row
is only skipped if it genuinely already has deep fundamentals — so the
fallback megacaps fall through to the real fetch and get fully populated.

SAFETY
------
* Targets ui/screener.py.
* Backs up to ui/screener.py.bak before writing.
* Idempotent: detects the new check and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_screener_skipcheck.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"

# The current skip-check (keys on shallow fields the offline fallback fills).
OLD = """        # Skip if we already have fundamentals (market_cap + pe + sector).
        already = (
            pd.notna(df.at[ridx, "market_cap"])
            and pd.notna(df.at[ridx, "pe"])
            and df.at[ridx, "sector"] not in (None, "", "—")
        )
        if already:
            continue"""

# New check: key on a DEEP field (gross_margin) that the offline fallback
# never provides, so fallback megacaps are not wrongly skipped.
NEW = """        # Skip only if we already have DEEP fundamentals. We key on
        # gross_margin specifically because the offline fallback fills the
        # shallow fields (market_cap + pe + sector) for megacaps but NOT the
        # deep ones — keying on pe/sector here would wrongly skip those names
        # and leave their margins/ROE/PEG/etc. permanently None.
        already = (
            pd.notna(df.at[ridx, "gross_margin"])
            and pd.notna(df.at[ridx, "roe"])
            and pd.notna(df.at[ridx, "market_cap"])
        )
        if already:
            continue"""


def _fail(msg: str) -> None:
    print(f"[fix_screener_skipcheck] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'pd.notna(df.at[ridx, "gross_margin"])' in src:
        print("[fix_screener_skipcheck] Already applied — skip-check keys on "
              "gross_margin. Nothing to do.")
        return

    if OLD not in src:
        _fail("could not find the exact skip-check block to replace. The file "
              "may have changed (whitespace/quotes). Not editing blindly.")

    src = src.replace(OLD, NEW, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_skipcheck] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Skip-check now keys on deep fields (gross_margin + roe +")
    print("    market_cap) instead of the shallow pe/sector the offline")
    print("    fallback pre-fills.")
    print()
    print("Fully restart Streamlit (to clear the _build_screener_frame cache),")
    print("then open Valuation / Profitability. The top megacap names (NVDA,")
    print("AAPL, MSFT, ...) should now fill their deep columns instead of None.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_skipcheck.py
