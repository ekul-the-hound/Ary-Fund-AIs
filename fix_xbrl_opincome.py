"""
fix_xbrl_opincome.py
===================

Add Operating Income to the XBRL concept map so it gets ingested from SEC
companyfacts (Total Assets and CapEx are already mapped).

CONTEXT
-------
``ingest_xbrl_facts`` in sec_fetcher.py walks ``XBRL_CONCEPT_MAP`` and writes
each mapped concept into the ``xbrl_facts`` table. The map already covers
total_assets (``Assets``) and capex
(``PaymentsToAcquirePropertyPlantAndEquipment``), but has NO entry for
operating income — so the screener's "Op Income" column can never fill from
XBRL. This adds the canonical operating-income concept (plus a common
fallback).

NOTE ON COVERAGE
----------------
``OperatingIncomeLoss`` is the standard US-GAAP tag and is widely reported,
but some companies — especially banks/financials (JPM, BAC, ...) — don't file
a clean GAAP operating-income line, so those will remain blank even after
ingestion. That's a filing reality, not a bug. Total Assets and CapEx have
broader coverage.

SAFETY
------
* Targets data/sec_fetcher.py.
* Backs up to data/sec_fetcher.py.bak before writing.
* Idempotent: detects the operating_income entry and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_xbrl_opincome.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("data") / "sec_fetcher.py"

# Insert the operating_income entry right after the total_assets entry
# (which is already present and stable).
ANCHOR = '''    "ticker.fundamental.total_assets": [
        "Assets",
    ],'''

INSERT = '''    "ticker.fundamental.total_assets": [
        "Assets",
    ],
    "ticker.fundamental.operating_income_ttm": [
        "OperatingIncomeLoss",
        # Fallback some filers use when OperatingIncomeLoss is absent:
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],'''


def _fail(msg: str) -> None:
    print(f"[fix_xbrl_opincome] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "ticker.fundamental.operating_income_ttm" in src:
        print("[fix_xbrl_opincome] Already applied — operating_income concept "
              "present. Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the total_assets map entry to anchor the insert. "
              "The map may have changed. Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_xbrl_opincome] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added operating_income_ttm -> OperatingIncomeLoss (+fallback)")
    print()
    print("Next: run the XBRL warm so operating income (and the already-mapped")
    print("total assets / capex) get ingested:")
    print("    python data/screener_data.py xbrl")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_xbrl_opincome.py
