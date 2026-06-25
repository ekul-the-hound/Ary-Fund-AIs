"""
fix_screener_xbrl_read.py
========================

Bridge the screener to read real Operating Income / Total Assets / CapEx from
the ``xbrl_facts`` table (populated by ``python data/screener_data.py xbrl``),
filling those three columns for rows where yfinance left them None.

HOW IT WORKS
------------
After the fundamentals + performance loops in ``_build_screener_frame``, this
inserts a step that — for rows still missing operating_income / total_assets /
capex — queries the latest annual (10-K) value of the matching US-GAAP concept
from xbrl_facts and fills the column. It's a LOCAL SQLite read (the SEC network
calls already happened during the xbrl warm), so there's no render-time network
cost.

Concept -> column:
    operating_income <- OperatingIncomeLoss
    total_assets     <- Assets
    capex            <- PaymentsToAcquirePropertyPlantAndEquipment

COVERAGE CAVEAT
---------------
Values only appear for symbols that were ingested AND that report the concept.
Some financials (banks) don't file a GAAP operating-income line, so Op Income
stays blank for them — a filing reality, not a bug. Total Assets and CapEx
have broader coverage. Run the xbrl warm first or these columns stay None.

SAFETY
------
* Targets ui/screener.py.
* Backs up to ui/screener.py.bak before writing.
* Idempotent: detects the XBRL bridge and does nothing on re-run.
* Verifies ast.parse before saving.
* Anchors on the stable "Step 6" marker, so it sits after the fundamentals
  and performance loops regardless of whether those patches are present.

Usage (from project root, venv active):
    python fix_screener_xbrl_read.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"

ANCHOR = "    # --- Step 6: derive rel_volume from volume / 30-day average ----"

INSERT = '''    # --- Step 5d: fill Op Income / Total Assets / CapEx from XBRL -------
    # Real from-filings values, read from the local xbrl_facts table
    # (populated by `python data/screener_data.py xbrl`). Local SQLite read —
    # no network at render time. Only fills rows where yfinance left these
    # None; symbols not ingested or not reporting the concept stay "—".
    try:
        import sqlite3 as _sqlite3
        _XBRL_COL_CONCEPT = {
            "operating_income": "OperatingIncomeLoss",
            "total_assets": "Assets",
            "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
        }
        _xbrl_db = "data/hedgefund.db"
        with _sqlite3.connect(_xbrl_db) as _xc:
            # Confirm the table exists (xbrl warm may not have run yet).
            _has_tbl = _xc.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='xbrl_facts'"
            ).fetchone() is not None
            if _has_tbl:
                for _ridx in df.index:
                    _sym = df.at[_ridx, "symbol"]
                    if not _sym:
                        continue
                    for _col, _concept in _XBRL_COL_CONCEPT.items():
                        if _col not in df.columns:
                            continue
                        # Don't overwrite a value already present.
                        if pd.notna(df.at[_ridx, _col]):
                            continue
                        _row = _xc.execute(
                            "SELECT value FROM xbrl_facts "
                            "WHERE ticker = ? AND concept = ? AND form = '10-K' "
                            "ORDER BY period_end DESC LIMIT 1",
                            (str(_sym).upper(), _concept),
                        ).fetchone()
                        if _row and _row[0] is not None:
                            df.at[_ridx, _col] = float(_row[0])
    except Exception:
        # XBRL is best-effort enrichment; never break the frame build over it.
        pass

''' + ANCHOR


def _fail(msg: str) -> None:
    print(f"[fix_screener_xbrl_read] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "# --- Step 5d: fill Op Income / Total Assets / CapEx from XBRL" in src:
        print("[fix_screener_xbrl_read] Already applied — XBRL read bridge "
              "present. Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the 'Step 6' anchor to insert before. The file "
              "may differ. Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_xbrl_read] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added Step 5d: read Op Income / Total Assets / CapEx from")
    print("    xbrl_facts for rows where yfinance left them None.")
    print()
    print("Make sure you've run the XBRL warm first:")
    print("    python data/screener_data.py xbrl")
    print("Then fully restart Streamlit and check Income statement (Op Income),")
    print("Balance sheet (Total Assets), and Cash flow (CapEx).")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_xbrl_read.py
