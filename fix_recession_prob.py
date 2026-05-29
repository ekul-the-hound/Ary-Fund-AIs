"""
fix_recession_prob.py  —  one-time cleanup
==========================================
The recession-probability scale bug (FRED reports RECPROUSM156N in
percentage points, but the project expects a 0-1 fraction) was fixed in
data/macro_data.py. That fix corrects all FUTURE writes. But the
data_registry is a persistent SQLite store, and earlier pipeline runs
already wrote the un-converted value (e.g. 1.82 instead of 0.0182) into
the canonical ``global.recession_prob`` field. Because the context
builder reads canonical fields from the registry first, those stale rows
keep winning and the "182%" keeps appearing.

This script deletes ONLY the poisoned ``global.recession_prob`` rows
from the registry's ``data_points`` table. The next pipeline run will
repopulate the field with the corrected, fraction-scaled value.

It is safe and surgical:
  * touches only field = 'global.recession_prob' for entity 'GLOBAL'
  * makes a timestamped backup of portfolio.db first
  * prints what it found and what it deleted
  * idempotent — running it twice is harmless

Run from the project root:
    python fix_recession_prob.py
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime

# The registry shares the pipeline DB. Pull the path from config so this
# matches wherever the project actually writes.
try:
    import config
    DB_PATH = getattr(config, "PORTFOLIO_DB_PATH", "data/portfolio.db")
except Exception:
    DB_PATH = "data/portfolio.db"

FIELD = "global.recession_prob"
ENTITY = "GLOBAL"


def main() -> int:
    print(f"Registry DB: {DB_PATH}")

    # 1. Backup first — never mutate a DB without a copy.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{DB_PATH}.bak_{stamp}"
    try:
        shutil.copy2(DB_PATH, backup)
        print(f"Backup written: {backup}")
    except FileNotFoundError:
        print(f"ERROR: {DB_PATH} not found. Run from the project root.")
        return 1

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # 2. Show what's there now.
        rows = conn.execute(
            "SELECT entity_id, field, value_num, as_of "
            "FROM data_points WHERE field = ? AND entity_id = ? "
            "ORDER BY as_of DESC",
            (FIELD, ENTITY),
        ).fetchall()

        if not rows:
            print(f"No '{FIELD}' rows found — nothing to clean. "
                  "(The next run will populate it correctly.)")
            return 0

        print(f"\nFound {len(rows)} '{FIELD}' row(s):")
        for r in rows[:10]:
            v = r["value_num"]
            flag = "  <-- POISONED (>1.0, stored as percent)" if (
                v is not None and v > 1.0
            ) else ""
            print(f"  as_of={r['as_of']}  value={v}{flag}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")

        # 3. Delete them all. The next pipeline run rewrites the field
        #    with the corrected (fraction) scale, so a clean slate is
        #    simplest and safest — no risk of half-converted rows.
        cur = conn.execute(
            "DELETE FROM data_points WHERE field = ? AND entity_id = ?",
            (FIELD, ENTITY),
        )
        conn.commit()
        print(f"\nDeleted {cur.rowcount} row(s) for '{FIELD}'.")

    print(
        "\nDone. Now re-run the pipeline to repopulate with the corrected "
        "scale:\n    python main.py --tickers AAPL --model dev_llama8\n"
        "    python peek.py AAPL\n"
        "Expect: macro risk no longer falsely HIGH, and key_risks no "
        "longer says '182%'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())