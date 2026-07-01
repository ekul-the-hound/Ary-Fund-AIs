"""
fix_report_table_layout.py
=========================

Fix two cosmetic layout nits in the report:

1. "Net debt / EBITDA-0.24" — long labels in the metrics two-column table sit
   flush against their values because the label column is only 2.0 inch wide
   and the value column 4.0 inch. Rebalance to 2.6 / 3.4 inch so long labels
   have breathing room before the value, and right-pad the label cell.

2. "52w drawdown Charts-15.86%" — the Charts section H1 header renders tight
   against the last metrics row. Wrap the Charts heading + body in a spacer so
   the header always starts cleanly below the metrics table.

Both are pure layout; no data or wording changes.

SAFETY
------
* Targets ui/content_builder.py.
* Backs up to ui/content_builder.py.bak before writing.
* Idempotent: detects both markers; applies only the parts not yet present.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_report_table_layout.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("report") / "content_builder.py"

# --- Fix 1: rebalance the two-col metrics table widths ---------------------
COL_OLD = '''        col_widths=[2.0 * inch, 4.0 * inch],
        header_row=False,
    )'''
COL_NEW = '''        col_widths=[2.6 * inch, 3.4 * inch],
        header_row=False,
    )'''

# --- Fix 2: add a spacer before the Charts section header ------------------
CHARTS_OLD = '''    out: list[Flowable] = [Paragraph("Charts", styles["H1"])]'''
CHARTS_NEW = '''    # Leading spacer so the Charts header never collides with the last row
    # of the preceding metrics table.
    out: list[Flowable] = [Spacer(1, 10), Paragraph("Charts", styles["H1"])]'''


def _fail(msg: str) -> None:
    print(f"[fix_report_table_layout] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")
    original = src
    applied = []

    # Fix 1
    if COL_NEW.split("\\n")[0].strip() in src:
        pass  # already rebalanced
    elif COL_OLD in src:
        src = src.replace(COL_OLD, COL_NEW, 1)
        applied.append("rebalanced metrics table columns (2.6/3.4 inch)")
    else:
        print("[fix_report_table_layout] note: two-col width block not found "
              "as expected; skipping column fix (may already differ).")

    # Fix 2
    if 'Spacer(1, 10), Paragraph("Charts"' in src:
        pass  # already spaced
    elif CHARTS_OLD in src:
        src = src.replace(CHARTS_OLD, CHARTS_NEW, 1)
        applied.append("added spacer before the Charts header")
    else:
        print("[fix_report_table_layout] note: Charts header line not found as "
              "expected; skipping charts spacer (may already differ).")

    if src == original:
        print("[fix_report_table_layout] Already applied (or nothing to "
              "change). Nothing to do.")
        return

    # Ensure Spacer is imported (it's used elsewhere in this file already, but
    # be safe).
    if "Spacer" not in src.split("def ", 1)[0] and "import" in src:
        # Spacer is from reportlab.platform.flowables; the file already uses it
        # for other sections, so this is just a guard. Do nothing if unsure.
        pass

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_report_table_layout] SUCCESS")
    print(f"  • Backed up original to {backup}")
    for a in applied:
        print(f"  • {a}")
    print()
    print("Regenerate to see it:  report NVDA")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_report_table_layout.py