"""
fix_app_v2_report_wire.py
========================

Wire the new ui/report_orchestrator.py into app_v2's report resolver so the
``report <ticker>`` command finds ``generate_report`` and produces a PDF.

CONTEXT
-------
app_v2.py resolves the report function by scanning modules
(``ui.pdf_renderer`` / ``pdf_renderer``) for ``generate_report`` /
``render_report`` / ``build_report``. pdf_renderer only exposes the low-level
``render_pdf`` (it writes a pre-built Story), so the scan found nothing and the
app said "Reports aren't available in this build."

report_orchestrator.py supplies the missing high-level ``generate_report``.
This patch adds it to the front of the module list the resolver scans.

SAFETY
------
* Targets ui/app_v2.py.
* Backs up to ui/app_v2.py.bak before writing.
* Idempotent: detects report_orchestrator in the list and does nothing.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_app_v2_report_wire.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "app_v2.py"

ANCHOR = '''    report_fn = None
    for modname in ("ui.pdf_renderer", "pdf_renderer"):'''

INSERT = '''    report_fn = None
    for modname in ("ui.report_orchestrator", "report_orchestrator",
                    "ui.pdf_renderer", "pdf_renderer"):'''


def _fail(msg: str) -> None:
    print(f"[fix_app_v2_report_wire] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "report_orchestrator" in src:
        print("[fix_app_v2_report_wire] Already applied — report_orchestrator "
              "in the resolver list. Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the report resolver loop to extend. The file may "
              "have changed. Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_app_v2_report_wire] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added report_orchestrator to the report resolver list")
    print()
    print("Make sure ui/report_orchestrator.py is in place, then restart")
    print("Streamlit and run `report NVDA` in the command bar. Watch the Jobs")
    print("tab; the PDF lands in reports/.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_app_v2_report_wire.py
