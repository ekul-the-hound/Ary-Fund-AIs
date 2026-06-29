"""
fix_jobs_open_latest.py
======================

Add an "Open latest report" button to the Jobs panel header (just under the
"Clear finished" button), so you can open the most recent report PDF without
hunting for its job row.

HOW IT WORKS
------------
On click, scans the project's ``reports/`` folder for the newest ``*.pdf`` by
modified time and opens it via ``os.startfile`` (works because Streamlit runs
locally on your machine). If there are no PDFs yet, shows a hint to run
``report <ticker>`` first.

Note (honest): os.startfile is Windows-only and only does anything when the
Streamlit server and your screen are the same machine — which is your setup.
Guarded with try/except so it never breaks the panel.

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the button marker and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_jobs_open_latest.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

ANCHOR = '''    head_l, head_r = st.columns([3, 1])
    head_l.markdown("**Background jobs**")
    if head_r.button("Clear finished", key="tray_clear",
                     use_container_width=True):
        S.clear_finished_jobs()
        st.rerun()'''

INSERT = '''    head_l, head_r = st.columns([3, 1])
    head_l.markdown("**Background jobs**")
    if head_r.button("Clear finished", key="tray_clear",
                     use_container_width=True):
        S.clear_finished_jobs()
        st.rerun()

    # Open the most recent report PDF (newest in reports/), if any.
    if head_r.button("Open latest report", key="tray_open_latest",
                     use_container_width=True):
        try:
            import os as _os
            from pathlib import Path as _P
            _root = _P(__file__).resolve().parent.parent
            _rdir = _root / "reports"
            _pdfs = sorted(_rdir.glob("*.pdf"),
                           key=lambda p: p.stat().st_mtime, reverse=True) \\
                if _rdir.exists() else []
            if _pdfs:
                _os.startfile(str(_pdfs[0]))  # noqa: B606 (Windows-only, local)
            else:
                st.caption("No reports yet — run `report <ticker>` first.")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"Couldn't open latest report: {_e}")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_open_latest] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'key="tray_open_latest"' in src:
        print("[fix_jobs_open_latest] Already applied — Open latest report "
              "button present. Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the 'Clear finished' header block to extend. The "
              "file may have changed. Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_open_latest] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added 'Open latest report' button under 'Clear finished'")
    print()
    print("Restart Streamlit. After running `report NVDA`, the button opens")
    print("the newest PDF in reports/ in your default viewer.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_open_latest.py
