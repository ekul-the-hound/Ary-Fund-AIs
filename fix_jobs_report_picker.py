"""
fix_jobs_report_picker.py
========================

Replace the single "Open latest report" button with a REPORT PICKER: a
dropdown of all PDFs in reports/ (newest first) plus Download + Open buttons.

WHY
---
1. "Open latest report" didn't work reliably (os.startfile runs server-side and
   can fail silently; and there may have been no fresh PDF because `gen` makes
   an opinion, not a report).
2. You want to choose WHICH of several saved reports to open.

The picker solves both: it lists every report so you can see what's actually
saved, lets you pick one, and offers BOTH a Download button (always works in
the browser) and an Open button (os.startfile, for local convenience). If Open
ever fails, Download is the reliable fallback.

WHAT IT DOES
------------
Replaces the ``tray_open_latest`` button block (added by
fix_jobs_open_latest.py) with:
  * st.selectbox of reports/*.pdf, newest first (shows name + date)
  * Download button for the selected PDF (reliable)
  * Open button for the selected PDF (os.startfile; local convenience)

If fix_jobs_open_latest.py was NOT applied, this instead inserts the picker
after the "Clear finished" block.

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the picker marker and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_jobs_report_picker.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

# The block added by fix_jobs_open_latest.py (preferred replacement target).
OPEN_LATEST_BLOCK = '''
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

# Fallback anchor if the open-latest patch wasn't applied.
CLEAR_BLOCK = '''    if head_r.button("Clear finished", key="tray_clear",
                     use_container_width=True):
        S.clear_finished_jobs()
        st.rerun()'''

PICKER = '''
    # --- Report picker: choose any saved report to open / download ---------
    try:
        import os as _os
        from pathlib import Path as _P
        _root = _P(__file__).resolve().parent.parent
        _rdir = _root / "reports"
        _pdfs = sorted(_rdir.glob("*.pdf"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \\
            if _rdir.exists() else []
        if _pdfs:
            import datetime as _dt
            def _lbl(p):
                ts = _dt.datetime.fromtimestamp(p.stat().st_mtime)
                return f"{p.name}  ·  {ts.strftime('%Y-%m-%d %H:%M')}"
            _labels = [_lbl(p) for p in _pdfs]
            _choice = st.selectbox(
                "Saved reports", _labels, index=0, key="report_picker",
                help="Newest first. Pick one to download or open.")
            _sel = _pdfs[_labels.index(_choice)]
            _dc, _oc, _ = st.columns([1, 1, 3])
            try:
                with open(_sel, "rb") as _fh:
                    _dc.download_button(
                        "⬇ Download", data=_fh.read(), file_name=_sel.name,
                        mime="application/pdf", key="picker_dl",
                        use_container_width=True)
            except Exception as _de:  # noqa: BLE001
                _dc.caption(f"download failed: {_de}")
            if _oc.button("Open", key="picker_open",
                          use_container_width=True):
                try:
                    _os.startfile(str(_sel))  # noqa: B606 (Windows-only, local)
                except Exception as _oe:  # noqa: BLE001
                    st.caption(f"Open failed ({_oe}); use Download instead.")
        else:
            st.caption("No reports saved yet — run `report <ticker>` to make one.")
    except Exception as _pe:  # noqa: BLE001
        st.caption(f"report picker unavailable: {_pe}")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_report_picker] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'key="report_picker"' in src:
        print("[fix_jobs_report_picker] Already applied — picker present. "
              "Nothing to do.")
        return

    # Prefer replacing the open-latest block; else append after Clear block.
    if OPEN_LATEST_BLOCK in src:
        src = src.replace(OPEN_LATEST_BLOCK, PICKER, 1)
        where = "replaced the 'Open latest report' button"
    elif CLEAR_BLOCK in src:
        src = src.replace(CLEAR_BLOCK, CLEAR_BLOCK + "\\n" + PICKER, 1)
        where = "inserted after the 'Clear finished' block"
    else:
        _fail("could not find either the open-latest block or the Clear-"
              "finished block to anchor the picker. Not editing blindly.")

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_report_picker] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print(f"  • {where}")
    print("  • Added a report picker (selectbox + Download + Open)")
    print()
    print("Restart Streamlit, open the Jobs tab. Under the header you'll see a")
    print("'Saved reports' dropdown listing every PDF in reports/. Pick one and")
    print("Download (always works) or Open. If the list is empty, run")
    print("`report NVDA` (NOT `gen` — gen makes an opinion, report makes the PDF).")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_report_picker.py
