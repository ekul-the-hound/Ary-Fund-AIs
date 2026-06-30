"""
fix_jobs_report_rows.py
======================

Replace the dropdown report-picker with a PER-REPORT ROW LIST: every saved PDF
gets its own row showing its name plus an "Open" button and a "Download"
button. One click on a row's Open button opens that specific report — no
selecting from a dropdown first.

WHY
---
You wanted a button by each report so you can pick which one with a single
click, and you wanted the open action kept (not removed). This lists all
reports at once, each with its own Open + Download.

WHAT IT DOES
------------
Replaces the selectbox picker (key="report_picker", added by
fix_jobs_report_picker.py) with a loop that renders, for each reports/*.pdf
(newest first):
    {name}  ·  {date}     [Open]  [Download]

Open uses os.startfile (works because Streamlit runs locally on this machine).
Download is the always-reliable browser fallback. Capped at the 12 most recent
so the panel doesn't grow unbounded.

If the selectbox picker wasn't applied, this instead inserts the row list after
the "Clear finished" block.

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the row-list marker and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_jobs_report_rows.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

# The selectbox picker block (added by fix_jobs_report_picker.py).
PICKER_BLOCK = '''
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

# Fallback anchor (Clear-finished block) if picker wasn't applied.
CLEAR_BLOCK = '''    if head_r.button("Clear finished", key="tray_clear",
                     use_container_width=True):
        S.clear_finished_jobs()
        st.rerun()'''

ROWS = '''
    # --- Saved reports: one row each, with its own Open + Download ----------
    try:
        import os as _os
        from pathlib import Path as _P
        import datetime as _dt
        _root = _P(__file__).resolve().parent.parent
        _rdir = _root / "reports"
        _pdfs = sorted(_rdir.glob("*.pdf"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \\
            if _rdir.exists() else []
        if _pdfs:
            st.markdown("**Saved reports**")
            for _i, _pdf in enumerate(_pdfs[:12]):
                _ts = _dt.datetime.fromtimestamp(_pdf.stat().st_mtime)
                _nc, _oc, _dc = st.columns([3, 1, 1])
                _nc.caption(f"{_pdf.name}  ·  {_ts.strftime('%Y-%m-%d %H:%M')}")
                if _oc.button("Open", key=f"rep_open_{_i}",
                              use_container_width=True):
                    try:
                        _os.startfile(str(_pdf))  # noqa: B606 (Windows-only)
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"Open failed ({_oe}); use Download.")
                try:
                    with open(_pdf, "rb") as _fh:
                        _dc.download_button(
                            "⬇", data=_fh.read(), file_name=_pdf.name,
                            mime="application/pdf", key=f"rep_dl_{_i}",
                            use_container_width=True)
                except Exception as _de:  # noqa: BLE001
                    _dc.caption("dl?")
        else:
            st.caption("No reports saved yet — run `report <ticker>` to make one.")
    except Exception as _pe:  # noqa: BLE001
        st.caption(f"saved-reports list unavailable: {_pe}")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_report_rows] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'key=f"rep_open_{_i}"' in src:
        print("[fix_jobs_report_rows] Already applied — per-report rows "
              "present. Nothing to do.")
        return

    if PICKER_BLOCK in src:
        src = src.replace(PICKER_BLOCK, ROWS, 1)
        where = "replaced the dropdown picker with per-report rows"
    elif CLEAR_BLOCK in src:
        src = src.replace(CLEAR_BLOCK, CLEAR_BLOCK + "\\n" + ROWS, 1)
        where = "inserted per-report rows after the 'Clear finished' block"
    else:
        _fail("could not find the dropdown picker or the Clear-finished block "
              "to anchor the rows. Not editing blindly.")

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_report_rows] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print(f"  • {where}")
    print("  • Each saved report now has its own Open + Download button")
    print()
    print("IMPORTANT: the list is empty until you generate a report. Run")
    print("    report NVDA")
    print("in the command bar (NOT `gen` — that makes an opinion, not a PDF).")
    print("Then the Jobs tab lists the PDF with one-click Open / Download.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_report_rows.py
