"""
fix_jobs_read_inapp.py
=====================

Add a "Read" button to each saved-report row that shows the report's content
INLINE in the app — no external PDF viewer, no download. Reads the HTML sidecar
that report_orchestrator now writes next to each PDF.

WHY
---
You wanted to read the report inside the app (like reading it in the terminal),
not open it externally. The orchestrator writes a {name}.html sidecar alongside
each {name}.pdf; this Read button renders that HTML inline via
st.components.v1.html. Open (external) and Download stay as options too.

WHAT IT DOES
------------
Replaces the per-report rows block (keys rep_open_/rep_dl_) with rows that have:
    {name} · {date}   [Read]  [Open]  [⬇]
"Read" toggles an inline panel showing the report content (from the .html
sidecar; falls back to a note if the sidecar is missing — e.g. a report made
before this change).

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the read-button marker and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_jobs_read_inapp.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

# The per-report rows block (added by fix_jobs_report_rows.py).
ROWS_BLOCK = '''
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

NEW_ROWS = '''
    # --- Saved reports: Read inline / Open / Download per row ---------------
    try:
        import os as _os
        from pathlib import Path as _P
        import datetime as _dt
        import streamlit.components.v1 as _components
        _root = _P(__file__).resolve().parent.parent
        _rdir = _root / "reports"
        _pdfs = sorted(_rdir.glob("*.pdf"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \\
            if _rdir.exists() else []
        if _pdfs:
            st.markdown("**Saved reports**")
            for _i, _pdf in enumerate(_pdfs[:12]):
                _ts = _dt.datetime.fromtimestamp(_pdf.stat().st_mtime)
                _nc, _rc, _oc, _dc = st.columns([3, 1, 1, 1])
                _nc.caption(f"{_pdf.name}  ·  {_ts.strftime('%Y-%m-%d %H:%M')}")
                _rk = f"rep_read_{_i}"
                if _rc.button("Read", key=f"rep_readbtn_{_i}",
                              use_container_width=True):
                    st.session_state[_rk] = not st.session_state.get(_rk, False)
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
                # Inline reader: show the HTML sidecar content.
                if st.session_state.get(_rk, False):
                    _html = _pdf.with_suffix(".html")
                    if _html.exists():
                        try:
                            _components.html(_html.read_text(encoding="utf-8"),
                                             height=520, scrolling=True)
                        except Exception as _he:  # noqa: BLE001
                            st.caption(f"couldn't render ({_he}).")
                    else:
                        st.caption("No inline text for this report (it predates "
                                   "the in-app reader). Regenerate it with "
                                   "`report <ticker>` to enable Read.")
        else:
            st.caption("No reports saved yet — run `report <ticker>` to make one.")
    except Exception as _pe:  # noqa: BLE001
        st.caption(f"saved-reports list unavailable: {_pe}")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_read_inapp] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'key=f"rep_readbtn_{_i}"' in src:
        print("[fix_jobs_read_inapp] Already applied — Read button present. "
              "Nothing to do.")
        return

    if ROWS_BLOCK not in src:
        _fail("could not find the per-report rows block to extend. Apply "
              "fix_jobs_report_rows.py first (and check it wasn't reformatted). "
              "Not editing blindly.")

    src = src.replace(ROWS_BLOCK, NEW_ROWS, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_read_inapp] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Each report row now has a 'Read' button that shows the report")
    print("    content inline (from the .html sidecar), plus Open + Download.")
    print()
    print("Make sure the updated report_orchestrator.py (which writes the .html")
    print("sidecar) is in ui/. Then run `report NVDA`, open the Jobs tab, and")
    print("click 'Read' to view the memo inline.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_read_inapp.py
