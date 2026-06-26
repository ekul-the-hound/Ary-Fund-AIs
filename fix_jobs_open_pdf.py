"""
fix_jobs_open_pdf.py
===================

Add an "Open PDF" button (and "Open folder") next to the Download button on
finished report jobs, so you can view the report from the UI without manually
navigating to reports/.

HOW IT WORKS
------------
Streamlit runs server-side, but you run it locally (same machine), so
``os.startfile(path)`` opens the PDF in your default viewer on your desktop —
exactly what you want for a local single-user workstation. We add:
  * "Open PDF"   → os.startfile(pdf_path)
  * "Open folder"→ os.startfile(folder)  (reveals the file in Explorer)
alongside the existing Download button.

Caveat (stated honestly): os.startfile is Windows-only and only does anything
when the Streamlit server and your screen are the same machine. That's your
setup. On a remote/shared deployment it would do nothing useful, so we guard it
with a try/except and fall back silently to the Download button.

PREREQUISITE
------------
fix_jobs_download.py must have been applied (this extends that block).

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the open-button marker and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_jobs_open_pdf.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

# Anchor: the download_button call added by fix_jobs_download.py. We insert the
# open buttons right after the st.download_button(...) block's caption.
ANCHOR = '''                        key=f"dl_{job.kind}_{job.ticker}_{_p.name}",
                        use_container_width=False,
                    )'''

INSERT = '''                        key=f"dl_{job.kind}_{job.ticker}_{_p.name}",
                        use_container_width=False,
                    )
                # Open buttons — work because Streamlit runs on this machine.
                import os as _os
                _oc1, _oc2, _oc3 = st.columns([1, 1, 3])
                if _oc1.button("Open PDF",
                               key=f"open_{job.kind}_{job.ticker}_{_p.name}"):
                    try:
                        _os.startfile(str(_p))  # noqa: B606 (Windows-only, local)
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"could not open ({_oe}); use Download.")
                if _oc2.button("Open folder",
                               key=f"openf_{job.kind}_{job.ticker}_{_p.name}"):
                    try:
                        _os.startfile(str(_p.parent))  # noqa: B606
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"could not open folder ({_oe}).")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_open_pdf] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if '_oc1.button("Open PDF"' in src:
        print("[fix_jobs_open_pdf] Already applied — Open PDF button present. "
              "Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the download_button block to extend. Apply "
              "fix_jobs_download.py first (and check it wasn't reformatted). "
              "Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_open_pdf] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added 'Open PDF' + 'Open folder' buttons to finished report jobs")
    print()
    print("Restart Streamlit, run `report NVDA`, open the Jobs tab, and click")
    print("'Open PDF' — it opens in your default viewer.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_open_pdf.py
