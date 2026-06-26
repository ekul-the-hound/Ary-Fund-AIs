"""
fix_jobs_download.py
===================

Add a "Download PDF" button to finished ``report`` jobs in the Jobs panel.

CONTEXT
-------
``report <ticker>`` runs generate_report on the job queue, which returns the
written PDF's path as ``job.result``. The job row showed "done" but offered no
way to open the file. This patch reads ``job.result`` for finished report jobs
and renders an ``st.download_button`` (when the file exists on disk).

SAFETY
------
* Targets ui/palette.py.
* Backs up to ui/palette.py.bak before writing.
* Idempotent: detects the download block and does nothing on re-run.
* Verifies ast.parse before saving.
* Defensive: only acts on kind == "report" with a real .pdf path that exists;
  otherwise the row renders exactly as before.

Usage (from project root, venv active):
    python fix_jobs_download.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "palette.py"

ANCHOR = '''    if job.state == S.JobState.ERROR and job.error:
        st.caption(f"↳ {job.error[:200]}")'''

INSERT = '''    if job.state == S.JobState.ERROR and job.error:
        st.caption(f"↳ {job.error[:200]}")

    # Finished report jobs expose the PDF path as job.result — offer it.
    if job.state == S.JobState.DONE and job.kind == "report":
        _pdf = getattr(job, "result", None)
        try:
            from pathlib import Path as _P
            if _pdf and str(_pdf).lower().endswith(".pdf") and _P(_pdf).exists():
                _p = _P(_pdf)
                with open(_p, "rb") as _fh:
                    st.download_button(
                        label=f"⬇ Download {_p.name}",
                        data=_fh.read(),
                        file_name=_p.name,
                        mime="application/pdf",
                        key=f"dl_{job.kind}_{job.ticker}_{_p.name}",
                        use_container_width=False,
                    )
                st.caption(f"↳ saved to {_p}")
            elif _pdf:
                st.caption(f"↳ report result: {_pdf}")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"↳ report ready but link failed: {_e}")'''


def _fail(msg: str) -> None:
    print(f"[fix_jobs_download] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if 'job.kind == "report"' in src and "Download" in src:
        print("[fix_jobs_download] Already applied — download block present. "
              "Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the _render_job_row error-caption anchor. The "
              "file may have changed. Not editing blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_jobs_download] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Finished report jobs now show a 'Download PDF' button")
    print()
    print("Restart Streamlit, run `report NVDA`, then open the Jobs tab —")
    print("the finished row has a download button and the saved path.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_jobs_download.py
