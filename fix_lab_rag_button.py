"""
fix_lab_rag_button.py
====================

Add a UI trigger for the RAG learning loop (#6) — a "RAG learning" view in the
Lab with two buttons that run the loop as background jobs and show the result.

WHY / HOW
---------
The learning loop (auditor -> scorer -> curator -> indexer) had no UI trigger;
it only ran headless from the scheduler. But the scheduler already wires the
loop with all its dependencies via ``RefreshScheduler._run_rag_learning()`` and
``._run_rag_audit()``. So this patch reuses that path rather than rebuilding the
curator/auditor/indexer chain:

* A module-level worker (``_rag_job_worker``) constructs a RefreshScheduler and
  calls the chosen method. It touches no Streamlit, so it's safe to run on the
  job queue.
* ``_render_rag_learning_panel`` adds two buttons — "Run learning cycle"
  (process recently-closed theses) and "Run audit" (re-evaluate + sample) —
  each submitting a background job via ui.state.submit_job, then polling and
  showing the returned dict.
* A new "RAG learning" option is added to the Lab's view radio.

CAVEATS (shown in the panel too)
--------------------------------
* Needs the embedder on the 768-dim backend. Until ARY_EMBED_BACKEND=ollama is
  set (with Ollama running), retrieval/indexing degrades — the loop may run but
  index poorly. The panel notes this.
* Returns {"rows": 0, "note": "..."} when prerequisites are missing (e.g. no
  recently-closed theses, or portfolio_db hooks absent). That's informative,
  not a crash.

SAFETY
------
* Targets ui/lab.py.
* Backs up to ui/lab.py.bak before writing.
* Idempotent: detects the panel and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_lab_rag_button.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "lab.py"

# --- 1. Worker + panel: insert before render_lab -------------------------
RENDER_LAB_ANCHOR = "def render_lab("

PANEL_INSERT = '''def _rag_job_worker(mode: str, db_path: str) -> dict:
    """Background-job worker: run the RAG learning loop via the scheduler.

    Reuses RefreshScheduler's already-wired loop (curator/auditor/indexer).
    Touches no Streamlit — safe for the job queue. ``mode`` is "learning" or
    "audit". Returns the scheduler method's result dict (or an error dict).
    """
    try:
        try:
            from data.refresh_scheduler import RefreshScheduler
        except Exception:
            from refresh_scheduler import RefreshScheduler  # type: ignore
        sched = RefreshScheduler(db_path=db_path)
        if mode == "audit":
            return sched._run_rag_audit()
        return sched._run_rag_learning()
    except Exception as e:  # noqa: BLE001
        return {"rows": 0, "note": f"error: {type(e).__name__}: {e}"}


def _render_rag_learning_panel() -> None:
    """Buttons to run the RAG learning loop / audit as background jobs."""
    st.markdown("#### RAG learning loop")
    st.caption(
        "Runs the self-indexing pipeline (auditor → scorer → curator → "
        "indexer) over recently-closed theses. P&L-weighted: winners get "
        "indexed into the vector store, losers get demoted."
    )

    # Job-queue handles (optional — fall back to blocking if unavailable).
    try:
        from ui import state as S
    except Exception:
        try:
            import state as S  # type: ignore
        except Exception:
            S = None

    _db = "data/hedgefund.db"

    st.info(
        "Note: this needs the 768-dim embedder. If ARY_EMBED_BACKEND=ollama "
        "isn't set (with Ollama running), indexing degrades. A result like "
        "`{'rows': 0, 'note': 'no_recently_closed_hook'}` means there were no "
        "closed theses to learn from — informative, not an error.",
        icon="ℹ️",
    )

    col_a, col_b = st.columns(2)
    run_learning = col_a.button("Run learning cycle", key="rag_run_learning",
                                use_container_width=True)
    run_audit = col_b.button("Run audit", key="rag_run_audit",
                             use_container_width=True)

    if S is None:
        # No job queue — run inline (blocks the UI briefly).
        if run_learning or run_audit:
            mode = "audit" if run_audit else "learning"
            with st.spinner(f"Running RAG {mode}… (this can take a minute)"):
                result = _rag_job_worker(mode, _db)
            st.write(result)
        return

    job_key = "rag_job_id"
    result_key = "rag_job_result"

    if run_learning:
        job_id = S.submit_job("rag", "LEARNING", _rag_job_worker, "learning",
                              _db, label="RAG learning cycle")
        st.session_state[job_key] = job_id
        st.session_state.pop(result_key, None)
    if run_audit:
        job_id = S.submit_job("rag", "AUDIT", _rag_job_worker, "audit",
                              _db, label="RAG audit")
        st.session_state[job_key] = job_id
        st.session_state.pop(result_key, None)

    # Poll the active job (same pattern as the analyzer section).
    if st.session_state.get(job_key):
        job = S.get_job(st.session_state[job_key])
        if job is not None:
            if job.state in (S.JobState.QUEUED, S.JobState.RUNNING):
                st.warning("RAG job running in the background — this panel "
                           "updates as it progresses.")
                S.maybe_autorefresh()
            elif job.state == S.JobState.DONE:
                st.session_state[result_key] = job.result
                st.session_state.pop(job_key, None)
            elif job.state == S.JobState.ERROR:
                st.error(f"RAG job failed: "
                         f"{getattr(job, 'error', 'unknown error')}")
                st.session_state.pop(job_key, None)

    if st.session_state.get(result_key) is not None:
        st.success("RAG job complete:")
        st.write(st.session_state[result_key])
    elif not st.session_state.get(job_key):
        st.caption("No RAG job has been run yet this session.")


def render_lab('''


# --- 2. Add "RAG learning" to the view radio -----------------------------
RADIO_ANCHOR = '''        view = st.radio(
            "View",
            ["Per-ticker bench", "Extended models", "Portfolio structure"],
            horizontal=True,
            key="lab_view")'''

RADIO_INSERT = '''        view = st.radio(
            "View",
            ["Per-ticker bench", "Extended models", "Portfolio structure",
             "RAG learning"],
            horizontal=True,
            key="lab_view")'''

# --- 3. Add the dispatch branch (before the final else) ------------------
DISPATCH_ANCHOR = '''        else:
            _render_structure_panel(held_tickers, price_loader)'''

DISPATCH_INSERT = '''        elif view == "RAG learning":
            _render_rag_learning_panel()
        else:
            _render_structure_panel(held_tickers, price_loader)'''


def _fail(msg: str) -> None:
    print(f"[fix_lab_rag_button] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "_render_rag_learning_panel" in src:
        print("[fix_lab_rag_button] Already applied — RAG learning panel "
              "present. Nothing to do.")
        return

    for anchor, name in (
        (RENDER_LAB_ANCHOR, "render_lab definition"),
        (RADIO_ANCHOR, "view radio"),
        (DISPATCH_ANCHOR, "structure-panel dispatch (final else)"),
    ):
        if anchor not in src:
            _fail(f"could not find the {name} anchor. The file may have "
                  "changed. Not editing blindly.")

    # Insert panel before render_lab (first occurrence is the def).
    src = src.replace(RENDER_LAB_ANCHOR, PANEL_INSERT, 1)
    src = src.replace(RADIO_ANCHOR, RADIO_INSERT, 1)
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_lab_rag_button] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added _rag_job_worker + _render_rag_learning_panel")
    print("  • Added 'RAG learning' view to the Lab radio + dispatch")
    print()
    print("Restart Streamlit, go to Lab → 'RAG learning', and click")
    print("'Run learning cycle' or 'Run audit'. Watch the panel + the terminal")
    print("(for the 384/768 embedder warning if ARY_EMBED_BACKEND isn't set).")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_lab_rag_button.py
