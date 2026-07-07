"""
past_fixes.py — archive of every one-off fix_*.py patch script, combined.

These 38 scripts were applied once each to patch target modules during the
project's build-out. Their changes already live in the real source files
(app_v2.py, screener_data.py, refresh_scheduler.py, ...); this file exists as a
single-file record of HOW each fix was applied, and keeps them runnable.

Every fix is idempotent — each checks an "already applied" marker before
touching its target, so re-running is safe and simply reports nothing to do.

Usage (from project root, venv active):
    python past_fixes.py --list                 # list every fix + one-line doc
    python past_fixes.py <name>                 # run one, e.g. context_registry_db
    python past_fixes.py --all                  # run all (each no-ops if applied)

Each original fix_<name>.py became the function _fix_<name>() below; its
module-level constants/helpers are now locals, so nothing collides across the 38.
"""
from __future__ import annotations

import argparse
import sys

def _fix_add_flow_destination() -> object:
    """
    fix_add_flow_destination.py
    ===========================
    Register the **Money Flow Graph** as a top-nav destination in
    ``ui/app_v2.py``.

    Follows the project's fix-script conventions:
      * backs the target up to ``ui/app_v2_py.bak`` before writing
      * idempotent — re-running is a no-op
      * anchor-based ``str.replace`` (no line numbers)
      * verifies the result with ``ast.parse`` before saving; restores on failure

    Three edits:
      1. add ``"Flow"`` to ``_DESTINATIONS``
      2. add a ``_render_flow_destination(backend)`` renderer
      3. route ``dest == "Flow"`` to it in ``main()``

    Run from the project root:
        python fix_add_flow_destination.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    # --- anchors (must match ui/app_v2.py verbatim) --------------------------
    ANCHOR_DESTINATIONS = (
        '_DESTINATIONS = ["Desk", "Board", "Screener", "Lab", "Analyzer", "Jobs"]'
    )
    REPLACE_DESTINATIONS = (
        '_DESTINATIONS = ["Desk", "Board", "Screener", "Lab", "Analyzer", '
        '"Flow", "Jobs"]'
    )

    ANCHOR_ROUTE = (
        '    elif dest == "Analyzer":\n'
        '        _render_analyzer_destination(backend, side["lookback"])\n'
    )
    REPLACE_ROUTE = (
        '    elif dest == "Analyzer":\n'
        '        _render_analyzer_destination(backend, side["lookback"])\n'
        '    elif dest == "Flow":\n'
        '        _render_flow_destination(backend)\n'
    )

    ANCHOR_FUNC = (
        'def _render_metrics_destination(backend: dict[str, Any]) -> None:'
    )
    FLOW_FUNC = '''def _render_flow_destination(backend: dict[str, Any]) -> None:
        """SEC money-flow network (ui/money_flow).

        Company-to-company capital & supply relationships assembled by
        data.money_graph and rendered by ui/money_flow_template.html.
        """
        try:
            from ui.money_flow import render_money_flow_destination
        except Exception:
            try:
                from money_flow import render_money_flow_destination  # type: ignore
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"Money Flow Graph unavailable: {e}. Ensure "
                    "`ui/money_flow.py`, `ui/money_flow_template.html`, and "
                    "`data/money_graph.py` are present.")
                return
        render_money_flow_destination(backend)


    '''
    REPLACE_FUNC = FLOW_FUNC + ANCHOR_FUNC


    def _find_target() -> Path:
        for p in (Path("ui/app_v2.py"), Path("app_v2.py")):
            if p.exists():
                return p
        print("ERROR: could not find ui/app_v2.py (run from the project root).")
        sys.exit(1)


    def main() -> None:
        target = _find_target()
        src = target.read_text(encoding="utf-8")

        already = (
            '"Flow"' in src
            and "def _render_flow_destination" in src
            and 'dest == "Flow"' in src
        )
        if already:
            print(f"✓ {target}: Flow destination already registered — no changes.")
            return

        # --- verify anchors exist before touching anything -------------------
        missing = []
        if ANCHOR_DESTINATIONS not in src and REPLACE_DESTINATIONS not in src:
            missing.append("_DESTINATIONS list")
        if ANCHOR_ROUTE not in src and 'dest == "Flow"' not in src:
            missing.append("Analyzer route block in main()")
        if ANCHOR_FUNC not in src:
            missing.append("_render_metrics_destination anchor")
        if missing:
            print("ERROR: expected anchors not found in ui/app_v2.py: "
                  + ", ".join(missing))
            print("The file may have drifted. Aborting without changes.")
            sys.exit(1)

        patched = src
        if REPLACE_DESTINATIONS not in patched:
            patched = patched.replace(ANCHOR_DESTINATIONS, REPLACE_DESTINATIONS, 1)
        if 'dest == "Flow"' not in patched:
            patched = patched.replace(ANCHOR_ROUTE, REPLACE_ROUTE, 1)
        if "def _render_flow_destination" not in patched:
            patched = patched.replace(ANCHOR_FUNC, REPLACE_FUNC, 1)

        # --- validate before writing ----------------------------------------
        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source failed to parse ({e}). No changes made.")
            sys.exit(1)

        backup = target.with_name(target.name.replace(".py", "_py.bak"))
        shutil.copy2(target, backup)
        target.write_text(patched, encoding="utf-8")

        print(f"✓ Backed up  {target} -> {backup}")
        print(f"✓ Patched    {target}")
        print("  • added \"Flow\" to _DESTINATIONS")
        print("  • added _render_flow_destination(backend)")
        print("  • routed dest == \"Flow\" in main()")
        print("\nRestart Streamlit:  streamlit run ui/app_v2.py")
    return main()


def _fix_app_v2_report_wire() -> object:
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
    return main()


def _fix_context_registry_db() -> object:
    """
    fix_context_registry_db.py
    =========================

    Fix build_agent_context reading the registry from the WRONG database.

    ROOT CAUSE
    ----------
    The registry (data_points: derived signals, fundamentals, macro) lives in
    data/hedgefund.db — that's where DerivedSignals writes, where the market/
    fundamentals cache lives, and what DataRegistry.DEFAULT_DB_PATH points to.
    Confirmed by row counts:
        hedgefund.db : 340 NVDA rows, 24 fields, 8 signal rows
        portfolio.db :   2 NVDA rows,  2 fields, 0 signal rows

    But build_agent_context does:
        reg = get_default_registry(db_path)
    where db_path is the PORTFOLIO db (config.PORTFOLIO_DB_PATH = data/portfolio.db).
    So it builds the registry from the nearly-empty portfolio.db and finds no
    signals -> context["derived_signals"] is empty -> risk_scanner._score_market
    gets nothing -> "Market: LOW, no data". Same for any registry-sourced field.

    THE FIX
    -------
    Resolve the registry DB to the canonical market/registry DB rather than the
    portfolio path:
        1. config.MARKET_DB_PATH or config.DATA_DB_PATH or config.REGISTRY_DB_PATH
           if any is set
        2. else DataRegistry.DEFAULT_DB_PATH (data/hedgefund.db)
        3. else fall back to the passed db_path (old behavior)
    The pipeline (for lazy backfill) keeps using the passed db_path unchanged —
    only the registry read is redirected to where the data actually is.

    After this, the derived signals in hedgefund.db reach the context, the
    market-metric injection (fix_market_risk_metrics) has real values to inject,
    and `gen NVDA` produces market (and macro, if FRED-backed) reasons.

    SAFETY
    ------
    * Targets data/pipeline.py.
    * Backs up to data/pipeline.py.bak before writing.
    * Idempotent: detects the resolver marker and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_context_registry_db.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "pipeline.py"

    ANCHOR = '''    pipe = _build_pipeline(db_path, cfg)
        reg = get_default_registry(db_path)'''

    INSERT = '''    pipe = _build_pipeline(db_path, cfg)
        # Registry lives in the canonical market/registry DB (hedgefund.db), NOT
        # the portfolio DB. Reading it from db_path (the portfolio path) returns an
        # almost-empty data_points table, so derived signals / macro never reach
        # the context. Resolve the registry DB explicitly.
        _reg_db = None
        try:
            for _attr in ("MARKET_DB_PATH", "DATA_DB_PATH", "REGISTRY_DB_PATH"):
                _v = getattr(cfg, _attr, None) if cfg else None
                if _v:
                    _reg_db = str(_v)
                    break
            if _reg_db is None:
                try:
                    from data.data_registry import DEFAULT_DB_PATH as _DEF
                except Exception:
                    from data_registry import DEFAULT_DB_PATH as _DEF  # type: ignore
                _reg_db = _DEF  # data/hedgefund.db
        except Exception:
            _reg_db = None
        reg = get_default_registry(_reg_db or db_path)'''


    def _fail(msg: str) -> None:
        print(f"[fix_context_registry_db] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_reg_db = None" in src and "MARKET_DB_PATH" in src:
            print("[fix_context_registry_db] Already applied — registry resolver "
                  "present. Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the get_default_registry(db_path) call in "
                  "build_agent_context. pipeline.py may have changed. Not editing "
                  "blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_context_registry_db] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • build_agent_context now reads the registry from the canonical")
        print("    market DB (hedgefund.db), not the portfolio DB.")
        print()
        print("NEXT — regenerate so the signals flow into the opinion:")
        print("    (ensure NVDA signals are fresh: recompute if needed)")
        print("    gen NVDA")
        print("    report NVDA")
        print()
        print("Then check `python _whymacro.py` — key_metrics[realized_vol] etc.")
        print("should now be real numbers, and Market reasons should populate.")
    return main()


def _fix_jobs_download() -> object:
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
    return main()


def _fix_jobs_open_latest() -> object:
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
    return main()


def _fix_jobs_open_pdf() -> object:
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
    return main()


def _fix_jobs_read_inapp() -> object:
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
    return main()


def _fix_jobs_report_picker() -> object:
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
    return main()


def _fix_jobs_report_rows() -> object:
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
    return main()


def _fix_lab_rag_button() -> object:
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
    return main()


def _fix_main_offline() -> object:
    """
    fix_main_offline.py — set HuggingFace offline mode at the entry point.

    Run from D:\\Ary Fund:   python fix_main_offline.py

    The reranker (and embedder) re-validate against huggingface.co on every
    startup. Setting HF_HUB_OFFLINE inside the reranker's lazy property was too
    late: the huggingface libraries read the flag at import time, and the embedder
    imports sentence_transformers before the reranker property runs. The robust
    fix is to set the env vars at the very top of main.py, before ANY import can
    pull in a huggingface library.

    This inserts an os.environ block immediately after
    `from __future__ import annotations` (which must remain the first statement).
    A network fallback isn't needed here: if the model isn't cached, the
    individual loaders still raise clearly, and you'd unset these for a one-time
    download. The model is cached on this machine.

    Surgical: inserts one block at a single stable anchor. Refuses to write if the
    anchor isn't found exactly once, or if the block is already present. Re-parses
    before and after; writes UTF-8 without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\main.py"

    ANCHOR = "from __future__ import annotations\n"

    BLOCK = (
        "from __future__ import annotations\n"
        "\n"
        "# Load cached HuggingFace models (reranker cross-encoder, embedder)\n"
        "# without re-validating against huggingface.co on every startup. Must be\n"
        "# set before any import pulls in a huggingface library, so it lives here\n"
        "# at the top of the entry point rather than in a lazy loader.\n"
        "import os as _os\n"
        "_os.environ.setdefault(\"HF_HUB_OFFLINE\", \"1\")\n"
        "_os.environ.setdefault(\"TRANSFORMERS_OFFLINE\", \"1\")\n"
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:
            src = f.read()

        if 'setdefault("HF_HUB_OFFLINE"' in src:
            print("Already patched — HF_HUB_OFFLINE block present in main.py.")
            return 0

        count = src.count(ANCHOR)
        if count == 0:
            print("ERROR: anchor 'from __future__ import annotations' not found. "
                  "No changes made. Paste the top of main.py and I'll adjust.")
            return 1
        if count > 1:
            print(f"ERROR: anchor found {count} times (expected 1). Aborting.")
            return 1

        patched = src.replace(ANCHOR, BLOCK, 1)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — HF offline env vars set at top of main.py.")
        return 0
    return main()


def _fix_market_risk_metrics() -> object:
    """
    fix_market_risk_metrics.py
    =========================

    Fix the report's "Market: LOW — No specific reasons recorded" by feeding the
    market-risk scorer the data it needs.

    ROOT CAUSE
    ----------
    risk_scanner._score_market() looks in the metrics dict for realized_vol /
    volatility, drawdown / max_drawdown, and rsi. But key_metrics is built by
    filing_analyzer.extract_key_metrics_for_agent(), which only produces
    FUNDAMENTAL fields — it never includes vol/drawdown/rsi. So _score_market
    always sees nothing and records ["no data"], which the report faithfully shows
    as "No specific reasons recorded."

    Meanwhile build_agent_context already computes those signals and exposes them
    in context["derived_signals"] (rsi_14, realized_vol_30d, drawdown). They just
    never get copied into key_metrics before the risk scan.

    THE FIX
    -------
    In main.py, immediately before risk_scanner.compute_risk_flags(...), copy the
    market signals from context["derived_signals"] into key_metrics under the names
    _score_market reads:
        realized_vol_30d -> realized_vol
        drawdown         -> drawdown
        rsi_14           -> rsi
    Only fills a field if it's present and key_metrics doesn't already have it.
    After this, re-run `gen NVDA` to regenerate the opinion WITH market reasons,
    then `report NVDA`.

    NOTE: this fixes MARKET reasons. MACRO reasons ("Macro: LOW — no data") need
    FRED data (FRED_API_KEY in .env) — that's a config item, not a code fix.

    SAFETY
    ------
    * Targets main.py.
    * Backs up to main.py.bak before writing.
    * Idempotent: detects the inject marker and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_market_risk_metrics.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("main.py")

    # Anchor: the compute_risk_flags call. We insert the injection right before it.
    ANCHOR = '''        risk_flags = risk_scanner.compute_risk_flags(
                ticker=ticker,
                metrics=key_metrics,
                macro=macro,
                agent_risks=agent_risks,
                config=cfg,
                peer_stats=_peer_slice,
            )'''

    INSERT = '''        # Wire market-risk signals into key_metrics so _score_market can
            # produce real reasons instead of "no data". build_agent_context
            # already computed these in context["derived_signals"]; they just
            # weren't copied into the metrics dict the scanner reads.
            try:
                _ds = (context.get("derived_signals")
                       if isinstance(context, dict) else None) or {}
                if isinstance(key_metrics, dict) and _ds:
                    _market_map = {
                        "realized_vol": ("realized_vol_30d", "realized_vol",
                                         "volatility"),
                        "drawdown": ("drawdown", "max_drawdown"),
                        "rsi": ("rsi", "rsi_14"),
                    }
                    for _dest, _srcs in _market_map.items():
                        if key_metrics.get(_dest) is not None:
                            continue
                        for _s in _srcs:
                            _v = _ds.get(_s)
                            if _v is not None:
                                key_metrics[_dest] = _v
                                break
            except Exception:  # noqa: BLE001 — never break the run over this
                pass

            risk_flags = risk_scanner.compute_risk_flags(
                ticker=ticker,
                metrics=key_metrics,
                macro=macro,
                agent_risks=agent_risks,
                config=cfg,
                peer_stats=_peer_slice,
            )'''


    def _fail(msg: str) -> None:
        print(f"[fix_market_risk_metrics] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_market_map = {" in src:
            print("[fix_market_risk_metrics] Already applied — market-metric "
                  "injection present. Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the compute_risk_flags call to anchor the fix. "
                  "main.py may have changed. Not editing blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_market_risk_metrics] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Market signals (realized_vol/drawdown/rsi) now injected into")
        print("    key_metrics before the risk scan.")
        print()
        print("NEXT: regenerate the opinion so it picks up the market reasons:")
        print("    gen NVDA      (recomputes the opinion WITH market risk reasons)")
        print("    report NVDA   (renders the report from the new opinion)")
        print()
        print("The Market section should then show vol/drawdown/RSI-based reasons")
        print("instead of 'no data'. (Macro still needs FRED_API_KEY in .env.)")
    return main()


def _fix_pulse_fixture() -> object:
    """
    fix_pulse_fixture.py — fix the stale-seed bug in the global-risk-pulse tests.

    Run from D:\\Ary Fund:   python fix_pulse_fixture.py

    ROOT CAUSE: the base_universe fixture seeds prices with _seed_universe's
    default end_date=datetime(2026, 5, 13). The pulse freshness filter excludes
    any ticker whose latest price is older than max_staleness_days (7) relative to
    as_of. The 15 tests using this fixture omit as_of (so it defaults to now), and
    once wall-clock time passed ~7 days beyond 2026-05-13, every seeded ticker
    became stale -> included_tickers=0 -> empty result -> 11 tests fail (and ~4
    more pass only vacuously on the empty result). This is a test-fixture
    staleness bug; the production pulse code is correct.

    FIX: seed the fixture's universe ending at datetime.now(), so the data is
    always fresh relative to the default as_of=now. The 9 as_of-pinned tests are
    untouched (they call _seed_universe directly with their own args and pin
    as_of=2026-05-13, so they stay internally consistent).

    Surgical: replaces only the one-line fixture body. Refuses to write unless it
    matches exactly once or is already patched. Verifies `datetime` is imported.
    Re-parses before and after; writes UTF-8 without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\tests\test_all.py"

    OLD = (
        "@pytest.fixture\n"
        "def base_universe(db: str) -> list[str]:\n"
        "    return _seed_universe(db)\n"
    )

    NEW = (
        "@pytest.fixture\n"
        "def base_universe(db: str) -> list[str]:\n"
        "    # Seed ending today so the data is fresh relative to the default\n"
        "    # as_of=now used by the fixture-based tests (the pulse freshness\n"
        "    # filter excludes prices older than max_staleness_days). The\n"
        "    # as_of-pinned tests call _seed_universe directly and are unaffected.\n"
        "    return _seed_universe(db, end_date=datetime.now())\n"
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:
            src = f.read()

        # Guard: datetime must be importable in this module for the new call.
        if "datetime" not in src:
            print("ERROR: 'datetime' not referenced in test_all.py — cannot rely "
                  "on it being imported. Aborting; tell me and I'll add the import.")
            return 1

        if "_seed_universe(db, end_date=datetime.now())" in src:
            print("Already patched — base_universe seeds end_date=datetime.now().")
            return 0

        count = src.count(OLD)
        if count == 0:
            print("ERROR: base_universe fixture block not found as expected. "
                  "No changes made. Paste the fixture and I'll match it.")
            return 1
        if count > 1:
            print(f"ERROR: fixture block found {count} times (expected 1). Aborting.")
            return 1

        patched = src.replace(OLD, NEW)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — base_universe now seeds end_date=datetime.now().")
        return 0
    return main()


def _fix_rag_to_tools() -> object:
    """
    fix_rag_to_tools.py
    ==================

    Move the RAG learning panel from the Lab's view radio to the sidebar "Tools"
    section (next to Metrics / Debug).

    WHAT IT DOES
    ------------
    app_v2.py:
      * Adds "RAG" to _UTILITY_DESTINATIONS (so a button appears in Tools).
      * Adds a render branch: when destination == "RAG", call lab's
        _render_rag_learning_panel().

    lab.py:
      * Removes "RAG learning" from the view radio and its dispatch branch
        (the panel function _render_rag_learning_panel + _rag_job_worker stay
        defined in lab.py and are imported by app_v2 — no need to move the code).

    SAFETY
    ------
    * Targets ui/app_v2.py and ui/lab.py.
    * Backs up each to .py.bak before writing.
    * Idempotent: detects prior application on each file.
    * Verifies ast.parse on each before saving.
    * If lab.py doesn't have the RAG radio entry (e.g. fix_lab_rag_button wasn't
      applied), it warns but still wires the Tools button — as long as the panel
      function exists.

    Usage (from project root, venv active):
        python fix_rag_to_tools.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    APP = Path("ui") / "app_v2.py"
    LAB = Path("ui") / "lab.py"

    # --- app_v2.py edits -----------------------------------------------------
    UTIL_ANCHOR = '_UTILITY_DESTINATIONS = ["Metrics", "Debug"]'
    UTIL_INSERT = '_UTILITY_DESTINATIONS = ["Metrics", "Debug", "RAG"]'

    # Add a dispatch branch. Anchor on the Metrics branch in the main router.
    DISPATCH_ANCHOR = '''    elif dest == "Metrics":
            _render_metrics_destination(backend)'''
    DISPATCH_INSERT = '''    elif dest == "Metrics":
            _render_metrics_destination(backend)
        elif dest == "RAG":
            _render_rag_destination()'''

    # Add the render function. Anchor before _render_metrics_destination.
    RENDERFN_ANCHOR = "def _render_metrics_destination(backend: dict[str, Any]) -> None:"
    RENDERFN_INSERT = '''def _render_rag_destination() -> None:
        """Render the RAG learning panel (defined in lab.py) as a Tools page."""
        st.markdown("### RAG learning")
        try:
            try:
                from ui.lab import _render_rag_learning_panel
            except Exception:
                from lab import _render_rag_learning_panel  # type: ignore
            _render_rag_learning_panel()
        except Exception as e:  # noqa: BLE001
            st.error(f"RAG learning panel unavailable: {e}")


    def _render_metrics_destination(backend: dict[str, Any]) -> None:'''

    # --- lab.py edits (remove the RAG radio option + dispatch) ---------------
    LAB_RADIO_WITH_RAG = '''        view = st.radio(
                "View",
                ["Per-ticker bench", "Extended models", "Portfolio structure",
                 "RAG learning"],
                horizontal=True,
                key="lab_view")'''
    LAB_RADIO_WITHOUT = '''        view = st.radio(
                "View",
                ["Per-ticker bench", "Extended models", "Portfolio structure"],
                horizontal=True,
                key="lab_view")'''

    LAB_DISPATCH_WITH_RAG = '''        elif view == "RAG learning":
                _render_rag_learning_panel()
            else:
                _render_structure_panel(held_tickers, price_loader)'''
    LAB_DISPATCH_WITHOUT = '''        else:
                _render_structure_panel(held_tickers, price_loader)'''


    def _fail(msg: str) -> None:
        print(f"[fix_rag_to_tools] ABORT: {msg}")
        sys.exit(1)


    def _patch_app() -> str:
        src = APP.read_text(encoding="utf-8")
        if '"RAG"' in src and "_render_rag_destination" in src:
            return "app: already applied"

        if UTIL_ANCHOR not in src:
            _fail("app_v2: could not find _UTILITY_DESTINATIONS anchor.")
        if DISPATCH_ANCHOR not in src:
            _fail("app_v2: could not find the Metrics dispatch branch anchor.")
        if RENDERFN_ANCHOR not in src:
            _fail("app_v2: could not find _render_metrics_destination anchor.")

        src = src.replace(UTIL_ANCHOR, UTIL_INSERT, 1)
        src = src.replace(DISPATCH_ANCHOR, DISPATCH_INSERT, 1)
        src = src.replace(RENDERFN_ANCHOR, RENDERFN_INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"app_v2 patched file does not parse ({e}); not saving.")

        shutil.copy2(APP, APP.with_suffix(".py.bak"))
        APP.write_text(src, encoding="utf-8")
        return "app: patched"


    def _patch_lab() -> str:
        if not LAB.exists():
            return "lab: not found (skipped)"
        src = LAB.read_text(encoding="utf-8")

        if "RAG learning" not in src:
            return ("lab: no 'RAG learning' radio entry found (nothing to remove; "
                    "panel fn assumed present)")

        changed = False
        if LAB_RADIO_WITH_RAG in src:
            src = src.replace(LAB_RADIO_WITH_RAG, LAB_RADIO_WITHOUT, 1)
            changed = True
        if LAB_DISPATCH_WITH_RAG in src:
            src = src.replace(LAB_DISPATCH_WITH_RAG, LAB_DISPATCH_WITHOUT, 1)
            changed = True

        if not changed:
            return "lab: RAG radio present but anchors didn't match (left as-is)"

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"lab patched file does not parse ({e}); not saving.")

        shutil.copy2(LAB, LAB.with_suffix(".py.bak"))
        LAB.write_text(src, encoding="utf-8")
        return "lab: RAG removed from Lab radio"


    def main() -> None:
        if not APP.exists():
            _fail(f"{APP} not found. Run from the project root (D:\\\\Ary Fund).")

        r_app = _patch_app()
        r_lab = _patch_lab()

        print("[fix_rag_to_tools] SUCCESS")
        print(f"  • {r_app}")
        print(f"  • {r_lab}")
        print()
        print("Restart Streamlit. 'RAG' now appears in the sidebar Tools section")
        print("(next to Metrics / Debug); the Lab radio no longer lists it.")
    return main()


def _fix_recession_prob() -> object:
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

    STATUS: Operational script — safe to delete after cleanup. Idempotent;
            not part of runtime contracts.
    """

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
    return main()


def _fix_redflag_dedup() -> object:
    """
    fix_redflag_dedup.py — stop one event being counted as multiple red flags.

    Run from D:\\Ary Fund:   python fix_redflag_dedup.py

    BUG (reproduced via diag_distressed.py): _find_red_flags dedups on the regex
    PATTERN, so the same pattern can't fire twice — but two DIFFERENT patterns can
    both match the same sentence (e.g. a "material weakness" pattern and a
    "controls not effective" pattern hitting one material-weakness sentence),
    producing two flags for one event. That inflates the red-flag count (and the
    -0.3-per-flag penalty) on distressed filings.

    FIX: positional dedup. Track the text spans already flagged and skip a match
    whose span overlaps one already recorded. Two patterns on the same sentence
    count once; genuinely separate events elsewhere in the text still count
    separately. Detection coverage is unchanged — only double-counting of a single
    location is removed.

    Surgical: replaces the scan loop in _find_red_flags. Refuses to write unless
    the block matches exactly once; re-parses before and after; writes UTF-8
    without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\agent\filing_analyzer.py"

    OLD = (
        '    hits: List[str] = []\n'
        '    seen: set = set()\n'
        '    for pat in (*_AFFIRMATIVE_SEVERE_PATTERNS, *_RED_FLAG_PATTERNS):\n'
        '        m = pat.search(text)\n'
        '        if not m:\n'
        '            continue\n'
        '        key = pat.pattern\n'
        '        if key in seen:\n'
        '            continue\n'
        '        seen.add(key)\n'
        '        # ~80-char context window for human review.\n'
        '        start = max(0, m.start() - 40)\n'
        '        end = min(len(text), m.end() + 40)\n'
        '        snippet = re.sub(r"\\s+", " ", text[start:end]).strip()\n'
        '        hits.append(snippet)\n'
        '    return hits\n'
    )

    NEW = (
        '    hits: List[str] = []\n'
        '    seen: set = set()\n'
        '    flagged_spans: List[tuple] = []  # (start, end) of matched cores\n'
        '    for pat in (*_AFFIRMATIVE_SEVERE_PATTERNS, *_RED_FLAG_PATTERNS):\n'
        '        m = pat.search(text)\n'
        '        if not m:\n'
        '            continue\n'
        '        key = pat.pattern\n'
        '        if key in seen:\n'
        '            continue\n'
        '        # Positional dedup: if this match overlaps a span already\n'
        '        # flagged, it is the SAME underlying event picked up by a second\n'
        '        # pattern (e.g. "material weakness" and "controls not effective"\n'
        '        # in one sentence). Count the event once.\n'
        '        ms, me = m.start(), m.end()\n'
        '        if any(ms < fe and me > fs for fs, fe in flagged_spans):\n'
        '            continue\n'
        '        seen.add(key)\n'
        '        flagged_spans.append((ms, me))\n'
        '        # ~80-char context window for human review.\n'
        '        start = max(0, ms - 40)\n'
        '        end = min(len(text), me + 40)\n'
        '        snippet = re.sub(r"\\s+", " ", text[start:end]).strip()\n'
        '        hits.append(snippet)\n'
        '    return hits\n'
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:
            src = f.read()

        if "flagged_spans: List[tuple]" in src:
            print("Already patched — positional dedup present.")
            return 0

        count = src.count(OLD)
        if count == 0:
            print("ERROR: expected _find_red_flags loop not found. No changes made.")
            print("Paste the function and I'll match your on-disk version.")
            return 1
        if count > 1:
            print(f"ERROR: block found {count} times (expected 1). Aborting.")
            return 1

        patched = src.replace(OLD, NEW)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — _find_red_flags now dedups overlapping matches "
              "(one event = one flag).")
        return 0
    return main()


def _fix_refresh_button() -> object:
    """
    fix_refresh_button.py
    ====================

    Add a "Refresh data" button to the Jobs panel that runs the data refresh
    (daily macro + hourly derived signals) for all watchlist tickers as a
    background job.

    WHY
    ---
    The risk scanner needs fresh derived signals (realized_vol / drawdown / rsi)
    and macro (VIX / recession prob / yield curve) in the registry. Those are
    produced by RefreshScheduler.run_daily() (macro -> registry) and run_hourly()
    (derived signals recompute for every ticker). The wiring already exists; it
    just has to be RUN. Without a periodic refresh, a ticker's signals go stale and
    `gen` reports "no data" for market/macro risk (exactly the NVDA symptom we
    chased down). This button makes the refresh one click.

    WHAT IT DOES
    ------------
    * Adds a background-job worker `_refresh_job_worker(db_path)` that constructs a
      RefreshScheduler on data/hedgefund.db (the canonical registry DB) and runs
      run_daily(force=True) then run_hourly(force=True). Touches no Streamlit, so
      it's safe on the job queue.
    * Adds a "Refresh data" button in the Jobs panel header (next to "Clear
      finished") that submits that worker. Progress shows in the Jobs list like
      any opinion/report job.

    After it finishes, `gen <ticker>` for any watchlist name produces complete
    market + macro risk reasons.

    SAFETY
    ------
    * Targets ui/palette.py.
    * Backs up to ui/palette.py.bak before writing.
    * Idempotent: detects both markers; applies only what's missing.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_refresh_button.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "palette.py"

    # --- Worker function: inserted before render_job_tray ----------------------
    WORKER_ANCHOR = "def render_job_tray("
    WORKER_CODE = '''def _refresh_job_worker(db_path: str) -> dict:
        """Background-job worker: run the data refresh via RefreshScheduler.

        Runs the daily cadence (macro FRED -> registry) and the hourly cadence
        (derived-signals recompute for every watchlist ticker), both forced so they
        execute regardless of the last-run interval. Writes to the canonical
        registry DB (data/hedgefund.db) so gen/risk-scan pick the data up.
        Touches no Streamlit — safe for the job queue.
        """
        try:
            try:
                from data.refresh_scheduler import RefreshScheduler
            except Exception:
                from refresh_scheduler import RefreshScheduler  # type: ignore
            # Registry lives in hedgefund.db, not the portfolio DB.
            reg_db = "data/hedgefund.db"
            sched = RefreshScheduler(db_path=reg_db)
            daily = sched.run_daily(force=True)
            hourly = sched.run_hourly(force=True)

            def _ok(results):
                try:
                    return sum(1 for r in results
                               if getattr(r, "status", "ok") == "ok")
                except Exception:  # noqa: BLE001
                    return len(results) if results else 0

            return {
                "daily_tasks": len(daily or []),
                "daily_ok": _ok(daily or []),
                "hourly_tasks": len(hourly or []),
                "hourly_ok": _ok(hourly or []),
                "note": "macro + derived signals refreshed for all tickers",
            }
        except Exception as e:  # noqa: BLE001
            return {"note": f"error: {type(e).__name__}: {e}"}


    '''

    # --- Button: inserted after the Clear finished block -----------------------
    BUTTON_ANCHOR = '''    if head_r.button("Clear finished", key="tray_clear",
                         use_container_width=True):
            S.clear_finished_jobs()
            st.rerun()'''

    BUTTON_CODE = '''    if head_r.button("Clear finished", key="tray_clear",
                         use_container_width=True):
            S.clear_finished_jobs()
            st.rerun()

        # Refresh data — runs macro + derived-signals refresh for all tickers as a
        # background job, so gen produces complete market/macro risk reasons.
        if head_r.button("Refresh data", key="tray_refresh",
                         use_container_width=True):
            try:
                _db = "data/hedgefund.db"
                S.submit_job("refresh", "ALL", _refresh_job_worker, _db)
                st.toast("Data refresh started — see the Jobs list.")
            except Exception as _re:  # noqa: BLE001
                st.caption(f"Couldn't start refresh: {_re}")'''


    def _fail(msg: str) -> None:
        print(f"[fix_refresh_button] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")
        original = src

        # Insert the worker (once).
        if "_refresh_job_worker" not in src:
            if WORKER_ANCHOR not in src:
                _fail("could not find render_job_tray to anchor the worker.")
            src = src.replace(WORKER_ANCHOR, WORKER_CODE + WORKER_ANCHOR, 1)

        # Insert the button (once).
        if 'key="tray_refresh"' not in src:
            if BUTTON_ANCHOR not in src:
                _fail("could not find the Clear-finished block to anchor the "
                      "button.")
            src = src.replace(BUTTON_ANCHOR, BUTTON_CODE, 1)

        if src == original:
            print("[fix_refresh_button] Already applied. Nothing to do.")
            return

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_refresh_button] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added _refresh_job_worker + 'Refresh data' button to the Jobs")
        print("    panel.")
        print()
        print("Restart Streamlit, open Jobs, click 'Refresh data'. When it's done,")
        print("`gen <ticker>` for any watchlist name has fresh market + macro data.")
        print("(First run may take a bit — it fetches macro and recomputes signals")
        print("for every ticker.)")
    return main()


def _fix_regime_return() -> object:
    """
    fix_regime_return.py
    ====================

    Surgical fix for the Quant Snapshot showing "Regime: —".

    THE BUG
    -------
    In ``derived_signals.py``, the per-ticker compute method classifies ``regime``
    and writes it to the registry, but:

      1. ``regime`` is computed INSIDE the ``if self.registry:`` block, so if the
         registry is absent it is never assigned, and
      2. the method's return dict OMITS ``regime`` entirely.

    The Desk's Quant Snapshot reads ``context["derived_signals"]["regime"]``, which
    is populated from this return value (and the registry mapping). Because the
    return never includes it, the Snapshot always shows a dash.

    THE FIX (minimal, two changes)
    ------------------------------
      1. Move the ``regime = self._classify_regime(...)`` computation to BEFORE the
         ``if self.registry:`` block, initialized so it is always defined. The
         registry write still happens inside the block (guarded by ``if regime``).
      2. Add ``"regime": regime`` to the return dict.

    This is intentionally the smallest change that fixes the problem without
    altering any scoring logic, thresholds, or the registry write behavior.

    SAFETY
    ------
    * Backs up the original to ``derived_signals.py.bak`` before writing.
    * Idempotent: re-running detects the fix is already applied and does nothing.
    * Verifies the file still parses (ast.parse) before saving; aborts on any
      structural surprise so it never leaves a broken file.
    * Operates on the LIVE file in the project root, not a stale copy.

    Usage (from the project root, venv active):
        python fix_regime_return.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "derived_signals.py"


    def _fail(msg: str) -> None:
        print(f"[fix_regime_return] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run this from the project root "
                  "(D:\\Ary Fund) with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        # --- Idempotency check ------------------------------------------------
        if '"regime": regime' in src or "'regime': regime" in src:
            print("[fix_regime_return] Already applied — return dict includes "
                  "'regime'. Nothing to do.")
            return

        original = src

        # --- Change 1: ensure `regime` is computed before the registry block --
        # The current code has, inside `if self.registry:`:
        #
        #         # Regime
        #         regime = self._classify_regime(close.iloc[-1], sma50, sma200, dd)
        #         if regime:
        #             self.registry.upsert_point(
        #                 ticker, "ticker", "ticker.signal.regime",
        #                 ...
        #
        # We hoist the classification out so `regime` is always defined, and leave
        # the registry write where it is (it already guards on `if regime:`).
        #
        # Strategy: find the line that assigns regime via _classify_regime and the
        # `if self.registry:` line, and insert a hoisted assignment right before
        # the registry block, then neutralize the in-block assignment to reuse the
        # hoisted value (so we don't classify twice).

        hoist_marker = "regime = self._classify_regime("
        if hoist_marker not in src:
            _fail("could not locate the regime classification line "
                  "('regime = self._classify_regime(...'). The file may have "
                  "changed; not editing blindly.")

        # Locate the `if self.registry:` that precedes the regime write. We anchor
        # on the relative-strength comment that follows the block to bound it.
        if "if self.registry:" not in src:
            _fail("could not locate 'if self.registry:' block. Not editing.")

        # Extract the exact regime-classification statement (full line, preserving
        # its argument list) so the hoisted copy is identical.
        start = src.index(hoist_marker)
        line_start = src.rfind("\n", 0, start) + 1
        line_end = src.index("\n", start)
        regime_line = src[line_start:line_end]            # e.g. "            regime = self._classify_regime(close.iloc[-1], sma50, sma200, dd)"
        regime_expr = regime_line.strip()                 # "regime = self._classify_regime(...)"

        # Build the hoisted version at the indentation of the `if self.registry:`
        # line (8 spaces in the current file). We detect that indent dynamically.
        reg_if_idx = src.index("if self.registry:")
        reg_if_line_start = src.rfind("\n", 0, reg_if_idx) + 1
        reg_indent = src[reg_if_line_start:reg_if_idx]    # leading whitespace

        hoisted = f"{reg_indent}# Regime (hoisted so it is always defined for the return)\n" \
                  f"{reg_indent}{regime_expr}\n"

        # Insert the hoisted assignment immediately before the `if self.registry:`
        # line.
        src = src[:reg_if_line_start] + hoisted + src[reg_if_line_start:]

        # Now neutralize the ORIGINAL in-block classification so we don't run it
        # twice: replace the in-block "regime = self._classify_regime(...)" line
        # with a no-op comment (the hoisted value is already in scope).
        # Note: after the insert above, the original line still exists later in the
        # string; replace its first occurrence that is still the full assignment.
        src = src.replace(
            regime_line + "\n",
            f"{regime_line[:len(regime_line) - len(regime_line.lstrip())]}"
            f"# (regime computed above; value reused here)\n",
            1,
        )

        # --- Change 2: add "regime": regime to the return dict ----------------
        # Target the specific return block of this method, anchored on its known
        # keys to avoid touching any other return in the file.
        return_anchor = (
            '            "rsi_14": rsi14, "sma_50": sma50, "sma_200": sma200,\n'
            '            "realized_vol_30d": rv30, "drawdown_252d": dd,\n'
            '            "rs_pairs": len(rs_results),\n'
        )
        if return_anchor not in src:
            # Try a whitespace-tolerant fallback search.
            _fail("could not locate the exact return dict to extend. The file "
                  "formatting may differ; not editing blindly. (Change 1 not "
                  "saved.)")

        return_with_regime = return_anchor + '            "regime": regime,\n'
        src = src.replace(return_anchor, return_with_regime, 1)

        # --- Verify it still parses ------------------------------------------
        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving. Original "
                  "untouched.")

        # --- Back up + write --------------------------------------------------
        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_regime_return] SUCCESS")
        print(f"  • Backed up original to {backup.name}")
        print("  • Hoisted regime computation before the registry block")
        print("  • Added 'regime' to the compute return dict")
        print()
        print("Next: regenerate derived signals for the active ticker so the")
        print("registry/return carries regime. Re-run analysis (`gen <ticker>`")
        print("from the v2 command bar, or `python main.py`), then the Quant")
        print("Snapshot regime will populate instead of showing a dash.")
        if original == src:  # pragma: no cover - defensive
            print("\nWARNING: no net change detected — please report this.")
    return main()


def _fix_report_table_layout() -> object:
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
    return main()


def _fix_reranker_offline() -> object:
    """
    fix_reranker_offline.py — skip the reranker's ~15s HuggingFace startup check.

    Run from D:\\Ary Fund:   python fix_reranker_offline.py

    The reranker re-validates the cross-encoder against huggingface.co on every
    startup (a string of HTTP HEAD/GET requests, ~15s). The model is already
    cached locally, so this network round-trip is pure startup latency.

    FIX: set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE before constructing the
    CrossEncoder so it loads from the local cache without hitting the network.
    Includes a fallback: if the offline load fails (e.g. a fresh machine without
    the cached model), it retries once with offline mode disabled so the model
    can still download. So startup is fast on a warm cache and still works on a
    cold one.

    Surgical: replaces only the model-load block in the `model` property.
    Refuses to write unless the block matches exactly once, re-parses with ast
    before and after, writes UTF-8 without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\rag\reranker.py"

    OLD = (
        '        if self._model is None:\n'
        '            try:\n'
        '                from sentence_transformers import CrossEncoder\n'
        '                self._model = CrossEncoder(self.model_name)\n'
        '                logger.info("Reranker loaded | %s", self.model_name)\n'
        '            except ImportError as e:\n'
        '                raise RuntimeError(\n'
        '                    "Reranker requires sentence-transformers. "\n'
        '                    "Install with: pip install sentence-transformers"\n'
        '                ) from e\n'
    )

    NEW = (
        '        if self._model is None:\n'
        '            try:\n'
        '                import os\n'
        '                from sentence_transformers import CrossEncoder\n'
        '                # Load from local cache without re-validating against the\n'
        '                # HuggingFace Hub on every startup (~15s of HTTP round-\n'
        '                # trips). The model is already cached locally.\n'
        '                _prev_hub = os.environ.get("HF_HUB_OFFLINE")\n'
        '                _prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")\n'
        '                os.environ["HF_HUB_OFFLINE"] = "1"\n'
        '                os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
        '                try:\n'
        '                    self._model = CrossEncoder(self.model_name)\n'
        '                except Exception:  # noqa: BLE001\n'
        '                    # Cold cache (e.g. fresh machine): allow a networked\n'
        '                    # download by restoring the prior offline setting and\n'
        '                    # retrying once.\n'
        '                    if _prev_hub is None:\n'
        '                        os.environ.pop("HF_HUB_OFFLINE", None)\n'
        '                    else:\n'
        '                        os.environ["HF_HUB_OFFLINE"] = _prev_hub\n'
        '                    if _prev_tf is None:\n'
        '                        os.environ.pop("TRANSFORMERS_OFFLINE", None)\n'
        '                    else:\n'
        '                        os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf\n'
        '                    self._model = CrossEncoder(self.model_name)\n'
        '                logger.info("Reranker loaded | %s", self.model_name)\n'
        '            except ImportError as e:\n'
        '                raise RuntimeError(\n'
        '                    "Reranker requires sentence-transformers. "\n'
        '                    "Install with: pip install sentence-transformers"\n'
        '                ) from e\n'
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:
            src = f.read()

        count = src.count(OLD)
        if count == 0:
            print("ERROR: expected reranker load block not found. No changes made.")
            if 'os.environ["HF_HUB_OFFLINE"] = "1"' in src:
                print("NOTE: looks like it's already patched.")
            return 1
        if count > 1:
            print(f"ERROR: block found {count} times (expected 1). Aborting.")
            return 1

        patched = src.replace(OLD, NEW)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — reranker now loads offline (warm cache) with a "
              "network fallback for a cold cache.")
        return 0
    return main()


def _fix_review_timeout() -> object:
    """
    fix_review_timeout.py
    =====================

    Fix for the thesis-review step always falling back to the deterministic
    writer with:

        agent.thesis_review | NVDA | LLM failed: timed out -> fallback

    ROOT CAUSE
    ----------
    The essay and review steps have mismatched time budgets for similar output:

        * thesis_essay._call_ollama_text   -> timeout floor 360s   (SUCCEEDS)
        * thesis_review._call_ollama_text  -> timeout floor 180s   (TIMES OUT)

    …yet the review is instructed to write AT LEAST 1200 words
    (``_REVIEW_MIN_WORDS = 1200``) with ``num_predict`` = MAX_TOKENS (4096). So the
    review is asked to generate as much or more text than the essay, in HALF the
    time, on an 8 GB RTX 2080. It reliably exceeds 180s and falls back — which is
    why every memo shows the amber "deterministic fallback" banner even though
    Ollama and llama3.1:8b are perfectly healthy.

    THE FIX
    -------
    Raise the review's timeout floor from 180.0 to 360.0, matching the essay step
    (the proven-working reference for ~1200-word generations on this hardware).
    This is the minimal change: it does not touch token budgets, prompts, models,
    or the word-count target — it just gives the review the same wall-clock budget
    the essay already gets.

    This single edit covers BOTH the review and the revision call, because both go
    through the same ``_call_ollama_text`` helper.

    SAFETY
    ------
    * Targets agent/thesis_review.py.
    * Backs up to agent/thesis_review.py.bak before writing.
    * Idempotent: detects the 360.0 floor and does nothing on re-run.
    * Verifies ast.parse before saving; aborts on any surprise.

    Usage (from project root, venv active):
        python fix_review_timeout.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("agent") / "thesis_review.py"

    OLD = 'timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 180.0)'
    NEW = ('timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 360.0)  '
           '# raised 180->360 to match the essay step; review writes 1200+ words')


    def _fail(msg: str) -> None:
        print(f"[fix_review_timeout] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        # Idempotency.
        if "360.0" in src and "180.0" not in src.split("_call_ollama_text", 1)[-1][:400]:
            # Crude but safe: if the review helper region already mentions 360.0
            # and no longer has the 180.0 floor, treat as applied.
            if NEW.split("#")[0].strip() in src:
                print("[fix_review_timeout] Already applied — review timeout floor "
                      "is 360.0. Nothing to do.")
                return

        if OLD not in src:
            # Maybe already patched, or formatting differs.
            if 'getattr(config, "AGENT_TIMEOUT", 30)), 360.0' in src:
                print("[fix_review_timeout] Already applied (360.0 floor present).")
                return
            _fail("could not find the review timeout line "
                  "('...AGENT_TIMEOUT...180.0'). The file may have changed; not "
                  "editing blindly.")

        src = src.replace(OLD, NEW, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_review_timeout] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Raised review/revision timeout floor 180s -> 360s")
        print()
        print("Re-run `gen NVDA` in the app. The review step should now complete")
        print("with the real LLM, and the amber 'deterministic fallback' banner")
        print("should disappear (model line: llama3.1:8b, no '(failed)').")
        print()
        print("If it STILL times out at 360s, the review generation is genuinely")
        print("too slow on this GPU — next options are lowering num_predict for the")
        print("review call or trimming _REVIEW_MIN_WORDS (1200). Tell me and I'll")
        print("patch that instead.")
    return main()


def _fix_risk_count() -> object:
    """
    fix_risk_count.py — reconcile the peer-path risk count with the agent path.

    Run from D:\\Ary Fund:   python fix_risk_count.py

    The peer path (_risk_count_for in data/pipeline.py) counts risk-cue sentences
    over UP TO 3 10-Ks + 3 10-Qs. The agent path counts over 1 10-K + 2 10-Qs.
    That mismatch is why MSFT contributes ~202 to its sector distribution but is
    scored as 119. This script changes ONLY the peer path's filing selection to
    1x 10-K + 2x 10-Q (no 8-Ks), so a ticker's contributed count matches the
    basis it is scored on.

    Surgical: it replaces exactly one block inside _risk_count_for and leaves the
    rest of pipeline.py byte-for-byte unchanged. It refuses to write if the
    expected block isn't found (so it can't silently corrupt the file), and it
    re-parses the result with ast before saving, restoring the original on any
    syntax error. Writes UTF-8 without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\data\pipeline.py"

    OLD = (
        '        filings = []\n'
        '        for kind in ("10-K", "10-Q"):\n'
        '            try:\n'
        '                cached = _sec._get_cached_filings(tk, kind, 3, None, None)\n'
        '            except Exception:  # noqa: BLE001\n'
        '                cached = []\n'
    )

    NEW = (
        '        filings = []\n'
        '        # Match the AGENT path\'s annual/quarterly basis (1x 10-K, 2x 10-Q)\n'
        '        # so a ticker\'s CONTRIBUTED count == the count it is SCORED as.\n'
        '        # 8-Ks are excluded: they rarely carry Item-1A risk language and\n'
        '        # their date-relative selection would make the distribution drift.\n'
        '        for kind, _n in (("10-K", 1), ("10-Q", 2)):\n'
        '            try:\n'
        '                cached = _sec._get_cached_filings(tk, kind, _n, None, None)\n'
        '            except Exception:  # noqa: BLE001\n'
        '                cached = []\n'
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:  # tolerate any stray BOM
            src = f.read()

        count = src.count(OLD)
        if count == 0:
            print("ERROR: expected block not found. The file may already be "
                  "patched or differs from what was expected. No changes made.")
            # Help diagnose: show whether the new form is already present.
            if 'for kind, _n in (("10-K", 1), ("10-Q", 2)):' in src:
                print("NOTE: the file already contains the reconciled selection — "
                      "looks like it's already patched.")
            return 1
        if count > 1:
            print(f"ERROR: expected block found {count} times (expected exactly 1). "
                  "Aborting to avoid an ambiguous edit.")
            return 1

        patched = src.replace(OLD, NEW)

        # Verify the patched source parses before writing.
        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        # Write UTF-8 WITHOUT BOM.
        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        # Re-read and re-verify what actually landed on disk.
        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — _risk_count_for now uses 1x 10-K + 2x 10-Q (no 8-Ks).")
        print("File parses clean, written without BOM.")
        return 0
    return main()


def _fix_risk_reasons_allclear() -> object:
    """
    fix_risk_reasons_allclear.py
    ==========================

    Stop the risk scanner from labeling an evaluated "all clear" as "no data".

    THE PROBLEM
    -----------
    compute_risk_flags builds per-category reasons like:
        "macro": macro_reasons or ["no data"]
    _score_macro returns an EMPTY reasons list ([]) in two very different cases:
      (a) it HAD the macro inputs (VIX, recession prob, yield curve) and found
          none of them elevated  -> genuine "all clear, LOW risk"
      (b) it had NO inputs at all -> genuine data gap
    The `or ["no data"]` collapses both into ["no data"], so a calm macro
    environment (VIX 16, recession prob 0.4%, curve normal) is reported
    identically to a broken data pipeline. This is exactly the ambiguity that made
    "Macro: LOW — no specific reasons recorded" impossible to interpret.

    THE FIX
    -------
    Distinguish the two by checking whether the scorer actually had usable inputs:
      * macro  had inputs if the macro dict has any of vix / recession_probability
        / yield_curve_spread / yield_curve_inverted
      * market had inputs if metrics has any of realized_vol / volatility /
        drawdown / max_drawdown / rsi
      * fundamental had inputs if metrics is non-empty
      * agent had inputs if the agent risk list was non-empty
    When inputs were present but no reason fired -> emit a single informative
    "within normal ranges" note instead of "no data". When inputs were truly
    absent -> keep "no data".

    This changes only the REASONS text, never the risk LEVELS.

    SAFETY
    ------
    * Targets agent/risk_scanner.py.
    * Backs up to agent/risk_scanner.py.bak before writing.
    * Idempotent: detects the marker and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_risk_reasons_allclear.py

    After applying, regenerate: `gen NVDA` then `report NVDA`.
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("agent") / "risk_scanner.py"

    ANCHOR = '''    reasons: Dict[str, List[str]] = {
            "fundamental": fundamental_reasons or ["no data"],
            "macro": macro_reasons or ["no data"],
            "market": market_reasons or ["no data"],
            "agent": agent_reasons or ["no data"],
        }'''

    INSERT = '''    # Distinguish "evaluated, nothing elevated" (all clear) from a genuine
        # data gap. Empty reasons + inputs present -> all-clear note; empty reasons
        # + no inputs -> "no data". Levels are unaffected.
        def _had(d, keys):
            try:
                return any(d.get(k) is not None for k in keys)
            except Exception:  # noqa: BLE001
                return False

        _macro_had = _had(mc, ("vix", "recession_probability",
                               "yield_curve_spread", "yield_curve_inverted"))
        _market_had = _had(m, ("realized_vol", "volatility", "drawdown",
                               "max_drawdown", "rsi"))
        _fund_had = bool(m)
        _agent_had = bool(ar)

        def _reasons_for(scored, had, clear_msg):
            if scored:
                return scored
            return [clear_msg] if had else ["no data"]

        reasons: Dict[str, List[str]] = {
            "fundamental": _reasons_for(
                fundamental_reasons, _fund_had,
                "fundamentals within normal ranges vs peers"),
            "macro": _reasons_for(
                macro_reasons, _macro_had,
                "macro indicators within normal ranges (VIX, recession odds, "
                "yield curve)"),
            "market": _reasons_for(
                market_reasons, _market_had,
                "price/volatility metrics within normal ranges"),
            "agent": _reasons_for(
                agent_reasons, _agent_had,
                "no additional risks flagged by the agent"),
        }'''


    def _fail(msg: str) -> None:
        print(f"[fix_risk_reasons_allclear] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_reasons_for(" in src:
            print("[fix_risk_reasons_allclear] Already applied — all-clear "
                  "labeling present. Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the reasons-assembly block. risk_scanner.py may "
                  "have changed. Not editing blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_risk_reasons_allclear] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Evaluated-but-clear categories now show a 'within normal")
        print("    ranges' note instead of 'no data'. Genuine gaps still say")
        print("    'no data'. Risk levels unchanged.")
        print()
        print("Regenerate to see it:")
        print("    gen NVDA")
        print("    report NVDA")
        print()
        print("Expected: Macro: LOW with 'macro indicators within normal ranges")
        print("(VIX, recession odds, yield curve)' instead of a blank.")
    return main()


def _fix_scheduler_bootstrap() -> object:
    """
    fix_scheduler_bootstrap.py
    =========================

    Fix the scheduler's "registry load failed: No module named 'data'" warning.

    CAUSE
    -----
    refresh_scheduler.py does ``from data.data_registry import ...`` (and
    data.market_data, data.macro_data, ...). Those resolve only if the project
    ROOT is on sys.path. When you run it as a script —
    ``python data/refresh_scheduler.py`` — the script's own directory (data/) is
    on sys.path, not the root, so ``import data.*`` fails and the registry silently
    falls back to a degraded mode. (The module form ``python -m data.refresh_
    scheduler`` would work, but the script form is what you ran.)

    FIX
    ---
    Insert the same project-root bootstrap screener_data.py uses: compute the root
    as this file's parent's parent, chdir there, and put it on sys.path. Then both
    the script form and the module form work, and the registry loads.

    SAFETY
    ------
    * Targets data/refresh_scheduler.py.
    * Backs up to data/refresh_scheduler.py.bak before writing.
    * Idempotent: detects the bootstrap marker and does nothing on re-run.
    * Verifies ast.parse before saving.
    * Inserts right after the HF offline-flags block (added earlier), before the
      other imports — so flags stay first and imports resolve.

    Usage (from project root, venv active):
        python fix_scheduler_bootstrap.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "refresh_scheduler.py"

    ANCHOR = '''import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")'''

    INSERT = ANCHOR + '''

    # --- Project-root bootstrap (so `from data.*` works when run as a script) ---
    # Running this file as `python data/refresh_scheduler.py` puts data/ on
    # sys.path, not the project root, so `import data.data_registry` fails and the
    # registry falls back to a degraded mode. Compute the root (this file's
    # parent's parent), chdir there, and add it to sys.path so both the script
    # form and the module form (`python -m data.refresh_scheduler`) resolve.
    import sys as _sys
    from pathlib import Path as _Path
    _SCHED_ROOT = _Path(__file__).resolve().parent.parent
    try:
        _os.chdir(_SCHED_ROOT)
    except Exception:
        pass
    if str(_SCHED_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_SCHED_ROOT))'''


    def _fail(msg: str) -> None:
        print(f"[fix_scheduler_bootstrap] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_SCHED_ROOT = _Path(__file__)" in src:
            print("[fix_scheduler_bootstrap] Already applied — bootstrap present. "
                  "Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the HF offline-flags block to anchor the "
                  "bootstrap. Did fix_scheduler_offline.py run? Not editing "
                  "blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_scheduler_bootstrap] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added project-root chdir + sys.path bootstrap")
        print()
        print("Re-run; the 'No module named data' warning should be gone:")
        print("    python data/refresh_scheduler.py status")
    return main()


def _fix_scheduler_offline() -> object:
    """
    fix_scheduler_offline.py
    =======================

    Propagate the HuggingFace offline flags to the refresh_scheduler entry point.

    CONTEXT
    -------
    ``main.py`` sets these at the very top so cached HF models (the reranker
    cross-encoder and the sentence-transformers embedder) load without
    re-validating against huggingface.co on every startup:

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    But ``refresh_scheduler.py`` is its OWN entry point (invoked from cron /
    APScheduler / a Streamlit button), and it never set these. So when the
    scheduler's daily cadence runs the RAG index / reranker, HuggingFace tries to
    phone home and re-validate — adding latency and failing outright if the box is
    offline.

    THE FIX
    -------
    Insert the same ``os.environ.setdefault`` block into refresh_scheduler.py,
    immediately AFTER ``from __future__ import annotations`` (which must remain the
    first statement) and BEFORE any other import — because the HF libraries read
    these env vars at import time, so the flags must be set before anything pulls
    them in.

    SAFETY
    ------
    * Targets refresh_scheduler.py.
    * Backs up to refresh_scheduler.py.bak before writing.
    * Idempotent: detects HF_HUB_OFFLINE and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_scheduler_offline.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "refresh_scheduler.py"

    ANCHOR = "from __future__ import annotations"

    INSERT = '''
    # Load cached HuggingFace models (reranker cross-encoder, embedder) without
    # re-validating against huggingface.co on every run. Must be set before any
    # import pulls in a huggingface library, so it lives here at the top of this
    # entry point (the scheduler is invoked independently of main.py, which sets
    # the same flags). See fix_scheduler_offline.py.
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")'''


    def _fail(msg: str) -> None:
        print(f"[fix_scheduler_offline] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "HF_HUB_OFFLINE" in src:
            print("[fix_scheduler_offline] Already applied — HF_HUB_OFFLINE present. "
                  "Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find 'from __future__ import annotations' to anchor "
                  "the insert. Not editing blindly.")

        # Insert right after the __future__ line (keeping __future__ first).
        src = src.replace(ANCHOR, ANCHOR + INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_scheduler_offline] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Inserted HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE at the top of")
        print("    refresh_scheduler.py (after __future__, before other imports)")
        print()
        print("The scheduler's RAG/reranker steps will no longer re-validate against")
        print("huggingface.co on each run.")
    return main()


def _fix_screener_data_xbrl() -> object:
    """
    fix_screener_data_xbrl.py
    ========================

    Add an ``xbrl`` subcommand to data/screener_data.py that ingests SEC XBRL
    companyfacts for the whole universe into the ``xbrl_facts`` table, so the
    screener can later read real Operating Income / Total Assets / CapEx from
    filings.

    WHAT IT ADDS
    ------------
    * ``_import_sec_fetcher()`` — tolerant import (data.sec_fetcher | sec_fetcher).
    * ``cmd_xbrl(symbols, *, sleep, limit)`` — loops the universe and calls the
      existing ``SECFetcher.ingest_xbrl_facts(ticker)`` for each, with progress and
      SEC-friendly throttling. (ingest_xbrl_facts already fetches companyfacts,
      writes xbrl_facts + the registry, and derives ratios — we just drive it over
      the universe.)
    * Wires ``xbrl`` into the argparse ``choices`` and the dispatch in ``main``.

    USAGE AFTER PATCH
    -----------------
        python data/screener_data.py xbrl

    HONEST COST NOTE
    ----------------
    This hits SEC ~once per symbol, and each companyfacts JSON is several MB, so a
    full-universe run takes a while (≈10-20 min) and downloads a few GB. XBRL only
    changes on new filings, so this is a once-per-quarter job, not daily. SEC asks
    for a descriptive User-Agent and ≤10 req/s — SECFetcher already throttles; the
    default --sleep here adds a little extra headroom. Be polite to SEC's servers.

    SAFETY
    ------
    * Targets data/screener_data.py.
    * Backs up to data/screener_data.py.bak before writing.
    * Idempotent: detects cmd_xbrl and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_data_xbrl.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "screener_data.py"

    # --- 1. Add _import_sec_fetcher after _import_market_data ----------------
    IMPORT_ANCHOR = '''def _import_market_data():
        for path in ("data.market_data", "market_data"):
            try:
                mod = __import__(path, fromlist=["MarketData"])
                return mod.MarketData
            except Exception:
                continue
        print("ERROR: could not import MarketData (tried data.market_data and "
              "market_data).")
        sys.exit(1)'''

    IMPORT_INSERT = IMPORT_ANCHOR + '''


    def _import_sec_fetcher():
        for path in ("data.sec_fetcher", "sec_fetcher"):
            try:
                mod = __import__(path, fromlist=["SECFetcher"])
                return mod.SECFetcher
            except Exception:
                continue
        print("ERROR: could not import SECFetcher (tried data.sec_fetcher and "
              "sec_fetcher).")
        sys.exit(1)


    def cmd_xbrl(symbols: list[str], *, sleep: float,
                 limit: "Optional[int]" = None) -> None:
        """Ingest SEC XBRL companyfacts for each symbol into the xbrl_facts table.

        Drives the existing SECFetcher.ingest_xbrl_facts over the universe. Each
        call fetches the company's companyfacts JSON (several MB), writes every
        mapped concept (revenue, net income, total assets, capex, operating
        income, ...) to xbrl_facts + the registry, and derives ratios.
        """
        SECFetcher = _import_sec_fetcher()
        sec = SECFetcher()

        if limit is not None:
            symbols = symbols[:limit]
        total = len(symbols)

        ingested = 0
        no_facts: list[str] = []
        failed: list[str] = []
        t0 = time.time()
        print(f"[xbrl] Ingesting SEC XBRL companyfacts for {total} symbols into "
              f"the xbrl_facts table…")
        print("[xbrl] Each companyfacts JSON is several MB; this is a once-per-"
              "quarter job. Be patient and kind to SEC's servers.")
        print()

        for i, sym in enumerate(symbols, 1):
            try:
                n = sec.ingest_xbrl_facts(sym)
                if n > 0:
                    ingested += 1
                else:
                    no_facts.append(sym)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{sym} ({type(e).__name__})")

            if i % 10 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else 0.0
                print(f"[xbrl] {i}/{total}  ingested={ingested}  "
                      f"no_facts={len(no_facts)}  failed={len(failed)}  "
                      f"~{eta:5.0f}s left  (last: {sym})")

            if sleep:
                time.sleep(sleep)

        dt = time.time() - t0
        print()
        print(f"[xbrl] Done. Ingested facts for {ingested}/{total} in {dt:.0f}s.")
        if no_facts:
            print(f"[xbrl] {len(no_facts)} symbol(s) had no us-gaap facts "
                  f"(e.g. foreign filers, ETFs): {', '.join(no_facts[:20])}"
                  + (" ..." if len(no_facts) > 20 else ""))
        if failed:
            print(f"[xbrl] {len(failed)} symbol(s) errored: {', '.join(failed[:20])}"
                  + (" ..." if len(failed) > 20 else ""))
        print()
        print("Now restart the screener — Op Income / Total Assets / CapEx should")
        print("fill from XBRL for names that report those concepts. (Some "
              "financials don't file a GAAP operating-income line, so those stay "
              "blank — that's a filing reality, not a bug.)")'''

    # --- 2. Add 'xbrl' to the argparse choices -------------------------------
    CHOICES_ANCHOR = '''    parser.add_argument("command", choices=["check", "warm", "all"],
                            help="check = report delisted; warm = fill cache; "
                                 "all = check then warm the survivors")'''

    CHOICES_INSERT = '''    parser.add_argument("command", choices=["check", "warm", "all", "xbrl"],
                            help="check = report delisted; warm = fill cache; "
                                 "all = check then warm the survivors; "
                                 "xbrl = ingest SEC XBRL facts (Op Income/Total "
                                 "Assets/CapEx) for the universe")'''

    # --- 3. Add the dispatch branch ------------------------------------------
    DISPATCH_ANCHOR = '''    elif args.command == "all":
            good = cmd_check(symbols, sleep=args.sleep)
            print(f"[all] Warming the {len(good)} symbols that passed check…\\n")
            cmd_warm(good, sleep=args.sleep, limit=args.limit)'''

    DISPATCH_INSERT = '''    elif args.command == "all":
            good = cmd_check(symbols, sleep=args.sleep)
            print(f"[all] Warming the {len(good)} symbols that passed check…\\n")
            cmd_warm(good, sleep=args.sleep, limit=args.limit)
        elif args.command == "xbrl":
            cmd_xbrl(symbols, sleep=args.sleep, limit=args.limit)'''


    def _fail(msg: str) -> None:
        print(f"[fix_screener_data_xbrl] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "def cmd_xbrl(" in src:
            print("[fix_screener_data_xbrl] Already applied — cmd_xbrl present. "
                  "Nothing to do.")
            return

        for anchor, name in (
            (IMPORT_ANCHOR, "_import_market_data (for cmd_xbrl insert)"),
            (CHOICES_ANCHOR, "argparse choices line"),
            (DISPATCH_ANCHOR, "dispatch 'all' branch"),
        ):
            if anchor not in src:
                _fail(f"could not find the {name} anchor. The file may have "
                      "changed. Not editing blindly.")

        src = src.replace(IMPORT_ANCHOR, IMPORT_INSERT, 1)
        src = src.replace(CHOICES_ANCHOR, CHOICES_INSERT, 1)
        src = src.replace(DISPATCH_ANCHOR, DISPATCH_INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_data_xbrl] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added _import_sec_fetcher() + cmd_xbrl()")
        print("  • Wired 'xbrl' into the command choices and dispatch")
        print()
        print("Run it (full universe — takes a while, downloads a few GB):")
        print("    python data/screener_data.py xbrl")
    return main()


def _fix_screener_derived() -> object:
    """
    fix_screener_derived.py
    ======================

    Fill the screener's DERIVED-RATIO and PER-SHARE columns (#8): FCF Margin,
    FCF Yield, ROIC, Cash/Share, Book/Share, Sales/Share, FCF/Share, Div/Share.

    These columns already exist in the screener's schema but were never populated
    (they showed "None"), because ``_fetch_fundamentals_one`` only mapped the raw
    yfinance fields, not the ratios/per-share values derived from them.

    WHAT THIS DOES
    --------------
    Rewrites the return dict of ``_fetch_fundamentals_one`` to ALSO compute, from
    fields it already fetches (no extra network/data calls):

        fcf_margin   = free_cash_flow / revenue        (%)
        fcf_yield    = free_cash_flow / market_cap      (%)
        roic (proxy) = net_income / (market_cap + total_debt)  (%)   [approximation]
        cash_per_share  = total_cash / shares_outstanding
        book_per_share  = price_proxy / price_to_book   (price ÷ P/B)
        sales_per_share = revenue / shares_outstanding
        fcf_per_share   = free_cash_flow / shares_outstanding
        div_per_share   = div_yield% * price_proxy / 100

    where price_proxy = market_cap / shares_outstanding.

    ROIC is explicitly a PROXY (uses market cap as an equity stand-in because book
    equity isn't in the fundamentals payload) — good for relative screening, not a
    substitute for a from-filings ROIC.

    Because these are pure arithmetic on already-fetched data, this adds ZERO
    extra data calls and no render-time cost.

    This does NOT touch _FUNDAMENTALS_LAZY_LIMIT or the skip-check — it only
    rewrites the _fetch_fundamentals_one return block, so prior fixes are
    preserved.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects the new fcf_margin computation and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_derived.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    # The current return block of _fetch_fundamentals_one (must match exactly).
    OLD = '''    return {
            "symbol":           symbol,
            "name":             f.get("name") or symbol,
            "sector":           f.get("sector") or "—",
            # Overview
            "market_cap":       _num(ov.get("market_cap")),
            "beta":             _num(ov.get("beta")),
            # Valuation
            "pe":               _num(val.get("trailing_pe")),
            "forward_pe":       _num(val.get("forward_pe")),
            "peg":              _num(val.get("peg_ratio")),
            "ps":               _num(val.get("price_to_sales")),
            "pb":               _num(val.get("price_to_book")),
            "ev_ebitda":        _num(val.get("ev_to_ebitda")),
            # Financials
            "revenue":          _num(fin.get("revenue")),
            "gross_profit":     _num(fin.get("gross_profit")),
            "ebitda":           _num(fin.get("ebitda")),
            "net_income":       _num(fin.get("net_income")),
            "fcf":              _num(fin.get("free_cash_flow")),
            "op_cash_flow":     _num(fin.get("operating_cash_flow")),
            "total_debt":       _num(fin.get("total_debt")),
            # yfinance returns debtToEquity as a percent already (e.g. 60.5),
            # not a decimal — keep as-is for display.
            "debt_to_equity":   _num(fin.get("debt_to_equity")),
            "current_ratio":    _num(fin.get("current_ratio")),
            "roe":              _pct(fin.get("return_on_equity")),
            "roa":              _pct(fin.get("return_on_assets")),
            "profit_margin":    _pct(fin.get("profit_margin")),
            "gross_margin":     _pct(fin.get("gross_margin")),
            "op_margin":        _pct(fin.get("operating_margin")),
            # Growth
            "revenue_growth":   _pct(gr.get("revenue_growth")),
            "eps_dil_growth":   _pct(gr.get("earnings_growth")),
            # Dividends
            # yfinance: dividendYield is already a percent in newer versions
            # (e.g. 0.74 = 0.74%), but historically was a decimal (0.0074).
            # We probe and normalize: a value < 1 is treated as decimal.
            "div_yield":        _div_yield_normalize(div.get("dividend_yield")),
            "div_payout":       _pct(div.get("payout_ratio")),
            "ex_div_date":      str(div.get("ex_dividend_date") or "—"),
            # Analyst
            "analyst_rating":   _normalize_recommendation(an.get("recommendation")),
        }'''

    NEW = '''    # --- Derived ratios + per-share (computed from already-fetched fields) ---
        _shares_out = _num(ov.get("shares_outstanding"))
        _revenue = _num(fin.get("revenue"))
        _fcf = _num(fin.get("free_cash_flow"))
        _net_income = _num(fin.get("net_income"))
        _total_debt = _num(fin.get("total_debt"))
        _mktcap = _num(ov.get("market_cap"))
        _pb = _num(val.get("price_to_book"))

        def _safe_div(a: Any, b: Any) -> float:
            try:
                if a is None or b is None or pd.isna(a) or pd.isna(b) or float(b) == 0.0:
                    return float("nan")
                return float(a) / float(b)
            except (TypeError, ValueError):
                return float("nan")

        _fcf_margin = _safe_div(_fcf, _revenue) * 100.0
        _fcf_yield = _safe_div(_fcf, _mktcap) * 100.0
        _invested = (_mktcap + _total_debt
                     if pd.notna(_mktcap) and pd.notna(_total_debt) else float("nan"))
        _roic = _safe_div(_net_income, _invested) * 100.0  # proxy (mkt cap as equity)
        _price_proxy = _safe_div(_mktcap, _shares_out)
        _cash_ps = _safe_div(_num(fin.get("total_cash")), _shares_out)
        _book_ps = _safe_div(_price_proxy, _pb)            # price ÷ P/B
        _sales_ps = _safe_div(_revenue, _shares_out)
        _fcf_ps = _safe_div(_fcf, _shares_out)
        _dy = _div_yield_normalize(div.get("dividend_yield"))
        _div_ps = (_dy / 100.0 * _price_proxy
                   if pd.notna(_dy) and pd.notna(_price_proxy) else float("nan"))

        return {
            "symbol":           symbol,
            "name":             f.get("name") or symbol,
            "sector":           f.get("sector") or "—",
            # Overview
            "market_cap":       _num(ov.get("market_cap")),
            "beta":             _num(ov.get("beta")),
            # Valuation
            "pe":               _num(val.get("trailing_pe")),
            "forward_pe":       _num(val.get("forward_pe")),
            "peg":              _num(val.get("peg_ratio")),
            "ps":               _num(val.get("price_to_sales")),
            "pb":               _num(val.get("price_to_book")),
            "ev_ebitda":        _num(val.get("ev_to_ebitda")),
            # Financials
            "revenue":          _num(fin.get("revenue")),
            "gross_profit":     _num(fin.get("gross_profit")),
            "ebitda":           _num(fin.get("ebitda")),
            "net_income":       _num(fin.get("net_income")),
            "fcf":              _num(fin.get("free_cash_flow")),
            "op_cash_flow":     _num(fin.get("operating_cash_flow")),
            "total_debt":       _num(fin.get("total_debt")),
            # yfinance returns debtToEquity as a percent already (e.g. 60.5),
            # not a decimal — keep as-is for display.
            "debt_to_equity":   _num(fin.get("debt_to_equity")),
            "current_ratio":    _num(fin.get("current_ratio")),
            "roe":              _pct(fin.get("return_on_equity")),
            "roa":              _pct(fin.get("return_on_assets")),
            "profit_margin":    _pct(fin.get("profit_margin")),
            "gross_margin":     _pct(fin.get("gross_margin")),
            "op_margin":        _pct(fin.get("operating_margin")),
            # Derived ratios (computed above)
            "fcf_margin":       _fcf_margin,
            "fcf_yield":        _fcf_yield,
            "roic":             _roic,
            # Per-share (computed above)
            "cash_per_share":   _cash_ps,
            "book_per_share":   _book_ps,
            "sales_per_share":  _sales_ps,
            "fcf_per_share":    _fcf_ps,
            "div_per_share":    _div_ps,
            # Growth
            "revenue_growth":   _pct(gr.get("revenue_growth")),
            "eps_dil_growth":   _pct(gr.get("earnings_growth")),
            # Dividends
            # yfinance: dividendYield is already a percent in newer versions
            # (e.g. 0.74 = 0.74%), but historically was a decimal (0.0074).
            # We probe and normalize: a value < 1 is treated as decimal.
            "div_yield":        _div_yield_normalize(div.get("dividend_yield")),
            "div_payout":       _pct(div.get("payout_ratio")),
            "ex_div_date":      str(div.get("ex_dividend_date") or "—"),
            # Analyst
            "analyst_rating":   _normalize_recommendation(an.get("recommendation")),
        }'''


    def _fail(msg: str) -> None:
        print(f"[fix_screener_derived] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if '_fcf_margin = _safe_div(_fcf, _revenue)' in src:
            print("[fix_screener_derived] Already applied — derived ratios present. "
                  "Nothing to do.")
            return

        if OLD not in src:
            _fail("could not find the exact _fetch_fundamentals_one return block. "
                  "The file may differ (whitespace/quotes/prior edits). Not editing "
                  "blindly.")

        src = src.replace(OLD, NEW, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_derived] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added derived ratios (FCF margin, FCF yield, ROIC proxy) and")
        print("    per-share values (cash/book/sales/fcf/div per share) to")
        print("    _fetch_fundamentals_one — computed from already-fetched fields.")
        print()
        print("Fully restart Streamlit (clears the _build_screener_frame cache),")
        print("then check Profitability (FCF Margin, ROIC) and Per share. Those")
        print("columns should now fill instead of None.")
        print()
        print("NOTE: ROIC is a PROXY (net income / (market cap + debt)) because")
        print("book equity isn't in the fundamentals payload. Good for relative")
        print("screening; not a from-filings ROIC.")
    return main()


def _fix_screener_limit() -> object:
    """
    fix_screener_limit.py
    =====================

    Raise the screener's fundamentals lazy-load cap so the warm-cached universe
    fills ALL category columns (Valuation, Profitability, Income statement,
    Balance sheet, Cash flow, Per share, Technicals), not just the top 60 by
    market cap.

    CONTEXT
    -------
    ``ui/screener.py`` reads fundamentals via
    ``MarketData.get_fundamentals(symbol, use_cache=True)`` — i.e. it reads the
    same 24h SQLite cache in data/hedgefund.db that ``data/screener_data.py warm``
    fills. But it only populates fundamentals for the top
    ``_FUNDAMENTALS_LAZY_LIMIT`` (= 60) symbols per render, because the cap was
    sized for COLD caches where each read is a live yfinance round-trip.

    Now that ``warm`` pre-fills the whole universe, those reads are near-instant
    cache hits, so the 60 cap is needlessly conservative — it's why every deep
    category tab shows ``None`` below the top 60.

    THE FIX
    -------
    Raise ``_FUNDAMENTALS_LAZY_LIMIT`` from 60 to 600 (covers the ~560-name
    universe with margin) and update the rationale comment to reflect the
    warm-cache workflow.

    TRADE-OFF (stated honestly)
    ---------------------------
    At 600, names whose cache entry has EXPIRED (>24h since the last ``warm``)
    will trigger live yfinance fetches on render and can slow the first render or
    briefly show ``None`` until re-warmed. Keep the cache fresh by re-running
    ``python data/screener_data.py warm`` daily (or wiring it into
    refresh_scheduler). With a fresh warm, the screener fills fully and fast.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects the 600 value and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_limit.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    OLD_BLOCK = """# Cap on lazy fundamentals fetches per render. Each call is a yfinance
    # round trip; even with 24h SQLite caching, the *first* render after a
    # clean start can hit dozens of network calls for the visible page.
    # 60 covers the default Streamlit dataframe height comfortably without
    # stalling the UI on cold caches.
    _FUNDAMENTALS_LAZY_LIMIT = 60"""

    NEW_BLOCK = """# Cap on lazy fundamentals reads per render. Each read goes through
    # MarketData.get_fundamentals(use_cache=True), which hits the 24h SQLite
    # cache in data/hedgefund.db. With the cache pre-warmed for the whole
    # universe (`python data/screener_data.py warm`), these are near-instant
    # cache hits, so the cap can cover the full ~560-name universe — this is
    # what fills the deep category columns (valuation, profitability, balance
    # sheet, etc.) for every row, not just the megacaps. Raised 60 -> 600.
    # NOTE: if the cache has expired (>24h since the last warm), names beyond
    # the warmed set fall back to live yfinance fetches and can slow the first
    # render; re-run `warm` daily (or via refresh_scheduler) to keep it fast.
    _FUNDAMENTALS_LAZY_LIMIT = 600"""


    def _fail(msg: str) -> None:
        print(f"[fix_screener_limit] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        # Idempotency.
        if "_FUNDAMENTALS_LAZY_LIMIT = 600" in src:
            print("[fix_screener_limit] Already applied — limit is 600. Nothing "
                  "to do.")
            return

        # Prefer replacing the whole commented block; fall back to just the line.
        if OLD_BLOCK in src:
            src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
        elif "_FUNDAMENTALS_LAZY_LIMIT = 60" in src:
            src = src.replace("_FUNDAMENTALS_LAZY_LIMIT = 60",
                              "_FUNDAMENTALS_LAZY_LIMIT = 600", 1)
            print("[fix_screener_limit] NOTE: comment block differed; bumped the "
                  "value only.")
        else:
            _fail("could not find '_FUNDAMENTALS_LAZY_LIMIT = 60'. The file may "
                  "have changed; not editing blindly.")

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_limit] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Raised _FUNDAMENTALS_LAZY_LIMIT 60 -> 600")
        print()
        print("Reload the Screener. With the cache warmed, the deep category tabs")
        print("(Valuation PEG/PS/PB, Profitability, Balance sheet, Cash flow, Per")
        print("share, Technicals) should now fill for the whole universe instead")
        print("of only the top 60 names.")
        print()
        print("If some rows still show None, those names' cache entries have")
        print("expired — re-run:  python data/screener_data.py warm")
    return main()


def _fix_screener_performance() -> object:
    """
    fix_screener_performance.py
    ==========================

    Restore the **Performance** columns (Perf 1W/1M/3M/6M/YTD/1Y + Vol 1M), which
    went all-None after the Technicals tab was removed.

    WHY THEY BROKE
    --------------
    Performance and Technicals were filled by the SAME per-row loop (the old
    "Step 5b"): it computed trailing returns AND RSI/MA/ATR in one pass. Removing
    that loop to kill the Technicals tab also removed the only thing populating the
    Performance columns. This patch adds back a SLIM loop that fills ONLY the
    Performance columns (no RSI/MA/ATR), so Technicals stays gone.

    WHAT THIS DOES
    --------------
    Inserts a performance-only loop into ``_build_screener_frame``, right after the
    fundamentals loop and before Step 6. For the top ``_TECHNICALS_LAZY_LIMIT``
    names by market cap it calls ``get_prices`` (warm cache) and fills the
    Performance columns via the existing ``_compute_performance`` helper. Capped
    for the same reason as before: per-row price-history math is real CPU work and
    low-value on micro-caps.

    Prereqs: run AFTER fix_screener_remove_tabs.py (which removed the combined
    loop). Relies on the _compute_performance helper and _TECHNICALS_LAZY_LIMIT
    constant left in place by that removal.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects the performance-only loop and does nothing on re-run.
    * Verifies the helper + constant exist before inserting (fails loudly if not).
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_performance.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    ANCHOR = '''        fetched += 1

        # --- Step 6: derive rel_volume from volume / 30-day average ----'''

    INSERT = '''        fetched += 1

        # --- Step 5c: performance (trailing returns + 1m vol) for top-N -----
        # Performance-only (no technicals). Capped to the largest names by market
        # cap; reads the warm price cache. Rows below the cap stay "—".
        try:
            from data.market_data import MarketData as _MD
        except Exception:
            try:
                from market_data import MarketData as _MD  # type: ignore
            except Exception:
                _MD = None

        if _MD is not None:
            _md_p = _MD()
            _top = df.dropna(subset=["market_cap"]).nlargest(
                _TECHNICALS_LAZY_LIMIT, "market_cap"
            ) if "market_cap" in df.columns else df.head(_TECHNICALS_LAZY_LIMIT)
            _perf_done = 0
            for ridx in _top.index:
                sym = df.at[ridx, "symbol"]
                if not sym:
                    continue
                # Skip if performance already present (idempotent across reruns).
                if pd.notna(df.at[ridx, "perf_1m"]) and pd.notna(df.at[ridx, "perf_1y"]):
                    continue
                try:
                    _px = _md_p.get_prices(sym, period="1y", use_cache=True)
                    perf = _compute_performance(_px)
                    for _col, _val in perf.items():
                        if _col in df.columns and pd.notna(_val):
                            df.at[ridx, _col] = _val
                    _perf_done += 1
                except Exception:
                    continue
            df.attrs["performance_fetched"] = _perf_done

        # --- Step 6: derive rel_volume from volume / 30-day average ----'''


    def _fail(msg: str) -> None:
        print(f"[fix_screener_performance] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "# --- Step 5c: performance" in src:
            print("[fix_screener_performance] Already applied — performance loop "
                  "present. Nothing to do.")
            return

        # Prereqs left in place by the removal patch.
        if "def _compute_performance" not in src:
            _fail("the _compute_performance helper is missing. Apply "
                  "fix_screener_technicals.py (which adds it) or restore it before "
                  "running this.")
        if "_TECHNICALS_LAZY_LIMIT" not in src:
            _fail("_TECHNICALS_LAZY_LIMIT is missing. It should remain after the "
                  "removal patch; restore it before running this.")

        if ANCHOR not in src:
            _fail("could not find the fundamentals-loop / Step 6 anchor. The file "
                  "may differ (did you run fix_screener_remove_tabs.py first?). "
                  "Not editing blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_performance] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added Step 5c: performance-only loop (trailing returns + 1m vol)")
        print("    for the top-N by market cap. Technicals stays removed.")
        print()
        print("Fully restart Streamlit, then check Performance — Perf 1W..1Y and")
        print("Vol 1M should fill for the top ~100 names (rows below stay '—').")
        print("Keep the price cache warm: python data/screener_data.py warm")
    return main()


def _fix_screener_remove_tabs() -> object:
    """
    fix_screener_remove_tabs.py
    ==========================

    Remove the **Extended hours** and **Technicals** category tabs from the
    screener, and tear out the per-row technicals/performance computation loop
    that fed the Technicals tab.

    WHY
    ---
    * Extended hours (pre/post-market close/chg/vol): yfinance's daily data
      doesn't carry these, so the columns were always None — a dead tab.
    * Technicals (RSI/MA/ATR): these read the price cache per row and only filled
      for a handful of names where the cache was warm enough, giving an
      inconsistent half-populated tab. With the tab gone, the Step 5b loop that
      computed them is pure wasted CPU on every render, so we remove it too.

    WHAT THIS DOES
    --------------
    1. Deletes the ``"Extended hours": [...]`` and ``"Technicals": [...]`` entries
       from ``_CATEGORY_COLUMNS``. Because ``_CATEGORIES`` is derived from the
       dict's keys, both radio tabs disappear automatically.
    2. Removes the ``# --- Step 5b ...`` technicals/performance loop from
       ``_build_screener_frame`` (added by fix_screener_technicals.py), restoring
       the direct hand-off to Step 6.

    This leaves Performance, Valuation, Dividends, Profitability, Income
    statement, Balance sheet, Cash flow, and Per share intact — including the
    derived ratios. The _compute_performance helper and _TECHNICALS_LAZY_LIMIT
    constant are left in place (harmless, unused) to keep the diff minimal; they
    can be removed later if desired.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: if both tabs are already gone, does nothing.
    * Verifies ast.parse before saving.
    * Tolerant: removes whichever pieces are present (the Step 5b loop may or may
      not exist depending on whether the technicals patch was applied).

    Usage (from project root, venv active):
        python fix_screener_remove_tabs.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    # --- 1. The two category entries to delete (exact text) ------------------
    EXTENDED_HOURS = '''    "Extended hours": [
            "symbol", "name", "premarket_close", "premarket_chg_pct",
            "premarket_vol", "postmarket_close", "postmarket_chg_pct",
            "postmarket_vol",
        ],
    '''

    TECHNICALS_CAT = '''    "Technicals": [
            "symbol", "name", "price", "change_pct", "rsi_14", "ma_50",
            "ma_200", "beta", "atr_14", "volatility_1m",
        ],
    '''

    # --- 2. The Step 5b loop to remove (added by fix_screener_technicals.py) --
    # We replace the whole inserted block back down to the Step 6 marker.
    STEP5B_START = "    # --- Step 5b: technicals + performance for the top-N by market cap ---"
    STEP6_MARKER = "    # --- Step 6: derive rel_volume from volume / 30-day average ----"


    def _fail(msg: str) -> None:
        print(f"[fix_screener_remove_tabs] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")
        original = src
        did = []

        # 1. Remove the two category entries.
        if EXTENDED_HOURS in src:
            src = src.replace(EXTENDED_HOURS, "", 1)
            did.append("Extended hours tab")
        if TECHNICALS_CAT in src:
            src = src.replace(TECHNICALS_CAT, "", 1)
            did.append("Technicals tab")

        # 2. Remove the Step 5b loop if present (collapse to the Step 6 marker).
        s = src.find(STEP5B_START)
        if s != -1:
            e = src.find(STEP6_MARKER, s)
            if e == -1:
                _fail("found the Step 5b start but not the Step 6 marker after it; "
                      "refusing to guess the block boundary.")
            src = src[:s] + src[e:]
            did.append("Step 5b technicals loop")

        if not did:
            print("[fix_screener_remove_tabs] Nothing to do — both tabs already "
                  "removed and no Step 5b loop present.")
            return

        if src == original:
            print("[fix_screener_remove_tabs] No changes made.")
            return

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_remove_tabs] SUCCESS")
        print(f"  • Backed up original to {backup}")
        for d in did:
            print(f"  • Removed: {d}")
        print()
        print("Fully restart Streamlit. The Extended hours and Technicals radio")
        print("tabs should be gone; the remaining tabs are unchanged.")
    return main()


def _fix_screener_reset() -> object:
    """
    fix_screener_reset.py
    =====================

    Fix for the Screener "Clear filters" button crashing with:

        StreamlitAPIException: `st.session_state._flt_mcap` cannot be modified
        after the widget with key `_flt_mcap` is instantiated.

    THE BUG
    -------
    ``_reset_filters()`` does ``st.session_state["_flt_mcap"] = []`` (and similar
    assignments). But ``_flt_mcap`` is a multiselect WIDGET key, and
    ``render_screener_tab`` calls ``_render_filter_bar()`` — which instantiates
    that widget — BEFORE the Clear-filters button runs. Streamlit forbids
    assigning to a widget-backed key after the widget has been created in the same
    script run. So clicking Clear -> rerun -> the widget is created first -> the
    button's ``_reset_filters()`` assignment throws.

    This only manifested once the screener became a standalone page (render order
    shifted); it was latent before.

    THE FIX (the Streamlit-blessed pattern)
    ---------------------------------------
    You cannot reset a widget by assignment after it exists — but you CAN delete
    its key before it is instantiated, which makes the widget re-initialize from
    its default next run. So:

      1. The button no longer calls ``_reset_filters()`` directly. It sets a flag
         ``st.session_state["_screener_do_reset"] = True`` and reruns.
      2. At the VERY TOP of ``render_screener_tab`` — before any widget is created
         — we check the flag; if set, we DELETE all filter widget keys (and the
         paired ``_widget`` / min/max keys) and clear the flag. Deletion before
         instantiation is allowed, so the widgets come back at their defaults with
         no exception.
      3. ``_reset_filters()`` is rewritten to DELETE keys rather than assign, and
         is what the top-of-render flag handler calls.

    This fixes the crash everywhere (the original app too), changes no filtering
    logic, and preserves the defaults.

    SAFETY
    ------
    * Targets the real path ``ui/screener.py``.
    * Backs up to ``ui/screener.py.bak`` before writing.
    * Idempotent: detects the flag pattern and does nothing on re-run.
    * Verifies ``ast.parse`` before saving; aborts on any structural surprise.

    Usage (from project root, venv active):
        python fix_screener_reset.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"


    def _fail(msg: str) -> None:
        print(f"[fix_screener_reset] ABORT: {msg}")
        sys.exit(1)


    # The new _reset_filters body: deletion instead of assignment.
    _NEW_RESET = '''def _reset_filters() -> None:
        """Clear all filter state by DELETING widget keys.

        Assigning to a widget-backed key after the widget is instantiated raises
        StreamlitAPIException. Deleting the key (so the widget re-initializes from
        its default on the next run) is the supported reset pattern. This function
        is invoked from the top of render_screener_tab BEFORE any widget is built,
        via the `_screener_do_reset` flag set by the Clear-filters button.
        """
        keys = list(_FILTER_DEFAULTS.keys()) + [
            f"{k}_widget" for k in _FILTER_DEFAULTS
        ] + [
            "_flt_mcap", "_flt_sector", "_flt_rating",
            "_flt_price_min", "_flt_price_max",
            "_flt_change_widget", "_flt_pe_widget", "_flt_epsg_widget",
            "_flt_dy_widget", "_flt_revg_widget", "_flt_peg_widget",
            "_flt_roe_widget", "_flt_beta_widget",
        ]
        for k in keys:
            st.session_state.pop(k, None)
    '''


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active. (Expected ui/screener.py per the "
                  "traceback path.)")

        src = TARGET.read_text(encoding="utf-8")
        original = src

        # --- Idempotency check ------------------------------------------------
        if "_screener_do_reset" in src:
            print("[fix_screener_reset] Already applied — '_screener_do_reset' "
                  "flag present. Nothing to do.")
            return

        # --- Change 1: replace the _reset_filters function body ---------------
        # Find 'def _reset_filters() -> None:' and replace through to the next
        # top-level 'def ' (or blank-line boundary). We bound it by locating the
        # function start and the following 'def ' at column 0.
        start_marker = "def _reset_filters() -> None:"
        if start_marker not in src:
            _fail("could not find 'def _reset_filters() -> None:'. File may have "
                  "changed; not editing blindly.")
        start = src.index(start_marker)
        # Find the next top-level def after this one.
        next_def = src.find("\ndef ", start + len(start_marker))
        if next_def == -1:
            _fail("could not bound the _reset_filters function. Not editing.")
        # Preserve trailing blank lines between functions: capture up to next_def.
        src = src[:start] + _NEW_RESET + src[next_def:]

        # --- Change 2: button sets a flag instead of calling _reset_filters ---
        old_button = (
            "        ):\n"
            "            _reset_filters()\n"
            "            st.rerun()\n"
        )
        new_button = (
            "        ):\n"
            "            # Set a flag and rerun; the actual key deletion happens at\n"
            "            # the top of render_screener_tab BEFORE widgets are built,\n"
            "            # which is the only safe place to reset widget-backed keys.\n"
            "            st.session_state[\"_screener_do_reset\"] = True\n"
            "            st.rerun()\n"
        )
        if old_button not in src:
            _fail("could not find the Clear-filters button call site "
                  "('_reset_filters()' + 'st.rerun()'). Not editing. (Change 1 "
                  "not saved.)")
        src = src.replace(old_button, new_button, 1)

        # --- Change 3: flag handler at the very top of render_screener_tab ----
        render_marker = (
            "def render_screener_tab() -> None:\n"
            '    """Render the entire Screener tab. Safe to call inside a `with tab_x:`."""\n'
        )
        if render_marker not in src:
            _fail("could not find render_screener_tab signature + docstring to "
                  "insert the reset handler. Not editing. (Earlier changes not "
                  "saved.)")
        handler = (
            render_marker
            + "    # Handle a pending Clear-filters request BEFORE any widget is\n"
              "    # instantiated. Deleting widget keys here (rather than assigning\n"
              "    # to them after the widgets exist) is the only crash-free reset.\n"
              "    if st.session_state.pop(\"_screener_do_reset\", False):\n"
              "        _reset_filters()\n"
        )
        src = src.replace(render_marker, handler, 1)

        # --- Verify parse -----------------------------------------------------
        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving. Original "
                  "untouched.")

        if src == original:  # pragma: no cover
            _fail("no net change produced — please report this.")

        # --- Back up + write --------------------------------------------------
        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_reset] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Rewrote _reset_filters to DELETE keys (not assign)")
        print("  • Clear-filters button now sets a flag + reruns")
        print("  • Reset handler runs at the top of render_screener_tab, before")
        print("    any widget is instantiated")
        print()
        print("Reload the Screener tab and click 'Clear filters' — it should reset")
        print("every filter without the StreamlitAPIException.")
    return main()


def _fix_screener_skipcheck() -> object:
    """
    fix_screener_skipcheck.py
    ========================

    Fix for the screener showing `None` in the DEEP fundamentals columns
    (PEG, P/S, P/B, gross/op/profit margin, ROE, ROA, FCF, balance-sheet, etc.)
    for exactly the big-cap names at the top of the table (NVDA, AAPL, MSFT,
    AMZN, META, ...), even though the cache and yfinance both have the data.

    ROOT CAUSE
    ----------
    In ``_build_screener_frame``:

      * Step 4 merges ``_OFFLINE_FALLBACK_STOCKS`` into the frame. Each fallback
        row provides ONLY shallow fields: market_cap, pe, sector (plus price etc.)
        — it does NOT include deep fields (margins, ROE, PEG, P/S, P/B, FCF...).

      * Step 5's lazy-fetch loop has an "already have fundamentals?" skip-check:

            already = (pd.notna(market_cap) and pd.notna(pe)
                       and sector not in (None, "", "-"))
            if already: continue

        For every fallback name, Step 4 already filled market_cap + pe + sector,
        so this check is True → the loop SKIPS the real fundamentals fetch for
        those names. They keep only the 3 shallow offline values, and the deep
        columns stay None forever.

    So the most prominent names (the fallback list = top megacaps) are precisely
    the ones that never get deep fundamentals. That's the bug you see.

    THE FIX
    -------
    Make the skip-check key on a DEEP field the offline fallback never provides
    (gross_margin) instead of the shallow pe/sector it does provide. Then a row
    is only skipped if it genuinely already has deep fundamentals — so the
    fallback megacaps fall through to the real fetch and get fully populated.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects the new check and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_skipcheck.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    # The current skip-check (keys on shallow fields the offline fallback fills).
    OLD = """        # Skip if we already have fundamentals (market_cap + pe + sector).
            already = (
                pd.notna(df.at[ridx, "market_cap"])
                and pd.notna(df.at[ridx, "pe"])
                and df.at[ridx, "sector"] not in (None, "", "—")
            )
            if already:
                continue"""

    # New check: key on a DEEP field (gross_margin) that the offline fallback
    # never provides, so fallback megacaps are not wrongly skipped.
    NEW = """        # Skip only if we already have DEEP fundamentals. We key on
            # gross_margin specifically because the offline fallback fills the
            # shallow fields (market_cap + pe + sector) for megacaps but NOT the
            # deep ones — keying on pe/sector here would wrongly skip those names
            # and leave their margins/ROE/PEG/etc. permanently None.
            already = (
                pd.notna(df.at[ridx, "gross_margin"])
                and pd.notna(df.at[ridx, "roe"])
                and pd.notna(df.at[ridx, "market_cap"])
            )
            if already:
                continue"""


    def _fail(msg: str) -> None:
        print(f"[fix_screener_skipcheck] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if 'pd.notna(df.at[ridx, "gross_margin"])' in src:
            print("[fix_screener_skipcheck] Already applied — skip-check keys on "
                  "gross_margin. Nothing to do.")
            return

        if OLD not in src:
            _fail("could not find the exact skip-check block to replace. The file "
                  "may have changed (whitespace/quotes). Not editing blindly.")

        src = src.replace(OLD, NEW, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_skipcheck] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Skip-check now keys on deep fields (gross_margin + roe +")
        print("    market_cap) instead of the shallow pe/sector the offline")
        print("    fallback pre-fills.")
        print()
        print("Fully restart Streamlit (to clear the _build_screener_frame cache),")
        print("then open Valuation / Profitability. The top megacap names (NVDA,")
        print("AAPL, MSFT, ...) should now fill their deep columns instead of None.")
    return main()


def _fix_screener_technicals() -> object:
    """
    fix_screener_technicals.py
    =========================

    Fill the screener's TECHNICALS (#6) and PERFORMANCE (#7) columns for the top
    names by market cap (Option 2 — capped, to keep the frame build fast).

    Columns filled (for the top _TECHNICALS_LAZY_LIMIT symbols by market cap):
        Technicals : rsi_14, ma_50, ma_200, atr_14, volatility_1m
                     (beta already comes from fundamentals)
        Performance: perf_1w, perf_1m, perf_3m, perf_6m, perf_ytd, perf_1y,
                     volatility_1m

    WHY CAPPED
    ----------
    These require per-row price-history computation (RSI/MA/ATR via
    MarketData.get_technicals, plus trailing returns from get_prices). Running
    that for the whole ~538-name universe on every frame rebuild is real CPU work.
    RSI/MA on a micro-cap is also low-value. So we cap at the top 100 by market
    cap; below the cap these columns stay "—" (the same lazy behavior the
    fundamentals columns used to have). Both data calls read the 24h price cache
    (pre-warmed by `python data/screener_data.py warm`), so it's cache reads, not
    network.

    WHAT THIS DOES
    --------------
    1. Adds ``_TECHNICALS_LAZY_LIMIT = 100``.
    2. Adds a ``_compute_performance(prices)`` helper (trailing returns + 1m vol).
    3. Adds a second populate loop in ``_build_screener_frame`` (after the
       fundamentals loop, before Step 6) that fills technicals+performance for the
       top-N rows.

    Does NOT touch the fundamentals limit, the skip-check, or the derived-ratio
    code — it only inserts the constant, the helper, and the new loop.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects _TECHNICALS_LAZY_LIMIT and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_screener_technicals.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    # --- 1. Constant: insert right after the fundamentals limit line ---------
    LIMIT_ANCHOR_600 = "_FUNDAMENTALS_LAZY_LIMIT = 600"
    LIMIT_ANCHOR_60 = "_FUNDAMENTALS_LAZY_LIMIT = 60"
    LIMIT_INSERT = '''

    # Cap on per-row technicals/performance computation per render. These require
    # price-history math (RSI/MA/ATR + trailing returns) for each row, so unlike
    # the cache-backed fundamentals we keep this cap small — RSI/MA on a micro-cap
    # is low-value and the cost scales with row count. Top-N by market cap only;
    # rows below the cap show "—" for technicals/performance.
    _TECHNICALS_LAZY_LIMIT = 100'''

    # --- 2. Helper: insert before _build_screener_frame ----------------------
    HELPER_ANCHOR = "@st.cache_data(ttl=300, show_spinner=False)\ndef _build_screener_frame() -> pd.DataFrame:"
    HELPER_INSERT = '''def _compute_performance(prices: "pd.DataFrame") -> dict[str, float]:
        """Trailing total returns (%) and 1-month realized vol (%) from a price
        frame (OHLCV with a Close column). Returns NaN for windows longer than the
        available history. Pure computation — no I/O."""
        out = {
            "perf_1w": float("nan"), "perf_1m": float("nan"),
            "perf_3m": float("nan"), "perf_6m": float("nan"),
            "perf_ytd": float("nan"), "perf_1y": float("nan"),
            "volatility_1m": float("nan"),
        }
        try:
            if prices is None or len(prices) == 0 or "Close" not in prices.columns:
                return out
            close = prices["Close"].dropna()
            if len(close) < 2:
                return out
            last = float(close.iloc[-1])

            def _ret(n: int) -> float:
                if len(close) <= n:
                    return float("nan")
                prev = float(close.iloc[-1 - n])
                if prev == 0.0:
                    return float("nan")
                return (last / prev - 1.0) * 100.0

            # Approx trading-day windows.
            out["perf_1w"] = _ret(5)
            out["perf_1m"] = _ret(21)
            out["perf_3m"] = _ret(63)
            out["perf_6m"] = _ret(126)
            out["perf_1y"] = _ret(252)

            # YTD: first close of the current calendar year.
            try:
                idx = close.index
                this_year = idx[-1].year
                ytd_slice = close[[ts.year == this_year for ts in idx]]
                if len(ytd_slice) >= 2 and float(ytd_slice.iloc[0]) != 0.0:
                    out["perf_ytd"] = (last / float(ytd_slice.iloc[0]) - 1.0) * 100.0
            except Exception:
                pass

            # 1-month realized vol: std of daily returns over ~21d, annualized %.
            rets = close.pct_change().dropna()
            if len(rets) >= 21:
                import numpy as _np
                out["volatility_1m"] = float(rets.iloc[-21:].std() * _np.sqrt(252) * 100.0)
        except Exception:
            pass
        return out


    @st.cache_data(ttl=300, show_spinner=False)
    def _build_screener_frame() -> pd.DataFrame:'''

    # --- 3. The technicals loop: insert after the fundamentals loop ----------
    LOOP_ANCHOR = '''        fetched += 1

        # --- Step 6: derive rel_volume from volume / 30-day average ----'''
    LOOP_INSERT = '''        fetched += 1

        # --- Step 5b: technicals + performance for the top-N by market cap ---
        # Capped (Option 2): per-row price-history math is real CPU work, so we
        # only compute it for the largest names. Reads the warm price cache.
        try:
            from data.market_data import MarketData as _MD
        except Exception:
            try:
                from market_data import MarketData as _MD  # type: ignore
            except Exception:
                _MD = None  # technicals unavailable; columns stay NaN

        if _MD is not None:
            _md_t = _MD()
            # df is not yet sorted by market cap here; pick the top-N by the
            # market_cap column we just populated.
            _top = df.dropna(subset=["market_cap"]).nlargest(
                _TECHNICALS_LAZY_LIMIT, "market_cap"
            ) if "market_cap" in df.columns else df.head(_TECHNICALS_LAZY_LIMIT)
            _tech_done = 0
            for ridx in _top.index:
                sym = df.at[ridx, "symbol"]
                if not sym:
                    continue
                # Skip if technicals already present (idempotent across reruns).
                if pd.notna(df.at[ridx, "rsi_14"]) and pd.notna(df.at[ridx, "ma_50"]):
                    continue
                try:
                    tech = _md_t.get_technicals(sym, period="1y") or {}
                    if tech.get("rsi_14") is not None:
                        df.at[ridx, "rsi_14"] = tech.get("rsi_14")
                    if tech.get("sma_50") is not None:
                        df.at[ridx, "ma_50"] = tech.get("sma_50")
                    if tech.get("sma_200") is not None:
                        df.at[ridx, "ma_200"] = tech.get("sma_200")
                    if tech.get("atr_14") is not None:
                        df.at[ridx, "atr_14"] = tech.get("atr_14")
                    # Performance + 1m vol from the price cache (same 24h cache).
                    _px = _md_t.get_prices(sym, period="1y", use_cache=True)
                    perf = _compute_performance(_px)
                    for _col, _val in perf.items():
                        if _col in df.columns and pd.notna(_val):
                            df.at[ridx, _col] = _val
                    _tech_done += 1
                except Exception:
                    continue
            df.attrs["technicals_fetched"] = _tech_done

        # --- Step 6: derive rel_volume from volume / 30-day average ----'''


    def _fail(msg: str) -> None:
        print(f"[fix_screener_technicals] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_TECHNICALS_LAZY_LIMIT" in src:
            print("[fix_screener_technicals] Already applied — _TECHNICALS_LAZY_LIMIT "
                  "present. Nothing to do.")
            return

        # 1. Insert the constant after whichever fundamentals-limit line exists.
        if LIMIT_ANCHOR_600 in src:
            src = src.replace(LIMIT_ANCHOR_600, LIMIT_ANCHOR_600 + LIMIT_INSERT, 1)
        elif LIMIT_ANCHOR_60 in src:
            src = src.replace(LIMIT_ANCHOR_60, LIMIT_ANCHOR_60 + LIMIT_INSERT, 1)
        else:
            _fail("could not find _FUNDAMENTALS_LAZY_LIMIT to anchor the new "
                  "constant. Not editing blindly.")

        # 2. Insert the helper before _build_screener_frame.
        if HELPER_ANCHOR not in src:
            _fail("could not find the _build_screener_frame definition (with its "
                  "@st.cache_data decorator) to anchor the helper. Not editing.")
        src = src.replace(HELPER_ANCHOR, HELPER_INSERT, 1)

        # 3. Insert the technicals loop after the fundamentals loop.
        if LOOP_ANCHOR not in src:
            _fail("could not find the fundamentals-loop / Step 6 anchor to insert "
                  "the technicals loop. The file may differ. Not editing. (Earlier "
                  "inserts not saved.)")
        src = src.replace(LOOP_ANCHOR, LOOP_INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_technicals] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added _TECHNICALS_LAZY_LIMIT = 100")
        print("  • Added _compute_performance() helper")
        print("  • Added Step 5b loop: technicals + performance for top-100 by mktcap")
        print()
        print("Fully restart Streamlit, then check Technicals (RSI/MA/ATR/Vol) and")
        print("Performance (Perf 1W..1Y) — they should fill for the top ~100 names")
        print("(rows below the cap stay '—').")
        print()
        print("If the frame build feels slow, lower _TECHNICALS_LAZY_LIMIT (e.g.")
        print("to 50). It reads the warm price cache, so keep that fresh with")
        print("`python data/screener_data.py warm`.")
    return main()


def _fix_screener_xbrl_read() -> object:
    """
    fix_screener_xbrl_read.py
    ========================

    Bridge the screener to read real Operating Income / Total Assets / CapEx from
    the ``xbrl_facts`` table (populated by ``python data/screener_data.py xbrl``),
    filling those three columns for rows where yfinance left them None.

    HOW IT WORKS
    ------------
    After the fundamentals + performance loops in ``_build_screener_frame``, this
    inserts a step that — for rows still missing operating_income / total_assets /
    capex — queries the latest annual (10-K) value of the matching US-GAAP concept
    from xbrl_facts and fills the column. It's a LOCAL SQLite read (the SEC network
    calls already happened during the xbrl warm), so there's no render-time network
    cost.

    Concept -> column:
        operating_income <- OperatingIncomeLoss
        total_assets     <- Assets
        capex            <- PaymentsToAcquirePropertyPlantAndEquipment

    COVERAGE CAVEAT
    ---------------
    Values only appear for symbols that were ingested AND that report the concept.
    Some financials (banks) don't file a GAAP operating-income line, so Op Income
    stays blank for them — a filing reality, not a bug. Total Assets and CapEx
    have broader coverage. Run the xbrl warm first or these columns stay None.

    SAFETY
    ------
    * Targets ui/screener.py.
    * Backs up to ui/screener.py.bak before writing.
    * Idempotent: detects the XBRL bridge and does nothing on re-run.
    * Verifies ast.parse before saving.
    * Anchors on the stable "Step 6" marker, so it sits after the fundamentals
      and performance loops regardless of whether those patches are present.

    Usage (from project root, venv active):
        python fix_screener_xbrl_read.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("ui") / "screener.py"

    ANCHOR = "    # --- Step 6: derive rel_volume from volume / 30-day average ----"

    INSERT = '''    # --- Step 5d: fill Op Income / Total Assets / CapEx from XBRL -------
        # Real from-filings values, read from the local xbrl_facts table
        # (populated by `python data/screener_data.py xbrl`). Local SQLite read —
        # no network at render time. Only fills rows where yfinance left these
        # None; symbols not ingested or not reporting the concept stay "—".
        try:
            import sqlite3 as _sqlite3
            _XBRL_COL_CONCEPT = {
                "operating_income": "OperatingIncomeLoss",
                "total_assets": "Assets",
                "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
            }
            _xbrl_db = "data/hedgefund.db"
            with _sqlite3.connect(_xbrl_db) as _xc:
                # Confirm the table exists (xbrl warm may not have run yet).
                _has_tbl = _xc.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='xbrl_facts'"
                ).fetchone() is not None
                if _has_tbl:
                    for _ridx in df.index:
                        _sym = df.at[_ridx, "symbol"]
                        if not _sym:
                            continue
                        for _col, _concept in _XBRL_COL_CONCEPT.items():
                            if _col not in df.columns:
                                continue
                            # Don't overwrite a value already present.
                            if pd.notna(df.at[_ridx, _col]):
                                continue
                            _row = _xc.execute(
                                "SELECT value FROM xbrl_facts "
                                "WHERE ticker = ? AND concept = ? AND form = '10-K' "
                                "ORDER BY period_end DESC LIMIT 1",
                                (str(_sym).upper(), _concept),
                            ).fetchone()
                            if _row and _row[0] is not None:
                                df.at[_ridx, _col] = float(_row[0])
        except Exception:
            # XBRL is best-effort enrichment; never break the frame build over it.
            pass

    ''' + ANCHOR


    def _fail(msg: str) -> None:
        print(f"[fix_screener_xbrl_read] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "# --- Step 5d: fill Op Income / Total Assets / CapEx from XBRL" in src:
            print("[fix_screener_xbrl_read] Already applied — XBRL read bridge "
                  "present. Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the 'Step 6' anchor to insert before. The file "
                  "may differ. Not editing blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_screener_xbrl_read] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added Step 5d: read Op Income / Total Assets / CapEx from")
        print("    xbrl_facts for rows where yfinance left them None.")
        print()
        print("Make sure you've run the XBRL warm first:")
        print("    python data/screener_data.py xbrl")
        print("Then fully restart Streamlit and check Income statement (Op Income),")
        print("Balance sheet (Total Assets), and Cash flow (CapEx).")
    return main()


def _fix_sector_offline() -> object:
    """
    fix_sector_offline.py — make _sector_for strictly offline.

    Run from D:\\Ary Fund:   python fix_sector_offline.py

    BUG: _sector_for calls _md.get_fundamentals(tk, use_cache=True), which is
    cache-FIRST, not cache-ONLY. On a stale/missing cache entry it falls through
    to a live yfinance fetch, which 404s for delisted/renamed tickers. That makes
    the "offline" peer recompute non-deterministic: it hits the network whenever
    the fundamentals cache has aged past its 24h TTL.

    FIX: read sector directly from the fundamentals_cache table (data_json blob),
    ignoring the TTL. Sector is effectively immutable (a company's GICS sector
    doesn't change), so any cached row is authoritative and there is never a
    reason to re-fetch it from the network. This makes the recompute fully
    offline and deterministic regardless of cache age.

    Surgical: replaces exactly the _sector_for function body and nothing else.
    Refuses to write unless the expected block is found exactly once, re-parses
    with ast before and after writing, and writes UTF-8 without BOM.
    """
    import ast
    import io
    import sys

    PATH = r"D:\Ary Fund\data\pipeline.py"

    OLD = (
        '    def _sector_for(tk: str):\n'
        '        """Cache-first sector tag (yfinance .info), offline. None if absent."""\n'
        '        if _md is None:\n'
        '            return None\n'
        '        try:\n'
        '            f = _md.get_fundamentals(tk, use_cache=True)\n'
        '        except Exception:  # noqa: BLE001\n'
        '            return None\n'
        '        if not isinstance(f, dict):\n'
        '            return None\n'
        '        s = f.get("sector")\n'
        '        return s.strip() if isinstance(s, str) and s.strip() else None\n'
    )

    NEW = (
        '    def _sector_for(tk: str):\n'
        '        """Sector tag read DIRECTLY from the fundamentals_cache table.\n'
        '\n'
        '        Strictly offline: no get_fundamentals (which is cache-FIRST and\n'
        '        falls through to a live yfinance fetch on a stale entry), and no\n'
        '        TTL check. Sector is immutable, so any cached row is authoritative.\n'
        '        Returns None if the ticker has no cached fundamentals row.\n'
        '        """\n'
        '        if _md is None:\n'
        '            return None\n'
        '        try:\n'
        '            import json as _json\n'
        '            import sqlite3 as _sqlite3\n'
        '            with _sqlite3.connect(_md.db_path) as _conn:\n'
        '                _row = _conn.execute(\n'
        '                    "SELECT data_json FROM fundamentals_cache WHERE ticker = ?",\n'
        '                    (tk,),\n'
        '                ).fetchone()\n'
        '            if not _row or not _row[0]:\n'
        '                return None\n'
        '            _data = _json.loads(_row[0])\n'
        '        except Exception:  # noqa: BLE001\n'
        '            return None\n'
        '        if not isinstance(_data, dict):\n'
        '            return None\n'
        '        s = _data.get("sector")\n'
        '        return s.strip() if isinstance(s, str) and s.strip() else None\n'
    )


    def main() -> int:
        with io.open(PATH, "r", encoding="utf-8-sig") as f:
            src = f.read()

        count = src.count(OLD)
        if count == 0:
            print("ERROR: expected _sector_for block not found. No changes made.")
            if "Sector tag read DIRECTLY from the fundamentals_cache" in src:
                print("NOTE: looks like it's already patched.")
            return 1
        if count > 1:
            print(f"ERROR: block found {count} times (expected 1). Aborting.")
            return 1

        patched = src.replace(OLD, NEW)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            print(f"ERROR: patched source does not parse ({e}). No changes written.")
            return 1

        with io.open(PATH, "w", encoding="utf-8", newline="") as f:
            f.write(patched)

        with io.open(PATH, "r", encoding="utf-8") as f:
            check = f.read()
        try:
            ast.parse(check)
        except SyntaxError as e:
            print(f"ERROR: file on disk does not parse after write ({e}).")
            return 1

        print("PATCHED OK — _sector_for now reads fundamentals_cache directly "
              "(offline, no TTL, no network fallthrough).")
        return 0
    return main()


def _fix_universe() -> object:
    """
    fix_universe.py
    ==============

    Prune dead / delisted tickers from the screener+analysis universe so they stop
    spamming "possibly delisted; no price data found" warnings on every warm/scan.

    WHAT GETS REMOVED
    -----------------
    The `check` command (data/screener_data.py) flagged 22 symbols that return no
    yfinance data. They fall into two groups:

      20 GENUINELY DELISTED / acquired / renamed:
        ANSS  CMA   CTLT  CTRA  DAY   DFS   FI    HES   HOLX  IPG
        JNPR  K     MMC   MRO   PARA  PSTG  PXD   RDFN  SQ    WBA
      (e.g. ANSS acquired by Synopsys, PXD by ExxonMobil, SQ renamed to XYZ,
       FI = Fiserv ticker change, etc.)

      2 ALIVE but FORMAT-MISMATCHED:
        BRK.B  BF.B
      These are NOT delisted — Berkshire-B and Brown-Forman-B are active. yfinance
      wants the HYPHEN form (BRK-B / BF-B), but normalize_ticker() stores dot-form,
      so the fetch fails. The proper fix is a dot->hyphen conversion in the FETCH
      layer (market_data / yfinance call), not here. Until that's done we exclude
      them too, so they don't spam warnings. RE-ADD them once the fetch layer maps
      .B-class tickers to the hyphen form.

    HOW
    ---
    Rather than surgically deleting string literals from the densely-packed
    SP500_TICKERS / EXTRA_LARGE_CAPS tuples (fragile), this adds a `_DELISTED`
    exclusion set and filters it out at the `US_UNIVERSE` construction line. The
    source tuples stay intact (so the historical membership is still visible), but
    the dead names never reach the final universe.

    SAFETY
    ------
    * Targets universe.py.
    * Backs up to universe.py.bak before writing.
    * Idempotent: detects `_DELISTED` and does nothing on re-run.
    * Verifies ast.parse AND imports the patched module to confirm the dead
      tickers are actually gone from US_UNIVERSE before keeping the change.

    Usage (from project root, venv active):
        python fix_universe.py
    """

    import ast
    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "universe.py"

    OLD = ("US_UNIVERSE: Tuple[str, ...] = tuple(sorted(set(SP500_TICKERS + "
           "EXTRA_LARGE_CAPS)))")

    NEW = '''# Symbols that no longer return data from yfinance. Filtered out of the
    # combined universe below so they don't spam "possibly delisted" warnings on
    # every warm/scan. See fix_universe.py for the full rationale.
    #
    # 20 genuinely delisted / acquired / renamed:
    _DELISTED_DEAD = frozenset({
        "ANSS", "CMA", "CTLT", "CTRA", "DAY", "DFS", "FI", "HES", "HOLX", "IPG",
        "JNPR", "K", "MMC", "MRO", "PARA", "PSTG", "PXD", "RDFN", "SQ", "WBA",
    })
    # 2 ALIVE but format-mismatched (yfinance wants BRK-B / BF-B, not the dot
    # form normalize_ticker stores). Excluded until the FETCH layer maps .B-class
    # tickers to the hyphen form; re-add them then.
    _DELISTED_FORMAT = frozenset({"BRK.B", "BF.B"})

    _DELISTED = _DELISTED_DEAD | _DELISTED_FORMAT

    US_UNIVERSE: Tuple[str, ...] = tuple(
        sorted(set(SP500_TICKERS + EXTRA_LARGE_CAPS) - _DELISTED)
    )'''


    def _fail(msg: str) -> None:
        print(f"[fix_universe] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "_DELISTED" in src:
            print("[fix_universe] Already applied — _DELISTED present. Nothing to do.")
            return

        if OLD not in src:
            _fail("could not find the US_UNIVERSE construction line to replace. "
                  "The file may have changed; not editing blindly.")

        patched = src.replace(OLD, NEW, 1)

        try:
            ast.parse(patched)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        # --- Verify by importing the PATCHED source from a temp file ---------
        # Confirm the dead tickers are actually gone before we commit the change.
        tmp = TARGET.with_name("_universe_patched_check.py")
        try:
            tmp.write_text(patched, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("_universe_check", tmp)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            univ = set(getattr(mod, "US_UNIVERSE", ()))
            leftover = {"ANSS", "PXD", "SQ", "BRK.B", "BF.B", "WBA"} & univ
            if leftover:
                _fail(f"verification failed — these should be gone but remain: "
                      f"{sorted(leftover)}")
            before = len(set(getattr(mod, "SP500_TICKERS", ()))
                         | set(getattr(mod, "EXTRA_LARGE_CAPS", ())))
            after = len(univ)
            print(f"[fix_universe] verified: universe {before} -> {after} "
                  f"({before - after} removed)")
        except SystemExit:
            raise
        except Exception as e:
            _fail(f"could not import patched module to verify ({e}); not saving.")
        finally:
            if tmp.exists():
                tmp.unlink()

        # --- Back up + write --------------------------------------------------
        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(patched, encoding="utf-8")

        print("[fix_universe] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added _DELISTED exclusion set (20 dead + 2 format-mismatched)")
        print("  • Filtered them out of US_UNIVERSE")
        print()
        print("Re-run `python data/screener_data.py check` — it should now report")
        print("0 dead symbols. The delisting warnings will stop on warm/scan.")
        print()
        print("NOTE: BRK.B and BF.B are ALIVE — they're excluded only because the")
        print("fetch layer doesn't yet map them to yfinance's hyphen form (BRK-B/")
        print("BF-B). Re-add them to the universe once that fetch-layer fix lands.")
    return main()


def _fix_xbrl_opincome() -> object:
    """
    fix_xbrl_opincome.py
    ===================

    Add Operating Income to the XBRL concept map so it gets ingested from SEC
    companyfacts (Total Assets and CapEx are already mapped).

    CONTEXT
    -------
    ``ingest_xbrl_facts`` in sec_fetcher.py walks ``XBRL_CONCEPT_MAP`` and writes
    each mapped concept into the ``xbrl_facts`` table. The map already covers
    total_assets (``Assets``) and capex
    (``PaymentsToAcquirePropertyPlantAndEquipment``), but has NO entry for
    operating income — so the screener's "Op Income" column can never fill from
    XBRL. This adds the canonical operating-income concept (plus a common
    fallback).

    NOTE ON COVERAGE
    ----------------
    ``OperatingIncomeLoss`` is the standard US-GAAP tag and is widely reported,
    but some companies — especially banks/financials (JPM, BAC, ...) — don't file
    a clean GAAP operating-income line, so those will remain blank even after
    ingestion. That's a filing reality, not a bug. Total Assets and CapEx have
    broader coverage.

    SAFETY
    ------
    * Targets data/sec_fetcher.py.
    * Backs up to data/sec_fetcher.py.bak before writing.
    * Idempotent: detects the operating_income entry and does nothing on re-run.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_xbrl_opincome.py
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "sec_fetcher.py"

    # Insert the operating_income entry right after the total_assets entry
    # (which is already present and stable).
    ANCHOR = '''    "ticker.fundamental.total_assets": [
            "Assets",
        ],'''

    INSERT = '''    "ticker.fundamental.total_assets": [
            "Assets",
        ],
        "ticker.fundamental.operating_income_ttm": [
            "OperatingIncomeLoss",
            # Fallback some filers use when OperatingIncomeLoss is absent:
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        ],'''


    def _fail(msg: str) -> None:
        print(f"[fix_xbrl_opincome] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")

        if "ticker.fundamental.operating_income_ttm" in src:
            print("[fix_xbrl_opincome] Already applied — operating_income concept "
                  "present. Nothing to do.")
            return

        if ANCHOR not in src:
            _fail("could not find the total_assets map entry to anchor the insert. "
                  "The map may have changed. Not editing blindly.")

        src = src.replace(ANCHOR, INSERT, 1)

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_xbrl_opincome] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print("  • Added operating_income_ttm -> OperatingIncomeLoss (+fallback)")
        print()
        print("Next: run the XBRL warm so operating income (and the already-mapped")
        print("total assets / capex) get ingested:")
        print("    python data/screener_data.py xbrl")
    return main()


def _fix_yahoo_symbol() -> object:
    """
    fix_yahoo_symbol.py
    ==================

    Fix BRK.B / BF.B (and any dot-class symbol) not fetching from yfinance.

    ROOT CAUSE
    ----------
    The universe stores class shares in dot-form (BRK.B, BF.B) — that's the
    canonical key used for the cache and registry. But yfinance/Yahoo expects
    hyphen-form (BRK-B, BF-B). The code passes the dot-form straight to
    yf.Ticker(), so Yahoo returns nothing and these names look delisted (they're in
    universe._DELISTED_FORMAT for exactly this reason).

    THE FIX
    -------
    Add a module-level _yahoo_symbol() helper that converts the class separator
    '.' -> '-' (BRK.B -> BRK-B), and apply it ONLY at the yf.Ticker() call sites.
    The `ticker` variable itself (used for the cache key and registry storage)
    stays in canonical dot-form, so nothing about caching/storage changes — only
    the symbol handed to Yahoo is translated.

    Note: this converts a single dot that separates a share class (e.g. BRK.B,
    BF.B, BRK.A). It leaves normal symbols untouched.

    SAFETY
    ------
    * Targets data/market_data.py.
    * Backs up to data/market_data.py.bak before writing.
    * Idempotent: detects the helper + wrapped calls; re-run is a no-op.
    * Verifies ast.parse before saving.

    Usage (from project root, venv active):
        python fix_yahoo_symbol.py

    After applying, BRK.B / BF.B fetch normally. If they were excluded from the
    universe via _DELISTED_FORMAT, you can re-include them (optional).
    """

    import ast
    import shutil
    import sys
    from pathlib import Path

    TARGET = Path("data") / "market_data.py"

    HELPER = '''

    def _yahoo_symbol(ticker: str) -> str:
        """Translate a canonical dot-class symbol to yfinance's hyphen form.

        Yahoo Finance spells share classes with a hyphen (BRK-B, BF-B), while this
        project stores them with a dot (BRK.B, BF.B). Convert only for the Yahoo
        call; the dot-form remains the cache/registry key everywhere else. Plain
        symbols pass through unchanged.
        """
        if not ticker:
            return ticker
        return ticker.replace(".", "-")

    '''

    # Insert the helper right after the `import yfinance as yf` line.
    IMPORT_ANCHOR = "import yfinance as yf"

    # Wrap every yf.Ticker(ticker) call. Two textual forms appear.
    CALL_OLD_A = "t = yf.Ticker(ticker)"
    CALL_NEW_A = "t = yf.Ticker(_yahoo_symbol(ticker))"
    CALL_OLD_B = "info = yf.Ticker(ticker).info or {}"
    CALL_NEW_B = "info = yf.Ticker(_yahoo_symbol(ticker)).info or {}"


    def _fail(msg: str) -> None:
        print(f"[fix_yahoo_symbol] ABORT: {msg}")
        sys.exit(1)


    def main() -> None:
        if not TARGET.exists():
            _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
                  "with the venv active.")

        src = TARGET.read_text(encoding="utf-8")
        original = src

        # 1. Insert the helper (once), after the yfinance import.
        if "_yahoo_symbol" not in src:
            if IMPORT_ANCHOR not in src:
                _fail("could not find 'import yfinance as yf' to anchor the helper.")
            src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)

        # 2. Wrap the call sites (idempotent — only replaces the unwrapped form).
        n_a = src.count(CALL_OLD_A)
        src = src.replace(CALL_OLD_A, CALL_NEW_A)
        n_b = src.count(CALL_OLD_B)
        src = src.replace(CALL_OLD_B, CALL_NEW_B)

        if src == original:
            print("[fix_yahoo_symbol] Already applied (helper + wrapped calls "
                  "present). Nothing to do.")
            return

        try:
            ast.parse(src)
        except SyntaxError as e:
            _fail(f"patched file does not parse ({e}); not saving.")

        backup = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, backup)
        TARGET.write_text(src, encoding="utf-8")

        print("[fix_yahoo_symbol] SUCCESS")
        print(f"  • Backed up original to {backup}")
        print(f"  • Added _yahoo_symbol() helper")
        print(f"  • Wrapped {n_a} yf.Ticker(ticker) call(s) + {n_b} .info call(s)")
        print()
        print("BRK.B / BF.B now fetch from Yahoo as BRK-B / BF-B, while the cache")
        print("and registry keep the canonical dot-form. Test:")
        print("    python -c \"from data.market_data import MarketData; "
              "print(MarketData().get_prices('BRK.B', period='5d').tail())\"")
        print()
        print("Optional: if BRK.B/BF.B were excluded via universe._DELISTED_FORMAT,")
        print("you can re-include them now that fetching works.")
    return main()


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------
FIXES = {
    "add_flow_destination": _fix_add_flow_destination,
    "app_v2_report_wire": _fix_app_v2_report_wire,
    "context_registry_db": _fix_context_registry_db,
    "jobs_download": _fix_jobs_download,
    "jobs_open_latest": _fix_jobs_open_latest,
    "jobs_open_pdf": _fix_jobs_open_pdf,
    "jobs_read_inapp": _fix_jobs_read_inapp,
    "jobs_report_picker": _fix_jobs_report_picker,
    "jobs_report_rows": _fix_jobs_report_rows,
    "lab_rag_button": _fix_lab_rag_button,
    "main_offline": _fix_main_offline,
    "market_risk_metrics": _fix_market_risk_metrics,
    "pulse_fixture": _fix_pulse_fixture,
    "rag_to_tools": _fix_rag_to_tools,
    "recession_prob": _fix_recession_prob,
    "redflag_dedup": _fix_redflag_dedup,
    "refresh_button": _fix_refresh_button,
    "regime_return": _fix_regime_return,
    "report_table_layout": _fix_report_table_layout,
    "reranker_offline": _fix_reranker_offline,
    "review_timeout": _fix_review_timeout,
    "risk_count": _fix_risk_count,
    "risk_reasons_allclear": _fix_risk_reasons_allclear,
    "scheduler_bootstrap": _fix_scheduler_bootstrap,
    "scheduler_offline": _fix_scheduler_offline,
    "screener_data_xbrl": _fix_screener_data_xbrl,
    "screener_derived": _fix_screener_derived,
    "screener_limit": _fix_screener_limit,
    "screener_performance": _fix_screener_performance,
    "screener_remove_tabs": _fix_screener_remove_tabs,
    "screener_reset": _fix_screener_reset,
    "screener_skipcheck": _fix_screener_skipcheck,
    "screener_technicals": _fix_screener_technicals,
    "screener_xbrl_read": _fix_screener_xbrl_read,
    "sector_offline": _fix_sector_offline,
    "universe": _fix_universe,
    "xbrl_opincome": _fix_xbrl_opincome,
    "yahoo_symbol": _fix_yahoo_symbol,
}


def _doc(fn) -> str:
    d = (fn.__doc__ or "").strip().splitlines()
    return d[0].strip() if d else "(no description)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Archive of applied one-off patch scripts.")
    p.add_argument("name", nargs="?", help="fix to run (see --list)")
    p.add_argument("--list", action="store_true", help="list all fixes")
    p.add_argument("--all", action="store_true",
                   help="run every fix (each no-ops if already applied)")
    args = p.parse_args(argv)

    if args.list or (not args.name and not args.all):
        print(f"{len(FIXES)} archived fixes:\n")
        for n, fn in sorted(FIXES.items()):
            print(f"  {n:28} {_doc(fn)}")
        print("\nRun one:  python past_fixes.py <name>")
        return 0

    if args.all:
        for n, fn in sorted(FIXES.items()):
            print(f"\n{'='*70}\n# {n}\n{'='*70}")
            try:
                fn()
            except SystemExit:
                pass
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR in {n}: {e}")
        return 0

    fn = FIXES.get(args.name)
    if fn is None:
        print(f"unknown fix: {args.name!r}. Use --list to see the {len(FIXES)} names.")
        return 1
    rv = fn()
    return rv if isinstance(rv, int) else 0


if __name__ == "__main__":
    sys.exit(main())
