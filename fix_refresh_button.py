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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_refresh_button.py
