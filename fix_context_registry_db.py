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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_context_registry_db.py
