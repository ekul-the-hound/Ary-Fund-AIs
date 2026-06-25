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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_performance.py
