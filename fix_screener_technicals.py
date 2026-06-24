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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_technicals.py
