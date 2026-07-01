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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_yahoo_symbol.py
