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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_universe.py