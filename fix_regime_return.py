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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_regime_return.py