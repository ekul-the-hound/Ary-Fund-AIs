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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_risk_reasons_allclear.py
