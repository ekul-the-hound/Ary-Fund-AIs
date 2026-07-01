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
from __future__ import annotations

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


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_market_risk_metrics.py
