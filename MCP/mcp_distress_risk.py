"""
mcp_distress_risk.py
====================

MCP server #1 of 3: DISTRESS & ACCOUNTING RISK.
Exposes ARY QUANT's fundamental-risk tools to MCP clients (Claude
Desktop, Claude Code, or any agent that speaks MCP).

Tools:
    altman_z(ticker)        — bankruptcy risk, safe/grey/distress zone
    piotroski_f(ticker)     — 9-point fundamental quality score
    beneish_m(ticker)       — earnings-manipulation screen
    risk_flags(ticker)      — composite fundamental/macro/market risk

All tools read fundamentals through the registry-first pipeline
(build_agent_context), so figures come from hedgefund.db — the same
grounded numbers the platform uses, never LLM-generated.

INSTALL (once, hedgefund_ai venv active):
    pip install "mcp[cli]" -c constraints_openbb.txt

RUN:
    python mcp_distress_risk.py            # stdio (Claude Desktop/Code)
    python mcp_distress_risk.py --http     # http://127.0.0.1:8801/mcp

REGISTER with Claude Code (from project root):
    claude mcp add ary-distress-risk -- python mcp_distress_risk.py
Or in Claude Desktop's claude_desktop_config.json:
    "ary-distress-risk": {
      "command": "D:\\Ary Fund\\hedgefund_ai\\Scripts\\python.exe",
      "args": ["D:\\Ary Fund\\mcp_distress_risk.py"],
      "cwd": "D:\\Ary Fund"
    }
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

# Run from anywhere: put the project root on sys.path.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

PORT = 8801
mcp = FastMCP("ary-distress-risk", host="127.0.0.1", port=PORT)

_PORTFOLIO_DB = "data/portfolio.db"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _jsonable(obj: Any, _depth: int = 0) -> Any:
    """Convert numpy/pandas scalars & containers to JSON-safe values."""
    if _depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return None if isinstance(obj, float) and not math.isfinite(obj) else obj
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return _jsonable(obj.item(), _depth + 1)
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = [_jsonable(v, _depth + 1) for v in list(obj)[:100]]
        if hasattr(obj, "__len__") and len(obj) > 100:
            seq.append(f"... ({len(obj) - 100} more truncated)")
        return seq
    return str(obj)


def _get_context(ticker: str) -> Dict[str, Any]:
    """Registry-first context for a ticker (metrics + macro)."""
    from data.pipeline import build_agent_context  # lazy: heavy import

    return build_agent_context(ticker.upper().strip(),
                               _PORTFOLIO_DB, SimpleNamespace())


def _extract_metrics(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the key-metrics dict out of the context, tolerating naming."""
    for key in ("metrics", "key_metrics", "fundamental_metrics"):
        m = ctx.get(key)
        if isinstance(m, dict) and m:
            return m
    return None


def _run(tool: str, ticker: str, fn) -> Dict[str, Any]:
    """Shared defensive wrapper: context -> metrics -> compute -> JSON."""
    try:
        ctx = _get_context(ticker)
        metrics = _extract_metrics(ctx)
        if not metrics:
            return {"error": f"no fundamentals available for {ticker!r} — "
                             "is it in the universe and backfilled? "
                             f"(context keys: {sorted(ctx.keys())})"}
        result = fn(metrics, ctx)
        if result is None:
            return {"error": f"{tool}: insufficient inputs for {ticker!r} "
                             "(required fields missing from fundamentals)"}
        return _jsonable({"ticker": ticker.upper(), **result})
    except Exception as e:  # noqa: BLE001
        return {"error": f"{tool} failed for {ticker!r}: {e}"}


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
@mcp.tool()
def altman_z(ticker: str, variant: Optional[str] = None) -> dict:
    """Altman Z bankruptcy score for a ticker.

    Returns the Z (or Z'' for non-manufacturing) score, its components,
    and the zone: safe / grey / distress. `variant` optionally forces
    'original' or 'revised'; default auto-selects by sector.
    """
    from agent.risk_scanner import compute_altman_z

    return _run("altman_z", ticker,
                lambda m, ctx: compute_altman_z(m, variant=variant))


@mcp.tool()
def piotroski_f(ticker: str) -> dict:
    """Piotroski F-Score (0-9) for a ticker.

    Nine binary fundamental-quality checks across profitability,
    leverage/liquidity, and operating efficiency. Higher is stronger;
    <= 3 is a weak-fundamentals flag.
    """
    from agent.risk_scanner import compute_piotroski_f

    return _run("piotroski_f", ticker,
                lambda m, ctx: compute_piotroski_f(m))


@mcp.tool()
def beneish_m(ticker: str) -> dict:
    """Beneish M-Score earnings-manipulation screen for a ticker.

    M > -1.78 flags elevated probability of earnings manipulation.
    Returns the score and its eight component indices.
    """
    from agent.risk_scanner import compute_beneish_m

    return _run("beneish_m", ticker,
                lambda m, ctx: compute_beneish_m(m))


@mcp.tool()
def risk_flags(ticker: str) -> dict:
    """Composite risk assessment for a ticker.

    ARY QUANT's combined scan: fundamental risk (sector-relative
    z-scores + distress models), macro risk, and market risk, merged
    into per-axis levels and an overall rating.
    """
    from agent.risk_scanner import compute_risk_flags

    def _fn(m: Dict[str, Any], ctx: Dict[str, Any]):
        return compute_risk_flags(
            ticker.upper(), m, ctx.get("macro") or {},
            agent_risks=[], config=SimpleNamespace())

    return _run("risk_flags", ticker, _fn)


if __name__ == "__main__":
    if "--http" in sys.argv:
        print(f"ary-distress-risk MCP on http://127.0.0.1:{PORT}/mcp")
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio

# D:\Ary Fund\mcp_distress_risk.py
