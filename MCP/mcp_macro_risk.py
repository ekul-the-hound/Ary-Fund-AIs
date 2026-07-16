"""
mcp_macro_risk.py
=================

MCP server #3 of 3: MACRO RISK.
Exposes ARY QUANT's market-wide risk gauges to MCP clients.

Tools:
    global_risk_pulse()       — ARY QUANT's composite market-wide risk
                                score over the universe/book
    recession_probability()   — FRED smoothed US recession probability
                                (RECPROUSM156N), correctly normalized

Notes:
    * recession_probability uses your FRED_API_KEY from .env.
    * global_risk_pulse computes over the price panel in hedgefund.db;
      first call can take a while on a large universe. It does NOT
      persist by default when called through MCP.
    * The FRED series reports PERCENTAGE POINTS (1.82 == 1.82%). Your
      MacroData layer already normalizes this — the historical 182%
      bug — so values here are the corrected ones.

INSTALL (once): pip install "mcp[cli]" -c constraints_openbb.txt
RUN:   python mcp_macro_risk.py          # stdio
       python mcp_macro_risk.py --http   # http://127.0.0.1:8803/mcp
REGISTER (Claude Code):
    claude mcp add ary-macro-risk -- python mcp_macro_risk.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

PORT = 8803
mcp = FastMCP("ary-macro-risk", host="127.0.0.1", port=PORT)

_MARKET_DB = "data/hedgefund.db"


def _jsonable(obj: Any, _depth: int = 0) -> Any:
    if _depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return None if isinstance(obj, float) and not math.isfinite(obj) else obj
    if hasattr(obj, "item"):
        try:
            return _jsonable(obj.item(), _depth + 1)
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "to_dict") and not isinstance(obj, dict):
        try:
            return _jsonable(obj.to_dict(), _depth + 1)
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


@mcp.tool()
def global_risk_pulse(tickers: Optional[List[str]] = None) -> dict:
    """ARY QUANT's composite market-wide risk pulse.

    Aggregates cross-sectional risk over the universe (or an explicit
    ticker list) into a single market-risk gauge with its components.
    First call on a large universe can take noticeably long — pass a
    ticker list (e.g. the book) for a fast, focused pulse.
    """
    try:
        from data.global_risk_pulse import recompute_global_risk_pulse
        # load .env for any keyed data the pulse touches
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:  # noqa: BLE001
            pass
        res = recompute_global_risk_pulse(
            universe=[t.upper().strip() for t in tickers] if tickers else None,
            db_path=_MARKET_DB,
            persist=False,
        )
        return _jsonable(res)
    except Exception as e:  # noqa: BLE001
        return {"error": f"global_risk_pulse failed: {e}"}


@mcp.tool()
def recession_probability() -> dict:
    """FRED smoothed US recession probability (RECPROUSM156N).

    Returns the latest probability with context from ARY QUANT's macro
    layer (which normalizes FRED's percentage-point convention — the
    raw series says 1.82 meaning 1.82%).
    """
    try:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:  # noqa: BLE001
            pass
        from data.macro_data import MacroData
        md = MacroData()
        return _jsonable(md.get_recession_probability())
    except Exception as e:  # noqa: BLE001
        return {"error": f"recession_probability failed: {e}"}


if __name__ == "__main__":
    if "--http" in sys.argv:
        print(f"ary-macro-risk MCP on http://127.0.0.1:{PORT}/mcp")
        mcp.run(transport="streamable-http")
    else:
        mcp.run()

# D:\Ary Fund\mcp_macro_risk.py
