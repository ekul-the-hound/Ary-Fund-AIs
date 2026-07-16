"""
mcp_market_risk.py
==================

MCP server #2 of 3: MARKET & PORTFOLIO RISK.
Exposes ARY QUANT's price-based quant risk models to MCP clients.

Tools (single-ticker):
    var_es(ticker, confidence, method)   — Value-at-Risk + Expected Shortfall
    volatility(ticker, method)           — realized + forecast volatility
    hmm_regime(ticker, n_states)         — Gaussian-HMM market regime
    monte_carlo_cone(ticker, ...)        — bootstrap MC outcome distribution
    kelly_sizing(ticker, ...)            — Kelly fraction from fitted returns

Tools (multi-ticker / book-level):
    hrp_weights(tickers)                 — hierarchical risk parity weights
    rmt_correlation(tickers)             — RMT-cleaned correlation summary
    mst_structure(tickers)               — minimum-spanning-tree edges

Prices come from MarketData's cache in hedgefund.db (yfinance-backed),
so repeated calls are cheap. Tools return compact JSON summaries, not
raw paths/matrices.

INSTALL (once): pip install "mcp[cli]" -c constraints_openbb.txt
RUN:   python mcp_market_risk.py          # stdio
       python mcp_market_risk.py --http   # http://127.0.0.1:8802/mcp
REGISTER (Claude Code):
    claude mcp add ary-market-risk -- python mcp_market_risk.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

PORT = 8802
mcp = FastMCP("ary-market-risk", host="127.0.0.1", port=PORT)

_MARKET_DB = "data/hedgefund.db"
_md = None  # lazy MarketData singleton


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
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
    if hasattr(obj, "to_dict") and not isinstance(obj, dict):  # Series/DF
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


def _market():
    global _md
    if _md is None:
        from data.market_data import MarketData  # lazy: heavy import
        _md = MarketData(db_path=_MARKET_DB)
    return _md


def _close_series(ticker: str, period: str = "1y"):
    """Close-price Series for a ticker, or None."""
    df = _market().get_prices(ticker.upper().strip(), period=period)
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("Close", "close", "Adj Close", "adj_close"):
        if col in df.columns:
            s = df[col].dropna()
            return s if len(s) >= 30 else None
    return None


def _close_panel(tickers: List[str], period: str = "1y"):
    """DataFrame of close prices, one column per ticker; None if < 2 usable."""
    import pandas as pd

    cols = {}
    for t in tickers:
        s = _close_series(t, period)
        if s is not None:
            cols[t.upper().strip()] = s
    if len(cols) < 2:
        return None
    return pd.DataFrame(cols).dropna()


def _returns(prices):
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna()


# ----------------------------------------------------------------------
# Single-ticker tools
# ----------------------------------------------------------------------
@mcp.tool()
def var_es(ticker: str, confidence: float = 0.95,
           method: str = "historical", period: str = "1y") -> dict:
    """Value-at-Risk and Expected Shortfall for a ticker's daily returns.

    method: historical | parametric | monte_carlo. Returns the headline
    VaR by the chosen method plus historical ES (tail-conditional mean).
    """
    try:
        from quant.var_es import var_es_report
        prices = _close_series(ticker, period)
        if prices is None:
            return {"error": f"insufficient price history for {ticker!r}"}
        rep = var_es_report(_returns(prices), confidence=confidence,
                            method=method)
        return _jsonable({"ticker": ticker.upper(), "period": period, **rep})
    except Exception as e:  # noqa: BLE001
        return {"error": f"var_es failed for {ticker!r}: {e}"}


@mcp.tool()
def volatility(ticker: str, method: str = "rolling",
               window: int = 20, period: str = "1y") -> dict:
    """Volatility snapshot + forecast for a ticker.

    method: rolling | ewma | garch. Returns realized (annualized) vol
    and the structured forecast from ARY QUANT's volatility module.
    """
    try:
        from quant.volatility import (annualize_volatility,
                                      forecast_volatility,
                                      realized_volatility)
        prices = _close_series(ticker, period)
        if prices is None:
            return {"error": f"insufficient price history for {ticker!r}"}
        rets = _returns(prices)
        out = {
            "ticker": ticker.upper(),
            "realized_vol_annualized": annualize_volatility(
                realized_volatility(rets)),
            "forecast": forecast_volatility(prices, method=method,
                                            window=window),
        }
        return _jsonable(out)
    except Exception as e:  # noqa: BLE001
        return {"error": f"volatility failed for {ticker!r}: {e}"}


@mcp.tool()
def hmm_regime(ticker: str, n_states: int = 2, period: str = "2y") -> dict:
    """Gaussian-HMM market-regime fit on a ticker's log returns.

    Returns state parameters (mean/vol per regime), the current state,
    and recent state occupancy — the 'what regime are we in' readout.
    """
    try:
        from quant.regime_hmm import fit_hmm_regime
        prices = _close_series(ticker, period)
        if prices is None:
            return {"error": f"insufficient price history for {ticker!r}"}
        res = fit_hmm_regime(prices, n_states=n_states)
        # Compact: replace any long per-day state array with a tail sample.
        for k in ("states", "state_sequence", "labels"):
            v = res.get(k)
            if isinstance(v, (list, tuple)) and len(v) > 20:
                res[k] = {"last_20": list(v[-20:]), "n_days": len(v)}
        return _jsonable({"ticker": ticker.upper(), **res})
    except Exception as e:  # noqa: BLE001
        return {"error": f"hmm_regime failed for {ticker!r}: {e}"}


@mcp.tool()
def monte_carlo_cone(ticker: str, horizon_days: int = 252,
                     n_simulations: int = 2000, period: str = "2y") -> dict:
    """Bootstrap Monte Carlo outcome distribution for a ticker.

    Resamples historical daily returns over the horizon and summarizes
    the terminal distribution (percentiles, loss probability). Downside
    view of 'where could this go'.
    """
    try:
        from quant.monte_carlo import simulate_monte_carlo
        prices = _close_series(ticker, period)
        if prices is None:
            return {"error": f"insufficient price history for {ticker!r}"}
        res = simulate_monte_carlo(prices, horizon_days=horizon_days,
                                   n_simulations=n_simulations)
        return _jsonable({"ticker": ticker.upper(), **res})
    except Exception as e:  # noqa: BLE001
        return {"error": f"monte_carlo_cone failed for {ticker!r}: {e}"}


@mcp.tool()
def kelly_sizing(ticker: str, risk_free_rate: float = 0.0,
                 fractional: float = 0.5, period: str = "2y") -> dict:
    """Kelly position fraction for a ticker from fitted return moments.

    Estimates annualized mean/variance from history and returns the
    continuous Kelly fraction, scaled by `fractional` (default half-
    Kelly — full Kelly is aggressive; this is deliberate).
    """
    try:
        from quant.kelly import kelly_continuous
        prices = _close_series(ticker, period)
        if prices is None:
            return {"error": f"insufficient price history for {ticker!r}"}
        rets = _returns(prices)
        mu = float(rets.mean()) * 252.0
        var = float(rets.var()) * 252.0
        frac = kelly_continuous(mu, var, risk_free_rate=risk_free_rate,
                                fractional=fractional)
        return _jsonable({
            "ticker": ticker.upper(),
            "annualized_mean_return": mu,
            "annualized_variance": var,
            "fractional": fractional,
            "kelly_fraction": frac,
            "note": "estimated from historical moments; garbage-in applies",
        })
    except Exception as e:  # noqa: BLE001
        return {"error": f"kelly_sizing failed for {ticker!r}: {e}"}


# ----------------------------------------------------------------------
# Multi-ticker / book-level tools
# ----------------------------------------------------------------------
@mcp.tool()
def hrp_weights(tickers: List[str], period: str = "1y") -> dict:
    """Hierarchical Risk Parity weights over a list of tickers.

    Clusters assets by correlation and allocates inverse-variance within
    the hierarchy. Give it the book (or a candidate book) — 3+ names.
    """
    try:
        from quant.hrp import hrp_from_prices
        panel = _close_panel(tickers, period)
        if panel is None:
            return {"error": "need >= 2 tickers with sufficient history"}
        return _jsonable(hrp_from_prices(panel))
    except Exception as e:  # noqa: BLE001
        return {"error": f"hrp_weights failed: {e}"}


@mcp.tool()
def rmt_correlation(tickers: List[str], period: str = "1y") -> dict:
    """RMT-cleaned correlation analysis over a set of tickers.

    Filters correlation-matrix noise via Marchenko-Pastur and reports
    the signal eigenstructure — how much real co-movement exists vs.
    noise. Compact summary, not the full matrix.
    """
    try:
        from quant.rmt import rmt_from_prices
        panel = _close_panel(tickers, period)
        if panel is None:
            return {"error": "need >= 2 tickers with sufficient history"}
        res = rmt_from_prices(panel)
        # Drop any full-matrix payloads; keep scalars/eigen summaries.
        res = {k: v for k, v in res.items()
               if not (hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2)}
        return _jsonable(res)
    except Exception as e:  # noqa: BLE001
        return {"error": f"rmt_correlation failed: {e}"}


@mcp.tool()
def mst_structure(tickers: List[str], period: str = "1y") -> dict:
    """Minimum Spanning Tree of a ticker set's correlation structure.

    Returns the tree's edges (who is most tightly linked to whom) —
    the concentration/contagion map of a book or sector.
    """
    try:
        from quant.mst import compute_mst
        panel = _close_panel(tickers, period)
        if panel is None:
            return {"error": "need >= 2 tickers with sufficient history"}
        return _jsonable(compute_mst(panel))
    except Exception as e:  # noqa: BLE001
        return {"error": f"mst_structure failed: {e}"}


if __name__ == "__main__":
    if "--http" in sys.argv:
        print(f"ary-market-risk MCP on http://127.0.0.1:{PORT}/mcp")
        mcp.run(transport="streamable-http")
    else:
        mcp.run()

# D:\Ary Fund\mcp_market_risk.py
