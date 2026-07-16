"""
openbb_widgets.py
=================

OpenBB Workspace custom backend for ARY QUANT: exposes the same 14 risk
tools as the three MCP servers, but as Workspace WIDGETS (tables you can
drag onto dashboards).

Categories (how they'll group in the Workspace widget menu):
    ARY Risk / Distress   — Altman Z, Piotroski F, Beneish M, Risk Flags
    ARY Risk / Market     — VaR/ES, Volatility, HMM Regime, Monte Carlo,
                            Kelly, HRP, RMT, MST
    ARY Risk / Macro      — Global Risk Pulse, Recession Probability

RUN (project root, hedgefund_ai venv active — openbb-platform-api is
already installed; it came with openbb):

    openbb-api --app openbb_widgets.py --port 6900

Then in OpenBB Workspace (pro.openbb.co):
    left sidebar -> Connections -> Add Data / Custom Backend ->
    name it "ARY QUANT", URL http://127.0.0.1:6900 -> Test -> Add.
    (If Test fails with a mixed-content error, that's the HTTPS issue —
    the Cloudflare tunnel or a permissive browser solves it.)

The launcher auto-generates widgets.json from these endpoints: function
parameters become widget inputs, docstring first lines become widget
descriptions, and list-of-dict responses render as tables.

DESIGN NOTES
    * Every endpoint returns table rows (list[dict]) — the most robust
      Workspace widget type. Errors raise HTTPException so the widget
      shows a readable error state instead of silently blanking.
    * Multi-ticker inputs are comma-separated strings ("AAPL,MSFT,NVDA")
      because widget text inputs are strings.
    * Ollama is NOT required — all 14 tools are pure quant/DB reads.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # keys for FRED-backed tools
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

app = FastAPI(
    title="ARY QUANT Risk Backend",
    description="ARY QUANT risk tools as OpenBB Workspace widgets.",
    version="0.1.0",
)

_PORTFOLIO_DB = "data/portfolio.db"
_MARKET_DB = "data/hedgefund.db"
_md = None  # lazy MarketData singleton


# ======================================================================
# Shared helpers (mirrors the MCP servers')
# ======================================================================
def _clean(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):
        try:
            return _clean(v.item())
        except Exception:  # noqa: BLE001
            return str(v)
    if isinstance(v, (dict, list, tuple, set)):
        return str(v)  # nested containers stringify inside a cell
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _rows(d: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """Flatten a (possibly nested) dict into Field/Value table rows."""
    out: List[Dict[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v and len(out) < 200:
            out.extend(_rows(v, prefix=f"{key}."))
        else:
            out.append({"Field": key, "Value": _clean(v)})
    return out[:200]


def _fail(msg: str) -> None:
    raise HTTPException(status_code=400, detail=msg)


def _market():
    global _md
    if _md is None:
        from data.market_data import MarketData
        _md = MarketData(db_path=_MARKET_DB)
    return _md


def _close_series(ticker: str, period: str = "1y"):
    df = _market().get_prices(ticker.upper().strip(), period=period)
    if df is None or getattr(df, "empty", True):
        return None
    for col in ("Close", "close", "Adj Close", "adj_close"):
        if col in df.columns:
            s = df[col].dropna()
            return s if len(s) >= 30 else None
    return None


def _close_panel(tickers: List[str], period: str = "1y"):
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


def _split(tickers: str) -> List[str]:
    out = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(out) < 2:
        _fail("provide at least two comma-separated tickers, "
              "e.g. AAPL,MSFT,NVDA")
    return out


def _metrics_for(ticker: str) -> Dict[str, Any]:
    from data.pipeline import build_agent_context
    ctx = build_agent_context(ticker.upper().strip(), _PORTFOLIO_DB,
                              SimpleNamespace())
    for key in ("metrics", "key_metrics", "fundamental_metrics"):
        m = ctx.get(key)
        if isinstance(m, dict) and m:
            return {"metrics": m, "macro": ctx.get("macro") or {}}
    _fail(f"no fundamentals available for {ticker!r} — is it in the "
          "universe and backfilled?")
    return {}  # unreachable


_DISTRESS = {"widget_config": {"category": "ARY Risk",
                               "subCategory": "Distress"}}
_MARKETRISK = {"widget_config": {"category": "ARY Risk",
                                 "subCategory": "Market"}}
_MACRO = {"widget_config": {"category": "ARY Risk", "subCategory": "Macro"}}


# ======================================================================
# Distress & accounting risk
# ======================================================================
@app.get("/altman_z", openapi_extra=_DISTRESS)
def altman_z(ticker: str, variant: Optional[str] = None) -> list:
    """Altman Z bankruptcy score: value, components, and zone."""
    from agent.risk_scanner import compute_altman_z
    res = compute_altman_z(_metrics_for(ticker)["metrics"], variant=variant)
    if res is None:
        _fail(f"insufficient inputs for Altman Z on {ticker!r}")
    return _rows({"ticker": ticker.upper(), **res})


@app.get("/piotroski_f", openapi_extra=_DISTRESS)
def piotroski_f(ticker: str) -> list:
    """Piotroski F-Score (0-9): fundamental quality checks."""
    from agent.risk_scanner import compute_piotroski_f
    res = compute_piotroski_f(_metrics_for(ticker)["metrics"])
    if res is None:
        _fail(f"insufficient inputs for Piotroski F on {ticker!r}")
    return _rows({"ticker": ticker.upper(), **res})


@app.get("/beneish_m", openapi_extra=_DISTRESS)
def beneish_m(ticker: str) -> list:
    """Beneish M-Score: earnings-manipulation screen."""
    from agent.risk_scanner import compute_beneish_m
    res = compute_beneish_m(_metrics_for(ticker)["metrics"])
    if res is None:
        _fail(f"insufficient inputs for Beneish M on {ticker!r}")
    return _rows({"ticker": ticker.upper(), **res})


@app.get("/risk_flags", openapi_extra=_DISTRESS)
def risk_flags(ticker: str) -> list:
    """ARY QUANT composite risk: fundamental, macro, and market levels."""
    from agent.risk_scanner import compute_risk_flags
    bundle = _metrics_for(ticker)
    res = compute_risk_flags(ticker.upper(), bundle["metrics"],
                             bundle["macro"], agent_risks=[],
                             config=SimpleNamespace())
    return _rows({"ticker": ticker.upper(), **(res or {})})


# ======================================================================
# Market & portfolio risk
# ======================================================================
@app.get("/var_es", openapi_extra=_MARKETRISK)
def var_es(ticker: str, confidence: float = 0.95,
           method: str = "historical", period: str = "1y") -> list:
    """Value-at-Risk and Expected Shortfall for a ticker's daily returns."""
    from quant.var_es import var_es_report
    prices = _close_series(ticker, period)
    if prices is None:
        _fail(f"insufficient price history for {ticker!r}")
    rep = var_es_report(_returns(prices), confidence=confidence,
                        method=method)
    return _rows({"ticker": ticker.upper(), "period": period, **rep})


@app.get("/volatility", openapi_extra=_MARKETRISK)
def volatility(ticker: str, method: str = "rolling", window: int = 20,
               period: str = "1y") -> list:
    """Realized volatility plus a rolling/EWMA/GARCH forecast."""
    from quant.volatility import (annualize_volatility, forecast_volatility,
                                  realized_volatility)
    prices = _close_series(ticker, period)
    if prices is None:
        _fail(f"insufficient price history for {ticker!r}")
    rets = _returns(prices)
    return _rows({
        "ticker": ticker.upper(),
        "realized_vol_annualized": annualize_volatility(
            realized_volatility(rets)),
        "forecast": forecast_volatility(prices, method=method,
                                        window=window),
    })


@app.get("/hmm_regime", openapi_extra=_MARKETRISK)
def hmm_regime(ticker: str, n_states: int = 2, period: str = "2y") -> list:
    """Gaussian-HMM market regime: state parameters and current state."""
    from quant.regime_hmm import fit_hmm_regime
    prices = _close_series(ticker, period)
    if prices is None:
        _fail(f"insufficient price history for {ticker!r}")
    res = fit_hmm_regime(prices, n_states=n_states)
    for k in ("states", "state_sequence", "labels"):
        v = res.get(k)
        if isinstance(v, (list, tuple)) and len(v) > 20:
            res[k] = f"[{len(v)} days, last 5: {list(v[-5:])}]"
    return _rows({"ticker": ticker.upper(), **res})


@app.get("/monte_carlo", openapi_extra=_MARKETRISK)
def monte_carlo(ticker: str, horizon_days: int = 252,
                n_simulations: int = 2000, period: str = "2y") -> list:
    """Bootstrap Monte Carlo outcome distribution for a ticker."""
    from quant.monte_carlo import simulate_monte_carlo
    prices = _close_series(ticker, period)
    if prices is None:
        _fail(f"insufficient price history for {ticker!r}")
    res = simulate_monte_carlo(prices, horizon_days=horizon_days,
                               n_simulations=n_simulations)
    return _rows({"ticker": ticker.upper(), **res})


@app.get("/kelly", openapi_extra=_MARKETRISK)
def kelly(ticker: str, risk_free_rate: float = 0.0,
          fractional: float = 0.5, period: str = "2y") -> list:
    """Kelly position fraction from fitted return moments (half-Kelly default)."""
    from quant.kelly import kelly_continuous
    prices = _close_series(ticker, period)
    if prices is None:
        _fail(f"insufficient price history for {ticker!r}")
    rets = _returns(prices)
    mu = float(rets.mean()) * 252.0
    var = float(rets.var()) * 252.0
    return _rows({
        "ticker": ticker.upper(),
        "annualized_mean_return": mu,
        "annualized_variance": var,
        "fractional": fractional,
        "kelly_fraction": kelly_continuous(mu, var,
                                           risk_free_rate=risk_free_rate,
                                           fractional=fractional),
    })


@app.get("/hrp", openapi_extra=_MARKETRISK)
def hrp(tickers: str = "AAPL,MSFT,NVDA", period: str = "1y") -> list:
    """Hierarchical Risk Parity weights over comma-separated tickers."""
    from quant.hrp import hrp_from_prices
    panel = _close_panel(_split(tickers), period)
    if panel is None:
        _fail("need >= 2 tickers with sufficient history")
    res = hrp_from_prices(panel)
    weights = res.get("weights")
    if isinstance(weights, dict):
        return [{"Ticker": t, "Weight": _clean(w)}
                for t, w in sorted(weights.items(),
                                   key=lambda kv: -float(kv[1]))]
    return _rows(res)


@app.get("/rmt", openapi_extra=_MARKETRISK)
def rmt(tickers: str = "AAPL,MSFT,NVDA,AMZN,GOOGL",
        period: str = "1y") -> list:
    """RMT-cleaned correlation summary: signal vs noise eigenstructure."""
    from quant.rmt import rmt_from_prices
    panel = _close_panel(_split(tickers), period)
    if panel is None:
        _fail("need >= 2 tickers with sufficient history")
    res = rmt_from_prices(panel)
    res = {k: v for k, v in res.items()
           if not (hasattr(v, "shape") and getattr(v, "ndim", 0) >= 2)}
    return _rows(res)


@app.get("/mst", openapi_extra=_MARKETRISK)
def mst(tickers: str = "AAPL,MSFT,NVDA,AMZN,GOOGL",
        period: str = "1y") -> list:
    """Minimum Spanning Tree edges: the book's correlation structure."""
    from quant.mst import compute_mst
    panel = _close_panel(_split(tickers), period)
    if panel is None:
        _fail("need >= 2 tickers with sufficient history")
    res = compute_mst(panel)
    edges = res.get("edges")
    if isinstance(edges, list) and edges and isinstance(edges[0], (list, tuple)):
        return [{"From": e[0], "To": e[1],
                 "Distance": _clean(e[2]) if len(e) > 2 else None}
                for e in edges[:100]]
    if isinstance(edges, list) and edges and isinstance(edges[0], dict):
        return [{k: _clean(v) for k, v in e.items()} for e in edges[:100]]
    return _rows(res)


# ======================================================================
# Macro risk
# ======================================================================
@app.get("/global_risk_pulse", openapi_extra=_MACRO)
def global_risk_pulse(tickers: str = "") -> list:
    """ARY QUANT's composite market-wide risk pulse (ticker list optional)."""
    from data.global_risk_pulse import recompute_global_risk_pulse
    universe = ([t.strip().upper() for t in tickers.split(",") if t.strip()]
                or None)
    res = recompute_global_risk_pulse(universe=universe,
                                      db_path=_MARKET_DB, persist=False)
    return _rows(res or {})


@app.get("/recession_probability", openapi_extra=_MACRO)
def recession_probability() -> list:
    """FRED smoothed US recession probability, correctly normalized."""
    from data.macro_data import MacroData
    return _rows(MacroData().get_recession_probability() or {})


@app.get("/")
def root() -> dict:
    return {"backend": "ARY QUANT Risk", "widgets": 14, "status": "ok"}

# D:\Ary Fund\openbb_widgets.py
# Launch: openbb-api --app openbb_widgets.py --port 6900
