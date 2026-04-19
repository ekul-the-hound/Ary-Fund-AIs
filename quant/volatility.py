"""
Volatility estimation and forecasting.

Conventions
-----------
- Returns are simple log returns: r_t = ln(P_t / P_{t-1}).
- "Daily" volatility = standard deviation of daily returns (unitless, decimal).
- Annualization factor defaults to 252 trading days.
- All functions accept a pd.Series indexed by date (or any monotonic index).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# --- Optional GARCH support ---------------------------------------------------
try:
    from arch import arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _ARCH_AVAILABLE = False


TRADING_DAYS = 252


# --- Internal helpers ---------------------------------------------------------


def _clean_prices(prices: pd.Series) -> pd.Series:
    """Drop NaNs, require positive prices, preserve order."""
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


# --- Public API ---------------------------------------------------------------


def compute_returns(prices: pd.Series, kind: str = "log") -> pd.Series:
    """
    Compute a clean return series from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series. NaNs and non-positive values are dropped.
    kind : {"log", "simple"}
        "log" -> ln(P_t / P_{t-1}); "simple" -> P_t / P_{t-1} - 1.

    Returns
    -------
    pd.Series
        Return series, one observation shorter than the cleaned input.
        Empty if fewer than 2 valid prices are available.
    """
    s = _clean_prices(prices)
    if len(s) < 2:
        return pd.Series(dtype=float)

    if kind == "log":
        r = np.log(s / s.shift(1))
    elif kind == "simple":
        r = s.pct_change()
    else:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")

    return r.dropna()


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling standard deviation of returns.

    Parameters
    ----------
    returns : pd.Series
    window : int, default 20
        Trailing window length. Must be >= 2.

    Returns
    -------
    pd.Series
        Rolling vol; leading values are NaN until the window fills.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    r = _clean_returns(returns)
    if r.empty:
        return pd.Series(dtype=float)
    return r.rolling(window=window, min_periods=window).std(ddof=1)


def realized_volatility(returns: pd.Series) -> float:
    """
    Sample standard deviation of the full return series (daily scale).
    Returns NaN if fewer than 2 observations.
    """
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1))


def annualize_volatility(
    daily_vol: float, periods_per_year: int = TRADING_DAYS
) -> float:
    """Scale a daily vol to annual by sqrt(periods_per_year)."""
    if daily_vol is None or not np.isfinite(daily_vol):
        return float("nan")
    return float(daily_vol * np.sqrt(periods_per_year))


def forecast_volatility(
    prices: pd.Series,
    method: str = "rolling",
    window: int = 20,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """
    Produce a structured volatility forecast from a price series.

    Parameters
    ----------
    prices : pd.Series
    method : {"rolling", "ewma", "garch"}
        - "rolling": trailing std over `window`.
        - "ewma":    exponentially-weighted std with span=window.
        - "garch":   GARCH(1,1) via the `arch` package. Falls back to
                     "rolling" if `arch` is not installed, and annotates the
                     fallback in the output.
    window : int
    periods_per_year : int

    Returns
    -------
    dict
        {
            "method": str,
            "latest_volatility": float | nan,   # daily scale
            "annualized_volatility": float | nan,
            "window": int,
            "n_obs": int,
            "note": str,
        }
    """
    returns = compute_returns(prices)
    n_obs = int(len(returns))
    note = ""
    resolved_method = method

    if n_obs < 2:
        return {
            "method": method,
            "latest_volatility": float("nan"),
            "annualized_volatility": float("nan"),
            "window": window,
            "n_obs": n_obs,
            "note": "Not enough data to estimate volatility.",
        }

    if method == "rolling":
        rv = rolling_volatility(returns, window=min(window, max(n_obs, 2)))
        latest = float(rv.dropna().iloc[-1]) if rv.dropna().size else realized_volatility(returns)
        if not rv.dropna().size:
            note = "Insufficient window; used full-sample realized volatility."

    elif method == "ewma":
        span = max(2, window)
        ew = returns.ewm(span=span, adjust=False).std()
        latest = float(ew.dropna().iloc[-1]) if ew.dropna().size else realized_volatility(returns)

    elif method == "garch":
        if not _ARCH_AVAILABLE:
            note = "arch package not installed; fell back to rolling volatility."
            resolved_method = "rolling"
            rv = rolling_volatility(returns, window=min(window, max(n_obs, 2)))
            latest = (
                float(rv.dropna().iloc[-1])
                if rv.dropna().size
                else realized_volatility(returns)
            )
        else:
            latest = _garch_latest_vol(returns)
            if latest is None or not np.isfinite(latest):
                note = "GARCH fit failed; fell back to realized volatility."
                resolved_method = "rolling"
                latest = realized_volatility(returns)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return {
        "method": resolved_method,
        "latest_volatility": float(latest) if np.isfinite(latest) else float("nan"),
        "annualized_volatility": annualize_volatility(latest, periods_per_year),
        "window": window,
        "n_obs": n_obs,
        "note": note,
    }


def garch_forecast(
    prices: pd.Series,
    horizon: int = 1,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """
    GARCH(1,1) one-step (or h-step) volatility forecast.

    If the `arch` package is not available, returns a structured dict with
    method='unavailable' rather than raising, so callers can degrade cleanly.
    """
    returns = compute_returns(prices)
    n_obs = int(len(returns))

    if not _ARCH_AVAILABLE:
        return {
            "method": "unavailable",
            "latest_volatility": float("nan"),
            "annualized_volatility": float("nan"),
            "horizon": horizon,
            "n_obs": n_obs,
            "note": "arch package not installed.",
        }

    if n_obs < 30:
        return {
            "method": "garch",
            "latest_volatility": float("nan"),
            "annualized_volatility": float("nan"),
            "horizon": horizon,
            "n_obs": n_obs,
            "note": "Too few observations for a stable GARCH fit (<30).",
        }

    try:
        # arch expects returns in percentage points for numerical stability
        am = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off")
        fc = res.forecast(horizon=horizon, reindex=False)
        # variance forecast for the final horizon step, back to decimal
        var_pct2 = float(fc.variance.values[-1, -1])
        latest = np.sqrt(var_pct2) / 100.0
        return {
            "method": "garch",
            "latest_volatility": float(latest),
            "annualized_volatility": annualize_volatility(latest, periods_per_year),
            "horizon": horizon,
            "n_obs": n_obs,
            "note": "",
        }
    except Exception as e:  # pragma: no cover - defensive
        return {
            "method": "garch",
            "latest_volatility": float("nan"),
            "annualized_volatility": float("nan"),
            "horizon": horizon,
            "n_obs": n_obs,
            "note": f"GARCH fit failed: {e!s}",
        }


# --- Internal: GARCH latest vol, for forecast_volatility ---------------------


def _garch_latest_vol(returns: pd.Series) -> Optional[float]:
    """Return the one-step GARCH vol forecast in decimal, or None on failure."""
    if not _ARCH_AVAILABLE or len(returns) < 30:
        return None
    try:
        am = arch_model(returns * 100, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off")
        fc = res.forecast(horizon=1, reindex=False)
        var_pct2 = float(fc.variance.values[-1, -1])
        return float(np.sqrt(var_pct2) / 100.0)
    except Exception:  # pragma: no cover - defensive
        return None
