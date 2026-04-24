"""
Hurst exponent — persistence / mean-reversion classifier.

The Hurst exponent H measures long-memory in a time series:

    H < 0.5   -> mean-reverting (anti-persistent)
    H = 0.5   -> random walk / Brownian motion
    H > 0.5   -> trending (persistent, long memory)

Most liquid equities hover near 0.5 over long windows. Values
meaningfully away from 0.5 can be informative on shorter horizons and
are useful for deciding between momentum and mean-reversion overlays.

Implementation
--------------
Rescaled-range (R/S) analysis on log returns. This is the classical
estimator and needs no external packages. Also returns a rolling Hurst
series so the UI can plot how persistence evolves over time.

Caveats (worth surfacing in the UI):
    - R/S is biased upward on short samples. Results with < 100 bars
      should be read with skepticism.
    - Returns with strong heteroskedasticity (GARCH effects) can fake
      persistence. Consider differencing or volatility-adjusting first.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# Window sizes used for the log-log regression. A geometric spread
# between min_lag and max_lag is standard practice.
_DEFAULT_MIN_LAG = 10
_DEFAULT_MAX_LAG = 100


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def _rs_statistic(series: np.ndarray) -> float:
    """Rescaled range for a single chunk. Returns NaN if undefined."""
    n = len(series)
    if n < 2:
        return np.nan
    mean = series.mean()
    deviations = series - mean
    cumulative = np.cumsum(deviations)
    r = cumulative.max() - cumulative.min()
    s = series.std(ddof=1)
    if s == 0 or not np.isfinite(s):
        return np.nan
    return r / s


def hurst_exponent(
    prices: pd.Series,
    min_lag: int = _DEFAULT_MIN_LAG,
    max_lag: int = _DEFAULT_MAX_LAG,
    num_lags: int = 20,
) -> Dict[str, Any]:
    """
    Estimate the Hurst exponent of a price series via R/S analysis.

    Parameters
    ----------
    prices : pd.Series
        Price series. Internally converted to log returns.
    min_lag, max_lag : int
        Range of chunk sizes for the R/S scan. Logarithmically spaced
        lags are generated between these bounds.
    num_lags : int, default 20
        How many chunk sizes to evaluate.

    Returns
    -------
    dict with:
        available     : bool
        reason        : str - populated when available=False
        hurst         : float | None
        interpretation: str - "mean-reverting" | "random walk" | "trending"
        n_returns     : int
        _series:
            lags     : np.ndarray - chunk sizes used
            rs       : np.ndarray - mean R/S at each chunk size
            fit_line : dict with 'x', 'y' for log-log regression plot
    """
    prices = _clean_prices(prices)
    if len(prices) < max(50, max_lag + 10):
        return {
            "available": False,
            "reason": (
                f"Not enough data for Hurst (need >= {max(50, max_lag + 10)} "
                f"bars, got {len(prices)})."
            ),
            "hurst": None,
            "interpretation": None,
            "n_returns": len(prices),
            "_series": {"lags": np.array([]), "rs": np.array([]), "fit_line": {}},
        }

    log_returns = np.log(prices / prices.shift(1)).dropna().values
    n = len(log_returns)

    # Cap max_lag so we always have >= 2 non-overlapping chunks.
    effective_max = min(max_lag, n // 2)
    if effective_max <= min_lag:
        return {
            "available": False,
            "reason": "Insufficient data to span min_lag -> max_lag.",
            "hurst": None,
            "interpretation": None,
            "n_returns": n,
            "_series": {"lags": np.array([]), "rs": np.array([]), "fit_line": {}},
        }

    # Logarithmically spaced lags
    lags = np.unique(
        np.logspace(
            np.log10(min_lag), np.log10(effective_max), num=num_lags
        ).astype(int)
    )
    lags = lags[lags >= 2]

    rs_values = []
    for lag in lags:
        num_chunks = n // lag
        if num_chunks < 1:
            rs_values.append(np.nan)
            continue
        rs_chunk = [
            _rs_statistic(log_returns[i * lag : (i + 1) * lag])
            for i in range(num_chunks)
        ]
        rs_chunk = [x for x in rs_chunk if np.isfinite(x)]
        rs_values.append(np.mean(rs_chunk) if rs_chunk else np.nan)

    rs_arr = np.array(rs_values, dtype=float)
    mask = np.isfinite(rs_arr) & (rs_arr > 0)
    if mask.sum() < 3:
        return {
            "available": False,
            "reason": "Too few valid R/S points to fit.",
            "hurst": None,
            "interpretation": None,
            "n_returns": n,
            "_series": {"lags": lags, "rs": rs_arr, "fit_line": {}},
        }

    # Linear regression in log-log space: log(R/S) = H * log(lag) + c
    log_lags = np.log(lags[mask])
    log_rs = np.log(rs_arr[mask])
    slope, intercept = np.polyfit(log_lags, log_rs, 1)
    hurst = float(slope)

    if hurst < 0.45:
        interp = "mean-reverting"
    elif hurst > 0.55:
        interp = "trending"
    else:
        interp = "random walk"

    fit_y = np.exp(intercept + slope * np.log(lags))
    fit_line = {"x": lags.astype(float), "y": fit_y}

    return {
        "available": True,
        "reason": "",
        "hurst": hurst,
        "interpretation": interp,
        "n_returns": n,
        "_series": {"lags": lags.astype(float), "rs": rs_arr, "fit_line": fit_line},
    }


def rolling_hurst(
    prices: pd.Series,
    window: int = 120,
    step: int = 5,
    min_lag: int = _DEFAULT_MIN_LAG,
    max_lag: int = 60,
) -> pd.Series:
    """
    Rolling Hurst exponent for visualising how persistence evolves.

    Parameters
    ----------
    prices : pd.Series
    window : int, default 120
        Rolling window size in bars. Hurst on very short windows is noisy;
        120 bars is a reasonable floor for daily data.
    step : int, default 5
        Stride between windows (in bars). A stride of 5 roughly
        corresponds to one estimate per trading week — keeps the chart
        readable.
    min_lag, max_lag : int
        Forwarded to the R/S scan within each window.

    Returns
    -------
    pd.Series of Hurst values indexed by the *end* date of each window.
    NaN where the window has too few valid points.
    """
    prices = _clean_prices(prices)
    if len(prices) < window + 1:
        return pd.Series(dtype=float)

    idx = prices.index
    results = {}
    for end in range(window, len(prices) + 1, step):
        chunk = prices.iloc[end - window : end]
        out = hurst_exponent(chunk, min_lag=min_lag, max_lag=max_lag)
        results[idx[end - 1]] = out["hurst"] if out["available"] else np.nan

    return pd.Series(results, name="hurst").sort_index()
