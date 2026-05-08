"""
Omori Law — power-law decay of post-crash volatility aftershocks.

Originally from seismology (Omori 1894), the law describes how the
RATE of aftershocks decays after a main earthquake.  Applied to
markets, it captures how volatility "events" (large absolute returns)
cluster and decay following a crash:

    n(t)  =  K * t^(-p)

    n(t) : event rate at time t (events / day)
    K    : amplitude (depends on threshold and crash severity)
    p    : decay exponent  (typical equity values: p ≈ 0.7 - 1.3)

Practical reading of p:
    p > 1.0  → fast recovery, volatility normalises quickly
    p ≈ 1.0  → classical Omori, persistent but decaying clustering
    p < 1.0  → slow recovery, long-memory volatility (regime damage)

Methodology (from quant-traderr-lab / Omori Law / Omori Pipeline.py):
    1. Identify a CRASH_DATE.
    2. Define multiple |return| thresholds (q ∈ {0.5%, 1%, 1.5%, 2%, 3%}).
    3. Bin trading days post-crash on a LOG scale (handles fast-then-slow).
    4. For each threshold, compute event rate per bin.
    5. Fit  n(t) = K * t^(-p)  via curve_fit on the log-log axes.

Crash detection (this module's addition)
----------------------------------------
The reference repo asks the user to specify the crash date.  We add
an automatic detector:  the day with the largest single-day |return|
in the analysis window, OR the day of maximum drawdown trough.

Reference
---------
quant-traderr-lab / Omori Law / Omori Pipeline.py

Design
------
Pure functions, structured dict returns, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


_DEFAULT_THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.030]


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── crash detection ────────────────────────────────────────────────

def detect_crash_date(
    prices: pd.Series,
    method: str = "max_drawdown",
) -> pd.Timestamp:
    """
    Detect the crash date in a price series.

    Parameters
    ----------
    prices : pd.Series with DatetimeIndex
    method : str
        "max_drawdown" → date of the deepest drawdown trough
        "worst_return" → date of the largest single-day loss
    """
    prices = _clean_prices(prices)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)

    if method == "worst_return":
        rets = prices.pct_change()
        return rets.idxmin()

    # max_drawdown — date of the lowest point relative to running peak
    running_peak = prices.cummax()
    drawdown = (prices - running_peak) / running_peak
    return drawdown.idxmin()


# ── core analysis ──────────────────────────────────────────────────

def compute_omori(
    prices: pd.Series,
    crash_date: Optional[Union[str, pd.Timestamp]] = None,
    thresholds: Optional[List[float]] = None,
    crash_detection: str = "max_drawdown",
    fit_threshold_index: int = 1,
    n_bins: int = 25,
) -> Dict[str, Any]:
    """
    Fit the Omori power law to volatility aftershocks following a crash.

    Parameters
    ----------
    prices : pd.Series
        Historical price series with DatetimeIndex.
    crash_date : str or pd.Timestamp, optional
        Date of the main shock. If None, auto-detected.
    thresholds : list of float, optional
        |return| thresholds (e.g. 0.01 = 1% absolute daily move).
        Default: [0.005, 0.010, 0.015, 0.020, 0.030].
    crash_detection : str
        "max_drawdown" or "worst_return" if crash_date is None.
    fit_threshold_index : int
        Which threshold to use for the power-law fit (default 1 = 1%).
    n_bins : int
        Number of log-spaced bins for rate calculation.

    Returns
    -------
    dict with keys:
        available : bool
        crash_date : pd.Timestamp
        crash_return : float            return on crash day
        thresholds : list[float]
        rates_data : dict[float -> dict]
            each entry: {bin_centers, event_rates, n_events_total}
        fit : dict
            {K, p, threshold_used, n_points, R2}
        aftershocks : pd.DataFrame
            cols: [date, return, abs_return, t_days]
        decay_label : str   "fast" / "classical" / "slow"
    """
    if thresholds is None:
        thresholds = list(_DEFAULT_THRESHOLDS)

    prices = _clean_prices(prices)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    if len(prices) < 60:
        return {
            "available": False,
            "reason": f"Need >= 60 bars; got {len(prices)}.",
        }

    returns = prices.pct_change().dropna()

    # Crash date
    if crash_date is None:
        crash_dt = detect_crash_date(prices, method=crash_detection)
    else:
        crash_dt = pd.to_datetime(crash_date)

    if crash_dt not in returns.index:
        # Snap to nearest available trading day
        idx = returns.index.get_indexer([crash_dt], method="nearest")[0]
        crash_dt = returns.index[idx]

    crash_return = float(returns.loc[crash_dt])

    # Aftershocks: trading days strictly after crash
    after_mask = returns.index > crash_dt
    aftershocks = pd.DataFrame({
        "date": returns.index[after_mask],
        "return": returns[after_mask].values,
        "abs_return": np.abs(returns[after_mask].values),
    })
    if aftershocks.empty:
        return {
            "available": False,
            "reason": "No data found after the crash date.",
        }
    aftershocks["t_days"] = np.arange(1, len(aftershocks) + 1)

    t_max = int(aftershocks["t_days"].max())

    # Log-spaced bin edges (capture fast-then-slow decay)
    bin_edges = np.unique(
        np.logspace(0, np.log10(max(t_max, 2)), n_bins).astype(int)
    )

    rates_data: Dict[float, Dict[str, Any]] = {}
    for q in thresholds:
        events = aftershocks[aftershocks["abs_return"] > q]
        bin_centers = []
        event_rates = []
        for i in range(len(bin_edges) - 1):
            t0, t1 = bin_edges[i], bin_edges[i + 1]
            dt = t1 - t0
            if dt <= 0:
                continue
            n_ev = int(((events["t_days"] >= t0) &
                        (events["t_days"] < t1)).sum())
            rate = n_ev / dt
            if rate > 0:
                bin_centers.append(float(np.sqrt(t0 * t1)))
                event_rates.append(float(rate))
        rates_data[q] = {
            "bin_centers": np.array(bin_centers),
            "event_rates": np.array(event_rates),
            "n_events_total": int((aftershocks["abs_return"] > q).sum()),
        }

    # Fit n(t) = K * t^(-p)  on the chosen threshold (in log space)
    fit_q = thresholds[
        min(max(0, fit_threshold_index), len(thresholds) - 1)
    ]
    fit_data = rates_data[fit_q]
    tc = fit_data["bin_centers"]
    rate = fit_data["event_rates"]

    fit_block: Dict[str, Any] = {
        "K": 0.0, "p": 0.0,
        "threshold_used": fit_q,
        "n_points": len(tc),
        "R2": 0.0,
        "fit_succeeded": False,
    }

    if len(tc) >= 3:
        try:
            from scipy.optimize import curve_fit

            def power_law(t, K, p):
                return K * np.power(t, -p)

            popt, _ = curve_fit(
                power_law, tc, rate, p0=[1.0, 0.7], maxfev=5000,
            )
            K_fit, p_fit = float(popt[0]), float(popt[1])

            # R^2 on log-log scale (more stable for power laws)
            log_y = np.log(rate)
            log_y_pred = np.log(power_law(tc, K_fit, p_fit))
            ss_res = np.sum((log_y - log_y_pred) ** 2)
            ss_tot = np.sum((log_y - log_y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fit_block.update({
                "K": K_fit, "p": p_fit, "R2": float(r2),
                "fit_succeeded": True,
            })
        except Exception:
            pass

    # Decay regime label
    p_val = fit_block["p"]
    if not fit_block["fit_succeeded"]:
        decay_label = "unknown"
    elif p_val > 1.0:
        decay_label = "fast_recovery"
    elif p_val > 0.7:
        decay_label = "classical_omori"
    else:
        decay_label = "slow_persistent"

    return {
        "available": True,
        "crash_date": crash_dt,
        "crash_return": crash_return,
        "thresholds": list(thresholds),
        "rates_data": rates_data,
        "fit": fit_block,
        "aftershocks": aftershocks.reset_index(drop=True),
        "decay_label": decay_label,
        "params": {
            "n_bins": int(n_bins),
            "fit_threshold_index": int(fit_threshold_index),
            "n_aftershock_days": int(len(aftershocks)),
        },
    }
