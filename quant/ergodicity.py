"""
Ergodicity Economics — ensemble average vs. time average.

Core insight (Ole Peters, 2019):
    For multiplicative dynamics, the ENSEMBLE average (across N agents)
    diverges from the TIME average (single agent over T steps).

    Given a coin-flip gamble:   x(t+1) = x(t) * r(t)
    where r(t) ∈ {UP_MULT, DOWN_MULT} with equal probability:

    - The EXPECTED growth factor  E[r] = (UP + DOWN) / 2  can be > 1
      (positive expected value).
    - But the TIME-AVERAGE growth rate  <log(r)> = (log(UP) + log(DOWN))/2
      is often < 0  (negative geometric mean → median agent loses).

    The ensemble average converges to E[r]^T  as N → infinity, but NO
    single agent experiences that trajectory.  This is the ergodicity
    gap: ensemble ≠ time average for non-ergodic (multiplicative) dynamics.

Application to finance:
    Stock returns are multiplicative.  An investment that looks good "on
    average" (ensemble) may be ruinous for an individual investor (time).
    This has direct implications for:
        - Position sizing (Kelly criterion resolves this by maximizing
          the TIME-average growth rate)
        - Risk of ruin in leveraged strategies
        - Why diversification matters mechanically, not just statistically

Reference
---------
quant-traderr-lab / Ergo / Ergo Pipeline.py
Simulates multiplicative random walks for N ∈ {1, 100, 10_000, 1_000_000}
and shows ensemble averages on linear + log scale.

This module also computes the ergodicity gap from REAL historical returns,
connecting the abstract theory to the user's actual data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── theoretical simulation ──────────────────────────────────────────

def simulate_ergodicity(
    T: int = 50,
    up_mult: float = 1.5,
    down_mult: float = 0.6,
    x0: float = 1.0,
    n_values: Optional[List[int]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Simulate multiplicative random walks for varying ensemble sizes N
    and return ensemble-average trajectories.

    Parameters
    ----------
    T : int
        Number of time steps.
    up_mult, down_mult : float
        Win / loss multipliers (equal probability).
    x0 : float
        Starting wealth.
    n_values : list of int
        Ensemble sizes to simulate. Default [1, 100, 10_000, 1_000_000].
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        ensemble_averages : dict  {N: np.ndarray of shape (T+1,)}
        expected_growth : float   E[r] = (up + down) / 2
        geometric_growth : float  exp(E[log(r)]) — time-average growth
        ergodicity_gap : float    log difference between ensemble & time avg
        T : int
        up_mult, down_mult : float
    """
    if n_values is None:
        n_values = [1, 100, 10_000, 1_000_000]

    rng = np.random.default_rng(random_state)

    results: Dict[int, np.ndarray] = {}

    for N in n_values:
        flips = rng.choice([up_mult, down_mult], size=(N, T))
        cum_prod = np.cumprod(flips, axis=1)
        paths = np.hstack([np.full((N, 1), x0), x0 * cum_prod])
        ensemble_avg = paths.mean(axis=0)
        results[N] = ensemble_avg

    # Theoretical quantities
    expected_growth = (up_mult + down_mult) / 2.0
    log_growth = (np.log(up_mult) + np.log(down_mult)) / 2.0
    geometric_growth = np.exp(log_growth)

    # The gap: ensemble grows at E[r]^t, time-average at exp(E[log r])^t
    ergodicity_gap = np.log(expected_growth) - log_growth

    return {
        "available": True,
        "ensemble_averages": results,
        "expected_growth": float(expected_growth),
        "geometric_growth": float(geometric_growth),
        "ergodicity_gap": float(ergodicity_gap),
        "T": T,
        "up_mult": float(up_mult),
        "down_mult": float(down_mult),
        "x0": float(x0),
    }


# ── empirical ergodicity analysis from real returns ─────────────────

def analyze_ergodicity_from_prices(
    prices: pd.Series,
    window: int = 252,
) -> Dict[str, Any]:
    """
    Compute the ergodicity gap from real historical returns.

    The ensemble (arithmetic) average return overestimates the growth
    experienced by a single buy-and-hold investor.  The gap between
    E[r] and exp(E[log(r)]) quantifies this effect.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.
    window : int, default 252
        Rolling window for the time-varying gap calculation.

    Returns
    -------
    dict with keys:
        available : bool
        arithmetic_mean : float    E[r]
        geometric_mean : float     exp(E[log(1+r)]) - 1
        ergodicity_gap : float     arithmetic - geometric
        volatility_drag : float    ≈ 0.5 * sigma^2  (Jensen's inequality)
        rolling_gap : pd.Series    time-varying ergodicity gap
    """
    prices = _clean_prices(prices)
    if len(prices) < 60:
        return {
            "available": False,
            "reason": f"Need >= 60 bars; got {len(prices)}.",
        }

    simple_ret = prices.pct_change().dropna()
    log_ret = np.log(1 + simple_ret)

    arith_mean = float(simple_ret.mean())
    geom_mean = float(np.exp(log_ret.mean()) - 1)
    gap = arith_mean - geom_mean
    vol = float(simple_ret.std(ddof=1))
    vol_drag = 0.5 * vol**2

    # Rolling ergodicity gap
    rolling_arith = simple_ret.rolling(window).mean()
    rolling_log = log_ret.rolling(window).mean()
    rolling_geom = np.exp(rolling_log) - 1
    rolling_gap = (rolling_arith - rolling_geom).dropna()

    return {
        "available": True,
        "arithmetic_mean": arith_mean,
        "geometric_mean": geom_mean,
        "ergodicity_gap": gap,
        "volatility_drag": vol_drag,
        "daily_volatility": vol,
        "rolling_gap": rolling_gap,
        "n_obs": int(len(simple_ret)),
    }
