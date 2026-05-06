"""
Monte Carlo portfolio simulation — bootstrap resampling.

Unlike the parametric GBM module (which assumes log-normal returns),
this engine resamples ACTUAL historical returns to build forward
projections.  This preserves the empirical distribution — fat tails,
skew, and any serial structure in the sample — rather than imposing
Gaussian assumptions.

    Eq_t = Eq_0 * prod(1 + r_t)

where each r_t is drawn (with replacement) from the historical return
vector.

Outputs:
    - Full paths array for plotting
    - Mean / median / percentile paths
    - Terminal distribution statistics
    - VaR and ES at the 95% level
    - Probability of loss

Reference
---------
quant-traderr-lab / Monte Carlo / Monte Carlo Pipeline.py
Uses bootstrap resampling (not parametric GBM) for the simulation engine.

Design
------
Pure functions, structured dict returns, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def simulate_monte_carlo(
    prices: pd.Series,
    horizon_days: int = 252,
    n_simulations: int = 5_000,
    start_capital: float = 10_000.0,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Bootstrap Monte Carlo simulation from historical price data.

    Parameters
    ----------
    prices : pd.Series
        Historical prices. Daily returns are computed internally.
    horizon_days : int, default 252
        Number of trading days to project forward.
    n_simulations : int, default 5_000
        Number of simulation paths.
    start_capital : float, default 10_000
        Starting portfolio value.
    random_state : int | None, default 42
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        available : bool
        params : dict
        paths : np.ndarray  shape (horizon_days + 1, n_simulations)
        mean_path : np.ndarray
        median_path : np.ndarray
        upper_95 : np.ndarray    (95th percentile — best case)
        lower_05 : np.ndarray    (5th percentile — worst case / VaR)
        terminal : dict with mean, median, std, p05, p95, prob_loss,
                   expected_return, var_95, es_95
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars; got {len(prices)}.",
        }

    returns = prices.pct_change().dropna().values

    rng = np.random.default_rng(random_state)

    # Bootstrap resampling: draw random historical returns
    random_idx = rng.integers(0, len(returns),
                              size=(horizon_days, n_simulations))
    sim_returns = returns[random_idx]

    # Cumulative product to build equity curves
    sim_paths = start_capital * np.cumprod(1 + sim_returns, axis=0)

    # Prepend start capital (day 0)
    day0 = np.full((1, n_simulations), start_capital)
    sim_paths = np.vstack([day0, sim_paths])

    # Summary statistics per day
    mean_path = np.mean(sim_paths, axis=1)
    median_path = np.median(sim_paths, axis=1)
    upper_95 = np.percentile(sim_paths, 95, axis=1)
    lower_05 = np.percentile(sim_paths, 5, axis=1)

    # Terminal distribution
    final = sim_paths[-1, :]
    losses = start_capital - final
    losses_positive = losses[losses > 0]

    var_95 = float(np.percentile(losses, 95)) if len(losses) > 0 else 0.0
    es_95 = (float(losses_positive.mean())
             if len(losses_positive) > 0 else 0.0)

    return {
        "available": True,
        "params": {
            "horizon_days": int(horizon_days),
            "n_simulations": int(n_simulations),
            "start_capital": float(start_capital),
            "n_historical_returns": int(len(returns)),
            "historical_mean_return": float(returns.mean()),
            "historical_vol": float(returns.std(ddof=1)),
        },
        "paths": sim_paths,
        "mean_path": mean_path,
        "median_path": median_path,
        "upper_95": upper_95,
        "lower_05": lower_05,
        "days": np.arange(horizon_days + 1),
        "terminal": {
            "mean": float(final.mean()),
            "median": float(np.median(final)),
            "std": float(final.std(ddof=1)),
            "p05": float(np.percentile(final, 5)),
            "p25": float(np.percentile(final, 25)),
            "p75": float(np.percentile(final, 75)),
            "p95": float(np.percentile(final, 95)),
            "prob_loss": float((final < start_capital).mean()),
            "expected_return": float(final.mean() / start_capital - 1.0),
            "var_95": var_95,
            "es_95": es_95,
        },
        "_series": {
            "terminal_values": final,
        },
    }
