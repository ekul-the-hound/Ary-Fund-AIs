"""
Geometric Brownian Motion — Monte Carlo price simulation.

Simulates forward price paths under the standard GBM assumption:

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Discrete-time solution (closed-form, exact, no time-stepping error):

    S_{t+1} = S_t * exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z )

where Z ~ N(0, 1).

Inputs are fit from the user's own price series so the simulation is
anchored on the ticker's actual empirical drift and volatility. The
drift is estimated from log returns; the tab UI can override it
(e.g. zero-drift fan for a pure "stress" chart).

Outputs include:
    - full paths array (for plotting the spaghetti fan)
    - terminal-price distribution summary (mean, std, percentiles, prob up)
    - per-day quantile bands (5/25/50/75/95) for a clean cone plot
    - VaR / ES on the terminal log return, in the same sign convention
      as ``var_es.py`` (positive number = loss)

Design notes
------------
- Pure NumPy; no scipy required.
- Default ``random_state=42`` for reproducibility in tests and screenshots.
- Defensive on short series and non-positive prices.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def estimate_gbm_params(prices: pd.Series) -> Dict[str, float]:
    """
    Estimate GBM drift and volatility from a price series.

    Returns annualized mu and sigma derived from daily log returns, so
    downstream simulation code can scale by dt = 1 / TRADING_DAYS.
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {"mu": 0.0, "sigma": 0.0, "n_obs": int(len(prices))}

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    daily_mu = float(log_ret.mean())
    daily_sigma = float(log_ret.std(ddof=1))
    return {
        "mu": daily_mu * TRADING_DAYS,
        "sigma": daily_sigma * np.sqrt(TRADING_DAYS),
        "n_obs": int(len(log_ret)),
    }


def simulate_gbm(
    spot: float,
    mu: float,
    sigma: float,
    horizon_days: int = 60,
    n_paths: int = 500,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run a GBM Monte Carlo simulation.

    Parameters
    ----------
    spot : float
        Current price (starting point S_0).
    mu : float
        Annualized drift (use 0.0 for a pure-volatility stress fan).
    sigma : float
        Annualized volatility.
    horizon_days : int, default 60
        Forecast horizon in trading days.
    n_paths : int, default 500
        Number of Monte Carlo paths. 500 is a nice UX default; 10k if
        you care about the tails.
    random_state : int | None, default 42
        Seed for reproducibility.

    Returns
    -------
    dict with:
        available         : bool
        reason            : str
        spot, mu, sigma, horizon_days, n_paths : (echoed inputs)
        terminal:
            mean, std, median
            p05, p25, p75, p95 : terminal price percentiles
            prob_up            : P(S_T > spot)
            expected_return    : mean(S_T)/spot - 1
            var_95             : 5% VaR on terminal log return (positive=loss)
            es_95              : 5% ES on terminal log return  (positive=loss)
        quantile_bands : dict of per-day arrays 'p05','p25','p50','p75','p95'
        _series:
            paths : np.ndarray, shape (n_paths, horizon_days + 1)
            days  : np.ndarray, shape (horizon_days + 1,)  values 0..H
    """
    if spot <= 0 or not np.isfinite(spot):
        return {
            "available": False,
            "reason": "Spot must be positive and finite.",
            "spot": spot, "mu": mu, "sigma": sigma,
            "horizon_days": horizon_days, "n_paths": n_paths,
            "terminal": {}, "quantile_bands": {},
            "_series": {"paths": np.array([]), "days": np.array([])},
        }
    if sigma < 0 or horizon_days < 1 or n_paths < 1:
        return {
            "available": False,
            "reason": "Invalid sigma / horizon / n_paths.",
            "spot": spot, "mu": mu, "sigma": sigma,
            "horizon_days": horizon_days, "n_paths": n_paths,
            "terminal": {}, "quantile_bands": {},
            "_series": {"paths": np.array([]), "days": np.array([])},
        }

    rng = np.random.default_rng(random_state)
    dt = 1.0 / TRADING_DAYS
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Shape: (n_paths, horizon_days). Exact GBM; no Euler error.
    Z = rng.standard_normal(size=(n_paths, horizon_days))
    log_increments = drift + diffusion * Z
    log_cum = np.cumsum(log_increments, axis=1)
    paths = np.zeros((n_paths, horizon_days + 1))
    paths[:, 0] = spot
    paths[:, 1:] = spot * np.exp(log_cum)

    terminal = paths[:, -1]
    # Log-return to match var_es.py sign convention (positive = loss)
    terminal_log_ret = np.log(terminal / spot)
    losses = -terminal_log_ret  # positive where terminal < spot

    q = np.quantile
    qbands = {
        "p05": q(paths, 0.05, axis=0),
        "p25": q(paths, 0.25, axis=0),
        "p50": q(paths, 0.50, axis=0),
        "p75": q(paths, 0.75, axis=0),
        "p95": q(paths, 0.95, axis=0),
    }

    var_95_losses = float(q(losses, 0.95))
    es_95_losses = (
        float(losses[losses >= var_95_losses].mean())
        if np.any(losses >= var_95_losses)
        else var_95_losses
    )

    return {
        "available": True,
        "reason": "",
        "spot": float(spot),
        "mu": float(mu),
        "sigma": float(sigma),
        "horizon_days": int(horizon_days),
        "n_paths": int(n_paths),
        "terminal": {
            "mean": float(terminal.mean()),
            "std": float(terminal.std(ddof=1)),
            "median": float(np.median(terminal)),
            "p05": float(q(terminal, 0.05)),
            "p25": float(q(terminal, 0.25)),
            "p75": float(q(terminal, 0.75)),
            "p95": float(q(terminal, 0.95)),
            "prob_up": float((terminal > spot).mean()),
            "expected_return": float(terminal.mean() / spot - 1.0),
            "var_95": var_95_losses,
            "es_95": es_95_losses,
        },
        "quantile_bands": qbands,
        "_series": {
            "paths": paths,
            "days": np.arange(horizon_days + 1),
        },
    }


def gbm_from_prices(
    prices: pd.Series,
    horizon_days: int = 60,
    n_paths: int = 500,
    drift_override: Optional[float] = None,
    random_state: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Convenience wrapper: fit GBM params from a price series, then simulate.

    Parameters
    ----------
    prices : pd.Series
        Historical price series. The last observation is used as spot.
    horizon_days, n_paths, random_state
        Forwarded to ``simulate_gbm``.
    drift_override : float | None
        If provided, override the estimated annual mu. Common choices:
            0.0             -> risk-neutral-ish / pure volatility stress
            risk_free_rate  -> Black-Scholes style drift

    Returns
    -------
    Same dict as ``simulate_gbm``, plus a top-level 'fit' block with the
    estimated params so the UI can show them side-by-side.
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars to fit GBM; got {len(prices)}.",
            "fit": {}, "terminal": {}, "quantile_bands": {},
            "_series": {"paths": np.array([]), "days": np.array([])},
        }

    fit = estimate_gbm_params(prices)
    spot = float(prices.iloc[-1])
    mu = fit["mu"] if drift_override is None else float(drift_override)

    sim = simulate_gbm(
        spot=spot,
        mu=mu,
        sigma=fit["sigma"],
        horizon_days=horizon_days,
        n_paths=n_paths,
        random_state=random_state,
    )
    sim["fit"] = {
        "estimated_mu": fit["mu"],
        "estimated_sigma": fit["sigma"],
        "n_obs": fit["n_obs"],
        "drift_used": mu,
        "drift_overridden": drift_override is not None,
    }
    return sim
