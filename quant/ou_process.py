"""
Ornstein-Uhlenbeck mean-reversion fit.

The OU process models a quantity that reverts to a long-run mean:

    dX_t = theta * (mu - X_t) * dt + sigma * dW_t

Intuition:
    mu        = long-run mean (the level the series pulls toward)
    theta     = speed of reversion (higher -> reverts faster)
    sigma     = instantaneous volatility
    half_life = ln(2) / theta  (days to close half the gap to mu)

This is the canonical fit quants use to decide whether a spread or a
log-price is behaving like a stationary mean-reverter. On raw equity
log-prices OU will usually fit poorly (prices drift) but it's still
useful as a diagnostic: a weak theta and a huge half-life is a hint
that the series is closer to a random walk than a mean-reverter, which
agrees with Hurst ~ 0.5.

Estimation
----------
We use the discrete AR(1) regression. If

    X_{t+1} = a + b * X_t + epsilon_t

then continuous-time OU parameters are recovered (with dt = 1 day) as:

    theta = -ln(b)
    mu    = a / (1 - b)
    sigma = stdev(epsilon) * sqrt(-2 * ln(b) / (1 - b^2))

This closed-form MLE avoids any scipy / optimizer dependency.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _clean_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "theta": None,
        "mu": None,
        "sigma": None,
        "half_life": None,
        "is_mean_reverting": False,
        "r_squared": None,
        "n_obs": 0,
        "_series": {
            "fitted": pd.Series(dtype=float),
            "residuals": pd.Series(dtype=float),
            "mean_line": pd.Series(dtype=float),
            "upper_band": pd.Series(dtype=float),
            "lower_band": pd.Series(dtype=float),
        },
    }


def fit_ou_process(
    series: pd.Series,
    use_log: bool = True,
    band_sigmas: float = 2.0,
) -> Dict[str, Any]:
    """
    Fit an OU process to a price or spread series via AR(1) regression.

    Parameters
    ----------
    series : pd.Series
        Price series (or a pre-built spread / residual series).
    use_log : bool, default True
        If True, fit OU on log(price). Set False when passing a spread
        that is already in return-space or in log form.
    band_sigmas : float, default 2.0
        Width of the +/- sigma band around mu used for plotting. 2.0 is
        a conventional "roughly 95% envelope" for a Gaussian OU.

    Returns
    -------
    dict with:
        available        : bool
        reason           : str
        theta            : float - mean-reversion speed (per bar)
        mu               : float - long-run mean (in fitted space)
        sigma            : float - instantaneous volatility
        half_life        : float - bars to revert halfway; inf if random walk
        is_mean_reverting: bool  - theta > 0 and half_life finite & < n_obs
        r_squared        : float - AR(1) R^2
        n_obs            : int
        _series:
            fitted     : pd.Series - fitted X_{t+1} values
            residuals  : pd.Series - X_{t+1} - fitted
            mean_line  : pd.Series - constant mu (in ORIGINAL price space)
            upper_band : pd.Series - mu + band_sigmas * sigma_stationary
            lower_band : pd.Series - mu - band_sigmas * sigma_stationary
    """
    s = _clean_series(series)
    if len(s) < 30:
        return _empty_result(f"Not enough data (need >= 30 bars, got {len(s)}).")

    if use_log:
        if (s <= 0).any():
            return _empty_result("Non-positive values; cannot take log.")
        X = np.log(s.values)
    else:
        X = s.values.astype(float)

    X_t = X[:-1]
    X_t1 = X[1:]

    # AR(1): X_{t+1} = a + b * X_t + eps
    b_num = np.sum((X_t - X_t.mean()) * (X_t1 - X_t1.mean()))
    b_den = np.sum((X_t - X_t.mean()) ** 2)
    if b_den == 0:
        return _empty_result("Zero variance in lagged series.")
    b = b_num / b_den
    a = X_t1.mean() - b * X_t.mean()

    residuals = X_t1 - (a + b * X_t)
    resid_std = residuals.std(ddof=1)

    # R^2 of the AR(1)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((X_t1 - X_t1.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # OU parameter recovery (dt = 1)
    # b in (0, 1) -> mean-reverting;  b >= 1 -> random walk / explosive
    if b <= 0 or b >= 1:
        theta = 0.0
        half_life = float("inf")
        is_mr = False
    else:
        theta = float(-np.log(b))
        half_life = float(np.log(2) / theta) if theta > 0 else float("inf")
        is_mr = half_life < len(s)

    if 1 - b != 0:
        mu = float(a / (1 - b))
    else:
        mu = float(X_t1.mean())

    if 0 < b < 1:
        sigma = float(resid_std * np.sqrt(-2 * np.log(b) / (1 - b ** 2)))
    else:
        sigma = float(resid_std)

    # Stationary stdev of the OU process (used for the band around mu)
    # Var_inf = sigma^2 / (2 * theta)
    if theta > 0 and np.isfinite(theta):
        stationary_std = float(sigma / np.sqrt(2 * theta))
    else:
        stationary_std = float(resid_std)

    fitted = pd.Series(a + b * X_t, index=s.index[1:], name="fitted_X")
    residuals_s = pd.Series(residuals, index=s.index[1:], name="residuals")

    # Build mean line and bands in ORIGINAL price space if use_log,
    # so the UI can overlay them directly on the price chart.
    if use_log:
        mean_line = pd.Series(np.exp(mu), index=s.index, name="mu")
        upper = pd.Series(
            np.exp(mu + band_sigmas * stationary_std), index=s.index, name="upper"
        )
        lower = pd.Series(
            np.exp(mu - band_sigmas * stationary_std), index=s.index, name="lower"
        )
    else:
        mean_line = pd.Series(mu, index=s.index, name="mu")
        upper = pd.Series(mu + band_sigmas * stationary_std, index=s.index, name="upper")
        lower = pd.Series(mu - band_sigmas * stationary_std, index=s.index, name="lower")

    return {
        "available": True,
        "reason": "",
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "is_mean_reverting": bool(is_mr),
        "r_squared": float(r_squared),
        "n_obs": int(len(X_t)),
        "_series": {
            "fitted": fitted,
            "residuals": residuals_s,
            "mean_line": mean_line,
            "upper_band": upper,
            "lower_band": lower,
        },
    }
