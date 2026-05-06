"""
Girsanov theorem — change of measure for risk-neutral pricing.

Shows the SAME Brownian path reinterpreted under two probability measures:

    P  (real-world):   drift = mu      (estimated from historical returns)
    Q  (risk-neutral): drift = r       (risk-free rate)

The Girsanov kernel  theta = (mu - r) / sigma  defines the exponential
martingale that converts one measure into the other:

    dQ/dP = exp(-0.5 * theta^2 * T  -  theta * W_T)

Practical value
---------------
- Terminal distributions differ: P(S_T > K | P) != P(S_T > K | Q).
  This is exactly why Black-Scholes prices do NOT equal real-world
  expected payoffs.  The "risk premium" gap between the two measures
  is the Girsanov drift shift.
- Visualising both path ensembles side-by-side makes the abstract
  measure-change concrete and helps explain why hedged portfolios
  use Q while forecast models use P.

Reference
---------
quant-traderr-lab / Girsanov / Girsanov Pipeline.py
Math: GBM under each measure shares Brownian noise dW_t but replaces
      the drift (mu -> r), producing different trajectory fans and
      different terminal densities.

Design
------
Pure functions, structured dict returns, no I/O.
All randomness seeded for reproducibility in tests and screenshots.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ── helpers ──────────────────────────────────────────────────────────

def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── core simulation ─────────────────────────────────────────────────

def simulate_girsanov(
    prices: pd.Series,
    risk_free_rate: float = 0.05,
    horizon_days: int = 252,
    n_paths: int = 140,
    strike_pct: float = 1.10,
    n_terminal_samples: int = 80_000,
    random_state: int = 3,
) -> Dict[str, Any]:
    """
    Simulate price paths under both P and Q measures sharing the same
    Brownian noise, following Girsanov's theorem.

    Parameters
    ----------
    prices : pd.Series
        Historical price series. Used to estimate mu and sigma.
    risk_free_rate : float, default 0.05
        Annualized risk-free rate (the Q-measure drift).
    horizon_days : int, default 252
        Simulation horizon in trading days.
    n_paths : int, default 140
        Number of path realizations per measure (for the path cloud).
    strike_pct : float, default 1.10
        Strike as a fraction of spot (1.10 = 10% OTM call).
    n_terminal_samples : int, default 80_000
        Monte Carlo draws for the terminal density histograms.
    random_state : int, default 3
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        available : bool
        params : dict  (mu, r, sigma, theta, S0, K)
        time : np.ndarray  shape (N+1,)
        paths_P : np.ndarray  shape (n_paths, N+1)
        paths_Q : np.ndarray  shape (n_paths, N+1)
        expected_P : np.ndarray  shape (N+1,)
        expected_Q : np.ndarray  shape (N+1,)
        terminal_P : np.ndarray  shape (n_terminal_samples,)
        terminal_Q : np.ndarray  shape (n_terminal_samples,)
        itm_prob_P : float
        itm_prob_Q : float
        radon_nikodym_mean : float  (sanity check: should be ~1)
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars; got {len(prices)}.",
        }

    # ── estimate parameters from historical data ──
    log_ret = np.log(prices / prices.shift(1)).dropna().values
    daily_mu = float(log_ret.mean())
    daily_sigma = float(log_ret.std(ddof=1))

    mu = daily_mu * TRADING_DAYS
    sigma = daily_sigma * np.sqrt(TRADING_DAYS)
    r = risk_free_rate
    S0 = float(prices.iloc[-1])
    K = S0 * strike_pct

    if sigma < 1e-9:
        return {"available": False, "reason": "Volatility ~ 0; degenerate."}

    # Girsanov drift-shift parameter
    theta = (mu - r) / sigma

    T = horizon_days / TRADING_DAYS
    N = horizon_days
    dt = T / N

    rng = np.random.default_rng(random_state)
    t = np.linspace(0.0, T, N + 1)

    # ── shared Brownian increments ──
    dW = rng.standard_normal((n_paths, N)) * np.sqrt(dt)
    W = np.concatenate([np.zeros((n_paths, 1)),
                        np.cumsum(dW, axis=1)], axis=1)

    # Under P: S_t = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
    drift_p = (mu - 0.5 * sigma**2) * t[None, :]
    paths_P = S0 * np.exp(drift_p + sigma * W)

    # Under Q: S_t = S0 * exp((r - 0.5*sigma^2)*t + sigma*W_t)
    drift_q = (r - 0.5 * sigma**2) * t[None, :]
    paths_Q = S0 * np.exp(drift_q + sigma * W)

    # Expected value curves  E[S_t] = S0 * exp(drift * t)
    expected_P = S0 * np.exp(mu * t)
    expected_Q = S0 * np.exp(r * t)

    # ── large terminal samples for density plots ──
    rng2 = np.random.default_rng(random_state + 1)
    W_T_big = rng2.standard_normal(n_terminal_samples) * np.sqrt(T)

    S_T_P = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * W_T_big)
    S_T_Q = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T_big)

    itm_prob_P = float((S_T_P > K).mean())
    itm_prob_Q = float((S_T_Q > K).mean())

    # Radon-Nikodym density (sanity: E_P[dQ/dP] should ≈ 1)
    rn_density = np.exp(-0.5 * theta**2 * T - theta * W_T_big)
    rn_mean = float(rn_density.mean())

    return {
        "available": True,
        "params": {
            "mu": float(mu),
            "r": float(r),
            "sigma": float(sigma),
            "theta": float(theta),
            "S0": float(S0),
            "K": float(K),
            "T": float(T),
            "n_paths": int(n_paths),
            "horizon_days": int(horizon_days),
        },
        "time": t,
        "paths_P": paths_P,
        "paths_Q": paths_Q,
        "expected_P": expected_P,
        "expected_Q": expected_Q,
        "terminal_P": S_T_P,
        "terminal_Q": S_T_Q,
        "itm_prob_P": itm_prob_P,
        "itm_prob_Q": itm_prob_Q,
        "radon_nikodym_mean": rn_mean,
    }
