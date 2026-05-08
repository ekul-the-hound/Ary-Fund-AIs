"""
Sequential Monte Carlo — Bootstrap Particle Filter.

Estimates a HIDDEN state x_t from noisy observations y_{1:t} when
the dynamics are nonlinear and / or non-Gaussian.  Where the Kalman
filter would fail (e.g. quadratic observation → bimodal posterior),
particle filters succeed by representing the posterior as a weighted
empirical distribution:

    p(x_t | y_{1:t})  ≈  Σ_i  w_t^(i)  δ(x_t - x_t^(i))

Algorithm (bootstrap PF, Gordon 1993):
    1. PROPAGATE:    x_t^(i)  =  f(x_{t-1}^(i))  +  v_t,  v_t ~ N(0, Q)
    2. WEIGHT:       w_t^(i)  ∝  w_{t-1}^(i)  *  p(y_t | x_t^(i))
    3. NORMALISE:    Σ w_t^(i) = 1
    4. RESAMPLE if effective sample size  ESS = 1 / Σ (w^(i))^2  < N/2
       using SYSTEMATIC resampling.

This module ships TWO things:

A) ``run_particle_filter_benchmark`` — the canonical Gordon 1993
   nonlinear benchmark used in the reference repo.  Useful as a
   teaching / validation tool.

       x_{t+1}  =  0.5 x_t + 25 x_t / (1 + x_t^2) + 8 cos(1.2 t) + v_t
       y_t      =  x_t^2 / 20  +  w_t

   The quadratic observation creates a BIMODAL posterior — perfect
   showcase of why Gaussian (Kalman) filters break.

B) ``run_volatility_particle_filter`` — applies the same machinery
   to a real price series.  The hidden state is log-volatility:

       h_{t+1}  =  φ * h_t  +  η_t,    η_t ~ N(0, Q)
       y_t      =  exp(h_t)  *  ε_t,   ε_t ~ N(0, 1)

   This is a stochastic-volatility (SV) model.  The particle filter
   recovers a smoothed latent vol path together with credible bands.

Reference
---------
quant-traderr-lab / Sequential Monte Carlo / Sequential Monte Carlo Pipeline.py

Design
------
Pure functions, structured dict returns, no I/O.  No external deps
beyond NumPy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── helpers ────────────────────────────────────────────────────────

def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling (Kitagawa 1996). O(N), low variance."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # guard against floating point
    return np.searchsorted(cdf, positions)


# ── A) Gordon 1993 benchmark ───────────────────────────────────────

def _gordon_state(x: np.ndarray, t: int) -> np.ndarray:
    return 0.5 * x + 25.0 * x / (1.0 + x * x) + 8.0 * np.cos(1.2 * t)


def _gordon_obs(x: np.ndarray) -> np.ndarray:
    return x * x / 20.0


def run_particle_filter_benchmark(
    n_particles: int = 300,
    T: int = 100,
    Q: float = 10.0,
    R: float = 1.0,
    prior_var: float = 5.0,
    ess_threshold_frac: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run the bootstrap particle filter on the Gordon (1993) nonlinear benchmark.

    Returns
    -------
    dict with keys:
        available : bool
        truth : np.ndarray (T,)
        observations : np.ndarray (T,)
        particles : np.ndarray (T, N)
        weights : np.ndarray (T, N)
        posterior_mean : np.ndarray (T,)
        posterior_std : np.ndarray (T,)
        ess : np.ndarray (T,)
        resampled : np.ndarray (T,) of bool
        rmse : float
        params : dict
    """
    rng = np.random.default_rng(random_state)

    # 1. Simulate ground truth + observations
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = rng.normal(0, np.sqrt(prior_var))
    y[0] = _gordon_obs(x[0]) + rng.normal(0, np.sqrt(R))
    for t in range(1, T):
        x[t] = _gordon_state(x[t-1], t) + rng.normal(0, np.sqrt(Q))
        y[t] = _gordon_obs(x[t]) + rng.normal(0, np.sqrt(R))

    # 2. Particle filter
    rng2 = np.random.default_rng(random_state + 1)
    N = n_particles

    particles_hist = np.zeros((T, N))
    weights_hist = np.zeros((T, N))
    post_mean = np.zeros(T)
    post_std = np.zeros(T)
    ess_hist = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)

    particles = rng2.normal(0, np.sqrt(prior_var), N)
    weights = np.full(N, 1.0 / N)

    for t in range(T):
        if t > 0:
            particles = _gordon_state(particles, t) + \
                rng2.normal(0, np.sqrt(Q), N)

        # Importance weights (log-space stabilisation)
        y_pred = _gordon_obs(particles)
        log_w = -0.5 * (y[t] - y_pred) ** 2 / R
        log_w -= log_w.max()
        w = np.exp(log_w) * weights
        w /= w.sum() + 1e-15
        weights = w

        particles_hist[t] = particles
        weights_hist[t] = weights
        post_mean[t] = float(np.sum(weights * particles))
        post_std[t] = float(
            np.sqrt(np.sum(weights * (particles - post_mean[t]) ** 2))
        )
        ess_hist[t] = 1.0 / (np.sum(weights ** 2) + 1e-15)

        if ess_hist[t] < ess_threshold_frac * N:
            idx = _systematic_resample(weights, rng2)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)
            resampled[t] = True

    rmse = float(np.sqrt(np.mean((post_mean - x) ** 2)))

    return {
        "available": True,
        "truth": x,
        "observations": y,
        "particles": particles_hist,
        "weights": weights_hist,
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ess": ess_hist,
        "resampled": resampled,
        "rmse": rmse,
        "params": {
            "n_particles": int(N),
            "T": int(T),
            "Q": float(Q),
            "R": float(R),
            "prior_var": float(prior_var),
            "ess_threshold_frac": float(ess_threshold_frac),
        },
    }


# ── B) Stochastic-volatility particle filter on real prices ────────

def run_volatility_particle_filter(
    prices: pd.Series,
    n_particles: int = 500,
    phi: float = 0.95,
    Q: float = 0.05,
    mu: Optional[float] = None,
    ess_threshold_frac: float = 0.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Estimate latent log-volatility from a price series via SMC.

    Stochastic-volatility model (mean-reverting AR(1) on log-vol):
        h_{t+1}  =  μ (1 - φ)  +  φ * h_t  +  η_t,    η_t ~ N(0, Q)
        y_t      =  exp(h_t / 2) * ε_t  (mean-zero return),  ε_t ~ N(0, 1)

    Parameters
    ----------
    prices : pd.Series
    n_particles : int
    phi : float ∈ (0, 1)
        Persistence of log-vol process. 0.95 = highly persistent.
    Q : float
        Innovation variance of log-vol.
    mu : float, optional
        Unconditional mean of log-vol.  If None, set to the log
        empirical variance of returns (standard initialisation).
    ess_threshold_frac : float
    random_state : int

    Returns
    -------
    dict with keys:
        available : bool
        log_returns : np.ndarray
        latent_vol : np.ndarray   exp(posterior mean of h_t / 2), per-day vol
        latent_vol_low / latent_vol_high : np.ndarray  (5/95 percentile bands)
        posterior_mean : np.ndarray  (h_t)
        posterior_std : np.ndarray
        ess : np.ndarray
        resampled : np.ndarray
        params : dict
    """
    prices = _clean_prices(prices)
    if len(prices) < 60:
        return {
            "available": False,
            "reason": f"Need >= 60 bars; got {len(prices)}.",
        }

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    log_ret = log_ret - log_ret.mean()  # center to remove drift
    T = len(log_ret)
    N = n_particles

    rng = np.random.default_rng(random_state)

    # Empirical estimate of the unconditional log-vol level
    if mu is None:
        mu = float(np.log(np.var(log_ret) + 1e-12))

    # Initialise particles at the unconditional distribution of h:
    #     h ~ N(mu, Q / (1 - phi²))
    stationary_var = Q / max(1.0 - phi**2, 1e-3)
    particles = mu + rng.normal(0.0, np.sqrt(stationary_var), N)
    weights = np.full(N, 1.0 / N)

    post_mean = np.zeros(T)
    post_std = np.zeros(T)
    post_low = np.zeros(T)
    post_high = np.zeros(T)
    ess_hist = np.zeros(T)
    resampled = np.zeros(T, dtype=bool)

    for t in range(T):
        if t > 0:
            # AR(1) transition with reversion to mu
            particles = (
                mu * (1.0 - phi)
                + phi * particles
                + rng.normal(0.0, np.sqrt(Q), N)
            )

        # Likelihood:  y_t | h_t  ~  N(0, exp(h_t))
        # log p(y_t | h_t) = -0.5 [ h_t + y_t^2 / exp(h_t) + log(2π) ]
        log_w = -0.5 * (
            particles
            + log_ret[t] ** 2 / np.exp(np.clip(particles, -30, 30))
        )
        log_w -= log_w.max()
        w = np.exp(log_w) * weights
        w /= w.sum() + 1e-15
        weights = w

        post_mean[t] = float(np.sum(weights * particles))
        post_std[t] = float(
            np.sqrt(np.sum(weights * (particles - post_mean[t]) ** 2))
        )

        # Weighted percentiles for credible band
        order = np.argsort(particles)
        cum_w = np.cumsum(weights[order])
        p5_idx = int(np.searchsorted(cum_w, 0.05))
        p95_idx = int(np.searchsorted(cum_w, 0.95))
        p5_idx = min(max(p5_idx, 0), N - 1)
        p95_idx = min(max(p95_idx, 0), N - 1)
        post_low[t] = float(particles[order[p5_idx]])
        post_high[t] = float(particles[order[p95_idx]])

        ess_hist[t] = 1.0 / (np.sum(weights ** 2) + 1e-15)
        if ess_hist[t] < ess_threshold_frac * N:
            idx = _systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.full(N, 1.0 / N)
            resampled[t] = True

    # Convert log-vol to per-day vol  σ_t = exp(h_t / 2)
    latent_vol = np.exp(post_mean / 2.0)
    latent_vol_low = np.exp(post_low / 2.0)
    latent_vol_high = np.exp(post_high / 2.0)

    return {
        "available": True,
        "log_returns": log_ret,
        "latent_vol": latent_vol,
        "latent_vol_low": latent_vol_low,
        "latent_vol_high": latent_vol_high,
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ess": ess_hist,
        "resampled": resampled,
        "params": {
            "n_particles": int(N),
            "phi": float(phi),
            "Q": float(Q),
            "mu": float(mu),
            "T": int(T),
            "ess_threshold_frac": float(ess_threshold_frac),
        },
    }