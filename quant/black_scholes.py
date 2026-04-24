"""
Black-Scholes option pricing, Greeks, and implied volatility.

Covers European options on non-dividend-paying or continuous-dividend-yield
underlyings:

    bs_price(S, K, T, r, sigma, option_type, q=0)
    bs_greeks(S, K, T, r, sigma, option_type, q=0)  -> dict of all 5 greeks
    implied_vol(price, S, K, T, r, option_type, q=0)

Sign & unit conventions
-----------------------
- S, K      : positive floats.
- T         : time to expiry in YEARS (e.g. 30 days = 30/365).
- r, q      : continuously compounded annual rates (decimal, 0.05 = 5%).
- sigma     : annual vol (decimal).
- theta     : per-year decay. Divide by 365 for per-calendar-day, or by 252
              for per-trading-day.
- vega      : change in price per +1.0 (absolute) in sigma. Divide by 100
              to get "per vol point" (the trader convention).
- rho       : change in price per +1.0 in r. Divide by 100 for "per bp * 100".

All computations are vectorized-friendly: pass numpy arrays for any of
S, K, T, r, sigma, q and you get arrays back.
"""

from __future__ import annotations

from typing import Dict, Union

import math
import numpy as np
from scipy import stats


Number = Union[float, np.ndarray]

_CALL = "call"
_PUT = "put"
_VALID_TYPES = (_CALL, _PUT)


# --- Helpers ------------------------------------------------------------------


def _validate_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot not in _VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return ot


def _d1_d2(S: Number, K: Number, T: Number, r: Number, sigma: Number, q: Number):
    """Black-Scholes d1 and d2 with continuous dividend yield q."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    sqrtT = np.sqrt(T)
    # guard against T=0 / sigma=0: those inputs will be handled by the price
    # function via intrinsic-value fallback
    denom = sigma * sqrtT
    # avoid division warnings; we'll mask later
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / denom
        d2 = d1 - denom
    return d1, d2, sqrtT


def _intrinsic(S: Number, K: Number, option_type: str) -> Number:
    if option_type == _CALL:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


# --- Pricing ------------------------------------------------------------------


def bs_price(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    option_type: str = "call",
    q: Number = 0.0,
) -> Number:
    """
    Black-Scholes-Merton price for a European option with continuous
    dividend yield q.

    At T=0 or sigma=0, returns the discounted intrinsic value (degenerate
    case handled explicitly to avoid NaNs).
    """
    ot = _validate_type(option_type)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    # Degenerate mask: T<=0 or sigma<=0 -> intrinsic of the FORWARD
    degenerate = (T_arr <= 0) | (sigma_arr <= 0)

    # Normal path
    d1, d2, _ = _d1_d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
    disc = np.exp(-r_arr * T_arr)
    div_disc = np.exp(-q_arr * T_arr)

    if ot == _CALL:
        price_normal = S_arr * div_disc * stats.norm.cdf(d1) - K_arr * disc * stats.norm.cdf(d2)
        intrinsic = np.maximum(S_arr * div_disc - K_arr * disc, 0.0)
    else:
        price_normal = K_arr * disc * stats.norm.cdf(-d2) - S_arr * div_disc * stats.norm.cdf(-d1)
        intrinsic = np.maximum(K_arr * disc - S_arr * div_disc, 0.0)

    out = np.where(degenerate, intrinsic, price_normal)

    if np.ndim(out) == 0:
        return float(out)
    return out


# --- Greeks -------------------------------------------------------------------


def bs_greeks(
    S: Number,
    K: Number,
    T: Number,
    r: Number,
    sigma: Number,
    option_type: str = "call",
    q: Number = 0.0,
) -> Dict[str, Number]:
    """
    All five first/second-order Black-Scholes Greeks.

    Returns
    -------
    dict with keys:
        price, delta, gamma, vega, theta, rho

    Conventions:
      vega  : per +1.0 in sigma (divide by 100 for per-vol-point)
      theta : per YEAR  (divide by 365 for per-calendar-day)
      rho   : per +1.0 in r (divide by 100 for per-100bp)
    """
    ot = _validate_type(option_type)

    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    q_arr = np.asarray(q, dtype=float)

    degenerate = (T_arr <= 0) | (sigma_arr <= 0)

    d1, d2, sqrtT = _d1_d2(S_arr, K_arr, T_arr, r_arr, sigma_arr, q_arr)
    disc = np.exp(-r_arr * T_arr)
    div_disc = np.exp(-q_arr * T_arr)
    phi_d1 = stats.norm.pdf(d1)  # N'(d1)

    price = bs_price(S_arr, K_arr, T_arr, r_arr, sigma_arr, ot, q_arr)

    if ot == _CALL:
        delta_n = div_disc * stats.norm.cdf(d1)
        theta_n = (
            -(S_arr * div_disc * phi_d1 * sigma_arr) / (2.0 * sqrtT)
            - r_arr * K_arr * disc * stats.norm.cdf(d2)
            + q_arr * S_arr * div_disc * stats.norm.cdf(d1)
        )
        rho_n = K_arr * T_arr * disc * stats.norm.cdf(d2)
    else:
        delta_n = -div_disc * stats.norm.cdf(-d1)
        theta_n = (
            -(S_arr * div_disc * phi_d1 * sigma_arr) / (2.0 * sqrtT)
            + r_arr * K_arr * disc * stats.norm.cdf(-d2)
            - q_arr * S_arr * div_disc * stats.norm.cdf(-d1)
        )
        rho_n = -K_arr * T_arr * disc * stats.norm.cdf(-d2)

    # gamma and vega are same formula for call and put
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_n = (div_disc * phi_d1) / (S_arr * sigma_arr * sqrtT)
    vega_n = S_arr * div_disc * phi_d1 * sqrtT

    # In degenerate regime (T<=0 or sigma<=0): delta = sign of intrinsic,
    # gamma/vega/theta/rho = 0. Use forward-adjusted intrinsic.
    if ot == _CALL:
        delta_deg = np.where(S_arr * div_disc > K_arr * disc, div_disc, 0.0)
    else:
        delta_deg = np.where(S_arr * div_disc < K_arr * disc, -div_disc, 0.0)
    zero = np.zeros_like(delta_deg)

    delta = np.where(degenerate, delta_deg, delta_n)
    gamma = np.where(degenerate, zero, gamma_n)
    vega = np.where(degenerate, zero, vega_n)
    theta = np.where(degenerate, zero, theta_n)
    rho = np.where(degenerate, zero, rho_n)

    # Scalar -> float
    def _scalarize(x):
        return float(x) if np.ndim(x) == 0 else x

    return {
        "price": _scalarize(price),
        "delta": _scalarize(delta),
        "gamma": _scalarize(gamma),
        "vega": _scalarize(vega),
        "theta": _scalarize(theta),
        "rho": _scalarize(rho),
    }


# --- Implied volatility -------------------------------------------------------


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
    sigma_init: float = 0.2,
) -> float:
    """
    Solve for Black-Scholes implied volatility from a market price.

    Uses Newton-Raphson on vega (fast, quadratic convergence), with a
    bisection fallback for pathological cases (vega collapses, deep ITM/OTM,
    price just outside no-arb bounds due to rounding).

    Parameters
    ----------
    price : float
        Market option price.
    S, K, T, r, q : floats
    option_type : "call" | "put"
    tol : float, default 1e-8
    max_iter : int, default 100
    sigma_init : float, default 0.20

    Returns
    -------
    float
        Implied volatility. Returns NaN if the price violates no-arbitrage
        bounds or the solver fails to converge.
    """
    ot = _validate_type(option_type)

    # Basic input sanity
    if not all(math.isfinite(x) for x in (price, S, K, T, r, q, sigma_init)):
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0 or price < 0:
        return float("nan")

    disc = math.exp(-r * T)
    div_disc = math.exp(-q * T)

    # No-arbitrage bounds
    if ot == _CALL:
        lower = max(S * div_disc - K * disc, 0.0)
        upper = S * div_disc
    else:
        lower = max(K * disc - S * div_disc, 0.0)
        upper = K * disc

    # Allow a tiny slack for numerical roundoff
    slack = 1e-10 * max(upper, 1.0)
    if price < lower - slack or price > upper + slack:
        return float("nan")
    if abs(price - lower) < 1e-12:
        # Deep OTM / at intrinsic; implied vol is effectively 0
        return 0.0

    # --- Newton-Raphson ---
    sigma = max(sigma_init, 1e-4)
    for _ in range(max_iter):
        g = bs_greeks(S, K, T, r, sigma, ot, q)
        diff = g["price"] - price
        if abs(diff) < tol:
            return float(sigma)
        v = g["vega"]
        if v < 1e-12:
            break  # vega collapsed; switch to bisection
        step = diff / v
        # dampen huge steps
        if step > 1.0:
            step = 1.0
        elif step < -1.0:
            step = -1.0
        sigma_new = sigma - step
        if sigma_new <= 1e-8:
            sigma_new = sigma / 2.0
        if abs(sigma_new - sigma) < tol:
            return float(sigma_new)
        sigma = sigma_new

    # --- Bisection fallback ---
    lo, hi = 1e-8, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(S, K, T, r, mid, ot, q)
        if abs(p_mid - price) < tol:
            return float(mid)
        if p_mid > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return float(0.5 * (lo + hi))

    return float("nan")
