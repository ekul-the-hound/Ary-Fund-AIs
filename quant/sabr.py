"""
SABR — Stochastic Alpha Beta Rho volatility model.

The industry standard for interest-rate and FX option pricing.
Captures the volatility SMILE and SKEW that Black-Scholes misses.

Dynamics (under the forward measure):

    dF      =  σ * F^β  *  dW_1
    dσ      =  α * σ    *  dW_2
    dW_1 dW_2  =  ρ * dt

Parameters (this is what makes SABR a 4-D family):
    α  : initial vol-of-vol               (controls overall smile curvature)
    β  : CEV exponent ∈ [0, 1]            (0 = Bachelier, 1 = log-normal)
    ρ  : correlation between F and σ      (negative ρ → equity-style skew)
    F₀ : forward price                    (anchors the surface)

Hagan et al. (2002) gave an asymptotic expansion that yields a
closed-form Black-Scholes implied volatility, avoiding any
Monte Carlo:

    σ_imp(K, T)  =  (α / D)  *  (z / χ(z))  *  (1 + correction * T)

where  D, z, χ(z)  depend on F, K, β, ρ, and α.

Practical risk applications
---------------------------
- Build an implied-vol surface for any underlying with at least one
  observed ATM vol (calibration via the ATM backbone).
- Visualise smile dynamics: vary ρ to see skew flip, vary α to see
  smile widen/contract.
- Stress test option books: shift the surface and re-price.

Reference
---------
quant-traderr-lab / SABR / SABR Pipeline.py
The reference repo's SABR Pipeline contains a partially-incorrect z
formula in places — this module uses the canonical Hagan (2002)
formulation throughout.

Design
------
Pure functions, structured dict returns, no I/O. Optional ATM
calibration from a user-supplied ATM vol observation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ── canonical Hagan (2002) implied vol ──────────────────────────────

def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: Optional[float] = None,
) -> float:
    """
    Hagan (2002) lognormal implied volatility under SABR.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike.
    T : float
        Time to expiry (years).
    alpha : float
        Initial vol of forward (σ₀).
    beta : float
        CEV exponent ∈ [0, 1].
    rho : float
        Correlation ∈ (-1, 1).
    nu : float, optional
        Vol-of-vol.  If None, defaults to alpha (single-vol-param SABR
        used in the reference repo).

    Returns
    -------
    Black-Scholes-equivalent implied volatility (annualised).
    """
    if nu is None:
        nu = alpha
    eps = 1e-12

    if K <= 0 or T <= 0 or alpha <= 0:
        return 0.0

    one_minus_beta = 1.0 - beta

    # ATM case — closed-form simplification
    if abs(F - K) < 1e-8:
        F_b = F ** one_minus_beta
        atm = alpha / max(F_b, eps)
        corr = (
            1.0
            + (
                (one_minus_beta ** 2 / 24.0)
                * (alpha ** 2) / max(F ** (2 * one_minus_beta), eps)
                + 0.25 * rho * beta * nu * alpha / max(F_b, eps)
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            ) * T
        )
        return max(atm * corr, 1e-6)

    log_FK = np.log(F / K)
    FK_avg = (F * K) ** (one_minus_beta / 2.0)

    # Canonical Hagan z and χ(z)
    z = (nu / alpha) * FK_avg * log_FK
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    chi_num = sqrt_term + z - rho
    chi_den = 1.0 - rho
    chi = np.log(max(chi_num, eps) / max(chi_den, eps))

    if abs(chi) < 1e-10:
        z_over_chi = 1.0
    else:
        z_over_chi = z / chi

    # Pre-factor
    denom = FK_avg * (
        1.0
        + (one_minus_beta ** 2 / 24.0) * log_FK ** 2
        + (one_minus_beta ** 4 / 1920.0) * log_FK ** 4
    )

    # Time correction
    corr = (
        1.0
        + (
            (one_minus_beta ** 2 / 24.0)
            * (alpha ** 2) / max(FK_avg ** 2, eps)
            + 0.25 * rho * beta * nu * alpha / max(FK_avg, eps)
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        ) * T
    )

    sigma = (alpha / max(denom, eps)) * z_over_chi * corr
    return max(float(sigma), 1e-6)


# ── full surface ────────────────────────────────────────────────────

def build_sabr_surface(
    F: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: Optional[float] = None,
    n_strikes: int = 60,
    n_maturities: int = 40,
    moneyness_range: tuple = (0.6, 1.4),
    maturity_range: tuple = (0.05, 3.0),
) -> Dict[str, Any]:
    """
    Build the full SABR implied volatility surface.

    Returns
    -------
    dict with keys:
        available : bool
        strikes : np.ndarray (n_strikes,)
        maturities : np.ndarray (n_maturities,)
        iv_surface : np.ndarray (n_maturities, n_strikes)
            Implied volatility (annualised, decimal).
        atm_term_structure : np.ndarray (n_maturities,)
            ATM IV at each maturity.
        smile_at_T0 : np.ndarray (n_strikes,)
            Smile at the shortest maturity.
        params : dict
        diagnostics : dict
            atm_iv, skew_25d, smile_curvature
    """
    if F <= 0 or alpha <= 0 or beta < 0 or beta > 1 or abs(rho) >= 1:
        return {
            "available": False,
            "reason": "Invalid SABR parameters.",
        }
    if nu is None:
        nu = alpha

    K_lo = F * moneyness_range[0]
    K_hi = F * moneyness_range[1]
    K_arr = np.linspace(K_lo, K_hi, n_strikes)
    T_arr = np.linspace(maturity_range[0], maturity_range[1], n_maturities)

    iv = np.zeros((n_maturities, n_strikes))
    for i, T in enumerate(T_arr):
        for j, K in enumerate(K_arr):
            iv[i, j] = sabr_implied_vol(F, K, T, alpha, beta, rho, nu)

    # ATM term structure
    atm_idx = int(np.argmin(np.abs(K_arr - F)))
    atm_term = iv[:, atm_idx]

    # Smile at the shortest maturity
    smile_T0 = iv[0, :]

    # Diagnostics
    atm_iv = float(iv[0, atm_idx])
    # 25-delta proxy: ±10% moneyness wings
    j_lo = int(np.argmin(np.abs(K_arr - F * 0.9)))
    j_hi = int(np.argmin(np.abs(K_arr - F * 1.1)))
    skew_25d = float(iv[0, j_lo] - iv[0, j_hi])  # >0 = put skew
    smile_curv = float(0.5 * (iv[0, j_lo] + iv[0, j_hi]) - iv[0, atm_idx])

    return {
        "available": True,
        "strikes": K_arr,
        "maturities": T_arr,
        "iv_surface": iv,
        "atm_term_structure": atm_term,
        "smile_at_T0": smile_T0,
        "params": {
            "F": float(F),
            "alpha": float(alpha),
            "beta": float(beta),
            "rho": float(rho),
            "nu": float(nu),
        },
        "diagnostics": {
            "atm_iv": atm_iv,
            "skew_25d": skew_25d,
            "smile_curvature": smile_curv,
        },
    }


# ── ATM calibration ────────────────────────────────────────────────

def calibrate_alpha_to_atm(
    F: float,
    T: float,
    sigma_atm: float,
    beta: float,
    rho: float = -0.3,
    nu: float = 0.4,
) -> float:
    """
    Solve for α that reproduces a given ATM implied volatility.

    Inverts the cubic correction in the Hagan ATM formula.

    Parameters
    ----------
    F, T : forward and maturity
    sigma_atm : observed (target) ATM IV
    beta, rho, nu : remaining SABR params (typically chosen by analyst)

    Returns
    -------
    alpha : float
    """
    F_b = F ** (1.0 - beta)

    # Hagan ATM cubic in α:
    #   c3 α³ + c2 α² + c1 α + c0 = 0
    # where σ_ATM ≈ (α / F_b) * (1 + [...] T)
    # Expanding and solving numerically (cubic root close to α₀ guess).
    c3 = (1.0 - beta) ** 2 / 24.0 * T / (F_b * F ** (2 * (1.0 - beta)))
    c2 = 0.25 * rho * beta * nu * T / (F_b ** 2)
    c1 = 1.0 / F_b * (1.0 + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2 * T)
    c0 = -sigma_atm

    coeffs = [c3, c2, c1, c0]
    roots = np.roots(coeffs)
    real_pos = [
        r.real for r in roots
        if np.isclose(r.imag, 0.0, atol=1e-6) and r.real > 0
    ]
    if not real_pos:
        # Fallback: linear approximation σ_ATM ≈ α / F_b
        return float(sigma_atm * F_b)
    return float(min(real_pos))


def calibrate_sabr_from_history(
    prices: pd.Series,
    beta: float = 0.5,
    rho: float = -0.3,
    nu: Optional[float] = None,
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """
    Quick calibration of SABR parameters from a historical price series.

    Estimates ATM vol from realised volatility, calibrates α via the
    Hagan ATM formula, and returns a calibrated parameter set ready
    to feed into ``build_sabr_surface``.

    This is an APPROXIMATION — production calibration uses observed
    option market data.  Use it for visualisation / what-if analysis.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.
    beta : float
        Held fixed (default 0.5 = stochastic CIR / Cox-Ingersoll-Ross-like).
    rho, nu : floats
        Held fixed at user-supplied values.
    horizon_days : int
        Default expiry for the ATM calibration (days).

    Returns
    -------
    dict with keys: available, F, alpha, beta, rho, nu, sigma_atm
    """
    prices = prices.dropna()
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars; got {len(prices)}.",
        }

    F = float(prices.iloc[-1])
    log_ret = np.log(prices / prices.shift(1)).dropna()
    sigma_realized = float(log_ret.std() * np.sqrt(252))

    if nu is None:
        # Default vol-of-vol from rolling-vol variability
        rolling_vol = log_ret.rolling(20).std() * np.sqrt(252)
        nu = float(np.nanstd(rolling_vol.dropna()) /
                   max(np.nanmean(rolling_vol.dropna()), 1e-6))
        nu = float(np.clip(nu, 0.1, 1.5))

    T = horizon_days / 252.0
    alpha = calibrate_alpha_to_atm(
        F=F, T=T, sigma_atm=sigma_realized, beta=beta, rho=rho, nu=nu,
    )

    return {
        "available": True,
        "F": F,
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "nu": float(nu),
        "sigma_atm": sigma_realized,
        "horizon_days": int(horizon_days),
    }