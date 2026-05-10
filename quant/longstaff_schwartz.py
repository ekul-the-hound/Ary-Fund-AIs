"""
Longstaff-Schwartz Method (LSM) — American option pricer.

A Monte Carlo + regression hybrid that prices American-style options
where early exercise matters.  Standard MC can only price European
options because it lacks the dynamic-programming structure needed
for early-exercise decisions.  LSM adds it back via a backward
induction:

    For t = T-1, ..., 1:
        On in-the-money (ITM) paths only:
            Regress  discounted future cashflow  ~  basis(S_t)
            cont_est(S_t)  =  fitted continuation value
        if intrinsic(S_t) > cont_est(S_t):
            exercise now:  CF[path] = intrinsic
        else:
            keep continuation

Final option value:
    V0  =  E[ exp(-r * τ_i) * CF_i ]   over all paths,
                                       τ_i = exercise time per path

The method also yields the EARLY-EXERCISE BOUNDARY S*(t): the strike
above which (for puts) early exercise dominates continuation.  Plotting
this boundary against time produces the iconic LSM curve.

Practical use in the Ary Fund workflow
--------------------------------------
- Price American puts on portfolio holdings (e.g. for hedging analysis).
- Quantify the "early-exercise premium" =  American − European.
  This number tells you how much extra premium the optionality of
  early exercise commands, which matters for protective puts and
  employee stock options.
- The boundary visualises WHEN early exercise becomes optimal — this
  is critical for any strategy that involves writing American options
  (you face exercise risk when spot crosses the boundary).

Reference
---------
quant-traderr-lab / longstaff schwartz / Longstaff schwartz Pipeline.py
The reference implementation prices an American PUT under risk-neutral
GBM with antithetic variance reduction and a {1, S, S²} basis.  This
module follows the same construction and adds:
    - support for both PUT and CALL
    - choice of basis (poly / Laguerre)
    - a calibrate-from-prices helper that estimates σ from a real series

Design
------
Pure functions, structured dict returns, no I/O.
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


# ── GBM paths under the risk-neutral measure ───────────────────────

def simulate_rn_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = True,
    random_state: int = 11,
) -> np.ndarray:
    """
    Risk-neutral GBM path simulator.

    Returns
    -------
    np.ndarray of shape (n_paths, n_steps + 1).  paths[:, 0] = S0.
    """
    rng = np.random.default_rng(random_state)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
        if Z.shape[0] < n_paths:
            extra = rng.standard_normal((n_paths - Z.shape[0], n_steps))
            Z = np.vstack([Z, extra])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    drift = (r - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt)
    incr = drift + diff * Z

    log_paths = np.cumsum(incr, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


# ── basis functions ────────────────────────────────────────────────

def _polynomial_basis(S: np.ndarray, degree: int = 2) -> np.ndarray:
    """[1, S, S², ...] basis matrix."""
    cols = [np.ones_like(S)]
    for d in range(1, degree + 1):
        cols.append(S ** d)
    return np.column_stack(cols)


def _laguerre_basis(S: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Weighted Laguerre basis as in the original Longstaff-Schwartz
    paper:  L_n(S) = exp(-S/2) * laguerre_n(S).
    """
    weight = np.exp(-S / 2.0)
    cols = [weight]  # L0
    if degree >= 1:
        cols.append(weight * (1.0 - S))                                # L1
    if degree >= 2:
        cols.append(weight * (1.0 - 2.0 * S + 0.5 * S ** 2))           # L2
    if degree >= 3:
        cols.append(weight * (1.0 - 3.0 * S + 1.5 * S ** 2
                              - (S ** 3) / 6.0))                       # L3
    return np.column_stack(cols)


def _build_basis(S: np.ndarray, basis: str, degree: int) -> np.ndarray:
    if basis == "laguerre":
        return _laguerre_basis(S, degree=min(max(degree, 1), 3))
    return _polynomial_basis(S, degree=max(degree, 1))


# ── core LSM ───────────────────────────────────────────────────────

def run_lsm(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: str = "put",
    basis: str = "polynomial",
    degree: int = 2,
    boundary_grid_n: int = 200,
) -> Dict[str, Any]:
    """
    Backward iteration of the Longstaff-Schwartz algorithm.

    Parameters
    ----------
    paths : np.ndarray (N, n_steps + 1)
        Simulated paths with paths[:, 0] = S0.
    K : float
        Strike.
    r : float
        Continuously-compounded risk-free rate.
    T : float
        Maturity (years).
    option_type : "put" or "call"
    basis : "polynomial" or "laguerre"
    degree : int
        Basis order. Default 2 = {1, S, S²}.
    boundary_grid_n : int
        Resolution of the strike grid used to recover S*(t).

    Returns
    -------
    dict with keys:
        american_price : float
        european_price : float
        early_premium : float
        std_error : float
        exercise_t : np.ndarray (N,)        time-step index of exercise per path
        cash_flow : np.ndarray (N,)
        boundary : np.ndarray (n_steps + 1,)  early-exercise boundary S*(t),
                                              NaN where boundary undefined
        n_early_exercises : int
        params : dict
    """
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] < 2:
        return {
            "available": False,
            "reason": "paths must be 2-D with at least 2 time steps.",
        }

    N, n_total = paths.shape
    n_steps = n_total - 1
    dt = T / n_steps

    if option_type == "put":
        intrinsic_fn = lambda S: np.maximum(K - S, 0.0)
    elif option_type == "call":
        intrinsic_fn = lambda S: np.maximum(S - K, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'.")

    # Terminal cashflow = intrinsic at T
    cash_flow = intrinsic_fn(paths[:, -1]).copy()
    exercise_t = np.full(N, n_steps, dtype=int)

    # European price (no early exercise)
    european_payoffs = intrinsic_fn(paths[:, -1])
    discount_T = np.exp(-r * T)
    european_price = float(np.mean(european_payoffs * discount_T))
    european_se = float(np.std(european_payoffs * discount_T, ddof=1) /
                        np.sqrt(N))

    boundary = np.full(n_total, np.nan)
    # At maturity, the trivial boundary is K
    boundary[-1] = K

    # Strike grid for boundary recovery
    if option_type == "put":
        S_grid = np.linspace(0.40 * K, K, boundary_grid_n)
    else:
        S_grid = np.linspace(K, 1.60 * K, boundary_grid_n)

    # Backward iteration
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        intrinsic = intrinsic_fn(S_t)
        itm = intrinsic > 0
        n_itm = int(itm.sum())
        if n_itm < 5:
            continue

        # Discounted future cashflow on ITM paths
        future_cf = (
            cash_flow[itm]
            * np.exp(-r * dt * (exercise_t[itm] - t))
        )
        S_itm = S_t[itm]
        X = _build_basis(S_itm, basis=basis, degree=degree)

        # OLS regression (np.linalg.lstsq is rank-aware and stable)
        coefs, *_ = np.linalg.lstsq(X, future_cf, rcond=None)
        cont_est = X @ coefs

        # Exercise where intrinsic > continuation
        ex_now = intrinsic[itm] > cont_est
        ex_idx = np.where(itm)[0][ex_now]
        cash_flow[ex_idx] = intrinsic[itm][ex_now]
        exercise_t[ex_idx] = t

        # Recover boundary: edge of the exercise region on S_grid
        X_grid = _build_basis(S_grid, basis=basis, degree=degree)
        cont_g = X_grid @ coefs
        intr_g = intrinsic_fn(S_grid)
        ex_mask = intr_g > cont_g
        if ex_mask.any():
            if option_type == "put":
                # For puts the boundary is the LARGEST S where exercise wins
                boundary[t] = float(S_grid[ex_mask].max())
            else:
                # For calls the boundary is the SMALLEST S where exercise wins
                boundary[t] = float(S_grid[ex_mask].min())

    # Discounted to t=0
    discounted = cash_flow * np.exp(-r * dt * exercise_t)
    american_price = float(np.mean(discounted))
    american_se = float(np.std(discounted, ddof=1) / np.sqrt(N))

    early_mask = exercise_t < n_steps
    n_early = int(early_mask.sum())

    return {
        "available": True,
        "american_price": american_price,
        "european_price": european_price,
        "early_premium": float(american_price - european_price),
        "american_se": american_se,
        "european_se": european_se,
        "exercise_t": exercise_t,
        "cash_flow": cash_flow,
        "boundary": boundary,
        "paths": paths,
        "n_early_exercises": n_early,
        "early_exercise_fraction": float(n_early / N),
        "params": {
            "K": float(K), "r": float(r), "T": float(T),
            "n_paths": int(N), "n_steps": int(n_steps),
            "option_type": option_type,
            "basis": basis, "degree": int(degree),
        },
    }


# ── high-level wrapper ────────────────────────────────────────────

def price_american_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "put",
    n_paths: int = 5000,
    n_steps: int = 50,
    basis: str = "polynomial",
    degree: int = 2,
    antithetic: bool = True,
    random_state: int = 11,
) -> Dict[str, Any]:
    """
    Convenience wrapper: simulate paths + run LSM in one call.

    Returns the same dict as ``run_lsm``, plus the GBM parameters.
    """
    paths = simulate_rn_gbm_paths(
        S0=S0, r=r, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths,
        antithetic=antithetic, random_state=random_state,
    )
    out = run_lsm(
        paths=paths, K=K, r=r, T=T,
        option_type=option_type, basis=basis, degree=degree,
    )
    if out.get("available"):
        out["params"].update({
            "S0": float(S0), "sigma": float(sigma),
            "antithetic": bool(antithetic),
        })
    return out


def price_american_from_prices(
    prices: pd.Series,
    K: Optional[float] = None,
    r: float = 0.05,
    horizon_days: int = 252,
    option_type: str = "put",
    n_paths: int = 5000,
    n_steps: int = 50,
    basis: str = "polynomial",
    degree: int = 2,
    random_state: int = 11,
) -> Dict[str, Any]:
    """
    Calibrate σ from a historical price series, then price an American
    option using LSM.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.  Last observation is used as S₀.
    K : float, optional
        Strike.  Defaults to S₀ (ATM).
    r : float
        Continuously-compounded risk-free rate.
    horizon_days : int
        Maturity in trading days (T = horizon_days / 252).
    option_type, n_paths, n_steps, basis, degree, random_state
        Forwarded to ``price_american_option``.

    Returns
    -------
    Same dict as ``run_lsm``, plus a "fit" block with σ estimate.
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars; got {len(prices)}.",
        }

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    sigma = float(log_ret.std(ddof=1) * np.sqrt(252))
    S0 = float(prices.iloc[-1])
    K_used = float(S0 if K is None else K)
    T = horizon_days / 252.0

    out = price_american_option(
        S0=S0, K=K_used, r=r, sigma=sigma, T=T,
        option_type=option_type, n_paths=n_paths, n_steps=n_steps,
        basis=basis, degree=degree, random_state=random_state,
    )
    if out.get("available"):
        out["fit"] = {
            "S0": S0,
            "sigma_estimated": sigma,
            "n_obs": int(len(log_ret)),
            "horizon_days": int(horizon_days),
        }
    return out
