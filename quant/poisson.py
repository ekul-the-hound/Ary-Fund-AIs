"""
Poisson intensity estimation for order-arrival and jump modeling.

Covers:

  - poisson_mle(arrivals, window) -> lambda hat
        Maximum-likelihood intensity for a homogeneous Poisson process
        over a fixed window. Dead simple but the right baseline.

  - exponential_decay_intensity(arrivals, decay, t_now) -> lambda hat
        EWMA-style intensity estimate that down-weights stale arrivals.
        Useful for real-time order-arrival rate tracking.

  - hawkes_mle(arrivals, T) -> dict
        Fit a univariate exponential-kernel Hawkes process:
            lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))
        via maximum likelihood. Handles order-flow clustering and
        self-exciting jump behavior that homogeneous Poisson misses.

  - simulate_poisson(lam, T, seed) -> arrival times
  - simulate_hawkes(mu, alpha, beta, T, seed) -> arrival times
        Deterministic (given seed) thinning-algorithm simulators.

  - detect_jumps(returns, z_threshold=4.0) -> dict
        Lee-Mykland-style jump detection on a returns series, returning
        jump times, sizes, and an estimated jump intensity.

Sign / unit convention
----------------------
- `arrivals` is a sorted numpy array of arrival times in "time units".
- `T` (horizon) is in the same time units.
- Intensities are in arrivals-per-time-unit.
Choose the unit (seconds, trading days, minutes) consistently.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import math
import numpy as np
import pandas as pd
from scipy import optimize


# --- Helpers ------------------------------------------------------------------


def _as_arrivals(arrivals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(arrivals, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("arrivals contains non-finite values")
    # Must be sorted ascending
    if arr.size > 1 and np.any(np.diff(arr) < 0):
        arr = np.sort(arr)
    return arr


# --- 1. Homogeneous Poisson MLE ----------------------------------------------


def poisson_mle(arrivals: Sequence[float], window: float) -> Dict[str, Any]:
    """
    MLE for a homogeneous Poisson process over [0, window].

        lambda_hat = N / window

    where N is the number of arrivals in the window.

    Parameters
    ----------
    arrivals : array-like
        Arrival times, any unit (must be consistent with `window`).
    window : float > 0
        Length of the observation window.

    Returns
    -------
    dict
        {
          "lambda_hat": float,
          "n_arrivals": int,
          "window": float,
          "std_error": float,   # sqrt(N) / window, from Poisson variance
          "ci_95": (float, float),
        }
    """
    arr = _as_arrivals(arrivals)
    if window <= 0:
        raise ValueError("window must be > 0")

    n = int(arr.size)
    lam = n / window
    # Poisson count variance = N; so var(lambda_hat) = N / window^2
    se = math.sqrt(n) / window
    # Wald CI (symmetric); for small N use the chi-square-based interval instead
    ci_lo = max(0.0, lam - 1.96 * se)
    ci_hi = lam + 1.96 * se

    return {
        "lambda_hat": float(lam),
        "n_arrivals": n,
        "window": float(window),
        "std_error": float(se),
        "ci_95": (float(ci_lo), float(ci_hi)),
    }


# --- 2. Exponential-decay (EWMA) intensity -----------------------------------


def exponential_decay_intensity(
    arrivals: Sequence[float],
    decay: float,
    t_now: Optional[float] = None,
) -> float:
    """
    Exponentially-weighted arrival intensity at time `t_now`.

        lambda_hat(t_now) = decay * sum_i exp(-decay * (t_now - t_i))

    for t_i <= t_now. This is the natural real-time estimate when you want
    to weight recent arrivals more heavily than old ones. Equivalent to a
    Hawkes process with alpha = 0 viewed through a decay filter.

    Parameters
    ----------
    arrivals : array-like
    decay : float > 0
        Rate of exponential decay (higher = faster forgetting).
    t_now : float, optional
        Current time. Defaults to the last arrival time.

    Returns
    -------
    float
        Estimated intensity at t_now, in arrivals per time unit.
    """
    if decay <= 0:
        raise ValueError("decay must be > 0")
    arr = _as_arrivals(arrivals)
    if arr.size == 0:
        return 0.0

    if t_now is None:
        t_now = float(arr[-1])

    # Only arrivals at or before t_now contribute
    mask = arr <= t_now
    if not np.any(mask):
        return 0.0
    dt = t_now - arr[mask]
    return float(decay * np.sum(np.exp(-decay * dt)))


# --- 3. Hawkes process MLE (exponential kernel) ------------------------------


def _hawkes_neg_log_lik(
    params: np.ndarray, arrivals: np.ndarray, T: float
) -> float:
    """Negative log-likelihood for univariate exp-kernel Hawkes on [0, T]."""
    mu, alpha, beta = params
    # Penalize non-positive or unstable parameters; optimizer will avoid
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e12
    # Stability: branching ratio alpha/beta must be < 1 for stationarity
    if alpha / beta >= 0.999:
        return 1e12

    n = arrivals.size
    if n == 0:
        # No events: likelihood is exp(-mu * T)
        return mu * T

    # Recursive intensity evaluation at each arrival (O(n))
    # lambda(t_i) = mu + alpha * R_i where
    #   R_i = sum_{t_j < t_i} exp(-beta * (t_i - t_j))
    #       = exp(-beta * (t_i - t_{i-1})) * (1 + R_{i-1})
    R = 0.0
    log_lam_sum = math.log(mu)  # contribution from t_0, R_0 = 0
    prev = arrivals[0]
    for i in range(1, n):
        dt = arrivals[i] - prev
        R = math.exp(-beta * dt) * (1.0 + R)
        lam_i = mu + alpha * R
        if lam_i <= 0:
            return 1e12
        log_lam_sum += math.log(lam_i)
        prev = arrivals[i]

    # Compensator: integral of lambda(s) ds from 0 to T
    # = mu * T + (alpha/beta) * sum_i [1 - exp(-beta * (T - t_i))]
    compensator = mu * T + (alpha / beta) * np.sum(
        1.0 - np.exp(-beta * (T - arrivals))
    )

    log_lik = log_lam_sum - compensator
    return -log_lik


def hawkes_mle(
    arrivals: Sequence[float],
    T: float,
    init: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Fit a univariate exponential-kernel Hawkes process to arrival data.

        lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

    Interpretation:
      - mu    : baseline exogenous arrival rate.
      - alpha : self-excitation magnitude per past arrival.
      - beta  : decay rate of self-excitation (higher = shorter memory).
      - alpha / beta is the "branching ratio": expected number of
        offspring arrivals triggered by each event. Must be < 1 for
        stationarity. Values near 1 indicate highly clustered flow.

    Parameters
    ----------
    arrivals : array-like
        Sorted arrival times in [0, T].
    T : float > 0
        Observation horizon.
    init : (mu, alpha, beta), optional
        Initial guess. Defaults to moment-based heuristic.

    Returns
    -------
    dict
        {
          "mu": float, "alpha": float, "beta": float,
          "branching_ratio": float,
          "log_likelihood": float,
          "n_arrivals": int,
          "converged": bool,
          "message": str,
        }
    """
    arr = _as_arrivals(arrivals)
    n = arr.size
    if T <= 0:
        raise ValueError("T must be > 0")

    base_empty = {
        "mu": float("nan"),
        "alpha": float("nan"),
        "beta": float("nan"),
        "branching_ratio": float("nan"),
        "log_likelihood": float("nan"),
        "n_arrivals": n,
        "converged": False,
        "message": "",
    }

    if n < 5:
        base_empty["message"] = (
            f"Insufficient arrivals ({n}) for reliable Hawkes fit; "
            f"returning NaN parameters."
        )
        return base_empty

    # Heuristic init
    if init is None:
        empirical_rate = n / T
        mu0 = 0.5 * empirical_rate
        beta0 = 1.0
        alpha0 = 0.5 * beta0  # branching ratio 0.5
        x0 = np.array([mu0, alpha0, beta0])
    else:
        x0 = np.asarray(init, dtype=float)

    bounds = [(1e-8, None), (0.0, None), (1e-6, None)]

    try:
        res = optimize.minimize(
            _hawkes_neg_log_lik,
            x0=x0,
            args=(arr, float(T)),
            method="L-BFGS-B",
            bounds=bounds,
        )
        mu, alpha, beta = res.x
        branching = alpha / beta if beta > 0 else float("nan")
        return {
            "mu": float(mu),
            "alpha": float(alpha),
            "beta": float(beta),
            "branching_ratio": float(branching),
            "log_likelihood": float(-res.fun),
            "n_arrivals": n,
            "converged": bool(res.success),
            "message": str(res.message),
        }
    except Exception as e:  # pragma: no cover - defensive
        base_empty["message"] = f"Optimizer failed: {e!s}"
        return base_empty


# --- 4. Simulators ------------------------------------------------------------


def simulate_poisson(
    lam: float, T: float, seed: Optional[int] = 42
) -> np.ndarray:
    """
    Simulate arrivals of a homogeneous Poisson process on [0, T].

    Returns a sorted numpy array of arrival times.
    """
    if lam < 0 or T <= 0:
        raise ValueError("lam must be >= 0 and T > 0")
    rng = np.random.default_rng(seed)
    if lam == 0:
        return np.empty(0, dtype=float)
    # Number of arrivals ~ Poisson(lam*T); times ~ Uniform(0, T)
    n = int(rng.poisson(lam * T))
    if n == 0:
        return np.empty(0, dtype=float)
    times = np.sort(rng.uniform(0.0, T, size=n))
    return times


def simulate_hawkes(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulate a univariate exp-kernel Hawkes process via Ogata's thinning.

    Deterministic given seed.
    """
    if mu <= 0 or alpha < 0 or beta <= 0 or T <= 0:
        raise ValueError("Invalid Hawkes parameters or horizon.")
    if alpha / beta >= 1.0:
        raise ValueError(
            f"Non-stationary Hawkes: branching ratio alpha/beta = "
            f"{alpha/beta:.3f} must be < 1."
        )

    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    # Running intensity state: intensity just BEFORE adding the next event is
    # lam_bar = mu + alpha * S, where S = sum exp(-beta*(t - t_i))
    S = 0.0

    while True:
        lam_bar = mu + alpha * S
        # Candidate inter-arrival time
        u = rng.uniform()
        if u <= 0:
            continue
        w = -math.log(u) / lam_bar
        t_new = t + w
        if t_new > T:
            break

        # Update S to time t_new (decay) BEFORE acceptance test
        S_new = S * math.exp(-beta * w)
        lam_new = mu + alpha * S_new
        # Accept with prob lam_new / lam_bar
        if rng.uniform() <= lam_new / lam_bar:
            events.append(t_new)
            # Accepted event contributes to S for future intensities
            S = S_new + 1.0
        else:
            S = S_new
        t = t_new

    return np.asarray(events, dtype=float)


# --- 5. Jump detection --------------------------------------------------------


def detect_jumps(
    returns: Sequence[float],
    z_threshold: float = 4.0,
) -> Dict[str, Any]:
    """
    Simple robust jump detector on a returns series.

    Flags observations whose return exceeds `z_threshold` standard deviations
    of a robust MAD-based scale estimate. Also reports an empirical jump
    intensity (jumps per observation).

    This is a light-weight stand-in for Lee-Mykland's test — suitable for
    equity/futures daily or intraday returns when you want a yes/no jump
    flag without the full bipower-variation machinery.

    Parameters
    ----------
    returns : array-like or pd.Series
    z_threshold : float, default 4.0
        Robust z-score threshold. 4.0 is conservative; 3.0 is lenient.

    Returns
    -------
    dict
        {
          "n_jumps": int,
          "jump_indices": np.ndarray of int,
          "jump_sizes": np.ndarray of float,
          "z_threshold": float,
          "scale_mad": float,          # MAD-based sigma estimate
          "intensity": float,          # jumps per observation
          "note": str,
        }
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna().to_numpy(dtype=float)
        index = returns.dropna().index
    else:
        r = np.asarray(returns, dtype=float).reshape(-1)
        r = r[np.isfinite(r)]
        index = None

    base = {
        "n_jumps": 0,
        "jump_indices": np.empty(0, dtype=int),
        "jump_sizes": np.empty(0, dtype=float),
        "jump_times": None,
        "z_threshold": float(z_threshold),
        "scale_mad": float("nan"),
        "intensity": 0.0,
        "note": "",
    }

    if r.size < 10:
        base["note"] = "Too few observations for jump detection (<10)."
        return base

    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    # MAD -> sigma scaling factor for Gaussian data
    scale = 1.4826 * mad
    if scale <= 0:
        base["note"] = "Zero MAD; cannot compute robust scale."
        return base

    z = (r - med) / scale
    mask = np.abs(z) >= z_threshold
    idx = np.flatnonzero(mask)
    sizes = r[idx]

    result = {
        "n_jumps": int(idx.size),
        "jump_indices": idx,
        "jump_sizes": sizes,
        "jump_times": (index[idx] if index is not None else None),
        "z_threshold": float(z_threshold),
        "scale_mad": float(scale),
        "intensity": float(idx.size / r.size),
        "note": "",
    }
    return result
