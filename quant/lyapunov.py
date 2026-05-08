"""
Lyapunov exponent — chaos analysis and phase-space reconstruction.

Two complementary analyses:

1. MAXIMAL LYAPUNOV EXPONENT (MLE)
   Measures sensitivity to initial conditions — the hallmark of chaos.
       λ > 0  →  chaotic (nearby trajectories diverge exponentially)
       λ ≈ 0  →  marginally stable / periodic
       λ < 0  →  dissipative / convergent

   Estimated via the Rosenstein (1993) algorithm:
       - Embed the time series into R^m using time-delay embedding
         x(t) = [s(t), s(t+τ), ..., s(t+(m-1)τ)]
       - For each point, find its nearest neighbor (excluding temporal
         neighbors within a Theiler window)
       - Track how the distance between each pair diverges over time
       - The slope of  <log(divergence)>  vs. time  = λ_max

2. METHOD OF ANALOGUES
   Finds historical periods whose phase-space trajectory most closely
   resembles the current market microstructure.  Uses KD-tree nearest-
   neighbor search in the embedded space.

   This answers: "Has the market behaved like THIS before? When?"

Reference
---------
quant-traderr-lab / Lyapunov Exponent / Lyapunov Pipeline.py
Phase-space reconstruction with time-delay embedding and KD-tree
nearest-neighbor search for historical analogues.

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


# ── time-delay embedding ────────────────────────────────────────────

def _embed_time_delay(
    series: np.ndarray,
    dim: int,
    tau: int,
) -> np.ndarray:
    """
    Time-delay embedding of a scalar series into R^dim.

    Returns array of shape (N - (dim-1)*tau, dim).
    """
    N = len(series)
    M = N - (dim - 1) * tau
    if M <= 0:
        raise ValueError(
            f"Series too short ({N}) for dim={dim}, tau={tau}."
        )
    embedded = np.zeros((M, dim))
    for d in range(dim):
        start = d * tau
        embedded[:, d] = series[start: start + M]
    return embedded


# ── Lyapunov exponent (Rosenstein 1993) ─────────────────────────────

def compute_lyapunov(
    prices: pd.Series,
    dim: int = 3,
    tau: int = 5,
    theiler_window: int = 20,
    max_divergence_steps: int = 50,
) -> Dict[str, Any]:
    """
    Estimate the maximal Lyapunov exponent from a price series.

    Parameters
    ----------
    prices : pd.Series
        Historical prices. Log returns are used for embedding.
    dim : int, default 3
        Embedding dimension.
    tau : int, default 5
        Time delay for embedding (in bars).
    theiler_window : int, default 20
        Minimum temporal separation for nearest-neighbor search
        (prevents identifying temporally adjacent points as neighbors).
    max_divergence_steps : int, default 50
        How many steps forward to track divergence.

    Returns
    -------
    dict with keys:
        available : bool
        lyapunov_exponent : float
        divergence_curve : np.ndarray   <log(divergence)> vs. time
        interpretation : str            "chaotic" / "stable" / "convergent"
        embedded_vectors : np.ndarray   phase-space embedding (for 3D viz)
        params : dict
    """
    prices = _clean_prices(prices)
    if len(prices) < 200:
        return {
            "available": False,
            "reason": f"Need >= 200 bars; got {len(prices)}.",
        }

    # Use log returns for stationarity
    log_ret = np.diff(np.log(prices.values.astype(np.float64)))
    # Normalize for numerical stability
    mu = log_ret.mean()
    sigma = log_ret.std() + 1e-9
    normalized = (log_ret - mu) / sigma

    try:
        vectors = _embed_time_delay(normalized, dim, tau)
    except ValueError as e:
        return {"available": False, "reason": str(e)}

    N = len(vectors)
    if N < max_divergence_steps + theiler_window + 10:
        return {
            "available": False,
            "reason": "Embedded series too short for divergence tracking.",
        }

    # For each point, find nearest neighbor (excluding Theiler window)
    from scipy.spatial import cKDTree

    tree = cKDTree(vectors)

    # Query: 2 nearest (self + 1 neighbor), then filter by Theiler
    nn_indices = np.full(N, -1, dtype=int)
    nn_dists = np.full(N, np.inf)

    for i in range(N):
        # Find enough candidates to skip Theiler window
        k = min(theiler_window + 5, N)
        dists, idxs = tree.query(vectors[i], k=k)

        for d, j in zip(dists, idxs):
            if j != i and abs(j - i) > theiler_window and d > 0:
                nn_indices[i] = j
                nn_dists[i] = d
                break

    # Track divergence: for each valid pair (i, nn[i]),
    # compute distance(vectors[i+k], vectors[nn[i]+k]) for k=0..max_steps
    valid_mask = nn_indices >= 0
    valid_idx = np.where(valid_mask)[0]

    # Only use pairs where both can be tracked forward
    usable = valid_idx[
        (valid_idx + max_divergence_steps < N) &
        (nn_indices[valid_idx] + max_divergence_steps < N)
    ]

    if len(usable) < 10:
        return {
            "available": False,
            "reason": "Too few valid nearest-neighbor pairs.",
        }

    # Accumulate log-divergence
    log_divs = np.zeros((len(usable), max_divergence_steps))
    for step in range(max_divergence_steps):
        for ui, i in enumerate(usable):
            j = nn_indices[i]
            dist = np.linalg.norm(vectors[i + step] - vectors[j + step])
            log_divs[ui, step] = np.log(dist + 1e-12)

    # Average divergence curve
    mean_log_div = log_divs.mean(axis=0)

    # Lyapunov exponent = slope of the linear region
    # Use first half of the curve for the fit (before saturation)
    fit_end = max_divergence_steps // 2
    t_fit = np.arange(fit_end)
    if fit_end >= 3:
        coeffs = np.polyfit(t_fit, mean_log_div[:fit_end], 1)
        lyap = float(coeffs[0])
    else:
        lyap = 0.0

    if lyap > 0.01:
        interpretation = "chaotic"
    elif lyap < -0.01:
        interpretation = "convergent"
    else:
        interpretation = "marginally_stable"

    return {
        "available": True,
        "lyapunov_exponent": lyap,
        "divergence_curve": mean_log_div,
        "interpretation": interpretation,
        "embedded_vectors": vectors,
        "params": {
            "dim": dim,
            "tau": tau,
            "theiler_window": theiler_window,
            "max_divergence_steps": max_divergence_steps,
            "n_valid_pairs": int(len(usable)),
            "n_embedded": int(N),
        },
    }


# ── method of analogues ─────────────────────────────────────────────

def find_analogues(
    prices: pd.Series,
    dim: int = 3,
    tau: int = 5,
    lookback: int = 60,
    n_analogues: int = 3,
) -> Dict[str, Any]:
    """
    Find historical periods with similar phase-space structure.

    Parameters
    ----------
    prices : pd.Series
        Historical prices.
    dim, tau : int
        Embedding parameters.
    lookback : int, default 60
        Length of the "current" trajectory to match.
    n_analogues : int, default 3
        Number of analogues to return.

    Returns
    -------
    dict with keys:
        available : bool
        analogues : list of dict, each with:
            index : int
            distance : float
            days_ago : float
        current_trajectory : np.ndarray
        embedded_vectors : np.ndarray
    """
    prices = _clean_prices(prices)
    if len(prices) < lookback * 3:
        return {
            "available": False,
            "reason": f"Need >= {lookback * 3} bars; got {len(prices)}.",
        }

    # Normalize log returns
    log_ret = np.diff(np.log(prices.values.astype(np.float64)))
    mu, sigma = log_ret.mean(), log_ret.std() + 1e-9
    normalized = (log_ret - mu) / sigma

    try:
        vectors = _embed_time_delay(normalized, dim, tau)
    except ValueError as e:
        return {"available": False, "reason": str(e)}

    if len(vectors) < lookback * 2:
        return {"available": False, "reason": "Insufficient embedded data."}

    current = vectors[-lookback:]

    # Search in history, excluding recent data (safety buffer)
    safety = lookback
    search_end = len(vectors) - lookback - safety
    if search_end < lookback:
        return {"available": False, "reason": "History too short."}

    search_space = vectors[:search_end]

    # Find nearest neighbors to the START of current trajectory
    from sklearn.neighbors import NearestNeighbors

    query = current[0].reshape(1, -1)
    k = min(n_analogues, len(search_space) - 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    nbrs.fit(search_space)
    distances, indices = nbrs.kneighbors(query)

    analogues = []
    for dist, idx in zip(distances[0], indices[0]):
        days_ago = (len(vectors) - idx) / 1.0  # in bars
        analogues.append({
            "index": int(idx),
            "distance": float(dist),
            "days_ago": float(days_ago),
        })

    return {
        "available": True,
        "analogues": analogues,
        "current_trajectory": current,
        "embedded_vectors": vectors,
        "params": {
            "dim": dim,
            "tau": tau,
            "lookback": lookback,
            "n_analogues": n_analogues,
        },
    }
