"""
Lempel-Ziv complexity — algorithmic complexity of price sequences.

Measures the "information content" of a discretized price series by
counting the number of distinct substrings encountered during a
sequential scan (the LZ76 algorithm).

Interpretation:
    Low LZ complexity  →  predictable / structured / trending
    High LZ complexity →  chaotic / random / noisy

The rolling LZ complexity trace acts as a REGIME DETECTOR:
    - A sharp INCREASE signals a transition from structured to noisy
      (breakout failure, regime disruption).
    - A sharp DECREASE signals emerging structure (trend formation,
      herding behavior).

Normalized complexity  c_norm = c(n) / (n / log2(n))  puts the raw
count on a [0, 1] scale where 1.0 = maximum complexity (pure random)
and 0.0 = perfectly predictable (constant).

Reference
---------
quant-traderr-lab / Lempel-Ziv / Lempel-Ziv Pipeline.py
Discretizes returns to binary (up/down), computes rolling LZ complexity,
and visualises in 3D phase space (time × price × entropy).

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


# ── core LZ76 algorithm ────────────────────────────────────────────

def _lz_complexity(binary_string: str) -> int:
    """
    Compute Lempel-Ziv complexity count (LZ76).

    Scans the string left-to-right, counting the number of distinct
    new substrings ("phrases") encountered.
    """
    n = len(binary_string)
    if n == 0:
        return 0

    u, v, w = 0, 1, 1
    complexity = 1

    while v + w <= n:
        if binary_string[v: v + w] in binary_string[u: v]:
            w += 1
        else:
            u, v, w = 0, v + w, 1
            complexity += 1

    return complexity


def _normalize_lz(c: int, n: int) -> float:
    """Normalize LZ count to [0, 1] using the Lempel-Ziv bound.

    The theoretical upper bound n/log2(n) is asymptotic; for small n the
    raw count can exceed it.  We clip to [0, 1] for clean interpretation.
    """
    if n <= 1:
        return 0.0
    log2n = np.log2(n) if n > 0 else 1.0
    bound = n / log2n if log2n > 0 else n
    return min(c / bound, 1.0) if bound > 0 else 0.0


# ── public API ──────────────────────────────────────────────────────

def compute_lz_complexity(
    prices: pd.Series,
    window: int = 30,
    threshold: str = "median",
) -> Dict[str, Any]:
    """
    Rolling Lempel-Ziv complexity analysis.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.
    window : int, default 30
        Rolling window size for the complexity calculation.
    threshold : str, default "median"
        Discretization method for returns:
            "zero"   → binary: 1 if return > 0, else 0
            "median" → binary: 1 if return > rolling median, else 0

    Returns
    -------
    dict with keys:
        available : bool
        lz_raw : np.ndarray       raw LZ complexity count (aligned to prices)
        lz_normalized : np.ndarray  normalized to [0, 1]
        prices : np.ndarray
        binary_sequence : str       full binary string for reference
        current_complexity : float  latest normalized LZ value
        regime_label : str          "structured" / "random" / "transitional"
        stats : dict                mean, std, min, max of normalized series
        phase_space : dict          time, price, entropy arrays for 3D plot
    """
    prices = _clean_prices(prices)
    if len(prices) < window + 10:
        return {
            "available": False,
            "reason": f"Need >= {window + 10} bars; got {len(prices)}.",
        }

    price_arr = prices.values.astype(np.float64)
    returns = np.diff(price_arr) / price_arr[:-1]

    # Discretize to binary
    if threshold == "median":
        # Rolling median threshold
        ret_series = pd.Series(returns)
        rolling_med = ret_series.rolling(window, min_periods=1).median()
        binary = (returns > rolling_med.values).astype(int)
    else:
        binary = (returns > 0).astype(int)

    binary_str_arr = binary.astype(str)
    full_binary = "".join(binary_str_arr.tolist())

    # Rolling LZ complexity
    n_returns = len(returns)
    lz_raw = np.full(len(price_arr), np.nan)
    lz_norm = np.full(len(price_arr), np.nan)

    for i in range(window, n_returns + 1):
        chunk = full_binary[i - window: i]
        c = _lz_complexity(chunk)
        lz_raw[i] = c
        lz_norm[i] = _normalize_lz(c, window)

    # Fill the first window with edge values
    first_valid = window
    if first_valid < len(lz_raw):
        lz_raw[:first_valid] = lz_raw[first_valid]
        lz_norm[:first_valid] = lz_norm[first_valid]

    # Current regime classification
    valid_norm = lz_norm[~np.isnan(lz_norm)]
    current = float(valid_norm[-1]) if len(valid_norm) > 0 else 0.5

    if current < 0.35:
        regime = "structured"
    elif current > 0.65:
        regime = "random"
    else:
        regime = "transitional"

    return {
        "available": True,
        "lz_raw": lz_raw,
        "lz_normalized": lz_norm,
        "prices": price_arr,
        "binary_sequence": full_binary,
        "current_complexity": current,
        "regime_label": regime,
        "params": {
            "window": window,
            "threshold": threshold,
            "n_samples": len(price_arr),
        },
        "stats": {
            "mean": float(np.nanmean(lz_norm)),
            "std": float(np.nanstd(lz_norm)),
            "min": float(np.nanmin(lz_norm)),
            "max": float(np.nanmax(lz_norm)),
        },
        "phase_space": {
            "time": np.arange(len(price_arr)),
            "price": price_arr,
            "entropy": lz_norm,
        },
    }
