"""
Hierarchical Risk Parity — correlation-based portfolio allocation.

Implements the HRP algorithm (Marcos López de Prado, 2016):

    1. Compute distance matrix from correlation:
           d(i,j) = sqrt( (1 - corr(i,j)) / 2 )
    2. Single-linkage hierarchical clustering → dendrogram.
    3. Quasi-diagonalize the covariance matrix using dendrogram ordering.
    4. Recursive bisection: split clusters, allocate by inverse variance.

HRP advantages over Markowitz mean-variance:
    - No matrix inversion → stable even with near-singular covariances.
    - Respects the hierarchical structure of asset correlations.
    - Out-of-sample performance tends to be more robust.

The 3D visualization stacks sorted cumulative returns across assets
over time, creating a "return landscape" surface that shows which
clusters contributed to portfolio performance.

Reference
---------
quant-traderr-lab / Hierarchical risk parity / HRP pipeline.py
Uses scipy linkage + recursive bisection for weight computation.

Design
------
Pure functions, structured dict returns, no I/O.
Accepts either a multi-asset DataFrame OR a dict of pd.Series.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────

def _get_cluster_var(cov: pd.DataFrame) -> float:
    """Inverse-variance portfolio variance for a cluster."""
    ivp = 1.0 / np.diag(cov.values)
    ivp /= ivp.sum()
    return float(ivp @ cov.values @ ivp)


def _hrp_weights(
    cov: pd.DataFrame,
    sort_idx: List[int],
) -> pd.Series:
    """Recursive bisection to allocate by inverse cluster variance."""
    w = pd.Series(1.0, index=range(len(sort_idx)))
    cluster_items = [sort_idx]

    while cluster_items:
        next_clusters = []
        for subset in cluster_items:
            if len(subset) <= 1:
                continue
            mid = len(subset) // 2
            left = subset[:mid]
            right = subset[mid:]

            cov_left = cov.iloc[left, left]
            cov_right = cov.iloc[right, right]

            var_left = _get_cluster_var(cov_left)
            var_right = _get_cluster_var(cov_right)

            alpha = 1.0 - var_left / (var_left + var_right + 1e-16)

            for i in left:
                w.iloc[i] *= alpha
            for i in right:
                w.iloc[i] *= (1.0 - alpha)

            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)

        cluster_items = next_clusters

    return w / w.sum()


# ── main computation ────────────────────────────────────────────────

def compute_hrp(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> Dict[str, Any]:
    """
    Compute HRP portfolio weights from a multi-asset return DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for each asset (columns = tickers, rows = dates).
    linkage_method : str, default "single"
        Hierarchical clustering linkage method.

    Returns
    -------
    dict with keys:
        available : bool
        weights : pd.Series   (ticker → weight)
        sorted_tickers : list  (dendrogram leaf order)
        linkage_matrix : np.ndarray
        correlation : pd.DataFrame
        covariance : pd.DataFrame
        cumulative_returns : pd.DataFrame  (sorted by final return)
        surface_data : np.ndarray  (for 3D visualization)
        n_assets : int
        n_days : int
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    returns = returns.dropna(axis=1, how="all").dropna()
    tickers = list(returns.columns)
    n_assets = len(tickers)

    if n_assets < 3:
        return {
            "available": False,
            "reason": f"Need >= 3 assets; got {n_assets}.",
        }
    if len(returns) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 days of returns; got {len(returns)}.",
        }

    corr = returns.corr()
    cov = returns.cov()

    # Distance matrix from correlation
    dist = ((1 - corr) / 2.0).pow(0.5)
    dist_vals = dist.values.copy()
    np.fill_diagonal(dist_vals, 0)

    condensed = squareform(dist_vals, checks=False)
    link = linkage(condensed, method=linkage_method)
    sort_idx = list(leaves_list(link))

    # HRP weights
    weights = _hrp_weights(cov, sort_idx)
    weights.index = [tickers[i] for i in sort_idx]

    # Cumulative returns for 3D surface
    cum_ret = (1 + returns).cumprod() - 1
    cum_pct = cum_ret * 100

    # Sort by final cumulative return (descending) for mountain shape
    final_rets = cum_pct.iloc[-1].sort_values(ascending=False)
    sorted_cols = final_rets.index.tolist()
    sorted_cum = cum_pct[sorted_cols]

    return {
        "available": True,
        "weights": weights,
        "sorted_tickers": sorted_cols,
        "dendrogram_order": [tickers[i] for i in sort_idx],
        "linkage_matrix": link,
        "correlation": corr,
        "covariance": cov,
        "cumulative_returns": sorted_cum,
        "surface_data": sorted_cum.values.astype(np.float64),
        "n_assets": n_assets,
        "n_days": len(returns),
    }


def hrp_from_prices(
    prices: pd.DataFrame,
    linkage_method: str = "single",
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute HRP from a multi-asset price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices for each asset (columns = tickers).

    Returns
    -------
    Same as compute_hrp, plus a 'fit' key with summary stats.
    """
    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    if prices.empty or len(prices.columns) < 3:
        return {
            "available": False,
            "reason": "Insufficient asset data.",
        }

    returns = prices.pct_change().dropna()
    result = compute_hrp(returns, linkage_method=linkage_method)

    if result["available"]:
        # Portfolio-level stats
        w = result["weights"]
        aligned_returns = returns[w.index] if set(w.index).issubset(returns.columns) else returns
        if set(w.index).issubset(aligned_returns.columns):
            port_ret = (aligned_returns[w.index] * w).sum(axis=1)
            result["portfolio_return"] = float(port_ret.mean() * 252)
            result["portfolio_vol"] = float(port_ret.std() * np.sqrt(252))
            sharpe = result["portfolio_return"] / (result["portfolio_vol"] + 1e-9)
            result["portfolio_sharpe"] = float(sharpe)

    return result
