"""
Market Correlation Structure — Minimum Spanning Tree (MST).

Strips a multi-asset correlation matrix down to its skeleton: the
N-1 edges that hold the market together.  This is the classical
Mantegna (1999) construction:

    1.  Compute pairwise correlation matrix from log returns.
    2.  Convert to a distance metric:
            d(i,j) = sqrt( 2 * (1 - corr(i,j)) )
        Range:  ρ = +1 → d = 0,  ρ = 0 → d = √2,  ρ = -1 → d = 2.
    3.  Build a complete graph with these distances as edge weights.
    4.  Extract the Minimum Spanning Tree (Kruskal / Prim).

The MST is the most parsimonious connected sub-graph that captures
the dominant correlation structure.  Useful diagnostics:

    - **Central hub**: the node with highest degree often acts as
      the market's "center of gravity" (e.g. a large-cap bellwether).
    - **Sector homophily**: fraction of intra-sector edges → measures
      how well sectors cluster.
    - **MST length** (sum of edge weights): a SHORTER MST means the
      market is more highly correlated → higher systemic risk.
      During crises, the MST contracts (everything moves together).

Reference
---------
quant-traderr-lab / MST / MST pipeline.py
Distance metric, complete graph, MST extraction, hub & sector analysis.

Implementation notes
--------------------
- Uses scipy.sparse.csgraph.minimum_spanning_tree (Kruskal-equivalent)
  to avoid a networkx dependency.
- Returns plain Python edge lists so the UI can render with Plotly /
  Cytoscape without pulling in graph libraries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _clean_prices_df(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill, drop fully-missing columns, drop residual NaN rows."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    p = prices.dropna(axis=1, how="all").ffill().dropna()
    return p


# ── core MST computation ────────────────────────────────────────────

def compute_mst(
    prices: pd.DataFrame,
    sectors: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build the Minimum Spanning Tree from a multi-asset price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices, columns = tickers.
    sectors : dict, optional
        Mapping ticker → sector name.  Used for homophily analysis.
        If None, sector metrics are skipped.

    Returns
    -------
    dict with keys:
        available : bool
        n_assets : int
        tickers : list[str]
        correlation : pd.DataFrame
        distance : pd.DataFrame
        edges : list of dict   each {source, target, weight}
        nodes : list of dict   each {id, sector, degree}
        hub_node : str         most connected node
        hub_degree : int
        hub_sector : str
        avg_degree : float
        mst_length : float           sum of all MST edge weights
        avg_edge_weight : float
        avg_pairwise_corr : float
        sector_homophily : float     fraction of intra-sector edges
        risk_label : str             "high_systemic" / "moderate" / "low"
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    prices = _clean_prices_df(prices)
    tickers = list(prices.columns)
    n = len(tickers)

    if n < 4:
        return {
            "available": False,
            "reason": f"Need >= 4 assets; got {n}.",
        }
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 days of prices; got {len(prices)}.",
        }

    # 1. Log returns and correlation
    returns = np.log(prices / prices.shift(1)).dropna()
    corr = returns.corr()
    corr_vals = corr.values

    # Average pairwise correlation (off-diagonal, upper triangle)
    iu = np.triu_indices_from(corr_vals, k=1)
    avg_corr = float(np.nanmean(corr_vals[iu]))

    # 2. Distance metric
    dist_vals = np.sqrt(np.clip(2.0 * (1.0 - corr_vals), 0.0, None))
    np.fill_diagonal(dist_vals, 0.0)
    dist = pd.DataFrame(dist_vals, index=tickers, columns=tickers)

    # 3. MST via scipy (returns sparse upper-triangular tree)
    sparse_dist = csr_matrix(dist_vals)
    mst_sparse = minimum_spanning_tree(sparse_dist)
    mst_dense = mst_sparse.toarray()

    # 4. Extract edges (symmetrize and dedupe)
    edges: List[Dict[str, Any]] = []
    degree = {t: 0 for t in tickers}

    rows, cols = mst_dense.nonzero()
    for r, c in zip(rows, cols):
        w = float(mst_dense[r, c])
        if w <= 0:
            continue
        u, v = tickers[r], tickers[c]
        edges.append({"source": u, "target": v, "weight": w})
        degree[u] += 1
        degree[v] += 1

    # 5. Hub & degree stats
    hub_node = max(degree, key=degree.get)
    hub_degree = degree[hub_node]
    avg_degree = (sum(degree.values()) / n) if n > 0 else 0.0

    # 6. Sector homophily
    sector_homophily = float("nan")
    sector_edge_breakdown: Dict[str, int] = {}
    cross_sector_edges = 0
    if sectors:
        for e in edges:
            s_u = sectors.get(e["source"], "Unknown")
            s_v = sectors.get(e["target"], "Unknown")
            if s_u == s_v:
                sector_edge_breakdown[s_u] = sector_edge_breakdown.get(s_u, 0) + 1
            else:
                cross_sector_edges += 1
        if len(edges) > 0:
            sector_homophily = (
                (len(edges) - cross_sector_edges) / len(edges)
            )

    # 7. Risk metrics
    mst_length = float(sum(e["weight"] for e in edges))
    avg_edge = mst_length / max(len(edges), 1)

    # Interpretation thresholds from the reference
    if avg_edge < 0.5:
        risk_label = "high_systemic"
    elif avg_edge < 1.0:
        risk_label = "moderate"
    else:
        risk_label = "low_systemic"

    # 8. Build node list with sector + degree (for graph rendering)
    nodes = []
    for t in tickers:
        nodes.append({
            "id": t,
            "sector": (sectors.get(t, "Unknown") if sectors else None),
            "degree": degree[t],
        })

    return {
        "available": True,
        "n_assets": n,
        "tickers": tickers,
        "correlation": corr,
        "distance": dist,
        "edges": edges,
        "nodes": nodes,
        "hub_node": hub_node,
        "hub_degree": int(hub_degree),
        "hub_sector": (
            sectors.get(hub_node, "Unknown") if sectors else None
        ),
        "avg_degree": float(avg_degree),
        "mst_length": float(mst_length),
        "avg_edge_weight": float(avg_edge),
        "avg_pairwise_corr": avg_corr,
        "sector_homophily": float(sector_homophily) if not np.isnan(sector_homophily) else None,
        "sector_edge_breakdown": sector_edge_breakdown,
        "cross_sector_edges": int(cross_sector_edges),
        "risk_label": risk_label,
    }


def compute_rolling_mst_length(
    prices: pd.DataFrame,
    window: int = 60,
    step: int = 5,
) -> Dict[str, Any]:
    """
    Rolling MST length over time — a single-number systemic-risk indicator.

    A FALLING rolling MST length signals tightening correlations
    (markets "lock together") and is a leading indicator of regime shifts
    and crises.

    Parameters
    ----------
    prices : pd.DataFrame
        Multi-asset prices.
    window : int
        Rolling window size in trading days.
    step : int
        Stride between consecutive MST computations.

    Returns
    -------
    dict with keys:
        available : bool
        timestamps : pd.DatetimeIndex
        mst_length : np.ndarray
        avg_edge_weight : np.ndarray
        avg_corr : np.ndarray
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    prices = _clean_prices_df(prices)
    if len(prices) < window + step or len(prices.columns) < 4:
        return {
            "available": False,
            "reason": f"Need >= {window + step} days and >= 4 assets.",
        }

    log_ret = np.log(prices / prices.shift(1)).dropna()

    timestamps = []
    mst_lens = []
    avg_edges = []
    avg_corrs = []

    for end in range(window, len(log_ret) + 1, step):
        chunk = log_ret.iloc[end - window: end]
        c = chunk.corr().values
        if np.isnan(c).any():
            continue
        d = np.sqrt(np.clip(2.0 * (1.0 - c), 0.0, None))
        np.fill_diagonal(d, 0.0)
        m = minimum_spanning_tree(csr_matrix(d)).toarray()
        edges = m[m > 0]
        if len(edges) == 0:
            continue
        timestamps.append(log_ret.index[end - 1])
        mst_lens.append(float(edges.sum()))
        avg_edges.append(float(edges.mean()))
        iu = np.triu_indices_from(c, k=1)
        avg_corrs.append(float(np.nanmean(c[iu])))

    return {
        "available": len(timestamps) > 0,
        "timestamps": pd.DatetimeIndex(timestamps),
        "mst_length": np.array(mst_lens),
        "avg_edge_weight": np.array(avg_edges),
        "avg_corr": np.array(avg_corrs),
        "params": {"window": window, "step": step},
    }
