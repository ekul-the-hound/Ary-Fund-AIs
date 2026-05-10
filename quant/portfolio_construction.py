"""
Portfolio construction methods.

Four allocators that share a common interface and live in one module so
the dashboard can compare them side-by-side without juggling four small
files:

    1.  Volatility Parity         — equal RISK contribution from each
                                    asset (1/σ weighting, no covariance).
    2.  Conviction-Vol Hybrid     — overlay analyst conviction scores on
                                    the volatility-parity baseline.
    3.  Mean-Variance Optimization — Markowitz max-Sharpe / min-variance
                                    with optional Ledoit-Wolf shrinkage.
    4.  Black-Litterman           — blends a market-equilibrium prior
                                    with explicit views to produce
                                    posterior expected returns, then
                                    runs MVO on the posterior.

All four return a uniform structure:

    {
      "available": bool,
      "weights": pd.Series  (ticker -> weight, sums to 1),
      "expected_return": float (annualised, when defined),
      "expected_vol": float (annualised),
      "sharpe": float,
      "concentration_hhi": float,
      "method": str,
      ... method-specific diagnostics ...
    }

Reference
---------
- HRP comes from quant-traderr-lab / Hierarchical risk parity / HRP pipeline.py
  (already implemented in batch 1; re-exported here for convenience and
  for the comparison helper).
- Volatility Parity, Mean-Variance, and Black-Litterman are textbook
  formulations grounded in the same Markowitz / risk-parity framework
  the repository builds on.  The Conviction-Vol Hybrid is an Ary Fund
  bespoke that fits naturally because the project already produces
  per-ticker conviction scores via the agent pipeline.

Design
------
Pure functions, structured dict returns, no I/O.  Inputs are either
a returns DataFrame or a prices DataFrame (the latter is normalized
internally).  Concentration is reported via the Herfindahl-Hirschman
Index so allocation balance is visible at a glance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TRADING_DAYS = 252


# ── shared helpers ──────────────────────────────────────────────────

def _clean_prices_df(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    return prices.dropna(axis=1, how="all").ffill().dropna()


def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return _clean_prices_df(prices).pct_change().dropna()


def _portfolio_stats(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Annualised return, vol, Sharpe, HHI for a weight vector."""
    aligned = returns.reindex(columns=weights.index).dropna(how="all")
    w = weights.reindex(aligned.columns).fillna(0.0).values
    daily = aligned.values @ w
    mu_a = float(np.mean(daily) * TRADING_DAYS)
    sigma_a = float(np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (mu_a - risk_free_rate) / sigma_a if sigma_a > 1e-12 else 0.0
    hhi = float(np.sum(w ** 2))
    return {
        "expected_return": mu_a,
        "expected_vol": sigma_a,
        "sharpe": float(sharpe),
        "concentration_hhi": hhi,
    }


def _ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage of the sample covariance to a constant-corr target."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.values)
        cov = pd.DataFrame(
            lw.covariance_ * TRADING_DAYS,
            index=returns.columns,
            columns=returns.columns,
        )
        return cov
    except Exception:
        return returns.cov() * TRADING_DAYS


# ── 1. Volatility Parity ───────────────────────────────────────────

def volatility_parity(
    returns_or_prices: pd.DataFrame,
    is_prices: bool = True,
    use_inverse_variance: bool = False,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Allocate proportional to 1 / σ_i (or 1 / σ_i² if inverse-variance).

    The simplest member of the risk-parity family.  Assumes ZERO
    cross-asset correlation, so each asset contributes the same amount
    of standalone volatility.  Compared to HRP, this is a cruder
    diagonalisation: HRP uses dendrogram structure, vol parity ignores
    correlations entirely.

    Parameters
    ----------
    returns_or_prices : pd.DataFrame
    is_prices : bool
        True if input is prices (returns are computed internally).
    use_inverse_variance : bool
        If True, weights ∝ 1/σ²  (commonly called "inverse-variance
        portfolio", IVP).  Default False uses 1/σ.

    Returns
    -------
    dict with keys: available, weights, expected_return, expected_vol,
    sharpe, concentration_hhi, vols, method.
    """
    rets = (_returns_from_prices(returns_or_prices)
            if is_prices else returns_or_prices.dropna())
    if rets.empty or len(rets.columns) < 2:
        return {"available": False, "reason": "Need >= 2 assets."}

    sigma_d = rets.std(ddof=1)
    if (sigma_d <= 0).any():
        return {"available": False, "reason": "Zero-vol asset present."}

    if use_inverse_variance:
        raw_w = 1.0 / (sigma_d ** 2)
    else:
        raw_w = 1.0 / sigma_d
    w = raw_w / raw_w.sum()

    stats = _portfolio_stats(w, rets, risk_free_rate=risk_free_rate)
    return {
        "available": True,
        "method": ("inverse_variance" if use_inverse_variance
                   else "volatility_parity"),
        "weights": w,
        "vols": (sigma_d * np.sqrt(TRADING_DAYS)),
        **stats,
    }


# ── 2. Conviction × Volatility Hybrid ──────────────────────────────

def conviction_vol_hybrid(
    returns_or_prices: pd.DataFrame,
    conviction: pd.Series,
    is_prices: bool = True,
    blend: float = 0.5,
    long_only: bool = True,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Blend analyst conviction scores with a volatility-parity baseline.

    conviction is expected to be a per-ticker score (e.g. agent-pipeline
    output, range typically [0, 1] or [-1, 1]).  Negative scores are
    clipped if long_only=True.

    Final weights:
        w_i  ∝   blend * conviction_i  +  (1 - blend) * (1 / σ_i)

    Each component is normalised to sum to 1 BEFORE blending so the
    blend coefficient is interpretable (0 = pure vol parity, 1 = pure
    conviction-weighted).

    Parameters
    ----------
    conviction : pd.Series
        Indexed by ticker; missing tickers default to 0.
    blend : float ∈ [0, 1]
        Mix coefficient.

    Returns
    -------
    dict with keys: available, weights, conviction_used, blend, ...
    """
    if not 0.0 <= blend <= 1.0:
        return {"available": False, "reason": "blend must be in [0, 1]."}

    rets = (_returns_from_prices(returns_or_prices)
            if is_prices else returns_or_prices.dropna())
    if rets.empty or len(rets.columns) < 2:
        return {"available": False, "reason": "Need >= 2 assets."}

    tickers = list(rets.columns)
    sigma_d = rets.std(ddof=1)
    if (sigma_d <= 0).any():
        return {"available": False, "reason": "Zero-vol asset present."}

    # Vol-parity component
    vol_w = (1.0 / sigma_d)
    vol_w = vol_w / vol_w.sum()

    # Conviction component (align, fill, clip if long-only)
    conv = conviction.reindex(tickers).astype(float).fillna(0.0)
    if long_only:
        conv = conv.clip(lower=0.0)
    if conv.sum() <= 0:
        # Degenerate conviction → fall back to vol parity
        w = vol_w
        actual_blend = 0.0
    else:
        conv_w = conv / conv.sum()
        w = blend * conv_w + (1.0 - blend) * vol_w
        # Renormalise (defensive — should already sum to 1)
        w = w / w.sum()
        actual_blend = blend

    stats = _portfolio_stats(w, rets, risk_free_rate=risk_free_rate)
    return {
        "available": True,
        "method": "conviction_vol_hybrid",
        "weights": w,
        "vol_parity_component": vol_w,
        "conviction_component": conv,
        "blend": float(actual_blend),
        **stats,
    }


# ── 3. Mean-Variance Optimization ──────────────────────────────────

def mean_variance(
    returns_or_prices: pd.DataFrame,
    is_prices: bool = True,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    long_only: bool = True,
    use_shrinkage: bool = True,
    expected_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Markowitz mean-variance optimisation.

    Three objectives are supported:

        "max_sharpe"     : maximise (μ_p - r_f) / σ_p
        "min_variance"   : minimise σ_p, ignoring μ
        "target_return"  : minimise σ_p subject to μ_p ≥ target_return

    Parameters
    ----------
    returns_or_prices : pd.DataFrame
    is_prices : bool
    objective : str
    risk_free_rate : float
        Annualised risk-free rate.
    target_return : float, optional
        Required for objective="target_return".
    long_only : bool
        Constrain weights to ≥ 0.
    use_shrinkage : bool
        Apply Ledoit-Wolf to the covariance estimate.  Strongly
        recommended for stability.
    expected_returns : pd.Series, optional
        Override historical mean returns.  Lets the caller plug in a
        Black-Litterman posterior or analyst forecast.  Annualised.

    Returns
    -------
    dict with keys: available, weights, expected_return, expected_vol,
    sharpe, concentration_hhi, frontier (small grid), method.
    """
    rets = (_returns_from_prices(returns_or_prices)
            if is_prices else returns_or_prices.dropna())
    if rets.empty or len(rets.columns) < 2:
        return {"available": False, "reason": "Need >= 2 assets."}

    n = len(rets.columns)
    tickers = list(rets.columns)

    cov = (_ledoit_wolf_shrinkage(rets) if use_shrinkage
           else rets.cov() * TRADING_DAYS)

    if expected_returns is None:
        mu = (rets.mean() * TRADING_DAYS).reindex(tickers)
    else:
        mu = expected_returns.reindex(tickers).astype(float)
        if mu.isna().any():
            mu = mu.fillna(rets.mean() * TRADING_DAYS)

    cov_vals = cov.reindex(index=tickers, columns=tickers).values
    mu_vals = mu.values

    # Try scipy.optimize for constrained problems; fall back to closed-form
    try:
        from scipy.optimize import minimize

        def port_vol(w):
            return float(np.sqrt(w @ cov_vals @ w))

        def neg_sharpe(w):
            v = port_vol(w)
            if v < 1e-12:
                return 1e6
            return -(float(w @ mu_vals) - risk_free_rate) / v

        x0 = np.full(n, 1.0 / n)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if objective == "target_return" and target_return is not None:
            constraints.append({
                "type": "ineq",
                "fun": lambda w: float(w @ mu_vals) - target_return,
            })
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n

        if objective == "min_variance":
            res = minimize(
                lambda w: float(w @ cov_vals @ w),
                x0, method="SLSQP", bounds=bounds, constraints=constraints,
            )
        elif objective == "target_return":
            res = minimize(
                lambda w: float(w @ cov_vals @ w),
                x0, method="SLSQP", bounds=bounds, constraints=constraints,
            )
        else:  # max_sharpe
            res = minimize(
                neg_sharpe,
                x0, method="SLSQP", bounds=bounds, constraints=constraints,
            )

        if not res.success:
            # SLSQP can fail for ill-conditioned covariances — fall back
            raise RuntimeError(res.message)
        w = pd.Series(res.x, index=tickers)
    except Exception:
        # Closed-form min-variance fallback (long-only via clipping)
        try:
            inv = np.linalg.pinv(cov_vals)
            ones = np.ones(n)
            w_unconstrained = inv @ ones / (ones @ inv @ ones)
            if long_only:
                w_unconstrained = np.clip(w_unconstrained, 0.0, None)
            w_unconstrained /= w_unconstrained.sum() or 1.0
            w = pd.Series(w_unconstrained, index=tickers)
        except Exception as e:
            return {
                "available": False,
                "reason": f"Optimisation failed: {e}",
            }

    # Stats
    port_mu = float(w.values @ mu_vals)
    port_vol = float(np.sqrt(w.values @ cov_vals @ w.values))
    sharpe = ((port_mu - risk_free_rate) / port_vol
              if port_vol > 1e-12 else 0.0)

    # Small efficient-frontier grid for plotting
    frontier = _build_frontier(mu_vals, cov_vals, tickers,
                               long_only=long_only, n_points=25)

    return {
        "available": True,
        "method": objective,
        "weights": w,
        "expected_return": port_mu,
        "expected_vol": port_vol,
        "sharpe": float(sharpe),
        "concentration_hhi": float(np.sum(w.values ** 2)),
        "expected_returns_used": mu,
        "covariance": cov,
        "frontier": frontier,
        "params": {
            "long_only": bool(long_only),
            "use_shrinkage": bool(use_shrinkage),
            "risk_free_rate": float(risk_free_rate),
            "target_return": (None if target_return is None
                              else float(target_return)),
        },
    }


def _build_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: Sequence[str],
    long_only: bool = True,
    n_points: int = 25,
) -> Dict[str, np.ndarray]:
    """Build the efficient frontier curve via target-return scanning."""
    try:
        from scipy.optimize import minimize
    except Exception:
        return {"target_returns": np.array([]),
                "frontier_vols": np.array([])}

    n = len(mu)
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    rets_grid = np.linspace(mu.min(), mu.max(), n_points)
    vols = np.full(n_points, np.nan)

    for i, target in enumerate(rets_grid):
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        x0 = np.full(n, 1.0 / n)
        try:
            res = minimize(
                lambda w: float(w @ cov @ w),
                x0, method="SLSQP", bounds=bounds, constraints=cons,
            )
            if res.success:
                vols[i] = float(np.sqrt(res.fun))
        except Exception:
            continue

    valid = ~np.isnan(vols)
    return {
        "target_returns": rets_grid[valid],
        "frontier_vols": vols[valid],
    }


# ── 4. Black-Litterman ─────────────────────────────────────────────

def black_litterman(
    returns_or_prices: pd.DataFrame,
    market_caps: Optional[pd.Series] = None,
    views: Optional[Dict[str, Any]] = None,
    is_prices: bool = True,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    use_shrinkage: bool = True,
) -> Dict[str, Any]:
    """
    Black-Litterman posterior expected returns + downstream MVO.

    Algorithm
    ---------
    1. Π = δ * Σ * w_mkt        (implied equilibrium returns)
    2. Posterior mean
            μ_BL = [(τΣ)^-1 + P^T Ω^-1 P]^-1
                   [(τΣ)^-1 Π + P^T Ω^-1 Q]
    3. Run mean-variance optimisation on μ_BL to get final weights.

    Parameters
    ----------
    returns_or_prices : pd.DataFrame
    market_caps : pd.Series, optional
        Per-ticker market cap.  Defines the equilibrium weights w_mkt.
        If None, equal-weight is used as the prior.
    views : dict, optional
        Views in either of two forms:
            {"absolute": {"AAPL": 0.12, "MSFT": 0.10}}
                expected annualised returns for individual tickers, OR
            {"relative": [{"long": "AAPL", "short": "MSFT", "delta": 0.03}]}
                "AAPL will outperform MSFT by 3%".
        If None, the posterior collapses to the prior (no view).
    risk_aversion : float
        δ in the equilibrium step.  ~2.5 is a common choice.
    tau : float
        Scales prior uncertainty.  Common values 0.025 - 0.10.
    risk_free_rate, long_only, use_shrinkage : forwarded to mean_variance.

    Returns
    -------
    dict with keys: available, weights, mu_prior, mu_posterior, omega,
    P, Q, plus all keys returned by the downstream MVO call.
    """
    rets = (_returns_from_prices(returns_or_prices)
            if is_prices else returns_or_prices.dropna())
    if rets.empty or len(rets.columns) < 2:
        return {"available": False, "reason": "Need >= 2 assets."}

    tickers = list(rets.columns)
    n = len(tickers)

    # Covariance (annualised)
    cov = (_ledoit_wolf_shrinkage(rets) if use_shrinkage
           else rets.cov() * TRADING_DAYS)
    Sigma = cov.reindex(index=tickers, columns=tickers).values

    # Equilibrium weights (market-cap or equal-weight)
    if market_caps is not None:
        mc = market_caps.reindex(tickers).astype(float).fillna(0.0)
        if mc.sum() > 0:
            w_mkt = (mc / mc.sum()).values
        else:
            w_mkt = np.full(n, 1.0 / n)
    else:
        w_mkt = np.full(n, 1.0 / n)

    # Π — implied equilibrium returns
    pi = risk_aversion * Sigma @ w_mkt

    # Build P (k × n) and Q (k,) from views
    P_rows: list = []
    Q_vals: list = []
    if views:
        if "absolute" in views:
            for tkr, q in views["absolute"].items():
                if tkr in tickers:
                    row = np.zeros(n)
                    row[tickers.index(tkr)] = 1.0
                    P_rows.append(row)
                    Q_vals.append(float(q))
        if "relative" in views:
            for v in views["relative"]:
                long_tkr = v.get("long")
                short_tkr = v.get("short")
                delta = float(v.get("delta", 0.0))
                if long_tkr in tickers and short_tkr in tickers:
                    row = np.zeros(n)
                    row[tickers.index(long_tkr)] = 1.0
                    row[tickers.index(short_tkr)] = -1.0
                    P_rows.append(row)
                    Q_vals.append(delta)

    if P_rows:
        P = np.array(P_rows)
        Q = np.array(Q_vals)
        # Ω: diagonal, view variance proportional to diag(P τΣ P^T)
        Omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))
        Omega = np.where(Omega <= 0, 1e-9, Omega)
        try:
            tau_sigma_inv = np.linalg.pinv(tau * Sigma)
            omega_inv = np.linalg.pinv(Omega)
            A = tau_sigma_inv + P.T @ omega_inv @ P
            b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
            mu_bl = np.linalg.pinv(A) @ b
        except np.linalg.LinAlgError:
            mu_bl = pi
    else:
        P = np.zeros((0, n))
        Q = np.zeros(0)
        Omega = np.zeros((0, 0))
        mu_bl = pi

    mu_posterior = pd.Series(mu_bl, index=tickers)
    mu_prior = pd.Series(pi, index=tickers)

    # Downstream MVO with the posterior μ
    mvo = mean_variance(
        rets, is_prices=False, objective="max_sharpe",
        risk_free_rate=risk_free_rate, long_only=long_only,
        use_shrinkage=use_shrinkage,
        expected_returns=mu_posterior,
    )
    if not mvo.get("available"):
        return mvo

    out = dict(mvo)
    out.update({
        "method": "black_litterman",
        "mu_prior": mu_prior,
        "mu_posterior": mu_posterior,
        "P": P, "Q": Q, "Omega": Omega,
        "equilibrium_weights": pd.Series(w_mkt, index=tickers),
        "params_bl": {
            "risk_aversion": float(risk_aversion),
            "tau": float(tau),
            "n_views": int(len(P_rows)),
        },
    })
    return out


# ── HRP re-export ──────────────────────────────────────────────────

def hierarchical_risk_parity(
    returns_or_prices: pd.DataFrame,
    is_prices: bool = True,
    linkage_method: str = "single",
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Thin wrapper that delegates to the batch-1 ``quant.hrp`` module
    and harmonises the output schema with the other allocators here.

    Returns a dict with the same SHARED keys (available, weights,
    expected_return, expected_vol, sharpe, concentration_hhi, method)
    plus all the rich HRP-specific outputs from the batch-1 module
    (linkage_matrix, dendrogram_order, sorted_tickers, surface_data, ...).
    """
    try:
        from . import hrp as _hrp_module  # type: ignore
    except (ImportError, ValueError):
        # When run as a flat script outside the quant package
        import hrp as _hrp_module  # type: ignore

    if is_prices:
        result = _hrp_module.hrp_from_prices(
            _clean_prices_df(returns_or_prices),
            linkage_method=linkage_method,
        )
    else:
        result = _hrp_module.compute_hrp(
            returns_or_prices.dropna(),
            linkage_method=linkage_method,
        )
    if not result.get("available"):
        return result

    # Compute the standard stats the other allocators report
    rets_df = (_returns_from_prices(returns_or_prices)
               if is_prices else returns_or_prices.dropna())
    stats = _portfolio_stats(
        result["weights"], rets_df, risk_free_rate=risk_free_rate,
    )
    out = dict(result)
    out["method"] = "hierarchical_risk_parity"
    out.update(stats)
    return out


# ── comparison helper ──────────────────────────────────────────────

def compare_allocations(
    prices: pd.DataFrame,
    conviction: Optional[pd.Series] = None,
    market_caps: Optional[pd.Series] = None,
    views: Optional[Dict[str, Any]] = None,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
) -> Dict[str, Any]:
    """
    Run all available allocators and return their weights / stats
    side-by-side for dashboard comparison.

    Returns
    -------
    dict with keys:
        weights : pd.DataFrame  (tickers × method)
        stats : pd.DataFrame    (method × [exp_ret, exp_vol, sharpe, hhi])
        full : dict             {method_name: full result dict}
    """
    full: Dict[str, Any] = {}

    full["volatility_parity"] = volatility_parity(
        prices, is_prices=True, risk_free_rate=risk_free_rate,
    )
    full["inverse_variance"] = volatility_parity(
        prices, is_prices=True, use_inverse_variance=True,
        risk_free_rate=risk_free_rate,
    )
    if conviction is not None:
        full["conviction_vol_hybrid"] = conviction_vol_hybrid(
            prices, conviction=conviction, is_prices=True,
            risk_free_rate=risk_free_rate, long_only=long_only,
        )
    full["mean_variance_max_sharpe"] = mean_variance(
        prices, is_prices=True, objective="max_sharpe",
        risk_free_rate=risk_free_rate, long_only=long_only,
    )
    full["mean_variance_min_variance"] = mean_variance(
        prices, is_prices=True, objective="min_variance",
        risk_free_rate=risk_free_rate, long_only=long_only,
    )
    full["black_litterman"] = black_litterman(
        prices, market_caps=market_caps, views=views,
        is_prices=True, risk_free_rate=risk_free_rate, long_only=long_only,
    )
    try:
        full["hrp"] = hierarchical_risk_parity(
            prices, is_prices=True, risk_free_rate=risk_free_rate,
        )
    except Exception as e:
        full["hrp"] = {"available": False, "reason": str(e)}

    # Stack into comparison frames
    weight_cols = {}
    stat_rows = []
    for name, res in full.items():
        if not res.get("available"):
            continue
        weight_cols[name] = res["weights"]
        stat_rows.append({
            "method": name,
            "expected_return": res.get("expected_return", np.nan),
            "expected_vol": res.get("expected_vol", np.nan),
            "sharpe": res.get("sharpe", np.nan),
            "concentration_hhi": res.get("concentration_hhi", np.nan),
        })

    weights_df = pd.DataFrame(weight_cols).fillna(0.0) if weight_cols else pd.DataFrame()
    stats_df = pd.DataFrame(stat_rows).set_index("method") if stat_rows else pd.DataFrame()

    return {
        "weights": weights_df,
        "stats": stats_df,
        "full": full,
    }
