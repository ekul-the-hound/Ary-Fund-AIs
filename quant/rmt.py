"""
Random Matrix Theory (RMT) — Marchenko-Pastur correlation cleaner.

Empirical correlation matrices estimated from finite samples are
contaminated by noise.  Marchenko-Pastur (1967) gives the EXACT
distribution of eigenvalues for a pure-noise correlation matrix
of N assets and T observations:

    λ_+  =  (1 + sqrt(N/T))^2     upper edge of the noise bulk
    λ_-  =  (1 - sqrt(N/T))^2     lower edge

Eigenvalues OUTSIDE this bulk carry genuine market signal:
    - The largest eigenvalue typically corresponds to the "market mode"
      (a single factor that lifts/drops all assets together).
    - The next few large eigenvalues correspond to sectoral or
      macroeconomic factors.
    - Everything inside [λ_-, λ_+] is noise — and using it directly
      in portfolio optimization is what makes Markowitz unstable.

RMT cleaning recipe (from Laloux et al. 1999, used in the reference):
    1. Compute the empirical correlation matrix C.
    2. Eigen-decompose:  C = V Λ V^T.
    3. For all  λ_i  <  λ_+, replace them with their mean
       (so total variance = N is preserved).
    4. Reconstruct  C_clean = V Λ_clean V^T.
    5. Restore the diagonal to 1 and clip to [-1, 1].

Reference
---------
quant-traderr-lab / RMT_Correlation_Filter / RMT_Pipeline.py

Outputs
-------
- Empirical eigenvalue spectrum
- Theoretical Marchenko-Pastur PDF for overlay
- Cleaned correlation matrix
- Number of "signal" eigenvalues (those above λ_+)
- Variance fraction explained by the signal eigenvalues
- The cleaned matrix is what should go into downstream portfolio
  optimisation (HRP, mean-variance, etc.)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _clean_prices_df(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    return prices.dropna(axis=1, how="all").ffill().dropna()


# ── Marchenko-Pastur PDF ────────────────────────────────────────────

def marchenko_pastur_pdf(
    eigenvalues: np.ndarray,
    Q: float,
    sigma2: float = 1.0,
) -> np.ndarray:
    """
    Theoretical Marchenko-Pastur eigenvalue density.

        f(λ) = (Q / (2π σ²)) * sqrt((λ_+ - λ)(λ - λ_-)) / λ
        for  λ ∈ [λ_-, λ_+]

    Parameters
    ----------
    eigenvalues : np.ndarray
        Points at which to evaluate the density.
    Q : float
        T / N, the aspect ratio (must be > 1).
    sigma2 : float
        Underlying variance (1.0 for a correlation matrix).
    """
    lam_plus = sigma2 * (1 + np.sqrt(1.0 / Q)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(1.0 / Q)) ** 2

    pdf = np.zeros_like(eigenvalues, dtype=float)
    mask = (eigenvalues > lam_minus) & (eigenvalues < lam_plus)
    if mask.any():
        x = eigenvalues[mask]
        pdf[mask] = (
            Q / (2 * np.pi * sigma2 * x)
            * np.sqrt(np.clip((lam_plus - x) * (x - lam_minus), 0, None))
        )
    return pdf


# ── core RMT filter ────────────────────────────────────────────────

def rmt_filter(
    correlation_matrix: np.ndarray,
    T: int,
    keep_market_mode: bool = True,
) -> Dict[str, Any]:
    """
    Apply Marchenko-Pastur eigenvalue cleaning to a correlation matrix.

    Parameters
    ----------
    correlation_matrix : np.ndarray (N x N)
    T : int
        Number of return observations used to compute the matrix.
    keep_market_mode : bool
        If True, the LARGEST eigenvalue (the market factor) is preserved
        as-is even if a user inadvertently sets it to noise.

    Returns
    -------
    dict with keys:
        empirical_eigenvalues : np.ndarray (descending)
        empirical_eigenvectors : np.ndarray
        cleaned_eigenvalues : np.ndarray
        cleaned_matrix : np.ndarray
        lambda_plus, lambda_minus : float    MP bulk edges
        Q : float                            T / N
        n_signal_eigenvalues : int           count above λ_+
        signal_variance_fraction : float     fraction of total variance
        market_mode_eigenvalue : float       largest eigenvalue
    """
    C = np.asarray(correlation_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("correlation_matrix must be square 2-D.")
    N = C.shape[0]

    # Symmetrize defensively (numerical noise from estimation)
    C = (C + C.T) / 2.0

    # 1. Eigen-decomposition (eigh = ascending order)
    evals, evecs = np.linalg.eigh(C)
    # Reorder descending for readability
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # 2. Marchenko-Pastur bulk edges
    Q = T / N
    if Q <= 1:
        # MP only defined for Q > 1; fall back to no filtering
        return {
            "available": False,
            "reason": (f"Q = T/N = {Q:.2f} <= 1; need more observations "
                       f"than assets (T > N)."),
        }
    lam_plus = (1 + np.sqrt(1.0 / Q)) ** 2
    lam_minus = (1 - np.sqrt(1.0 / Q)) ** 2

    # 3. Identify noise eigenvalues
    noise_mask = evals < lam_plus
    if keep_market_mode and noise_mask[0]:
        noise_mask[0] = False  # never wipe the market mode

    n_signal = int((~noise_mask).sum())
    n_noise = int(noise_mask.sum())

    # 4. Replace noise eigenvalues with their mean
    cleaned_evals = evals.copy()
    if n_noise > 0:
        cleaned_evals[noise_mask] = evals[noise_mask].mean()

    # 5. Reconstruct cleaned matrix
    cleaned = evecs @ np.diag(cleaned_evals) @ evecs.T
    np.fill_diagonal(cleaned, 1.0)
    np.clip(cleaned, -1.0, 1.0, out=cleaned)

    signal_var = float(evals[~noise_mask].sum())
    total_var = float(evals.sum())
    signal_frac = signal_var / total_var if total_var > 0 else 0.0

    return {
        "available": True,
        "empirical_eigenvalues": evals,
        "empirical_eigenvectors": evecs,
        "cleaned_eigenvalues": cleaned_evals,
        "cleaned_matrix": cleaned,
        "lambda_plus": float(lam_plus),
        "lambda_minus": float(lam_minus),
        "Q": float(Q),
        "N": int(N),
        "T": int(T),
        "n_signal_eigenvalues": n_signal,
        "n_noise_eigenvalues": n_noise,
        "signal_variance_fraction": float(signal_frac),
        "market_mode_eigenvalue": float(evals[0]),
    }


def rmt_from_prices(
    prices: pd.DataFrame,
    keep_market_mode: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute the correlation matrix from prices,
    then apply RMT cleaning.

    Parameters
    ----------
    prices : pd.DataFrame
        Multi-asset prices (columns = tickers).
    keep_market_mode : bool

    Returns
    -------
    Same as rmt_filter, plus:
        tickers : list[str]
        empirical_correlation : pd.DataFrame
        cleaned_correlation : pd.DataFrame
        mp_density : dict with x and y arrays for plotting the MP curve
    """
    prices = _clean_prices_df(prices)
    tickers = list(prices.columns)
    N = len(tickers)
    if N < 4 or len(prices) < N + 30:
        return {
            "available": False,
            "reason": (f"Need >= 4 assets and T > N + 30; "
                       f"got N={N}, T={len(prices)}."),
        }

    log_ret = np.log(prices / prices.shift(1)).dropna()
    T = len(log_ret)
    corr = log_ret.corr().values

    result = rmt_filter(corr, T=T, keep_market_mode=keep_market_mode)
    if not result.get("available"):
        return result

    # Add named DataFrames and the MP density curve for plotting
    result["tickers"] = tickers
    result["empirical_correlation"] = pd.DataFrame(
        corr, index=tickers, columns=tickers,
    )
    result["cleaned_correlation"] = pd.DataFrame(
        result["cleaned_matrix"], index=tickers, columns=tickers,
    )

    # MP density curve for overlay plots
    lam_max = max(result["empirical_eigenvalues"][0] * 1.05,
                  result["lambda_plus"] * 1.05)
    x = np.linspace(0.0, lam_max, 400)
    y = marchenko_pastur_pdf(x, Q=result["Q"], sigma2=1.0)
    result["mp_density"] = {"x": x, "y": y}

    return result
