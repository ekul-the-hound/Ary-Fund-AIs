"""
Value at Risk (VaR) and Expected Shortfall (ES).

Sign convention
---------------
VaR and ES are reported as **positive numbers representing losses**.
For example, a one-day 95% VaR of 0.023 means "a 5% chance of losing at
least 2.3% in one day". ES at 95% is the mean loss *given* the loss
exceeds VaR, also reported positive.

Default method is historical simulation. Parametric (normal) and Monte
Carlo helpers are provided for comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# --- Helpers ------------------------------------------------------------------


def _clean_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _validate_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


# --- Public API ---------------------------------------------------------------


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical-simulation VaR.

    Parameters
    ----------
    returns : pd.Series
    confidence : float, default 0.95
        E.g. 0.95 -> 95% VaR (5th percentile of returns).

    Returns
    -------
    float
        VaR as a positive loss magnitude. NaN if too few observations.
    """
    _validate_confidence(confidence)
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    alpha = 1.0 - confidence
    # Quantile at alpha is typically negative; flip sign so loss is positive.
    q = float(np.quantile(r.values, alpha))
    return float(-q)


def historical_es(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Expected Shortfall (aka Conditional VaR).
    Mean loss in the tail beyond VaR, reported positive.
    """
    _validate_confidence(confidence)
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    alpha = 1.0 - confidence
    q = float(np.quantile(r.values, alpha))
    tail = r[r <= q]
    if tail.empty:
        return float(-q)
    return float(-tail.mean())


def parametric_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Parametric (Gaussian) VaR: assumes returns ~ N(mu, sigma^2).
    Useful as a sanity-check against the historical estimate.
    """
    _validate_confidence(confidence)
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma == 0.0:
        return 0.0
    z = stats.norm.ppf(1.0 - confidence)  # negative
    var = mu + z * sigma  # likely negative
    return float(-var)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_sims: int = 10_000,
    seed: Optional[int] = 42,
) -> float:
    """
    Monte Carlo VaR under a Gaussian fit to historical returns.
    Deterministic given `seed`.
    """
    _validate_confidence(confidence)
    r = _clean_returns(returns)
    if len(r) < 2:
        return float("nan")
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    rng = np.random.default_rng(seed)
    sims = rng.normal(loc=mu, scale=max(sigma, 1e-12), size=int(n_sims))
    q = float(np.quantile(sims, 1.0 - confidence))
    return float(-q)


def var_es_report(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> dict:
    """
    Structured VaR/ES report.

    Parameters
    ----------
    returns : pd.Series
    confidence : float, default 0.95
    method : {"historical", "parametric", "monte_carlo"}
        Which estimator is used for the headline VaR. ES is always
        historical (tail-conditional mean of the sample).

    Returns
    -------
    dict
        {
          "method": str,
          "confidence": float,
          "var": float,          # positive loss
          "es": float,           # positive loss
          "n_obs": int,
          "sign_convention": "positive = loss",
          "summary": str,
        }
    """
    _validate_confidence(confidence)
    r = _clean_returns(returns)
    n_obs = int(len(r))

    if n_obs < 2:
        return {
            "method": method,
            "confidence": confidence,
            "var": float("nan"),
            "es": float("nan"),
            "n_obs": n_obs,
            "sign_convention": "positive = loss",
            "summary": "Not enough observations to compute VaR/ES.",
        }

    if method == "historical":
        var = historical_var(r, confidence)
    elif method == "parametric":
        var = parametric_var(r, confidence)
    elif method == "monte_carlo":
        var = monte_carlo_var(r, confidence)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    es = historical_es(r, confidence)
    pct = int(round(confidence * 100))

    summary = (
        f"At {pct}% confidence over the sample of {n_obs} observations, "
        f"the estimated one-period VaR is {var:.2%} and ES is {es:.2%} "
        f"(losses expressed as positive numbers)."
    )

    return {
        "method": method,
        "confidence": confidence,
        "var": float(var),
        "es": float(es),
        "n_obs": n_obs,
        "sign_convention": "positive = loss",
        "summary": summary,
    }
