"""
Kelly criterion — full implementation.

Covers:
    - Discrete Kelly (win/loss bet)        -> kelly_discrete
    - Continuous Kelly (Gaussian returns)  -> kelly_continuous
    - Multi-asset Kelly (mean-var optimal) -> kelly_multi_asset
    - Drawdown-constrained Kelly           -> kelly_with_drawdown_cap
    - Combined report with rationale       -> kelly_report

Notes
-----
Full Kelly is the mathematically optimal long-run growth rate but is
*extremely* aggressive in practice: parameter estimation error alone makes
full-Kelly sizing routinely lose 50%+ of capital in realistic backtests.
Practitioners universally use fractional Kelly (typically 0.25x - 0.5x)
and/or a drawdown constraint. Both are first-class citizens here.

Sign convention: all returns are decimal (0.05 = 5%); all fractions are
decimal (0.1 = 10% of capital). Variances are on the same horizon as
expected returns.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


# --- Safety bounds ------------------------------------------------------------

# Absolute hard cap on any single-asset fraction. Kelly alone can recommend
# leverage; this stops 3x / 5x blow-ups from bad parameter estimates.
_MAX_SINGLE_FRACTION = 1.0
# Default fractional-Kelly haircut.
_DEFAULT_KELLY_MULTIPLIER = 0.5


# --- Helpers ------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _as_1d_array(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d_cov(sigma: Sequence[Sequence[float]], n: int) -> np.ndarray:
    arr = np.asarray(sigma, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"cov shape {arr.shape} does not match mu length {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("cov contains non-finite values")
    # Symmetrize to guard against tiny numerical asymmetry
    arr = 0.5 * (arr + arr.T)
    return arr


# --- 1. Discrete Kelly --------------------------------------------------------


def kelly_discrete(
    win_prob: float,
    win_payoff: float = 1.0,
    loss_payoff: float = 1.0,
    fractional: float = 1.0,
) -> float:
    """
    Kelly fraction for a discrete win/loss bet.

        f* = (p * b - q * a) / (a * b)

    where p = win_prob, q = 1-p, b = win_payoff (net), a = loss_payoff (net
    amount risked per unit staked, typically 1.0).

    For even-money bets (a=b=1): f* = p - q = 2p - 1.

    Parameters
    ----------
    win_prob : float in (0, 1)
    win_payoff : float > 0
        Net payoff on a win per unit staked.
    loss_payoff : float > 0
        Net loss per unit staked on a loss. Usually 1.0.
    fractional : float in [0, 1], default 1.0
        Fractional-Kelly haircut. 0.5 is common ("half Kelly").

    Returns
    -------
    float
        Recommended fraction of capital, clamped to [0, _MAX_SINGLE_FRACTION].
        Returns 0 if the bet has no edge.
    """
    p = _safe_float(win_prob, 0.0)
    b = _safe_float(win_payoff, 0.0)
    a = _safe_float(loss_payoff, 0.0)
    mult = _clamp(_safe_float(fractional, 1.0), 0.0, 1.0)

    if not (0.0 < p < 1.0) or b <= 0 or a <= 0:
        return 0.0

    q = 1.0 - p
    f_star = (p * b - q * a) / (a * b)
    if f_star <= 0:
        return 0.0
    return float(_clamp(f_star * mult, 0.0, _MAX_SINGLE_FRACTION))


# --- 2. Continuous Kelly (single asset, Gaussian) ----------------------------


def kelly_continuous(
    expected_return: float,
    variance: float,
    risk_free_rate: float = 0.0,
    fractional: float = 1.0,
) -> float:
    """
    Kelly fraction for a single asset with Gaussian returns.

        f* = (mu - r_f) / sigma^2

    Maximizes expected log-wealth. Same-horizon convention for mu, variance,
    and r_f (e.g. all annual or all daily).

    Parameters
    ----------
    expected_return : float
        Expected arithmetic return mu.
    variance : float > 0
        Variance of returns (NOT standard deviation).
    risk_free_rate : float, default 0.0
    fractional : float in [0, 1], default 1.0

    Returns
    -------
    float
        Kelly fraction, clamped to [0, _MAX_SINGLE_FRACTION]. 0 if no edge.
    """
    mu = _safe_float(expected_return, 0.0)
    var = _safe_float(variance, 0.0)
    rf = _safe_float(risk_free_rate, 0.0)
    mult = _clamp(_safe_float(fractional, 1.0), 0.0, 1.0)

    if var <= 0:
        return 0.0

    edge = mu - rf
    if edge <= 0:
        return 0.0

    f_star = edge / var
    return float(_clamp(f_star * mult, 0.0, _MAX_SINGLE_FRACTION))


# --- 3. Multi-asset Kelly -----------------------------------------------------


def kelly_multi_asset(
    mu: Sequence[float],
    cov: Sequence[Sequence[float]],
    risk_free_rate: float = 0.0,
    fractional: float = 1.0,
    max_gross_leverage: float = 1.0,
    ridge: float = 1e-8,
) -> Dict[str, Any]:
    """
    Multi-asset continuous Kelly.

        f* = Sigma^{-1} (mu - r_f * 1)

    This is the Merton / mean-variance solution in continuous time: the
    vector of fractions that maximizes expected log wealth under a
    Gaussian return model. Interpretation:
      - Positive f_i means long asset i (fraction of capital).
      - Negative f_i means short.
      - Sum(|f_i|) is gross leverage; sum(f_i) is net exposure.

    Parameters
    ----------
    mu : (n,) array-like
        Expected returns per asset (same horizon).
    cov : (n, n) array-like
        Covariance matrix of returns. Symmetrized internally.
    risk_free_rate : float, default 0.0
    fractional : float in [0, 1], default 1.0
        Apply this haircut uniformly to every weight.
    max_gross_leverage : float > 0, default 1.0
        If the raw solution has gross leverage (sum of abs weights) above
        this, all weights are scaled down proportionally. Default 1.0 means
        "no leverage allowed" — change to 2.0 for a 2x book, etc.
    ridge : float, default 1e-8
        Tikhonov regularization added to the diagonal before inversion to
        stabilize near-singular covariance matrices. Set to 0 to disable.

    Returns
    -------
    dict
        {
          "weights": np.ndarray shape (n,),
          "gross_leverage": float,
          "net_exposure": float,
          "expected_return": float,     # w.T @ (mu - rf)
          "expected_variance": float,   # w.T @ Sigma @ w
          "sharpe_like": float,         # mu_p / sigma_p, 0 if var = 0
          "scaled_for_leverage": bool,
          "note": str,
        }
    """
    mu_arr = _as_1d_array(mu, "mu")
    n = mu_arr.size
    cov_arr = _as_2d_cov(cov, n)
    rf = _safe_float(risk_free_rate, 0.0)
    mult = _clamp(_safe_float(fractional, 1.0), 0.0, 1.0)
    max_lev = max(_safe_float(max_gross_leverage, 1.0), 1e-8)
    ridge = max(_safe_float(ridge, 0.0), 0.0)

    # Stabilize covariance
    if ridge > 0:
        cov_reg = cov_arr + ridge * np.eye(n)
    else:
        cov_reg = cov_arr

    excess = mu_arr - rf

    # Solve Sigma w = excess rather than explicit inversion (numerically safer)
    try:
        w_raw = np.linalg.solve(cov_reg, excess)
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse for truly singular covariance
        w_raw = np.linalg.pinv(cov_reg) @ excess

    # Apply fractional-Kelly multiplier
    w = w_raw * mult

    # Scale down if gross leverage exceeds cap
    gross = float(np.sum(np.abs(w)))
    scaled = False
    if gross > max_lev and gross > 0:
        w = w * (max_lev / gross)
        gross = float(np.sum(np.abs(w)))
        scaled = True

    net = float(np.sum(w))
    exp_ret = float(w @ excess)
    exp_var = float(w @ cov_arr @ w)
    sigma_p = math.sqrt(max(exp_var, 0.0))
    sharpe_like = float(exp_ret / sigma_p) if sigma_p > 0 else 0.0

    note_parts = []
    if mult < 1.0:
        note_parts.append(f"Applied fractional Kelly = {mult:.2f}.")
    if scaled:
        note_parts.append(
            f"Scaled weights down to gross leverage cap {max_lev:.2f}."
        )
    if ridge > 0:
        note_parts.append(f"Used ridge = {ridge:.0e} for covariance stability.")

    return {
        "weights": w,
        "gross_leverage": gross,
        "net_exposure": net,
        "expected_return": exp_ret,
        "expected_variance": exp_var,
        "sharpe_like": sharpe_like,
        "scaled_for_leverage": scaled,
        "note": " ".join(note_parts),
    }


# --- 4. Drawdown-constrained Kelly -------------------------------------------


def kelly_with_drawdown_cap(
    expected_return: float,
    variance: float,
    max_drawdown: float = 0.20,
    risk_free_rate: float = 0.0,
    fractional: float = 1.0,
) -> Dict[str, Any]:
    """
    Kelly size capped by a drawdown tolerance.

    Uses the Busseti-Ryu-Boyd result for Gaussian GBM: betting a constant
    fraction f on an asset with Sharpe S means the probability of ever
    hitting a drawdown of D (as a fraction of peak) is bounded by:

        P(max_drawdown >= D) <= D^(1 / f_norm)

    where f_norm = f / f_kelly is the fraction of full Kelly used. Setting
    this bound to a tolerance alpha and solving:

        f_norm <= log(D) / log(alpha)   (both negative, ratio positive)

    We pick f such that P(DD >= max_drawdown) <= 0.10 by default, then take
    the min of that cap and the requested fractional Kelly.

    Parameters
    ----------
    expected_return : float
    variance : float > 0
    max_drawdown : float in (0, 1), default 0.20
        The drawdown level the investor wants to bound the probability of.
    risk_free_rate : float, default 0.0
    fractional : float in [0, 1], default 1.0
        User's baseline fractional-Kelly preference.

    Returns
    -------
    dict
        {
          "fraction": float,             # final recommended fraction
          "full_kelly": float,           # unconstrained continuous Kelly
          "drawdown_cap_fraction": float,# fraction of full-Kelly allowed
          "max_drawdown_target": float,
          "tail_probability": float,     # alpha used, default 0.10
          "binding_constraint": str,     # "fractional" | "drawdown" | "none"
          "note": str,
        }
    """
    mu = _safe_float(expected_return, 0.0)
    var = _safe_float(variance, 0.0)
    dd = _safe_float(max_drawdown, 0.0)
    rf = _safe_float(risk_free_rate, 0.0)
    mult = _clamp(_safe_float(fractional, 1.0), 0.0, 1.0)

    tail_prob = 0.10  # bound: P(DD >= max_drawdown) <= 10%

    base = {
        "fraction": 0.0,
        "full_kelly": 0.0,
        "drawdown_cap_fraction": 0.0,
        "max_drawdown_target": dd,
        "tail_probability": tail_prob,
        "binding_constraint": "none",
        "note": "",
    }

    if var <= 0:
        base["note"] = "Non-positive variance; no position sized."
        return base
    if not (0.0 < dd < 1.0):
        base["note"] = "max_drawdown must be in (0, 1); no position sized."
        return base

    full_kelly = (mu - rf) / var
    if full_kelly <= 0:
        base["note"] = "Non-positive edge; no long position recommended."
        return base

    # Drawdown cap: f_norm = f / full_kelly <= log(dd) / log(tail_prob)
    # Both log values are negative, so the ratio is positive.
    dd_cap_ratio = math.log(dd) / math.log(tail_prob)
    dd_cap_ratio = _clamp(dd_cap_ratio, 0.0, 1.0)  # never exceed full Kelly
    f_from_dd = full_kelly * dd_cap_ratio

    # User's fractional preference
    f_from_user = full_kelly * mult

    # Take the tighter of the two; also enforce global single-asset cap
    if f_from_user <= f_from_dd:
        chosen = f_from_user
        binding = "fractional"
    else:
        chosen = f_from_dd
        binding = "drawdown"

    chosen = _clamp(chosen, 0.0, _MAX_SINGLE_FRACTION)

    note = (
        f"Full Kelly would recommend {full_kelly:.2%}. "
        f"Drawdown cap ({dd:.0%} with {tail_prob:.0%} tail prob) "
        f"permits up to {f_from_dd:.2%}. "
        f"Fractional-Kelly preference ({mult:.2f}x) permits {f_from_user:.2%}. "
        f"Chose {chosen:.2%} ({binding} constraint binds)."
    )

    return {
        "fraction": float(chosen),
        "full_kelly": float(full_kelly),
        "drawdown_cap_fraction": float(dd_cap_ratio),
        "max_drawdown_target": float(dd),
        "tail_probability": tail_prob,
        "binding_constraint": binding,
        "note": note,
    }


# --- 5. Combined report -------------------------------------------------------


def kelly_report(
    capital: float,
    expected_return: float,
    vol: float,
    risk_free_rate: float = 0.0,
    fractional: float = _DEFAULT_KELLY_MULTIPLIER,
    max_drawdown: Optional[float] = 0.20,
) -> Dict[str, Any]:
    """
    End-to-end Kelly sizing for a single asset, with optional drawdown cap.

    Returns dollar size, percent of capital, full-Kelly vs chosen, and a
    plain-English rationale suitable for agent prompts / UI / logs.
    """
    cap = _safe_float(capital, 0.0)
    mu = _safe_float(expected_return, 0.0)
    sigma = _safe_float(vol, 0.0)
    var = sigma * sigma

    base = {
        "recommended_dollar_size": 0.0,
        "recommended_percent": 0.0,
        "fraction": 0.0,
        "full_kelly": 0.0,
        "method": "none",
        "binding_constraint": "none",
        "inputs": {
            "capital": cap,
            "expected_return": mu,
            "vol": sigma,
            "risk_free_rate": _safe_float(risk_free_rate, 0.0),
            "fractional": fractional,
            "max_drawdown": max_drawdown,
        },
        "rationale": "",
    }

    if cap <= 0:
        base["rationale"] = "Non-positive capital; no position taken."
        return base
    if sigma <= 0:
        base["rationale"] = "Non-positive volatility; cannot size a position."
        return base

    if max_drawdown is None:
        full_kelly = kelly_continuous(
            expected_return=mu,
            variance=var,
            risk_free_rate=risk_free_rate,
            fractional=1.0,
        )
        f = kelly_continuous(
            expected_return=mu,
            variance=var,
            risk_free_rate=risk_free_rate,
            fractional=fractional,
        )
        binding = "fractional" if f < full_kelly else "none"
        method = "fractional_kelly"
    else:
        dd_result = kelly_with_drawdown_cap(
            expected_return=mu,
            variance=var,
            max_drawdown=max_drawdown,
            risk_free_rate=risk_free_rate,
            fractional=fractional,
        )
        f = dd_result["fraction"]
        full_kelly = dd_result["full_kelly"]
        binding = dd_result["binding_constraint"]
        method = "drawdown_capped_kelly"

    dollar = cap * f
    pct = f  # f is already fraction of capital

    if full_kelly <= 0:
        rationale = "No positive edge; zero allocation."
    else:
        rationale = (
            f"Full Kelly = {full_kelly:.2%} of capital. "
            f"Recommended = {pct:.2%} (${dollar:,.0f}) via {method}, "
            f"with {binding} constraint binding."
        )

    return {
        "recommended_dollar_size": float(round(dollar, 2)),
        "recommended_percent": float(round(pct, 4)),
        "fraction": float(f),
        "full_kelly": float(full_kelly),
        "method": method,
        "binding_constraint": binding,
        "inputs": base["inputs"],
        "rationale": rationale,
    }
