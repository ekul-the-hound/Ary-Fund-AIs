"""
Position sizing and risk allocation.

Includes:
  - kelly_fraction          : edge/odds -> optimal fraction (with safe clamp).
  - volatility_target_size  : dollar allocation to hit a target portfolio vol.
  - position_sizing_report  : combined, opinionated report with rationale.

All outputs are clamped to sensible ranges to avoid absurd recommendations.
"""

from __future__ import annotations

from typing import Dict, Any

import math


# --- Safety bounds ------------------------------------------------------------

# Cap any fractional allocation at 50% of capital for a single position.
_MAX_FRACTION = 0.50
# Default Kelly haircut — few practitioners use full Kelly.
_DEFAULT_KELLY_MULTIPLIER = 0.5


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


# --- Kelly --------------------------------------------------------------------


def kelly_fraction(edge: float, odds: float = 1.0) -> float:
    """
    Kelly criterion fraction for a discrete bet.

    Using the generalized form:
        f* = edge / odds

    where:
      - `edge` is the expected return per unit staked (e.g. 0.05 = +5%).
      - `odds` is the net payoff multiple on a win (1.0 = even money).

    Parameters
    ----------
    edge : float
    odds : float, default 1.0
        Must be > 0.

    Returns
    -------
    float
        Recommended fraction of capital, clamped to [0, _MAX_FRACTION].
        Negative edges return 0 (no position).
    """
    edge = _safe_float(edge, 0.0)
    odds = _safe_float(odds, 1.0)
    if odds <= 0:
        return 0.0
    if edge <= 0:
        return 0.0
    f = edge / odds
    return float(_clamp(f, 0.0, _MAX_FRACTION))


# --- Vol-target sizing --------------------------------------------------------


def volatility_target_size(
    capital: float,
    target_vol: float,
    asset_vol: float,
) -> float:
    """
    Dollar allocation sized to hit a target portfolio volatility.

        size = capital * (target_vol / asset_vol)

    Both vols must be on the same horizon (e.g. both daily or both annualized).

    Parameters
    ----------
    capital : float
        Total capital available (dollars). Must be > 0.
    target_vol : float
        Desired portfolio volatility for this position (decimal, e.g. 0.01 = 1%).
    asset_vol : float
        Current volatility of the asset (decimal).

    Returns
    -------
    float
        Dollar size, clamped to [0, capital * _MAX_FRACTION].
        Returns 0.0 if inputs are non-positive or invalid.
    """
    capital = _safe_float(capital, 0.0)
    target_vol = _safe_float(target_vol, 0.0)
    asset_vol = _safe_float(asset_vol, 0.0)

    if capital <= 0 or target_vol <= 0 or asset_vol <= 0:
        return 0.0

    raw_size = capital * (target_vol / asset_vol)
    return float(_clamp(raw_size, 0.0, capital * _MAX_FRACTION))


# --- Combined report ----------------------------------------------------------


def position_sizing_report(
    capital: float,
    expected_return: float,
    vol: float,
    risk_budget: float = 0.01,
    kelly_multiplier: float = _DEFAULT_KELLY_MULTIPLIER,
) -> Dict[str, Any]:
    """
    Combined position sizing recommendation.

    Strategy
    --------
    Compute two candidate sizes and take the more conservative:

      1. Vol-target sizing using `risk_budget` as the target portfolio vol
         contribution from this position.
      2. Fractional Kelly using `expected_return / vol^2` as the edge-to-
         variance ratio, scaled by `kelly_multiplier`.

    The minimum is clamped to at most _MAX_FRACTION of capital.

    Parameters
    ----------
    capital : float
        Total capital in dollars. Must be > 0.
    expected_return : float
        Expected return for the position over the horizon (decimal).
    vol : float
        Volatility of the position over the same horizon (decimal).
        Must be > 0 to size a position.
    risk_budget : float, default 0.01
        Target volatility contribution (e.g. 0.01 = 1% portfolio vol).
    kelly_multiplier : float, default 0.5
        Fractional-Kelly haircut in [0, 1].

    Returns
    -------
    dict
        {
          "recommended_dollar_size": float,
          "recommended_percent": float,  # of capital
          "method": str,
          "risk_budget": float,
          "inputs": {...},
          "components": {
              "vol_target_size": float,
              "kelly_size": float,
          },
          "rationale": str,
        }
    """
    capital = _safe_float(capital, 0.0)
    er = _safe_float(expected_return, 0.0)
    vol = _safe_float(vol, 0.0)
    risk_budget = _safe_float(risk_budget, 0.0)
    kelly_multiplier = _clamp(_safe_float(kelly_multiplier, 0.0), 0.0, 1.0)

    base_result = {
        "recommended_dollar_size": 0.0,
        "recommended_percent": 0.0,
        "method": "none",
        "risk_budget": risk_budget,
        "inputs": {
            "capital": capital,
            "expected_return": er,
            "vol": vol,
            "kelly_multiplier": kelly_multiplier,
        },
        "components": {"vol_target_size": 0.0, "kelly_size": 0.0},
        "rationale": "",
    }

    if capital <= 0:
        base_result["rationale"] = "Non-positive capital; no position taken."
        return base_result
    if vol <= 0:
        base_result["rationale"] = "Non-positive volatility input; cannot size a position."
        return base_result
    if er <= 0:
        base_result["rationale"] = (
            "Expected return is non-positive; no long position recommended."
        )
        return base_result

    # 1. Vol-target sizing
    vt_size = volatility_target_size(capital, risk_budget, vol)

    # 2. Fractional Kelly using edge/variance
    kelly_f = _clamp(er / (vol ** 2), 0.0, _MAX_FRACTION)
    kelly_size = capital * kelly_f * kelly_multiplier
    kelly_size = _clamp(kelly_size, 0.0, capital * _MAX_FRACTION)

    # Pick the more conservative
    if vt_size <= kelly_size:
        chosen = vt_size
        method = "volatility_target"
    else:
        chosen = kelly_size
        method = "fractional_kelly"

    pct = chosen / capital if capital > 0 else 0.0

    # Rationale text
    if vol > 2 * risk_budget:
        vol_note = "Position sized down due to elevated volatility."
    elif vol < 0.5 * risk_budget:
        vol_note = "Position sized up given compressed volatility."
    else:
        vol_note = "Volatility is within normal range for the risk budget."

    sharpe_like = er / vol if vol > 0 else 0.0
    edge_note = (
        "Attractive expected return-to-risk ratio."
        if sharpe_like >= 0.5
        else "Modest expected return-to-risk ratio."
    )

    rationale = (
        f"{vol_note} {edge_note} "
        f"Chose the more conservative of vol-target (${vt_size:,.0f}) "
        f"and fractional Kelly (${kelly_size:,.0f}); "
        f"final size is ${chosen:,.0f} ({pct:.1%} of capital) via {method}."
    )

    return {
        "recommended_dollar_size": float(round(chosen, 2)),
        "recommended_percent": float(round(pct, 4)),
        "method": method,
        "risk_budget": risk_budget,
        "inputs": {
            "capital": capital,
            "expected_return": er,
            "vol": vol,
            "kelly_multiplier": kelly_multiplier,
        },
        "components": {
            "vol_target_size": float(round(vt_size, 2)),
            "kelly_size": float(round(kelly_size, 2)),
        },
        "rationale": rationale,
    }
