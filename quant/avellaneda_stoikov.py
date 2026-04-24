"""
Avellaneda-Stoikov optimal market making.

Implements the closed-form solution from Avellaneda & Stoikov (2008),
"High-frequency trading in a limit order book", for an inventory-averse
market maker quoting around a mid price.

Core quantities
---------------
Reservation (indifference) price:
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

Optimal half-spread around the reservation price:
    delta = (gamma * sigma^2 * (T - t)) / 2 + (1 / gamma) * ln(1 + gamma / k)

Optimal bid / ask quotes:
    p_bid = r - delta
    p_ask = r + delta

Parameters
----------
s       : current mid price
q       : current inventory (signed; + = long, - = short)
t       : current time (0 <= t <= T)
T       : terminal time (end of quoting session)
sigma   : volatility of the mid (per time unit; same unit as T - t)
gamma   : risk-aversion coefficient (> 0). Higher -> tighter inventory
          control, asymmetric quotes bias harder when inventory nonzero.
k       : order-arrival intensity parameter. Poisson arrivals at distance
          delta from the mid have rate lambda(delta) = A * exp(-k * delta);
          only k enters the spread formula directly.

The `k` parameter is typically calibrated from executed-trade data —
see `quant.poisson` for estimators that feed into this.

This module does NOT place orders or talk to any exchange. It computes
structured quote recommendations that downstream execution or UI layers
can consume.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# --- Helpers ------------------------------------------------------------------


def _validate_as_inputs(
    mid: float, T: float, t: float, sigma: float, gamma: float, k: float
) -> None:
    if not math.isfinite(mid) or mid <= 0:
        raise ValueError(f"mid must be positive & finite, got {mid}")
    if not math.isfinite(T) or T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if not math.isfinite(t) or t < 0 or t > T:
        raise ValueError(f"t must be in [0, T], got t={t}, T={T}")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if not math.isfinite(k) or k <= 0:
        raise ValueError(f"k must be > 0, got {k}")


# --- Reservation price --------------------------------------------------------


def reservation_price(
    mid: float,
    inventory: float,
    gamma: float,
    sigma: float,
    time_to_horizon: float,
) -> float:
    """
    Avellaneda-Stoikov reservation (indifference) price.

        r = s - q * gamma * sigma^2 * (T - t)

    The reservation price shifts AWAY from the mid in the direction that
    would reduce inventory: if long (q>0), r < mid, so the market maker's
    quotes are skewed down to encourage selling.
    """
    if not math.isfinite(mid) or mid <= 0:
        raise ValueError(f"mid must be positive & finite, got {mid}")
    if gamma <= 0 or sigma <= 0 or time_to_horizon < 0:
        raise ValueError("gamma > 0, sigma > 0, time_to_horizon >= 0 required")
    q = float(inventory)
    return float(mid - q * gamma * sigma * sigma * time_to_horizon)


# --- Optimal spread -----------------------------------------------------------


def optimal_spread(
    gamma: float,
    sigma: float,
    time_to_horizon: float,
    k: float,
) -> float:
    """
    Half-spread around the reservation price (so full spread = 2 * delta).

        delta = (gamma * sigma^2 * (T - t)) / 2 + (1 / gamma) * ln(1 + gamma / k)

    The first term is the inventory-risk premium (grows with horizon and
    vol); the second is the monopoly rent from Poisson fills (shrinks as
    fills arrive faster, i.e. large k).
    """
    if gamma <= 0 or sigma <= 0 or time_to_horizon < 0 or k <= 0:
        raise ValueError("gamma>0, sigma>0, T-t>=0, k>0 required")
    inv_term = 0.5 * gamma * sigma * sigma * time_to_horizon
    intensity_term = (1.0 / gamma) * math.log(1.0 + gamma / k)
    return float(inv_term + intensity_term)


# --- Full quote computation ---------------------------------------------------


def compute_quotes(
    mid: float,
    inventory: float,
    t: float,
    T: float,
    sigma: float,
    gamma: float,
    k: float,
    tick_size: Optional[float] = None,
    max_inventory: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute optimal bid/ask quotes under Avellaneda-Stoikov.

    Parameters
    ----------
    mid : float
        Current mid price.
    inventory : float
        Signed inventory (positive = long, negative = short).
    t : float
        Current time (same units as T).
    T : float
        Terminal time (end of quoting session). T - t drives urgency.
    sigma : float
        Mid-price volatility (per time unit consistent with T).
    gamma : float
        Risk-aversion coefficient (> 0). Typical values: 0.01 - 1.0
        depending on asset tick and inventory scale.
    k : float
        Order-arrival intensity decay coefficient. Typically calibrated
        from executed fills using lambda(delta) = A * exp(-k * delta).
    tick_size : float, optional
        If provided, bid is floored to a tick and ask is ceiled to a tick.
    max_inventory : float, optional
        If provided and |inventory| >= max_inventory, one-sided quoting
        is recommended to force inventory back toward zero.

    Returns
    -------
    dict
        {
          "mid": float,
          "reservation_price": float,
          "optimal_half_spread": float,
          "bid": float,
          "ask": float,
          "bid_offset_from_mid": float,   # positive number
          "ask_offset_from_mid": float,
          "full_spread": float,
          "inventory_skew": float,        # r - s; sign matches -inventory
          "one_sided": None | "bid_only" | "ask_only",
          "time_to_horizon": float,
          "note": str,
        }
    """
    _validate_as_inputs(mid, T, t, sigma, gamma, k)

    tau = T - t
    r = reservation_price(mid, inventory, gamma, sigma, tau)
    delta = optimal_spread(gamma, sigma, tau, k)

    bid = r - delta
    ask = r + delta

    # Snap to tick if provided
    if tick_size is not None and tick_size > 0:
        bid = math.floor(bid / tick_size) * tick_size
        ask = math.ceil(ask / tick_size) * tick_size

    # Inventory-limit one-sided quoting
    one_sided = None
    note_parts = []
    if max_inventory is not None and max_inventory > 0:
        if inventory >= max_inventory:
            # Too long — only show an ask
            one_sided = "ask_only"
            note_parts.append(
                f"Inventory {inventory:.2f} >= max_inventory {max_inventory:.2f}; "
                f"suppressing bid."
            )
        elif inventory <= -max_inventory:
            one_sided = "bid_only"
            note_parts.append(
                f"Inventory {inventory:.2f} <= -max_inventory {-max_inventory:.2f}; "
                f"suppressing ask."
            )

    # Sanity: bid should not exceed ask
    if bid >= ask:
        note_parts.append(
            "Degenerate: bid >= ask after rounding; "
            "widen tick_size or reduce gamma."
        )

    return {
        "mid": float(mid),
        "reservation_price": float(r),
        "optimal_half_spread": float(delta),
        "bid": float(bid),
        "ask": float(ask),
        "bid_offset_from_mid": float(mid - bid),
        "ask_offset_from_mid": float(ask - mid),
        "full_spread": float(ask - bid),
        "inventory_skew": float(r - mid),
        "one_sided": one_sided,
        "time_to_horizon": float(tau),
        "note": " ".join(note_parts),
    }


# --- Calibration helper -------------------------------------------------------


def calibrate_k_from_fills(
    offsets: np.ndarray,
    fill_counts: np.ndarray,
) -> Dict[str, Any]:
    """
    Calibrate the order-arrival decay coefficient k from empirical fill data.

    Fits the Avellaneda-Stoikov intensity model

        lambda(delta) = A * exp(-k * delta)

    via a log-linear regression:

        log(lambda_i) = log(A) - k * delta_i

    Parameters
    ----------
    offsets : array-like of float
        Distances from mid at which fill counts were observed (positive).
    fill_counts : array-like of float
        Observed fill rate (or count per unit time) at each offset. Must
        be strictly positive for log to be defined; zeros are dropped.

    Returns
    -------
    dict
        {
          "A": float,
          "k": float,
          "r_squared": float,
          "n_points": int,
          "note": str,
        }
    """
    x = np.asarray(offsets, dtype=float).reshape(-1)
    y = np.asarray(fill_counts, dtype=float).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("offsets and fill_counts must have the same shape")

    # Filter to strictly positive, finite values
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0) & (x >= 0)
    x = x[mask]
    y = y[mask]
    n = int(x.size)

    base = {
        "A": float("nan"),
        "k": float("nan"),
        "r_squared": float("nan"),
        "n_points": n,
        "note": "",
    }
    if n < 2:
        base["note"] = "Need at least 2 positive fill-rate observations."
        return base

    log_y = np.log(y)
    # Fit log_y = a + b * x  with  a = log(A),  b = -k
    try:
        b, a = np.polyfit(x, log_y, 1)
    except np.linalg.LinAlgError as e:
        base["note"] = f"Polyfit failed: {e!s}"
        return base

    A = float(math.exp(a))
    k = float(-b)

    # R^2 on log scale
    resid = log_y - (a + b * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    note = ""
    if k <= 0:
        note = "Warning: estimated k <= 0. Fill rate does not decay with distance; model misspecified or data pathological."

    return {
        "A": A,
        "k": k,
        "r_squared": r2,
        "n_points": n,
        "note": note,
    }
