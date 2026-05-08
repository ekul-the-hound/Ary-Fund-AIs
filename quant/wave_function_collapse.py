"""
Wave Function Collapse — Schrödinger-style price uncertainty.

Borrows the language of quantum mechanics to visualise risk around
a known FUTURE EVENT (earnings, FOMC, options expiry, macro print).

Pre-event:
    - We don't know what the price will be.
    - The probability distribution over prices is the "wave function":
            ψ(x) ∝ exp(-(x - x₀)² / (2σ²)) * exp(i p x)
      where x₀ is the expected center, σ is uncertainty, p is momentum.
    - |ψ(x)|² is a probability density.

Time evolution (free-particle Schrödinger via split-step Fourier):
    Spreads the wave packet over time.  This is the visual analog of
    rising option premium as expiry approaches: the distribution
    "smears out" because more outcomes become possible.

At the measurement date:
    The wave function COLLAPSES — a single realised price is sampled.
    Whether that price is in the ±1σ region or far in the tail is
    informative about whether the market's pre-event pricing was
    correct (or whether there was a regime break).

Practical reading
-----------------
    distance from center  =  surprise
    surprise / sigma      =  z-score of the event
        |z| < 1   →  in-line with pricing
        1 < |z| < 2 →  notable but within 1-sigma envelope
        |z| > 2   →  tail event — pricing was wrong / regime break

Reference
---------
quant-traderr-lab / Wave Function Collapse / Wave Function Collapse pipeline.py
Gaussian packet, free-particle Schrödinger, FFT-based evolution.

Design
------
Pure functions, structured dict returns, no I/O.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── core wave function model ────────────────────────────────────────

def compute_wave_function_collapse(
    prices: pd.Series,
    measurement_date: Optional[Union[str, pd.Timestamp]] = None,
    momentum_lookback: int = 30,
    volatility_lookback: int = 60,
    sigma_multiplier: float = 1.5,
    momentum_scale: float = 0.1,
    evolution_time: float = 0.5,
    n_grid: int = 2048,
) -> Dict[str, Any]:
    """
    Build a Gaussian wave packet ψ(x) for the price distribution prior
    to a measurement date, evolve it under free-particle Schrödinger
    dynamics, and identify the realised price (collapse).

    Parameters
    ----------
    prices : pd.Series
        Historical price series with DatetimeIndex.
    measurement_date : str or pd.Timestamp, optional
        Date of the event / collapse.  Defaults to the LAST date in the series.
    momentum_lookback : int
        Days used to estimate drift/momentum (the phase factor).
    volatility_lookback : int
        Days used to estimate the wave packet width σ.
    sigma_multiplier : float
        Visual spread multiplier on realised vol.  1.5 widens the packet
        slightly relative to the historical std (matches the reference).
    momentum_scale : float
        Phase scaling factor (purely a visual parameter).
    evolution_time : float
        Time evolution duration (in dimensionless units).  Larger
        values produce more spreading.
    n_grid : int
        Grid resolution.  Power of 2 for FFT speed.

    Returns
    -------
    dict with keys:
        available : bool
        x_grid : np.ndarray              price-space grid
        psi_initial : np.ndarray         |ψ_0(x)|² before evolution
        prob_density : np.ndarray        |ψ(x)|² after evolution (normalized)
        center : float                   expected price ⟨x⟩
        sigma : float                    uncertainty (1-σ width)
        momentum : float                 drift (real units)
        realized_price : float
        measurement_date : pd.Timestamp
        z_score : float                  (realized - center) / sigma
        surprise_label : str             "in_line" / "notable" / "tail_event"
        params : dict
    """
    prices = _clean_prices(prices)
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    if len(prices) < volatility_lookback + 1:
        return {
            "available": False,
            "reason": (f"Need >= {volatility_lookback + 1} bars; "
                       f"got {len(prices)}."),
        }

    # Default measurement date = last date
    if measurement_date is None:
        m_date = prices.index[-1]
    else:
        m_date = pd.to_datetime(measurement_date)
        if m_date not in prices.index:
            idx = prices.index.get_indexer([m_date], method="nearest")[0]
            m_date = prices.index[idx]

    # All data strictly BEFORE the measurement date
    pre_mask = prices.index < m_date
    pre = prices[pre_mask]
    if len(pre) < volatility_lookback:
        return {
            "available": False,
            "reason": ("Not enough data prior to measurement_date for the "
                       "requested lookbacks."),
        }

    recent = pre.tail(momentum_lookback)
    vol_window = pre.tail(volatility_lookback)

    center = float(recent.iloc[-1])
    price_std = float(vol_window.std() * sigma_multiplier)
    if price_std <= 0:
        return {"available": False, "reason": "Degenerate volatility ~ 0."}

    # Drift / momentum
    trend = (recent.iloc[-1] - recent.iloc[0]) / max(len(recent), 1)
    momentum = float(trend * momentum_scale)

    realized = float(prices.loc[m_date])

    # Build price-space grid wide enough to cover both wave + realized
    x_min = min(center - price_std * 5, realized - price_std * 2)
    x_max = max(center + price_std * 5, realized + price_std * 2)
    x = np.linspace(x_min, x_max, n_grid)
    dx = x[1] - x[0]

    # 1. Gaussian wave packet ψ_0(x)
    envelope = np.exp(-0.5 * ((x - center) / price_std) ** 2)
    phase = np.exp(1j * momentum * x)
    psi = envelope * phase
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi_initial_density = np.abs(psi) ** 2

    # 2. Free-particle Schrödinger via split-step Fourier
    k = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=dx)
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * np.exp(-0.5j * (k ** 2) * evolution_time)
    psi_evolved = np.fft.ifft(psi_k)

    prob_density = np.abs(psi_evolved) ** 2
    if prob_density.max() > 0:
        prob_density = prob_density / prob_density.max()  # normalize for plot

    # Surprise / collapse classification
    z_score = (realized - center) / price_std
    abs_z = abs(z_score)
    if abs_z < 1.0:
        surprise = "in_line"
    elif abs_z < 2.0:
        surprise = "notable"
    else:
        surprise = "tail_event"

    return {
        "available": True,
        "x_grid": x,
        "psi_initial": psi_initial_density,
        "prob_density": prob_density,
        "center": center,
        "sigma": price_std,
        "momentum": momentum,
        "realized_price": realized,
        "measurement_date": m_date,
        "z_score": float(z_score),
        "surprise_label": surprise,
        "params": {
            "momentum_lookback": int(momentum_lookback),
            "volatility_lookback": int(volatility_lookback),
            "sigma_multiplier": float(sigma_multiplier),
            "momentum_scale": float(momentum_scale),
            "evolution_time": float(evolution_time),
            "n_grid": int(n_grid),
        },
    }