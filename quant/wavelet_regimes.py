"""
Wavelet Volatility Regimes — Continuous Wavelet Transform.

A wavelet transform decomposes a time series simultaneously in
TIME and SCALE (≈ frequency).  Where an FFT tells you "the average
spectral content over the whole series", a wavelet tells you "at THIS
moment, the dominant cycle is THIS long".

That makes wavelets the right tool for non-stationary signals like
returns: the dominant volatility scale shifts as regimes change.

Continuous Wavelet Transform with a Morlet wavelet:

    W(s, t) = (1 / √s) ∫  x(τ) ψ*((τ - t) / s) dτ

    ψ(η)  =  π^(-1/4) exp(i ω₀ η) exp(-η²/2)         (complex Morlet)

    POWER  =  |W(s, t)|²

The power surface (scale × time) is the "scalogram".  Hot spots are
localized volatility regimes.

Reference
---------
quant-traderr-lab / Wavelet Transform / Wavelet_Pipeline.py
The reference uses pywt.cwt with 'cmor1.5-1.0'.  This module
reimplements the Morlet CWT via FFT directly so the project doesn't
need PyWavelets installed.

Outputs
-------
    - The full time-frequency power matrix (for heatmap rendering)
    - Global (time-averaged) power spectrum
    - Dominant scale per timestamp (for regime tracking)
    - Cone of influence boundaries (mark unreliable edges)
    - Regime-shift markers based on changes in dominant scale

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


# ── Morlet wavelet CWT (FFT-based) ──────────────────────────────────

def _morlet_cwt(
    signal: np.ndarray,
    scales: np.ndarray,
    omega0: float = 6.0,
) -> np.ndarray:
    """
    Continuous Wavelet Transform with a complex Morlet wavelet,
    computed via FFT.  This is the same construction as Torrence &
    Compo (1998).

    Parameters
    ----------
    signal : 1-D array, length N
    scales : 1-D array of scales (in samples)
    omega0 : Morlet center frequency (6 ≈ standard)

    Returns
    -------
    cwt : complex array of shape (len(scales), N)
    """
    N = len(signal)
    # Pad to next power of 2 for FFT efficiency
    N_pad = int(2 ** np.ceil(np.log2(N)))
    sig_padded = np.zeros(N_pad, dtype=complex)
    sig_padded[:N] = signal - signal.mean()
    fft_signal = np.fft.fft(sig_padded)

    # Angular frequencies
    k = np.zeros(N_pad)
    k[1:N_pad // 2 + 1] = (
        np.arange(1, N_pad // 2 + 1) * 2 * np.pi / N_pad
    )
    k[N_pad // 2 + 1:] = -k[1:N_pad // 2][::-1]

    cwt = np.zeros((len(scales), N), dtype=complex)
    for i, s in enumerate(scales):
        # Morlet wavelet in Fourier space (Torrence & Compo eqn 12)
        # Ψ̂(s ω) = π^(-1/4) * H(ω) * exp(-(s ω - ω₀)²/2)
        sk = s * k
        norm = np.sqrt(2 * np.pi * s) * (np.pi ** -0.25)
        psi_hat = norm * np.exp(-0.5 * (sk - omega0) ** 2) * (k > 0)
        cwt_padded = np.fft.ifft(fft_signal * psi_hat)
        cwt[i, :] = cwt_padded[:N]
    return cwt


def _morlet_period(scale: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """Convert Morlet wavelet scale to Fourier period."""
    return 4.0 * np.pi * scale / (omega0 + np.sqrt(2.0 + omega0 ** 2))


# ── public API ──────────────────────────────────────────────────────

def compute_wavelet_regimes(
    prices: pd.Series,
    n_scales: int = 64,
    min_scale: float = 1.0,
    max_scale_ratio: float = 0.5,
    omega0: float = 6.0,
    smoothing_window: int = 5,
) -> Dict[str, Any]:
    """
    Wavelet-based volatility-regime decomposition of a price series.

    Parameters
    ----------
    prices : pd.Series
        Historical prices.
    n_scales : int
        Number of wavelet scales.
    min_scale : float
        Smallest scale (in samples).
    max_scale_ratio : float
        Largest scale as fraction of series length.
    omega0 : float
        Morlet wavelet center frequency.  6.0 is standard.
    smoothing_window : int
        Optional smoothing applied to the dominant-scale series for
        regime stability (set to 1 to disable).

    Returns
    -------
    dict with keys:
        available : bool
        z_score : np.ndarray
            Standardised log-returns (the input to the CWT).
        power : np.ndarray (n_scales, T)
            Wavelet power |W|².
        scales : np.ndarray
        periods : np.ndarray              Fourier periods (in samples)
        cone_of_influence : np.ndarray (T,)
            Maximum reliable period at each time.
        global_power : np.ndarray (n_scales,)
            Time-averaged power spectrum (for global plot).
        dominant_period : np.ndarray (T,)
            Period with maximum power per timestamp.
        regime_label : np.ndarray (T,)   one of "high_freq" / "mid" / "low_freq"
        regime_shifts : list[int]        indices where the regime changes
        params : dict
    """
    prices = _clean_prices(prices)
    if len(prices) < 60:
        return {
            "available": False,
            "reason": f"Need >= 60 bars; got {len(prices)}.",
        }

    log_ret = np.log(prices / prices.shift(1)).dropna().values
    if log_ret.std() == 0:
        return {"available": False, "reason": "Degenerate returns ~ 0."}

    z = (log_ret - log_ret.mean()) / log_ret.std()
    T = len(z)

    # Build scale grid (log-spaced)
    max_scale = max_scale_ratio * T
    scales = np.logspace(
        np.log10(min_scale), np.log10(max_scale), n_scales,
    )

    # CWT
    cwt = _morlet_cwt(z, scales, omega0=omega0)
    power = np.abs(cwt) ** 2

    # Cone of influence: e-folding time of the Morlet wavelet
    # (Torrence & Compo eqn 12-13).  COI period = √2 * scale * dt-conversion
    coi_periods = np.sqrt(2.0) * _morlet_period(
        np.minimum(np.arange(T), T - 1 - np.arange(T)), omega0=omega0,
    )

    # Periods (Fourier-equivalent)
    periods = _morlet_period(scales, omega0=omega0)

    # Global (time-averaged) power spectrum
    global_power = power.mean(axis=1)

    # Dominant period per timestamp
    dominant_idx = np.argmax(power, axis=0)
    dominant_period = periods[dominant_idx]

    # Smooth dominant-period series
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        dominant_period_smooth = np.convolve(
            dominant_period, kernel, mode="same",
        )
    else:
        dominant_period_smooth = dominant_period

    # Regime labels by dominant period tertile
    p_lo = np.percentile(periods, 33)
    p_hi = np.percentile(periods, 66)
    regime_label = np.empty(T, dtype=object)
    regime_label[dominant_period_smooth <= p_lo] = "high_freq"
    regime_label[
        (dominant_period_smooth > p_lo) & (dominant_period_smooth <= p_hi)
    ] = "mid"
    regime_label[dominant_period_smooth > p_hi] = "low_freq"

    # Regime shift indices: where the label changes
    shifts = list(np.where(regime_label[1:] != regime_label[:-1])[0] + 1)

    return {
        "available": True,
        "z_score": z,
        "power": power,
        "scales": scales,
        "periods": periods,
        "cone_of_influence": coi_periods,
        "global_power": global_power,
        "dominant_period": dominant_period_smooth,
        "regime_label": regime_label,
        "regime_shifts": shifts,
        "params": {
            "n_scales": int(n_scales),
            "omega0": float(omega0),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
            "smoothing_window": int(smoothing_window),
            "n_samples": int(T),
        },
    }