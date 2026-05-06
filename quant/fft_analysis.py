"""
Fast Fourier Transform — rolling frequency decomposition.

Decomposes a price series into dominant cyclical components using a
ROLLING window approach to ensure NO LOOKAHEAD BIAS. The filtered
signal at time t is computed using ONLY data available up to time t.

For each window [t - W, t]:
    1. Detrend (remove linear fit)
    2. Compute FFT of the detrended window
    3. Keep only the top-N frequency components by amplitude
    4. Inverse FFT to reconstruct the filtered signal
    5. Take the LAST point as the causal estimate for time t

Outputs:
    - Reconstructed (filtered) price series
    - Noise residual  (original - reconstructed)
    - Rolling dominant cycle length (days)
    - Signal-to-noise ratio
    - Full power spectrum of the most recent window

Reference
---------
quant-traderr-lab / FFT / FFT pipeline.py
Rolling FFT with detrending, top-N filtering, and causal point
estimation.

Application
-----------
- Identify dominant market cycles (e.g. 20-day, 60-day oscillations)
- Measure signal-to-noise ratio (higher = more structured price action)
- Detect when the dominant cycle SHIFTS (regime change in frequency domain)
- Volatility-regime filtering: noise residual variance tracks regime changes
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


def compute_rolling_fft(
    prices: pd.Series,
    window_size: int = 126,
    top_n_components: int = 10,
) -> Dict[str, Any]:
    """
    Rolling FFT decomposition of a price series.

    Parameters
    ----------
    prices : pd.Series
        Historical price series.
    window_size : int, default 126
        Rolling window in trading days (~6 months).
    top_n_components : int, default 10
        Number of frequency components to keep per window.

    Returns
    -------
    dict with keys:
        available : bool
        original : np.ndarray
        reconstructed : np.ndarray  (NaN for first window_size bars)
        noise : np.ndarray
        cycle_history : np.ndarray  (dominant cycle in days per window)
        snr_db : float              signal-to-noise ratio in dB
        noise_reduction : float     fraction of variance explained
        avg_dominant_cycle : float  mean dominant cycle (days)
        latest_cycle : float        most recent dominant cycle (days)
        latest_spectrum : dict      power spectrum of the most recent window
    """
    prices = _clean_prices(prices)
    y = prices.values.astype(np.float64)

    if len(y) < window_size + 10:
        return {
            "available": False,
            "reason": (f"Need >= {window_size + 10} bars; "
                       f"got {len(y)}."),
        }

    n_samples = len(y)

    reconstructed = np.full(n_samples, np.nan)
    cycle_history = []
    latest_spectrum = {}

    for t in range(window_size, n_samples + 1):
        window = y[t - window_size: t]
        x_window = np.arange(len(window))

        # 1. Detrend (linear)
        poly_coeffs = np.polyfit(x_window, window, 1)
        trend_line = np.polyval(poly_coeffs, x_window)
        detrended = window - trend_line

        # 2. FFT
        fft_coeffs = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(window))

        # 3. Filter: keep top-N by amplitude
        amplitudes = np.abs(fft_coeffs)
        sorted_idx = np.argsort(amplitudes)[::-1]

        filtered = np.zeros_like(fft_coeffs)
        filtered[sorted_idx[:top_n_components]] = (
            fft_coeffs[sorted_idx[:top_n_components]]
        )

        # 4. Inverse FFT
        recon_detrended = np.fft.ifft(filtered).real
        recon_window = recon_detrended + trend_line

        # 5. Causal: take last point only
        reconstructed[t - 1] = recon_window[-1]

        # 6. Dominant cycle
        valid = [i for i in sorted_idx if freqs[i] > 0]
        if valid:
            top_freq = freqs[valid[0]]
            cycle_history.append(1.0 / top_freq)
        else:
            cycle_history.append(np.nan)

        # Store latest full spectrum
        if t == n_samples:
            pos_mask = freqs > 0
            latest_spectrum = {
                "frequencies": freqs[pos_mask],
                "amplitudes": amplitudes[pos_mask],
                "periods": 1.0 / freqs[pos_mask],
            }

    cycle_arr = np.array(cycle_history)
    noise = y - reconstructed

    # Quality metrics (on valid data only)
    valid_mask = ~np.isnan(reconstructed)
    y_valid = y[valid_mask]
    recon_valid = reconstructed[valid_mask]
    noise_valid = noise[valid_mask]

    if len(y_valid) > 0 and np.mean(noise_valid**2) > 0:
        signal_power = np.mean(recon_valid**2)
        noise_power = np.mean(noise_valid**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        noise_reduction = 1.0 - np.std(noise_valid) / np.std(y_valid)
    else:
        snr_db = float("inf")
        noise_reduction = 1.0

    valid_cycles = cycle_arr[~np.isnan(cycle_arr)]
    avg_cycle = float(np.mean(valid_cycles)) if len(valid_cycles) > 0 else 0.0
    latest_cycle = float(valid_cycles[-1]) if len(valid_cycles) > 0 else 0.0

    return {
        "available": True,
        "params": {
            "window_size": int(window_size),
            "top_n_components": int(top_n_components),
            "n_samples": int(n_samples),
        },
        "original": y,
        "reconstructed": reconstructed,
        "noise": noise,
        "cycle_history": cycle_arr,
        "snr_db": float(snr_db),
        "noise_reduction": float(noise_reduction),
        "avg_dominant_cycle": avg_cycle,
        "latest_cycle": latest_cycle,
        "latest_spectrum": latest_spectrum,
    }


def compute_single_fft(
    prices: pd.Series,
    top_n_components: int = 10,
) -> Dict[str, Any]:
    """
    Non-rolling FFT on the full price series. Useful for a single
    snapshot power-spectrum analysis. NOT causal — uses the full series.

    Returns
    -------
    dict with frequencies, amplitudes, periods, reconstructed, noise.
    """
    prices = _clean_prices(prices)
    y = prices.values.astype(np.float64)

    if len(y) < 30:
        return {"available": False, "reason": f"Need >= 30 bars; got {len(y)}."}

    x = np.arange(len(y))
    poly = np.polyfit(x, y, 1)
    trend = np.polyval(poly, x)
    detrended = y - trend

    coeffs = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(len(y))
    amps = np.abs(coeffs)

    sorted_idx = np.argsort(amps)[::-1]
    filtered = np.zeros_like(coeffs)
    filtered[sorted_idx[:top_n_components]] = coeffs[sorted_idx[:top_n_components]]

    recon = np.fft.ifft(filtered).real + trend
    noise = y - recon

    pos_mask = freqs > 0
    return {
        "available": True,
        "frequencies": freqs[pos_mask],
        "amplitudes": amps[pos_mask],
        "periods": 1.0 / freqs[pos_mask],
        "original": y,
        "reconstructed": recon,
        "noise": noise,
        "trend": trend,
    }
