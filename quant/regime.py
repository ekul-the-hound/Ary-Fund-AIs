"""
Market regime detection.

A pragmatic, rule-based classifier that combines three signals:

  1. Trend         — fast vs slow moving average crossover.
  2. Drawdown      — peak-to-current decline over the window.
  3. Volatility    — current rolling vol vs its own historical median.

These are combined into a single label with a confidence score. The
implementation avoids fragile regime-switching model fits and is
deterministic, fast, and test-friendly.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from quant.volatility import compute_returns


# Drawdown thresholds for "bear" classification
_BEAR_DRAWDOWN = 0.10       # -10% peak-to-current -> leans bearish
_SEVERE_DRAWDOWN = 0.20     # -20% -> strongly bearish

# Volatility classification: compare current vol to its historical median
_HIGH_VOL_MULT = 1.5        # current >= 1.5x median -> high vol
_LOW_VOL_MULT = 0.75        # current <= 0.75x median -> low vol


def _safe_series(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def detect_regime(
    prices: pd.Series,
    window: int = 60,
    fast_ma: int = 20,
    slow_ma: int = 50,
) -> Dict[str, Any]:
    """
    Classify the current market regime from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series (daily cadence assumed but not required).
    window : int, default 60
        Lookback for the volatility state and drawdown reference.
    fast_ma, slow_ma : int
        Moving-average windows for the trend signal.

    Returns
    -------
    dict
        {
          "regime": str,                 # one of: bull, bear, risk_off,
                                         # high_vol, low_vol, neutral
          "confidence": float,           # 0..1
          "trend": "up" | "down" | "flat",
          "volatility_state": "high" | "low" | "normal",
          "signals": {
              "ma_fast_above_slow": bool,
              "recent_drawdown": float,  # positive number, e.g. 0.12 = -12%
              "rolling_volatility": float,
              "median_volatility": float,
              "n_obs": int,
          },
          "note": str,
        }
    """
    if fast_ma >= slow_ma:
        raise ValueError("fast_ma must be < slow_ma")
    if window < 5:
        raise ValueError("window must be >= 5")

    s = _safe_series(prices)
    n_obs = int(len(s))

    if n_obs < max(slow_ma, window) + 2:
        return {
            "regime": "neutral",
            "confidence": 0.0,
            "trend": "flat",
            "volatility_state": "normal",
            "signals": {
                "ma_fast_above_slow": False,
                "recent_drawdown": float("nan"),
                "rolling_volatility": float("nan"),
                "median_volatility": float("nan"),
                "n_obs": n_obs,
            },
            "note": (
                f"Insufficient data: need at least {max(slow_ma, window) + 2} "
                f"observations, got {n_obs}."
            ),
        }

    # --- Trend signal ---------------------------------------------------------
    ma_fast = s.rolling(fast_ma).mean().iloc[-1]
    ma_slow = s.rolling(slow_ma).mean().iloc[-1]
    ma_fast_above_slow = bool(ma_fast > ma_slow)

    # Relative gap between the MAs — used for trend confidence
    ma_gap = float((ma_fast - ma_slow) / ma_slow) if ma_slow else 0.0
    if ma_gap > 0.01:
        trend = "up"
    elif ma_gap < -0.01:
        trend = "down"
    else:
        trend = "flat"

    # --- Drawdown over the window --------------------------------------------
    recent = s.iloc[-window:]
    peak = float(recent.max())
    current = float(recent.iloc[-1])
    recent_drawdown = float(max(0.0, (peak - current) / peak)) if peak > 0 else 0.0

    # --- Volatility state -----------------------------------------------------
    returns = compute_returns(s)
    # Rolling vol series, then compare the latest to its own historical median
    vol_window = min(window, max(10, len(returns) // 4))
    roll_vol = returns.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    roll_vol_clean = roll_vol.dropna()

    if roll_vol_clean.empty:
        current_vol = float(returns.std(ddof=1)) if len(returns) >= 2 else float("nan")
        median_vol = current_vol
    else:
        current_vol = float(roll_vol_clean.iloc[-1])
        median_vol = float(roll_vol_clean.median())

    if median_vol and np.isfinite(median_vol) and median_vol > 0:
        vol_ratio = current_vol / median_vol
    else:
        vol_ratio = 1.0

    if vol_ratio >= _HIGH_VOL_MULT:
        volatility_state = "high"
    elif vol_ratio <= _LOW_VOL_MULT:
        volatility_state = "low"
    else:
        volatility_state = "normal"

    # --- Combine into a single regime label -----------------------------------
    # Priority: severe drawdown > drawdown+high vol > trend > vol state
    regime, confidence, note = _classify(
        trend=trend,
        ma_fast_above_slow=ma_fast_above_slow,
        ma_gap=ma_gap,
        drawdown=recent_drawdown,
        vol_state=volatility_state,
        vol_ratio=vol_ratio,
    )

    return {
        "regime": regime,
        "confidence": float(round(confidence, 3)),
        "trend": trend,
        "volatility_state": volatility_state,
        "signals": {
            "ma_fast_above_slow": ma_fast_above_slow,
            "recent_drawdown": float(round(recent_drawdown, 4)),
            "rolling_volatility": float(round(current_vol, 6)) if np.isfinite(current_vol) else float("nan"),
            "median_volatility": float(round(median_vol, 6)) if np.isfinite(median_vol) else float("nan"),
            "n_obs": n_obs,
        },
        "note": note,
    }


def _classify(
    trend: str,
    ma_fast_above_slow: bool,
    ma_gap: float,
    drawdown: float,
    vol_state: str,
    vol_ratio: float,
) -> tuple[str, float, str]:
    """
    Combine signals into (regime, confidence, note).

    Confidence is a bounded blend of:
      - magnitude of MA gap (capped),
      - drawdown severity,
      - vol deviation from its median.
    """
    # Base confidences per signal, each in [0, 1]
    trend_conf = min(abs(ma_gap) * 20.0, 1.0)          # 5% gap -> 1.0
    dd_conf = min(drawdown / _SEVERE_DRAWDOWN, 1.0)    # 20% dd -> 1.0
    vol_conf = min(abs(np.log(max(vol_ratio, 1e-6))) / np.log(2.0), 1.0)
    # vol_conf: ratio of 2x or 0.5x -> 1.0

    # Severe drawdown dominates
    if drawdown >= _SEVERE_DRAWDOWN:
        conf = max(dd_conf, 0.8)
        return "risk_off", conf, "Severe drawdown dominates classification."

    # Meaningful drawdown + elevated vol -> risk_off
    if drawdown >= _BEAR_DRAWDOWN and vol_state == "high":
        conf = min(1.0, 0.5 + 0.5 * max(dd_conf, vol_conf))
        return "risk_off", conf, "Drawdown paired with elevated volatility."

    # Clear bear without severe drawdown
    if trend == "down" and not ma_fast_above_slow:
        conf = min(1.0, 0.4 + 0.6 * max(trend_conf, dd_conf))
        return "bear", conf, "Downtrend with fast MA below slow MA."

    # Clear bull
    if trend == "up" and ma_fast_above_slow and drawdown < _BEAR_DRAWDOWN:
        conf = min(1.0, 0.4 + 0.6 * max(trend_conf, 1.0 - dd_conf))
        return "bull", conf, "Uptrend with fast MA above slow MA."

    # No clear trend — classify by vol state
    if vol_state == "high":
        return "high_vol", min(1.0, 0.3 + 0.7 * vol_conf), "Flat trend, elevated volatility."
    if vol_state == "low":
        return "low_vol", min(1.0, 0.3 + 0.7 * vol_conf), "Flat trend, compressed volatility."

    return "neutral", 0.3, "No decisive trend, drawdown, or vol signal."
