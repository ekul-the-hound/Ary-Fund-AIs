"""
Hidden Markov Model regime detection.

A probabilistic companion to ``regime.py`` (which is rule-based). This
module fits a Gaussian HMM on daily log-returns and labels each day with
its most likely hidden state. States are then sorted by sample mean
return and remapped to human labels (bearish / neutral / bullish).

Design notes
------------
- Deliberately *in addition to* ``regime.py``, not a replacement. The
  rule-based classifier is deterministic and testable; HMM is
  probabilistic and better at picking up subtle regime transitions.
  A good research habit is to look at both and pay attention when they
  disagree.
- ``hmmlearn`` is a soft dependency. If it is not installed, callers
  get a clear result dict with ``available=False`` rather than a crash.
- Fits are seeded with ``random_state=42`` by default so the same
  inputs produce the same output — important for test stability.

Conventions
-----------
- Input is a price series; internally converted to log returns.
- State labels are consistent across calls: states sorted lowest mean
  return -> highest. With 2 states: ["bearish", "bullish"].
  With 3 states: ["bearish", "neutral", "bullish"].
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# --- Optional HMM support ----------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    _HMMLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    _HMMLEARN_AVAILABLE = False


_LABELS_BY_COUNT: Dict[int, List[str]] = {
    2: ["bearish", "bullish"],
    3: ["bearish", "neutral", "bullish"],
    4: ["crisis", "bearish", "neutral", "bullish"],
}


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


def _empty_result(n_states: int, reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "n_states": n_states,
        "current_state": None,
        "current_label": None,
        "current_probability": None,
        "state_labels": _LABELS_BY_COUNT.get(n_states, []),
        "state_means": [],
        "state_vols": [],
        "transition_matrix": [],
        "_series": {
            "states": pd.Series(dtype=float),
            "labels": pd.Series(dtype=object),
            "probabilities": pd.DataFrame(),
            "returns": pd.Series(dtype=float),
        },
    }


def fit_hmm_regime(
    prices: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit a Gaussian HMM on log returns and label each day with its state.

    Parameters
    ----------
    prices : pd.Series
        Price series, daily cadence assumed.
    n_states : int, default 2
        Number of hidden regimes. 2 = bull/bear, 3 = bear/neutral/bull.
    n_iter : int, default 200
        Baum-Welch iterations. Converges quickly for daily equity data.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        available          : bool - False if hmmlearn missing or fit failed
        reason             : str  - populated when available=False
        n_states           : int
        current_state      : int  - sorted state index (0 = worst regime)
        current_label      : str  - human label ("bullish", "bearish", ...)
        current_probability: float - posterior prob of current_state today
        state_labels       : list[str] - labels for each sorted state
        state_means        : list[float] - daily mean return by sorted state
        state_vols         : list[float] - daily stdev by sorted state
        transition_matrix  : list[list[float]] - row-major, sorted order
        _series:
            states        : pd.Series[int]    - sorted state per bar
            labels        : pd.Series[str]
            probabilities : pd.DataFrame       - one column per sorted state
            returns       : pd.Series[float]   - log returns used for fit
    """
    if not _HMMLEARN_AVAILABLE:
        return _empty_result(
            n_states,
            "hmmlearn is not installed. Install with: pip install hmmlearn",
        )

    if n_states not in _LABELS_BY_COUNT:
        return _empty_result(n_states, f"Unsupported n_states={n_states}")

    prices = _clean_prices(prices)
    min_bars = max(60, 10 * n_states)
    if len(prices) < min_bars:
        return _empty_result(
            n_states,
            f"Not enough data (need >= {min_bars} bars, got {len(prices)}).",
        )

    log_returns = np.log(prices / prices.shift(1)).dropna()
    X = log_returns.values.reshape(-1, 1)

    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        model.fit(X)
        raw_states = model.predict(X)
        raw_probs = model.predict_proba(X)  # shape (T, n_states)
    except Exception as e:  # pragma: no cover - numerical failure
        return _empty_result(n_states, f"HMM fit failed: {e}")

    # Sort states by mean return (ascending). This makes state indices
    # comparable across different fits / tickers: 0 is always the worst
    # regime, n-1 is always the best.
    raw_means = model.means_.flatten()
    raw_vars = np.array([np.sqrt(c[0, 0]) for c in model.covars_])
    sort_order = np.argsort(raw_means)
    remap = {old: new for new, old in enumerate(sort_order)}

    sorted_states = np.array([remap[s] for s in raw_states])
    sorted_probs = raw_probs[:, sort_order]
    sorted_means = raw_means[sort_order]
    sorted_vols = raw_vars[sort_order]
    sorted_trans = model.transmat_[sort_order][:, sort_order]

    labels = _LABELS_BY_COUNT[n_states]
    state_series = pd.Series(sorted_states, index=log_returns.index, name="state")
    label_series = state_series.map(lambda s: labels[s])
    prob_df = pd.DataFrame(
        sorted_probs,
        index=log_returns.index,
        columns=[f"p_{lbl}" for lbl in labels],
    )

    current_state = int(sorted_states[-1])
    current_prob = float(sorted_probs[-1, current_state])

    return {
        "available": True,
        "reason": "",
        "n_states": n_states,
        "current_state": current_state,
        "current_label": labels[current_state],
        "current_probability": current_prob,
        "state_labels": labels,
        "state_means": [float(m) for m in sorted_means],
        "state_vols": [float(v) for v in sorted_vols],
        "transition_matrix": sorted_trans.tolist(),
        "_series": {
            "states": state_series,
            "labels": label_series,
            "probabilities": prob_df,
            "returns": log_returns,
        },
    }
