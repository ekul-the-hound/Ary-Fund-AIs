"""
agent/risk_scanner.py
=====================

Rule-based risk-flag engine. **No LLM calls, no DB writes.**

Inputs
------
- Normalised metric snapshot from
  :func:`agent.filing_analyzer.extract_key_metrics_for_agent`.
- Macro dict from ``pipeline.build_agent_context(...)["macro"]``.
- ``agent_risks``: list of free-form risk strings from
  :func:`agent.base_agent.ask_agent` (the LLM's take).

Output
------
A JSON-serialisable dict with stable shape::

    {
      "levels": {
          "fundamental": "HIGH" | "MEDIUM" | "LOW",
          "macro":       "HIGH" | "MEDIUM" | "LOW",
          "market":      "HIGH" | "MEDIUM" | "LOW",
          "agent":       "HIGH" | "MEDIUM" | "LOW",
          "combined":    "HIGH" | "MEDIUM" | "LOW",
      },
      "reasons": {
          "fundamental": [str, ...],
          "macro":       [str, ...],
          "market":      [str, ...],
          "agent":       [str, ...],
      },
    }

All thresholds live in ``config.RISK_THRESHOLDS`` so tuning is a one-file
change. Every sub-scorer is tolerant of missing data: if there's nothing to
assess, it returns ``"LOW"`` with an explanatory reason rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Canonical level ordering, used by the combiner. HIGH > MEDIUM > LOW.
_LEVEL_RANK: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_RANK_LEVEL: Dict[int, str] = {v: k for k, v in _LEVEL_RANK.items()}

# Fallback thresholds, used only if ``config.RISK_THRESHOLDS`` is absent or
# missing a key. Keeps the module importable in isolation (e.g. unit tests).
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "debt_ebitda_high": 3.0,
    "debt_ebitda_medium": 2.0,
    "interest_coverage_low": 2.0,
    "fcf_yield_low": 0.02,
    "recession_prob_high": 0.6,
    "recession_prob_medium": 0.35,
    "vix_high": 28.0,
    "vix_medium": 20.0,
    "drawdown_high": 0.25,
    "drawdown_medium": 0.15,
    "realized_vol_high": 0.45,
    "realized_vol_medium": 0.30,
}


# Alias groups for commonly-misnamed metric keys. The scanner may be called
# with metrics from either the internal snake_case schema
# (``debt_ebitda``, ``p_e``, ``ev_ebitda``) or more industry-standard names
# (``debt_to_ebitda``, ``pe_ratio``, ``ev_to_ebitda``). Rather than fail
# silently when only one side is present, we accept all known spellings.
_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "debt_ebitda":     ("debt_ebitda", "debt_to_ebitda"),
    "net_debt_ebitda": ("net_debt_ebitda", "net_debt_to_ebitda"),
    "p_e":             ("p_e", "pe_ratio", "pe"),
    "ev_ebitda":       ("ev_ebitda", "ev_to_ebitda"),
    "fcf_yield":       ("fcf_yield", "free_cash_flow_yield"),
    "yield_curve_spread": (
        "yield_curve_spread",
        "yield_curve_10y_2y",
        "yield_spread_10y2y",
    ),
}


def _read(d: Dict[str, Any], canonical: str) -> Any:
    """Read ``d[canonical]`` trying every known alias, in priority order.

    Returns the first non-None value found, or None. Falls back to a plain
    ``.get(canonical)`` if the key is not in the alias map.
    """
    for key in _METRIC_ALIASES.get(canonical, (canonical,)):
        v = d.get(key)
        if v is not None:
            return v
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_risk_flags(
    ticker: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    agent_risks: List[str],
    config: Any,
) -> Dict[str, Any]:
    """Compute fundamental / macro / market / agent / combined risk levels.

    Parameters
    ----------
    ticker:
        Ticker symbol, used only for logging.
    metrics:
        Metric snapshot from
        :func:`agent.filing_analyzer.extract_key_metrics_for_agent`.
    macro:
        Macro dict from ``pipeline.build_agent_context``. Expected-ish keys:
        ``recession_probability``, ``vix``, ``yield_curve_inverted``
        (bool), ``yield_curve_spread`` (e.g. 10y-2y in decimal).
    agent_risks:
        List of free-form risk strings produced by the LLM. Strings
        prefixed with ``HIGH:`` / ``MEDIUM:`` / ``LOW:`` are weighted by
        their explicit severity; unprefixed strings count as MEDIUM.
    config:
        The project's ``config`` module. Must expose
        ``RISK_THRESHOLDS: Dict[str, float]``; falls back to sane defaults
        otherwise.

    Returns
    -------
    dict
        See module docstring. Shape is always stable; empty ``reasons`` lists
        get a single ``"no data"`` entry so downstream UIs don't render blanks.
    """
    thresholds = _load_thresholds(config)
    m = metrics or {}
    mc = macro or {}
    ar = list(agent_risks or [])

    fundamental_level, fundamental_reasons = _score_fundamental(m, thresholds)
    macro_level, macro_reasons = _score_macro(mc, thresholds)
    market_level, market_reasons = _score_market(m, thresholds)
    agent_level, agent_reasons = _score_agent(ar)

    levels: Dict[str, str] = {
        "fundamental": fundamental_level,
        "macro": macro_level,
        "market": market_level,
        "agent": agent_level,
    }
    combined = _combine_levels(levels)
    levels["combined"] = combined

    reasons: Dict[str, List[str]] = {
        "fundamental": fundamental_reasons or ["no data"],
        "macro": macro_reasons or ["no data"],
        "market": market_reasons or ["no data"],
        "agent": agent_reasons or ["no data"],
    }

    logger.info(
        "risk_scanner | %s | fund=%s | macro=%s | market=%s | agent=%s | combined=%s",
        ticker,
        fundamental_level,
        macro_level,
        market_level,
        agent_level,
        combined,
    )

    return {"levels": levels, "reasons": reasons}


# =============================================================================
# INTERNAL: SUB-SCORERS
# =============================================================================

def _score_fundamental(
    metrics: Dict[str, Any],
    t: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Assess leverage, cash-flow quality, and coverage.

    Rules (each independent; worst level wins):
        - debt/EBITDA > ``debt_ebitda_high``      -> HIGH
        - debt/EBITDA > ``debt_ebitda_medium``    -> MEDIUM
        - 3+ years of negative FCF                -> HIGH
        - interest coverage < ``interest_coverage_low`` -> HIGH
        - fcf_yield < ``fcf_yield_low``           -> MEDIUM
        - margin_trend == "compressing"           -> MEDIUM
    """
    reasons: List[str] = []
    level = "LOW"

    debt_ebitda = _f(_read(metrics, "debt_ebitda"))
    if debt_ebitda is not None:
        if debt_ebitda > t["debt_ebitda_high"]:
            level = _max_level(level, "HIGH")
            reasons.append(
                f"debt/EBITDA {debt_ebitda:.2f}x exceeds {t['debt_ebitda_high']:.1f}x"
            )
        elif debt_ebitda > t["debt_ebitda_medium"]:
            level = _max_level(level, "MEDIUM")
            reasons.append(
                f"debt/EBITDA {debt_ebitda:.2f}x above {t['debt_ebitda_medium']:.1f}x"
            )

    flag = metrics.get("cash_flow_negative_3_years")
    if flag is True:
        level = _max_level(level, "HIGH")
        reasons.append("3+ consecutive years of negative free cash flow")
    elif flag is None:
        # Flag not pre-computed — derive the streak from raw history if
        # available. Accepts common spellings. An explicit ``False`` is
        # respected as a caller assertion and skips this branch.
        hist = (
            metrics.get("free_cash_flow_3y")
            or metrics.get("fcf_history")
            or metrics.get("free_cash_flow_history")
        )
        if isinstance(hist, (list, tuple)) and len(hist) >= 3:
            tail = [_f(x) for x in hist[-3:]]
            if all(x is not None and x < 0 for x in tail):
                level = _max_level(level, "HIGH")
                reasons.append("3+ consecutive years of negative free cash flow")

    ic = _f(metrics.get("interest_coverage"))
    if ic is not None and ic < t["interest_coverage_low"]:
        level = _max_level(level, "HIGH")
        reasons.append(
            f"interest coverage {ic:.2f}x below {t['interest_coverage_low']:.1f}x"
        )

    fcf_y = _f(_read(metrics, "fcf_yield"))
    if fcf_y is not None and fcf_y < t["fcf_yield_low"]:
        level = _max_level(level, "MEDIUM")
        reasons.append(
            f"FCF yield {fcf_y * 100:.2f}% below {t['fcf_yield_low'] * 100:.1f}%"
        )

    if metrics.get("margin_trend") == "compressing":
        level = _max_level(level, "MEDIUM")
        reasons.append("operating margins compressing")

    return level, reasons


def _score_macro(
    macro: Dict[str, Any],
    t: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Assess recession probability, VIX regime, and yield-curve state."""
    reasons: List[str] = []
    level = "LOW"

    rp = _f(macro.get("recession_probability"))
    if rp is not None:
        if rp > t["recession_prob_high"]:
            level = _max_level(level, "HIGH")
            reasons.append(
                f"recession probability {rp * 100:.0f}% "
                f"above {t['recession_prob_high'] * 100:.0f}%"
            )
        elif rp > t["recession_prob_medium"]:
            level = _max_level(level, "MEDIUM")
            reasons.append(
                f"recession probability {rp * 100:.0f}% "
                f"above {t['recession_prob_medium'] * 100:.0f}%"
            )

    vix = _f(macro.get("vix"))
    if vix is not None:
        if vix > t["vix_high"]:
            level = _max_level(level, "HIGH")
            reasons.append(f"VIX {vix:.1f} in stress regime (>{t['vix_high']:.0f})")
        elif vix > t["vix_medium"]:
            level = _max_level(level, "MEDIUM")
            reasons.append(f"VIX {vix:.1f} elevated (>{t['vix_medium']:.0f})")

    # Yield-curve signal. Accept either an explicit boolean or a numeric spread.
    inverted = macro.get("yield_curve_inverted")
    spread = _f(_read(macro, "yield_curve_spread"))
    if inverted is True or (spread is not None and spread < 0):
        level = _max_level(level, "MEDIUM")
        if spread is not None:
            reasons.append(f"yield curve inverted (10y-2y spread {spread * 100:.0f} bps)")
        else:
            reasons.append("yield curve inverted")

    return level, reasons


def _score_market(
    metrics: Dict[str, Any],
    t: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Assess realised volatility, drawdown, and technical extremes.

    Pulls from fields that ``data/market_data.py`` surfaces alongside
    fundamentals (``realized_vol``, ``drawdown`` / ``max_drawdown``, ``rsi``).
    """
    reasons: List[str] = []
    level = "LOW"

    vol = _f(metrics.get("realized_vol") or metrics.get("volatility"))
    if vol is not None:
        if vol > t["realized_vol_high"]:
            level = _max_level(level, "HIGH")
            reasons.append(
                f"realised vol {vol * 100:.0f}% above {t['realized_vol_high'] * 100:.0f}%"
            )
        elif vol > t["realized_vol_medium"]:
            level = _max_level(level, "MEDIUM")
            reasons.append(
                f"realised vol {vol * 100:.0f}% above {t['realized_vol_medium'] * 100:.0f}%"
            )

    # Drawdown is conventionally stored as a negative decimal or positive magnitude.
    # Normalise to a positive magnitude for comparison.
    dd_raw = _f(metrics.get("drawdown") or metrics.get("max_drawdown"))
    if dd_raw is not None:
        dd = abs(dd_raw)
        if dd > t["drawdown_high"]:
            level = _max_level(level, "HIGH")
            reasons.append(
                f"drawdown {dd * 100:.0f}% exceeds {t['drawdown_high'] * 100:.0f}%"
            )
        elif dd > t["drawdown_medium"]:
            level = _max_level(level, "MEDIUM")
            reasons.append(
                f"drawdown {dd * 100:.0f}% above {t['drawdown_medium'] * 100:.0f}%"
            )

    rsi = _f(metrics.get("rsi"))
    if rsi is not None:
        if rsi >= 75:
            level = _max_level(level, "MEDIUM")
            reasons.append(f"RSI {rsi:.0f} overbought")
        elif rsi <= 25:
            level = _max_level(level, "MEDIUM")
            reasons.append(f"RSI {rsi:.0f} oversold")

    return level, reasons


def _score_agent(agent_risks: List[str]) -> Tuple[str, List[str]]:
    """Weight LLM-reported risks by their explicit severity prefix.

    Spec rule: if the agent produced >3 risks, escalate combined risk. That's
    implemented here as the agent-component level — combined then rolls up.
    """
    if not agent_risks:
        return "LOW", []

    reasons: List[str] = []
    high_count = 0
    medium_count = 0

    for raw in agent_risks:
        s = str(raw).strip()
        if not s:
            continue
        reasons.append(s)
        tag = _extract_severity_tag(s)
        if tag == "HIGH":
            high_count += 1
        elif tag == "MEDIUM":
            medium_count += 1
        else:
            # Untagged risks default to MEDIUM — better to surface than ignore.
            medium_count += 1

    if high_count >= 2 or len(reasons) > 3:
        return "HIGH", reasons
    if high_count >= 1 or medium_count >= 2:
        return "MEDIUM", reasons
    return "LOW", reasons


def _extract_severity_tag(s: str) -> Optional[str]:
    """Return 'HIGH'/'MEDIUM'/'LOW' if the string starts with such a tag."""
    head = s[:8].upper()
    for tag in ("HIGH", "MEDIUM", "LOW"):
        if head.startswith(tag):
            return tag
    return None


# =============================================================================
# INTERNAL: COMBINER
# =============================================================================

def _combine_levels(levels: Dict[str, str]) -> str:
    """Roll up component levels into a single combined risk level.

    Policy:
        - Any single HIGH          -> HIGH
        - 2+ MEDIUM                -> HIGH
        - Exactly 1 MEDIUM         -> MEDIUM
        - Else                     -> LOW
    """
    values = [levels.get(k, "LOW") for k in ("fundamental", "macro", "market", "agent")]
    if any(v == "HIGH" for v in values):
        return "HIGH"
    medium_count = sum(1 for v in values if v == "MEDIUM")
    if medium_count >= 2:
        return "HIGH"
    if medium_count == 1:
        return "MEDIUM"
    return "LOW"


def _max_level(a: str, b: str) -> str:
    """Return the more severe of two risk levels."""
    return _RANK_LEVEL[max(_LEVEL_RANK.get(a, 0), _LEVEL_RANK.get(b, 0))]


# =============================================================================
# INTERNAL: HELPERS
# =============================================================================

def _load_thresholds(config: Any) -> Dict[str, float]:
    """Pull ``RISK_THRESHOLDS`` from ``config``, backfilling missing keys."""
    base = dict(_DEFAULT_THRESHOLDS)
    user_thresholds = getattr(config, "RISK_THRESHOLDS", None) or {}
    if isinstance(user_thresholds, dict):
        base.update({k: float(v) for k, v in user_thresholds.items() if v is not None})
    return base


def _f(x: Any) -> Optional[float]:
    """Coerce to float, returning None on failure or NaN."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v