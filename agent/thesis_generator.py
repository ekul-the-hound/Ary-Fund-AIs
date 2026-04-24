"""
agent/thesis_generator.py
=========================

Heuristic 1-year thesis generator. **No LLM calls (for now).**

Purpose
-------
Produce a single structured opinion per ticker — outlook, price direction,
confidence, and short risk/opportunity lists — that ``main.py`` can persist
via ``portfolio_db.save_agent_opinion``.

Design contract
---------------
This implementation uses simple, auditable heuristics that combine three
bias signals (fundamentals, macro, filings tone) into a single scalar, then
dampen it by the combined risk level. It deliberately returns the **same
JSON shape** an LLM-driven replacement would emit, so swapping to
``base_agent.ask_agent`` later is a drop-in.

Return shape::

    {
      "ticker":            str,
      "outlook":           "bullish" | "neutral" | "bearish",
      "time_horizon":      "1Y",
      "price_direction":   "strong_up" | "moderate_up" | "flat"
                            | "moderate_down" | "strong_down",
      "confidence":        float in [0.0, 1.0],
      "key_risks":         List[str],
      "key_opportunities": List[str],
      "rationale":         str,
      "bias_score":        float in [-1.0, 1.0],   # internal, kept for debugging
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Component weights when combining the three bias signals. Fundamentals
# dominate on a 1-year horizon; macro is a meaningful but secondary driver;
# filings tone is a soft signal that mostly resolves ties.
_WEIGHT_FUNDAMENTALS: float = 0.55
_WEIGHT_MACRO: float = 0.30
_WEIGHT_FILINGS: float = 0.15

# Risk-level dampening multipliers applied to the raw bias score. HIGH
# combined risk compresses conviction; LOW leaves it untouched.
_RISK_DAMPENERS: Dict[str, float] = {
    "HIGH": 0.5,
    "MEDIUM": 0.8,
    "LOW": 1.0,
}

# Bias -> outlook bucket boundaries.
# INVARIANT: _OUTLOOK_BULLISH must equal _DIR_MODERATE_UP, and
# _OUTLOOK_BEARISH must equal _DIR_MODERATE_DOWN.  If these pairs ever
# diverge, a bias in the gap (e.g. 0.16) would produce "bullish" outlook
# but "flat" direction — an internally contradictory verdict.  Keep them
# in sync whenever you tune these constants.
_OUTLOOK_BULLISH: float = 0.15   # was 0.20 — aligned to _DIR_MODERATE_UP
_OUTLOOK_BEARISH: float = -0.15  # was -0.20 — aligned to _DIR_MODERATE_DOWN

# Bias -> price_direction bucket boundaries. Five symmetric buckets.
_DIR_STRONG_UP: float = 0.50
_DIR_MODERATE_UP: float = 0.15   # must equal _OUTLOOK_BULLISH
_DIR_MODERATE_DOWN: float = -0.15  # must equal _OUTLOOK_BEARISH
_DIR_STRONG_DOWN: float = -0.50

# Max items surfaced in key_risks / key_opportunities to keep prompts and
# DB rows compact.
_MAX_LIST_ITEMS: int = 5


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_thesis(
    ticker: str,
    filings_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
) -> Dict[str, Any]:
    """Produce a 1-year thesis dict for a single ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    filings_summary:
        Output of :func:`agent.filing_analyzer.summarize_filings_by_year`.
    metrics:
        Output of :func:`agent.filing_analyzer.extract_key_metrics_for_agent`.
    macro:
        Macro dict from ``pipeline.build_agent_context``.
    risk_flags:
        Output of :func:`agent.risk_scanner.compute_risk_flags`.

    Returns
    -------
    dict
        See module docstring. Shape is always stable even with missing inputs.
    """
    fs = filings_summary or {}
    m = metrics or {}
    mc = macro or {}
    rf = risk_flags or {}

    # Three component biases, each in roughly [-1, +1].
    fund_bias = _score_fundamentals_bias(m)
    macro_bias = _score_macro_bias(mc)
    filings_bias = _score_filings_bias(fs)

    raw_bias = (
        _WEIGHT_FUNDAMENTALS * fund_bias
        + _WEIGHT_MACRO * macro_bias
        + _WEIGHT_FILINGS * filings_bias
    )
    raw_bias = _clip(raw_bias, -1.0, 1.0)

    # Dampen by combined risk level before bucketising.
    adjusted_bias = _apply_risk_penalty(raw_bias, rf)

    outlook = _bias_to_outlook(adjusted_bias)
    price_direction = _bias_to_direction(adjusted_bias)
    confidence = _compute_confidence(
        bias=adjusted_bias,
        fund_bias=fund_bias,
        macro_bias=macro_bias,
        filings_bias=filings_bias,
        risk_flags=rf,
        metrics=m,
    )

    key_risks = _collect_key_risks(rf, fs)
    key_opportunities = _collect_opportunities(m, fs)
    rationale = _build_rationale(
        ticker=ticker,
        fund_bias=fund_bias,
        macro_bias=macro_bias,
        filings_bias=filings_bias,
        adjusted_bias=adjusted_bias,
        rf=rf,
    )

    logger.info(
        "thesis | %s | outlook=%s | direction=%s | conf=%.2f | bias=%.2f "
        "(fund=%.2f, macro=%.2f, filings=%.2f)",
        ticker,
        outlook,
        price_direction,
        confidence,
        adjusted_bias,
        fund_bias,
        macro_bias,
        filings_bias,
    )

    return {
        "ticker": ticker,
        "outlook": outlook,
        "time_horizon": "1Y",
        "price_direction": price_direction,
        "confidence": round(confidence, 3),
        "key_risks": key_risks,
        "key_opportunities": key_opportunities,
        "rationale": rationale,
        "bias_score": round(adjusted_bias, 3),
    }


# =============================================================================
# INTERNAL: COMPONENT BIAS SCORERS
# =============================================================================

def _score_fundamentals_bias(metrics: Dict[str, Any]) -> float:
    """Combine growth, leverage, valuation, quality into a bias in [-1, +1].

    Each sub-signal contributes roughly equally; missing signals contribute
    0.0 (neutral) rather than penalising unnecessarily.
    """
    signals: List[float] = []

    # --- Growth -------------------------------------------------------------
    g3 = _f(metrics.get("revenue_growth_3y"))
    if g3 is not None:
        # Map CAGR to bias: >15% -> +1, 0% -> 0, <-10% -> -1.
        if g3 >= 0.15:
            signals.append(1.0)
        elif g3 >= 0.08:
            signals.append(0.5)
        elif g3 >= 0.0:
            signals.append(g3 / 0.08 * 0.5)  # 0..0.5 linear
        elif g3 >= -0.10:
            signals.append(g3 / 0.10)        # -1..0 linear
        else:
            signals.append(-1.0)

    # --- Leverage (lower is better) ----------------------------------------
    de = _f(metrics.get("debt_ebitda"))
    if de is not None:
        if de < 1.0:
            signals.append(0.6)
        elif de < 2.0:
            signals.append(0.2)
        elif de < 3.0:
            signals.append(-0.2)
        elif de < 4.5:
            signals.append(-0.6)
        else:
            signals.append(-1.0)

    # --- Valuation (FCF yield) ---------------------------------------------
    fy = _f(metrics.get("fcf_yield"))
    if fy is not None:
        if fy >= 0.08:
            signals.append(1.0)
        elif fy >= 0.05:
            signals.append(0.5)
        elif fy >= 0.02:
            signals.append(0.0)
        elif fy >= 0.0:
            signals.append(-0.4)
        else:
            signals.append(-0.9)

    # --- Quality: cash conversion and ROIC ---------------------------------
    cc = _f(metrics.get("cash_conversion"))
    if cc is not None:
        # OCF/NI > 1.0 is a quality signal; < 0.7 is a red flag.
        if cc >= 1.1:
            signals.append(0.5)
        elif cc >= 0.9:
            signals.append(0.2)
        elif cc >= 0.7:
            signals.append(-0.2)
        else:
            signals.append(-0.7)

    roic = _f(metrics.get("roic"))
    if roic is not None:
        if roic >= 0.20:
            signals.append(0.8)
        elif roic >= 0.10:
            signals.append(0.4)
        elif roic >= 0.05:
            signals.append(0.0)
        elif roic >= 0.0:
            signals.append(-0.3)
        else:
            signals.append(-0.8)

    # --- Margin trend (categorical) ----------------------------------------
    trend = metrics.get("margin_trend")
    if trend == "expanding":
        signals.append(0.5)
    elif trend == "compressing":
        signals.append(-0.5)
    # "flat" or None -> 0 (omitted)

    # --- Cash-flow streak override -----------------------------------------
    if metrics.get("cash_flow_negative_3_years") is True:
        signals.append(-1.0)

    if not signals:
        return 0.0
    return _clip(sum(signals) / len(signals), -1.0, 1.0)


def _score_macro_bias(macro: Dict[str, Any]) -> float:
    """Map macro backdrop to a bias. Risk-off regimes skew negative."""
    signals: List[float] = []

    rp = _f(macro.get("recession_probability"))
    if rp is not None:
        # 0% -> +0.5 (supportive), 30% -> 0, 60%+ -> -1
        if rp <= 0.10:
            signals.append(0.5)
        elif rp <= 0.30:
            signals.append(0.2)
        elif rp <= 0.50:
            signals.append(-0.3)
        else:
            signals.append(-1.0)

    vix = _f(macro.get("vix"))
    if vix is not None:
        if vix <= 15:
            signals.append(0.5)
        elif vix <= 20:
            signals.append(0.1)
        elif vix <= 28:
            signals.append(-0.3)
        else:
            signals.append(-0.9)

    spread = _f(macro.get("yield_curve_spread"))
    inverted = macro.get("yield_curve_inverted")
    if inverted is True or (spread is not None and spread < 0):
        signals.append(-0.4)
    elif spread is not None and spread > 0.01:
        signals.append(0.2)

    if not signals:
        return 0.0
    return _clip(sum(signals) / len(signals), -1.0, 1.0)


def _score_filings_bias(filings_summary: Dict[str, Any]) -> float:
    """Map management tone and red flags to a bias."""
    tone = filings_summary.get("management_tone", "neutral")
    red_flags = filings_summary.get("red_flags") or []
    risk_factors = filings_summary.get("risk_factors") or []

    tone_score = {
        "confident": 0.5,
        "neutral": 0.0,
        "cautious": -0.3,
        "defensive": -0.9,
    }.get(tone, 0.0)

    # Each red flag knocks meaningful bias off; cap the hit.
    red_flag_penalty = min(0.8, 0.3 * len(red_flags))

    # A huge list of risk factors is mild negative signal (lazy-prices style).
    rf_penalty = 0.0
    if len(risk_factors) >= 15:
        rf_penalty = 0.2

    bias = tone_score - red_flag_penalty - rf_penalty
    return _clip(bias, -1.0, 1.0)


# =============================================================================
# INTERNAL: COMBINATION, BUCKETISATION, CONFIDENCE
# =============================================================================

def _apply_risk_penalty(bias: float, risk_flags: Dict[str, Any]) -> float:
    """Dampen the magnitude of bias based on combined risk level.

    High combined risk compresses conviction in **either** direction — a
    HIGH-risk bullish thesis should be held with less confidence than a
    LOW-risk one, so we scale both signs.
    """
    combined = _get_combined_level(risk_flags)
    mult = _RISK_DAMPENERS.get(combined, 1.0)
    return _clip(bias * mult, -1.0, 1.0)


def _bias_to_outlook(bias: float) -> str:
    """Three-bucket outlook: bullish / neutral / bearish."""
    if bias >= _OUTLOOK_BULLISH:
        return "bullish"
    if bias <= _OUTLOOK_BEARISH:
        return "bearish"
    return "neutral"


def _bias_to_direction(bias: float) -> str:
    """Five-bucket price direction."""
    if bias >= _DIR_STRONG_UP:
        return "strong_up"
    if bias >= _DIR_MODERATE_UP:
        return "moderate_up"
    if bias <= _DIR_STRONG_DOWN:
        return "strong_down"
    if bias <= _DIR_MODERATE_DOWN:
        return "moderate_down"
    return "flat"


def _compute_confidence(
    bias: float,
    fund_bias: float,
    macro_bias: float,
    filings_bias: float,
    risk_flags: Dict[str, Any],
    metrics: Dict[str, Any],
) -> float:
    """Compute confidence from signal strength, agreement, and data coverage.

    Three factors, multiplied:

        1. Strength:      |bias|, rewarded for conviction.
        2. Agreement:     how aligned the three components are (std dev proxy).
        3. Coverage:      fraction of expected metric fields actually present.

    Further dampened by combined risk level so HIGH risk caps confidence.
    """
    # Strength: 0.0 when bias is 0, 1.0 at extreme.
    strength = min(1.0, abs(bias) * 1.5)

    # Agreement: punish high variance between components. Same-sign components
    # with similar magnitudes score near 1.0; opposite signs collapse this.
    components = [fund_bias, macro_bias, filings_bias]
    mean = sum(components) / len(components)
    variance = sum((c - mean) ** 2 for c in components) / len(components)
    # variance in [0, ~1]; map to agreement in [0, 1].
    agreement = max(0.0, 1.0 - variance)

    # Coverage: how many of the key metric fields are non-None.
    coverage = _metric_coverage(metrics)

    # Combine, then cap by risk.
    base = 0.3 + 0.7 * (0.4 * strength + 0.3 * agreement + 0.3 * coverage)
    combined = _get_combined_level(risk_flags)
    risk_cap = {"HIGH": 0.65, "MEDIUM": 0.85, "LOW": 1.0}.get(combined, 1.0)

    return _clip(base * risk_cap, 0.0, 1.0)


def _metric_coverage(metrics: Dict[str, Any]) -> float:
    """Fraction of core metric fields that are present (non-None)."""
    expected = (
        "revenue_growth_3y", "debt_ebitda", "p_e", "ev_ebitda",
        "fcf_yield", "cash_conversion", "roic", "interest_coverage",
    )
    if not metrics:
        return 0.0
    have = sum(1 for k in expected if metrics.get(k) is not None)
    return have / len(expected)


# =============================================================================
# INTERNAL: RISK / OPPORTUNITY COLLECTION
# =============================================================================

def _collect_key_risks(
    risk_flags: Dict[str, Any],
    filings_summary: Dict[str, Any],
) -> List[str]:
    """Surface the most important risks from risk_flags + filings red flags.

    Priority: HIGH-level risk reasons first, then filings red flags, then
    MEDIUM-level reasons. Deduplicated, capped at _MAX_LIST_ITEMS. Level
    strings are compared case-insensitively so either ``"HIGH"`` or
    ``"high"`` works.

    ``risk_flags["reasons"]`` may arrive in two shapes:

    - **Dict** keyed by component (``fundamental``, ``macro``, ...) — this
      is what the real risk_scanner emits.
    - **List** of free-form strings — simpler callers (and some tests) use
      this flat shape.

    Both are accepted; the dict form enables the HIGH-then-MEDIUM priority
    ordering, while the list form just surfaces each reason once tagged by
    the worst component level.
    """
    levels = (risk_flags or {}).get("levels", {})
    reasons = (risk_flags or {}).get("reasons", {})

    def _lvl(key: str) -> str:
        v = levels.get(key)
        return str(v).upper() if v is not None else ""

    def _reasons_for(component: str) -> List[str]:
        """Pull the per-component reasons list, tolerating dict or list."""
        if isinstance(reasons, dict):
            return list(reasons.get(component, []) or [])
        # List form: there's no component breakdown, so we only surface
        # these under the worst component level to avoid duplication.
        return []

    def _flat_reasons() -> List[str]:
        """Any reasons in the flat-list shape (not component-tagged)."""
        if isinstance(reasons, list):
            return [str(x) for x in reasons if x and str(x) != "no data"]
        return []

    ordered: List[str] = []

    # High-priority reasons first (dict form only).
    for component in ("fundamental", "macro", "market", "agent"):
        if _lvl(component) == "HIGH":
            for r in _reasons_for(component):
                if r and r != "no data":
                    ordered.append(f"{component}: {r}")

    # If reasons was a flat list and any component is HIGH, surface the
    # list items — they're all we have.
    if isinstance(reasons, list) and any(
        _lvl(c) == "HIGH" for c in ("fundamental", "macro", "market", "agent")
    ):
        ordered.extend(_flat_reasons())

    # Filings red flags next.
    for rf in (filings_summary or {}).get("red_flags", []) or []:
        ordered.append(f"filing red flag: {rf}")

    # Then medium-priority reasons (dict form only).
    for component in ("fundamental", "macro", "market", "agent"):
        if _lvl(component) == "MEDIUM":
            for r in _reasons_for(component):
                if r and r != "no data":
                    ordered.append(f"{component}: {r}")

    # If reasons was a flat list and no HIGH was surfaced but MEDIUM exists,
    # still surface the list items.
    if (
        isinstance(reasons, list)
        and not any(_lvl(c) == "HIGH" for c in ("fundamental", "macro", "market", "agent"))
        and any(_lvl(c) == "MEDIUM" for c in ("fundamental", "macro", "market", "agent"))
    ):
        ordered.extend(_flat_reasons())

    return _dedup_preserve_order(ordered)[:_MAX_LIST_ITEMS]


def _collect_opportunities(
    metrics: Dict[str, Any],
    filings_summary: Dict[str, Any],
) -> List[str]:
    """Derive simple, data-grounded opportunity statements."""
    out: List[str] = []

    g3 = _f(metrics.get("revenue_growth_3y"))
    if g3 is not None and g3 >= 0.10:
        out.append(f"3Y revenue CAGR {g3 * 100:.1f}% indicates durable top-line growth")

    roic = _f(metrics.get("roic"))
    if roic is not None and roic >= 0.15:
        out.append(f"ROIC {roic * 100:.0f}% suggests capital-efficient compounding")

    fy = _f(metrics.get("fcf_yield"))
    if fy is not None and fy >= 0.06:
        out.append(f"FCF yield {fy * 100:.1f}% offers attractive cash-on-cash return")

    cc = _f(metrics.get("cash_conversion"))
    if cc is not None and cc >= 1.1:
        out.append(f"OCF/NI {cc:.2f}x indicates high earnings quality")

    de = _f(metrics.get("debt_ebitda"))
    if de is not None and de < 1.0:
        out.append(f"Low leverage (debt/EBITDA {de:.2f}x) provides strategic flexibility")

    if metrics.get("margin_trend") == "expanding":
        out.append("Operating margins expanding over trailing period")

    tone = (filings_summary or {}).get("management_tone")
    if tone == "confident" and not (filings_summary or {}).get("red_flags"):
        out.append("Confident management tone with no flagged disclosure issues")

    return out[:_MAX_LIST_ITEMS]


# =============================================================================
# INTERNAL: RATIONALE
# =============================================================================

def _build_rationale(
    ticker: str,
    fund_bias: float,
    macro_bias: float,
    filings_bias: float,
    adjusted_bias: float,
    rf: Dict[str, Any],
) -> str:
    """Short prose explanation of how the verdict was reached."""
    combined = _get_combined_level(rf)
    return (
        f"{ticker}: fundamentals bias {fund_bias:+.2f}, macro bias "
        f"{macro_bias:+.2f}, filings bias {filings_bias:+.2f}. "
        f"Combined risk {combined}. Post-dampening bias {adjusted_bias:+.2f}."
    )


# =============================================================================
# INTERNAL: HELPERS
# =============================================================================

def _get_combined_level(risk_flags: Dict[str, Any]) -> str:
    """Pull the combined risk level, defaulting to MEDIUM if absent.

    Normalizes to uppercase so either ``"HIGH"`` or ``"high"`` input works —
    the real risk scanner emits uppercase, but older callers (and tests
    that simulate older flag dicts) may pass lowercase.
    """
    if not risk_flags:
        return "MEDIUM"
    raw = (risk_flags.get("levels") or {}).get("combined", "MEDIUM")
    level = str(raw).upper() if raw is not None else "MEDIUM"
    if level not in _RISK_DAMPENERS:
        return "MEDIUM"
    return level


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


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out