"""
agent/thesis_generator.py
=========================

Heuristic 1-year thesis generator. **No LLM calls (for now).**

Purpose
-------
Produce a structured opinion per ticker — outlook, price direction,
confidence, risks, opportunities, **a multi-section Markdown thesis
note**, and a back-of-envelope **valuation stub** — that ``main.py``
can persist via ``portfolio_db.save_agent_opinion`` and write to disk
as an Obsidian-compatible Markdown file.

Design contract
---------------
This implementation uses simple, auditable heuristics that combine three
bias signals (fundamentals, macro, filings tone) into a single scalar, then
dampen it by the combined risk level. It deliberately returns the **same
JSON shape** an LLM-driven replacement would emit, so swapping to
``base_agent.ask_agent`` later is a drop-in.

Determinism
-----------
Every output field is a pure function of the inputs. No ``random``,
no ``datetime.now()``, no dict-iteration-order dependencies. The
``created`` field in the YAML frontmatter takes its date from
``macro['as_of']`` if present, else from the deterministic placeholder
``'unknown'`` — never from wall-clock time. This is enforced by the
existing ``TestDeterminism.test_same_inputs_produce_same_output`` test,
which does strict ``a == b`` equality on two back-to-back calls.

Return shape::

    {
      # Existing fields (preserved exactly — downstream depends on them)
      "ticker":            str,
      "outlook":           "bullish" | "neutral" | "bearish",
      "time_horizon":      "1Y",
      "price_direction":   "strong_up" | "moderate_up" | "flat"
                            | "moderate_down" | "strong_down",
      "confidence":        float in [0.10, 0.95],
      "key_risks":         List[str],
      "key_opportunities": List[str],
      "rationale":         str,            # unchanged — kept for old UIs
      "bias_score":        float in [-1.0, 1.0],

      # New fields (added for Phase-X multi-section output)
      "short_rationale":   str,            # compact 1-line for UI tables
      "thesis_markdown":   str,            # YAML frontmatter + 6-section MD
      "valuation":         {               # numeric stub, all may be None
          "current_price":      Optional[float],
          "trailing_pe":        Optional[float],
          "forward_pe":         Optional[float],
          "implied_eps":        Optional[float],
          "target_multiple":    Optional[float],
          "fair_value_low":     Optional[float],
          "fair_value_mid":     Optional[float],
          "fair_value_high":    Optional[float],
          "upside_to_mid_pct":  Optional[float],
          "basis":              str,         # human-readable method note
          "missing_inputs":     List[str],   # explicit when data absent
      },
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional


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

# -----------------------------------------------------------------------------
# Multi-section Markdown / valuation / monitoring constants
# -----------------------------------------------------------------------------
# Bull/bear bullets per section. Each section emits 3-6 lines; we cap at 6
# because beyond that the section reads as a list rather than reasoning.
_MAX_BULLETS_PER_SECTION: int = 6

# Valuation: target multiple = base + growth_bonus * growth_factor, capped.
# At 0% growth -> 22x; at 30%+ growth -> 40x (the cap). Tunable but the
# exact numbers matter less than that the function is monotone in growth.
_VAL_BASE_MULTIPLE: float = 22.0
_VAL_MAX_MULTIPLE: float = 40.0
_VAL_MULTIPLE_PER_PCT_GROWTH: float = 60.0  # 30%*60 = 18, plus base = 40 (cap)

# Fair value range half-width as fraction of mid (e.g. 0.15 => ±15%).
_VAL_RANGE_HALF_WIDTH: float = 0.15

# Monitoring triggers (always emitted with these exact predicates so the
# field has a stable shape; values may be templated by the per-ticker call)
_MONITOR_PRICE_MOVE_PCT: float = 0.15           # ±15% from entry
_MONITOR_REVENUE_DECEL_PP: float = 5.0          # >5pp Y/Y decel


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_thesis(
    ticker: str,
    filings_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
    peer_stats: Optional[Dict[str, Any]] = None,
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
    filings_bias = _score_filings_bias(fs, peer_stats=peer_stats)

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

    # ------------------------------------------------------------------
    # NEW multi-section + valuation + monitoring output
    # ------------------------------------------------------------------
    valuation = _build_valuation(m, fund_bias)
    monitoring = _build_monitoring_plan(m, mc, rf)
    bull_bullets = _build_bull_bullets(m, fs, fund_bias, macro_bias, filings_bias)
    bear_bullets = _build_bear_bullets(m, fs, rf, fund_bias, macro_bias, filings_bias)
    catalysts = _build_catalysts(m, mc, fs)
    short_rationale = _build_short_rationale(
        ticker=ticker,
        outlook=outlook,
        price_direction=price_direction,
        adjusted_bias=adjusted_bias,
        confidence=confidence,
        rf=rf,
    )
    thesis_markdown = _build_thesis_markdown(
        ticker=ticker,
        outlook=outlook,
        price_direction=price_direction,
        adjusted_bias=adjusted_bias,
        confidence=confidence,
        fund_bias=fund_bias,
        macro_bias=macro_bias,
        filings_bias=filings_bias,
        rf=rf,
        m=m,
        mc=mc,
        fs=fs,
        bull=bull_bullets,
        bear=bear_bullets,
        catalysts=catalysts,
        key_risks=key_risks,
        valuation=valuation,
        monitoring=monitoring,
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

        # New fields (added without removing or renaming any existing field).
        "short_rationale": short_rationale,
        "thesis_markdown": thesis_markdown,
        "valuation": valuation,
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


def _score_filings_bias(
    filings_summary: Dict[str, Any],
    peer_stats: Optional[Dict[str, Any]] = None,
) -> float:
    """Map management tone, red flags, and (sector-relative) risk count to a bias.

    Parameters
    ----------
    filings_summary:
        Output of ``summarize_filings_by_year`` — carries ``management_tone``,
        ``red_flags`` and ``risk_factor_count`` (true, uncapped).
    peer_stats:
        Optional ``{metric: {mean, std, n}}`` slice for THIS ticker's sector
        (from ``peer_stats.peer_stats_for_sector``). When it contains a
        ``risk_factor_count`` entry, the risk-count penalty is applied
        SECTOR-RELATIVELY. Absent / thin sector -> no risk-count penalty.
    """
    tone = filings_summary.get("management_tone", "neutral")
    red_flags = filings_summary.get("red_flags") or []

    tone_score = {
        "confident": 0.5,
        "neutral": 0.0,
        "cautious": -0.3,
        "defensive": -0.9,
    }.get(tone, 0.0)

    # Each red flag knocks meaningful bias off; cap the hit.
    red_flag_penalty = min(0.8, 0.3 * len(red_flags))

    # Risk-factor COUNT penalty — SECTOR-RELATIVE.
    #
    # An absolute threshold (">=15 risks -> penalty") is useless: every
    # large-cap 10-K lists 15-100+ risk-cue sentences, so it fires
    # universally. What's actually informative is listing materially MORE
    # risk than your sector peers. We z-score the ticker's true
    # ``risk_factor_count`` against the sector distribution and penalize only
    # the right tail. No peer stats (unknown/thin sector) -> no penalty, so
    # the term degrades gracefully to neutral rather than to a false signal.
    rf_penalty = _risk_count_penalty(
        int(filings_summary.get("risk_factor_count") or 0),
        peer_stats,
    )

    bias = tone_score - red_flag_penalty - rf_penalty
    return _clip(bias, -1.0, 1.0)


# How many sample std-devs above the sector mean before the risk count
# counts as a real signal, and the maximum penalty it can apply.
_RISK_Z_THRESHOLD = 1.0   # only the upper tail (>1σ above peers) is penalized
_RISK_Z_FULLSCALE = 3.0   # z at which the penalty saturates
_RISK_MAX_PENALTY = 0.3   # cap, comparable to one red flag


def _risk_count_penalty(
    count: int,
    peer_stats: Optional[Dict[str, Any]],
) -> float:
    """Penalty in [0, _RISK_MAX_PENALTY] for an unusually high risk count.

    Linear ramp from ``_RISK_Z_THRESHOLD`` to ``_RISK_Z_FULLSCALE`` standard
    deviations above the sector mean. Returns 0.0 when peer stats are
    missing, the sector lacks a ``risk_factor_count`` distribution, or the
    std is degenerate — i.e. whenever a sector-relative judgement can't be
    made, so the term never invents a signal.
    """
    if not peer_stats or not isinstance(peer_stats, dict):
        return 0.0
    stats = peer_stats.get("risk_factor_count")
    if not isinstance(stats, dict):
        return 0.0
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None or std <= 0:
        return 0.0
    z = (count - mean) / std
    if z <= _RISK_Z_THRESHOLD:
        return 0.0
    # Linear ramp; clamp to full-scale.
    frac = (z - _RISK_Z_THRESHOLD) / (_RISK_Z_FULLSCALE - _RISK_Z_THRESHOLD)
    frac = max(0.0, min(1.0, frac))
    return _RISK_MAX_PENALTY * frac


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

    Returns a value in [0.10, 0.95]. Designed to **move visibly** between
    tickers — sparse-data tickers should land below 0.40, conviction-
    plus-clean-data tickers above 0.75.

    Five factors, weighted-sum:

        1. Strength      (weight 0.30): |bias|, rewarded for conviction.
        2. Agreement     (weight 0.20): how aligned the three sub-biases are.
        3. Coverage      (weight 0.25): fraction of core metrics present.
        4. Signal count  (weight 0.15): how many sub-biases are non-trivial.
        5. Risk dampener (multiplicative): HIGH=0.7, MEDIUM=0.9, LOW=1.0.

    The previous formula (``0.3 + 0.7 * weighted_sum``) compressed the
    output band into roughly [0.45, 0.65] for typical inputs, which is
    why every ticker's confidence looked the same. The new band is
    [0.10, 0.95] with a 0.5 mid-point only when factors are genuinely
    middling.
    """
    # Strength: 0.0 when bias is 0, 1.0 at |bias| >= 0.5. Scaled up from
    # 1.5 -> 2.0 so a bias of 0.25 reaches strength=0.50 instead of 0.38.
    strength = min(1.0, abs(bias) * 2.0)

    # Agreement: low variance across sub-biases means components agree.
    components = [fund_bias, macro_bias, filings_bias]
    mean = sum(components) / len(components)
    variance = sum((c - mean) ** 2 for c in components) / len(components)
    agreement = max(0.0, 1.0 - variance * 4.0)  # variance ~0.25 -> agreement 0

    # Coverage: how many of the key metric fields are non-None.
    coverage = _metric_coverage(metrics)

    # Signal count: how many sub-biases are meaningful (|x| >= 0.05).
    # If only one of three components has a real signal, drop confidence.
    nontrivial = sum(1 for c in components if abs(c) >= 0.05)
    signal_count_score = nontrivial / 3.0

    # Weighted sum (weights sum to 0.90; remaining 0.10 is the floor).
    weighted = (
        0.10
        + 0.30 * strength
        + 0.20 * agreement
        + 0.25 * coverage
        + 0.15 * signal_count_score
    )

    # Risk dampener (less aggressive than before so MEDIUM doesn't crush
    # confidence on its own).
    combined = _get_combined_level(risk_flags)
    risk_mult = {"HIGH": 0.70, "MEDIUM": 0.90, "LOW": 1.0}.get(combined, 1.0)

    return _clip(weighted * risk_mult, 0.10, 0.95)


def _metric_coverage(metrics: Dict[str, Any]) -> float:
    """Fraction of core metric fields that are present (non-None).

    Returns a value in [0.0, 1.0]. With the expanded snapshot from
    ``filing_analyzer`` (which now also carries ``revenue_growth_yoy``,
    ``gross_margin``, ``operating_margin``, ``profit_margin``, and
    ``free_cash_flow``), the expected list is broader so coverage is
    a more honest signal than before.
    """
    expected = (
        "revenue_growth_3y", "revenue_growth_yoy",
        "gross_margin", "operating_margin",
        "debt_ebitda", "p_e", "ev_ebitda",
        "fcf_yield", "cash_conversion", "roic", "interest_coverage",
        "free_cash_flow", "market_cap",
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
# INTERNAL: SHORT RATIONALE (one-liner for UI table)
# =============================================================================

def _build_short_rationale(
    ticker: str,
    outlook: str,
    price_direction: str,
    adjusted_bias: float,
    confidence: float,
    rf: Dict[str, Any],
) -> str:
    """One-line summary suitable for the dashboard table.

    Pattern: ``"NVDA: bullish · moderate_up · conf 0.78 · bias +0.42 · risk LOW"``
    """
    combined = _get_combined_level(rf)
    return (
        f"{ticker}: {outlook} · {price_direction} · "
        f"conf {confidence:.2f} · bias {adjusted_bias:+.2f} · risk {combined}"
    )


# =============================================================================
# INTERNAL: VALUATION STUB
# =============================================================================

def _build_valuation(
    metrics: Dict[str, Any],
    fund_bias: float,
) -> Dict[str, Any]:
    """Back-of-envelope multiple-times-earnings valuation.

    Returns a dict with all numeric fields (any of which may be ``None``
    if the corresponding input is missing), a ``basis`` string explaining
    the method, and a ``missing_inputs`` list naming every input that
    couldn't be filled. The dict's shape is **stable** regardless of how
    much input data is present — downstream consumers can always look
    up the same keys without ``KeyError``.

    Method
    ------
    1. Recover forward EPS from ``current_price / forward_pe`` (or
       trailing P/E if forward isn't available — flagged in ``basis``).
    2. Compute a target multiple as a monotone function of 3Y revenue
       CAGR: ``target = base + per_pct * growth_pct``, capped at
       ``_VAL_MAX_MULTIPLE``. At 0% growth the multiple equals
       ``_VAL_BASE_MULTIPLE`` (22x); at 30%+ growth it saturates to
       40x. This is a sanity check, not a DCF.
    3. Fair-value range = ``mid × (1 ± _VAL_RANGE_HALF_WIDTH)``.

    Missing-data
    ------------
    If ``current_price`` or any P/E is unavailable, the EPS recovery
    fails and *all* downstream numbers stay ``None``. The Markdown
    renderer then prints "valuation unavailable — missing: [...]"
    instead of a misleading table of zeros.
    """
    missing: List[str] = []

    price = _f(metrics.get("price")) or _f(metrics.get("current_price"))
    trailing_pe = _f(metrics.get("p_e")) or _f(metrics.get("pe_ratio"))
    forward_pe = _f(metrics.get("forward_pe")) or _f(metrics.get("forward_pe_ratio"))

    if price is None:
        missing.append("current_price")
    if forward_pe is None and trailing_pe is None:
        missing.append("p_e")  # at least one PE needed

    # Pick the PE to recover EPS from; prefer forward.
    pe_used = forward_pe if forward_pe is not None else trailing_pe
    pe_label = "forward_pe" if forward_pe is not None else "p_e"

    implied_eps: Optional[float] = None
    if price is not None and pe_used is not None and pe_used > 0:
        implied_eps = price / pe_used

    # Target multiple from growth.
    g3 = _f(metrics.get("revenue_growth_3y"))
    if g3 is None:
        # Fall back to YoY if available; explicit about which.
        g3 = _f(metrics.get("revenue_growth_yoy"))
        growth_source = "revenue_growth_yoy" if g3 is not None else None
    else:
        growth_source = "revenue_growth_3y"

    if g3 is None:
        missing.append("revenue_growth_3y")
        # Without growth, fall back to base multiple (don't extrapolate).
        target_multiple = _VAL_BASE_MULTIPLE
        target_basis = "base multiple (no growth signal)"
    else:
        growth_pct = max(0.0, g3) * 100.0  # decline doesn't lower the floor below base
        target_multiple = min(
            _VAL_MAX_MULTIPLE,
            _VAL_BASE_MULTIPLE + (growth_pct / 100.0) * _VAL_MULTIPLE_PER_PCT_GROWTH,
        )
        # Modest negative tilt if combined fundamentals are weak (bear bias
        # → less generous multiple). +/-2x sensitivity, never crosses base.
        target_multiple = max(
            _VAL_BASE_MULTIPLE * 0.85,
            target_multiple + 2.0 * fund_bias,
        )
        target_multiple = min(_VAL_MAX_MULTIPLE, target_multiple)
        target_basis = f"base {_VAL_BASE_MULTIPLE:.0f}x + growth premium (from {growth_source})"

    fair_value_mid: Optional[float] = None
    fair_value_low: Optional[float] = None
    fair_value_high: Optional[float] = None
    upside_to_mid_pct: Optional[float] = None
    if implied_eps is not None:
        fair_value_mid = implied_eps * target_multiple
        fair_value_low = fair_value_mid * (1.0 - _VAL_RANGE_HALF_WIDTH)
        fair_value_high = fair_value_mid * (1.0 + _VAL_RANGE_HALF_WIDTH)
        if price is not None and price > 0:
            upside_to_mid_pct = (fair_value_mid / price - 1.0) * 100.0

    if implied_eps is None:
        basis = f"valuation unavailable; missing: {', '.join(missing) or '?'}"
    else:
        basis = (
            f"implied EPS = price / {pe_label} = "
            f"{price:.2f} / {pe_used:.2f} = {implied_eps:.2f}; "
            f"target multiple {target_multiple:.1f}x ({target_basis}); "
            f"mid = EPS × target = {implied_eps:.2f} × {target_multiple:.1f} = "
            f"{fair_value_mid:.2f}; range ±{_VAL_RANGE_HALF_WIDTH:.0%}."
        )

    return {
        "current_price": price,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "implied_eps": implied_eps,
        "target_multiple": round(target_multiple, 2),
        "fair_value_low": fair_value_low,
        "fair_value_mid": fair_value_mid,
        "fair_value_high": fair_value_high,
        "upside_to_mid_pct": (
            round(upside_to_mid_pct, 2) if upside_to_mid_pct is not None else None
        ),
        "basis": basis,
        "missing_inputs": missing,
    }


# =============================================================================
# INTERNAL: MONITORING PLAN
# =============================================================================

def _build_monitoring_plan(
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    rf: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the four monitoring triggers as machine-checkable predicates.

    Shape is fixed — every key is always present so downstream code can
    reference predicates by name without defensive ``.get()`` chains.
    Each predicate has a human-readable description and the data it
    needs to be evaluated against, so the Obsidian template can render
    a status table and a future scheduler can poll the triggers.
    """
    price = _f(metrics.get("price")) or _f(metrics.get("current_price"))
    next_earnings = (
        metrics.get("next_earnings_date")
        or metrics.get("earnings_date")
        or macro.get("next_earnings_date")
    )

    return {
        "price": {
            "trigger": "price_move",
            "predicate": f"abs(price/entry - 1) >= {_MONITOR_PRICE_MOVE_PCT}",
            "threshold_pct": _MONITOR_PRICE_MOVE_PCT * 100.0,
            "entry_price": price,
        },
        "earnings": {
            "trigger": "earnings_release",
            "predicate": (
                "next reported quarter posts EPS or revenue surprise "
                "outside +/- 5% of consensus"
            ),
            "next_earnings_date": next_earnings,  # may be None
        },
        "risk": {
            "trigger": "risk_escalation",
            "predicate": (
                "any axis in risk_flags.levels moves "
                "LOW->MEDIUM or MEDIUM->HIGH"
            ),
            "current_levels": dict((rf or {}).get("levels") or {}),
        },
        "fundamentals": {
            "trigger": "fundamentals_deterioration",
            "predicate": (
                f"next 10-Q shows revenue_growth_yoy "
                f"deceleration of >{_MONITOR_REVENUE_DECEL_PP:.0f} "
                "percentage points vs. prior quarter"
            ),
            "current_revenue_growth_yoy": _f(metrics.get("revenue_growth_yoy")),
        },
    }


# =============================================================================
# INTERNAL: BULL / BEAR / CATALYSTS BULLETS
# =============================================================================

def _build_bull_bullets(
    metrics: Dict[str, Any],
    filings_summary: Dict[str, Any],
    fund_bias: float,
    macro_bias: float,
    filings_bias: float,
) -> List[str]:
    """Bull-case bullets — 3-6 short statements grounded in positive signals.

    Each bullet derives from one data point. We iterate over a fixed
    tuple of (predicate, formatter) pairs so order is deterministic.
    """
    bullets: List[str] = []

    g3 = _f(metrics.get("revenue_growth_3y"))
    if g3 is not None and g3 >= 0.10:
        bullets.append(
            f"3Y revenue CAGR {g3 * 100:.1f}% indicates durable top-line growth."
        )

    roic = _f(metrics.get("roic"))
    if roic is not None and roic >= 0.15:
        bullets.append(
            f"ROIC {roic * 100:.0f}% suggests capital-efficient compounding."
        )

    fy = _f(metrics.get("fcf_yield"))
    if fy is not None and fy >= 0.05:
        bullets.append(
            f"FCF yield {fy * 100:.1f}% offers attractive cash-on-cash return."
        )

    cc = _f(metrics.get("cash_conversion"))
    if cc is not None and cc >= 1.0:
        bullets.append(
            f"OCF/NI {cc:.2f}x indicates high earnings quality."
        )

    de = _f(metrics.get("debt_ebitda")) or _f(metrics.get("debt_to_ebitda"))
    if de is not None and de < 1.5:
        bullets.append(
            f"Low leverage (debt/EBITDA {de:.2f}x) provides strategic flexibility."
        )

    if metrics.get("margin_trend") == "expanding":
        bullets.append("Operating margins expanding over trailing period.")

    tone = (filings_summary or {}).get("management_tone")
    if tone == "confident" and not (filings_summary or {}).get("red_flags"):
        bullets.append(
            "Confident management tone with no flagged disclosure issues."
        )

    if macro_bias >= 0.20:
        bullets.append(
            f"Macro backdrop supportive (macro bias {macro_bias:+.2f})."
        )

    if not bullets:
        # Pure-bear ticker: explicit empty-state, not a misleading bullet.
        bullets.append("No materially positive sub-signals on current snapshot.")

    return bullets[:_MAX_BULLETS_PER_SECTION]


def _build_bear_bullets(
    metrics: Dict[str, Any],
    filings_summary: Dict[str, Any],
    rf: Dict[str, Any],
    fund_bias: float,
    macro_bias: float,
    filings_bias: float,
) -> List[str]:
    """Bear-case bullets — 3-6 statements from negative signals and risks."""
    bullets: List[str] = []

    g3 = _f(metrics.get("revenue_growth_3y"))
    if g3 is not None and g3 < 0.0:
        bullets.append(
            f"3Y revenue CAGR {g3 * 100:.1f}% — top-line is contracting."
        )

    de = _f(metrics.get("debt_ebitda")) or _f(metrics.get("debt_to_ebitda"))
    if de is not None and de >= 4.0:
        bullets.append(
            f"Elevated leverage (debt/EBITDA {de:.2f}x) restricts strategic optionality."
        )

    ic = _f(metrics.get("interest_coverage"))
    if ic is not None and ic < 2.0:
        bullets.append(
            f"Interest coverage {ic:.1f}x — near distress thresholds."
        )

    fy = _f(metrics.get("fcf_yield"))
    if fy is not None and fy < 0.0:
        bullets.append(
            f"Negative FCF yield ({fy * 100:.1f}%) — cash burn vs. market cap."
        )

    if metrics.get("cash_flow_negative_3_years") is True:
        bullets.append(
            "Free cash flow negative for three consecutive years — structural, not cyclical."
        )

    if metrics.get("margin_trend") == "compressing":
        bullets.append("Operating margins compressing over trailing period.")

    cc = _f(metrics.get("cash_conversion"))
    if cc is not None and cc < 0.7:
        bullets.append(
            f"OCF/NI {cc:.2f}x — earnings quality concern (paper earnings, no cash)."
        )

    tone = (filings_summary or {}).get("management_tone")
    if tone in ("cautious", "defensive"):
        bullets.append(
            f"Management tone {tone} in recent filings — softening confidence signal."
        )

    if macro_bias <= -0.30:
        bullets.append(
            f"Macro headwind (macro bias {macro_bias:+.2f}) compresses earnings multiple."
        )

    if not bullets:
        bullets.append("No materially negative sub-signals on current snapshot.")

    return bullets[:_MAX_BULLETS_PER_SECTION]


def _build_catalysts(
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    filings_summary: Dict[str, Any],
) -> List[str]:
    """Forward-looking catalysts. Includes earnings cadence, macro events,
    and any filing-flagged disclosures (e.g. CFO transitions, guidance)."""
    out: List[str] = []

    next_earnings = (
        metrics.get("next_earnings_date")
        or metrics.get("earnings_date")
        or macro.get("next_earnings_date")
    )
    if next_earnings:
        out.append(
            f"Next earnings release: **{next_earnings}** — consensus beat/miss "
            "and forward guidance are the primary thesis tests."
        )
    else:
        out.append(
            "Earnings cadence: next report due within the standard 90-day cycle; "
            "beat-and-raise extends the runway, in-line without raised guidance is "
            "the bear trigger."
        )

    # 8-K-style guidance signals
    for year_filings in (filings_summary or {}).get("by_year", {}).values():
        if isinstance(year_filings, dict):
            for kind, text in year_filings.items():
                if kind == "8-K" and text and "guidance" in str(text).lower():
                    out.append(f"Recent 8-K guidance disclosure: {text}")

    # Macro catalysts that affect every ticker
    if _f(macro.get("recession_probability")) is not None:
        rp = _f(macro.get("recession_probability"))
        if rp is not None and rp >= 0.30:
            out.append(
                f"Macro: elevated recession probability ({rp * 100:.0f}%) — "
                "NFP and CPI prints become outsized risk events."
            )

    if not out:
        out.append("No specific catalysts identified from current inputs.")

    return out[:_MAX_BULLETS_PER_SECTION]


# =============================================================================
# INTERNAL: THESIS MARKDOWN COMPOSER
# =============================================================================

def _build_thesis_markdown(
    *,
    ticker: str,
    outlook: str,
    price_direction: str,
    adjusted_bias: float,
    confidence: float,
    fund_bias: float,
    macro_bias: float,
    filings_bias: float,
    rf: Dict[str, Any],
    m: Dict[str, Any],
    mc: Dict[str, Any],
    fs: Dict[str, Any],
    bull: List[str],
    bear: List[str],
    catalysts: List[str],
    key_risks: List[str],
    valuation: Dict[str, Any],
    monitoring: Dict[str, Any],
) -> str:
    """Compose the multi-section thesis Markdown note.

    Layout (in order):
      1. YAML frontmatter (Obsidian-friendly)
      2. ``# {TICKER} — 1Y Thesis`` title
      3. **Verdict** one-liner
      4. ## Snapshot
      5. ## Bull case
      6. ## Bear case
      7. ## Catalysts
      8. ## Risks
      9. ## Valuation summary
      10. ## Monitoring plan

    The output is a *pure function* of the inputs — same inputs always
    yield byte-identical output. The only place a date appears is the
    YAML ``created`` field, and it is sourced from
    ``macro['as_of']`` (deterministic), falling back to the literal
    string ``'unknown'`` (also deterministic) when absent.
    """
    sector = (m or {}).get("sector") or "Unknown"
    combined = _get_combined_level(rf)
    created = mc.get("as_of") if isinstance(mc.get("as_of"), str) else "unknown"

    # --- YAML frontmatter ---
    yaml_lines = [
        "---",
        "type: thesis_note",
        f"ticker: {ticker}",
        f"sector: {sector}",
        "status: open",
        f"created: {created}",
        f"outlook: {outlook}",
        f"confidence: {round(confidence, 2)}",
        f"bias_score: {round(adjusted_bias, 2)}",
        f"combined_risk: {combined}",
        "---",
        "",
    ]

    # --- Title + verdict ---
    body = [
        f"# {ticker} — 1Y Thesis",
        "",
        (
            f"**Verdict:** {outlook}, {price_direction} · "
            f"confidence {confidence:.2f} · bias {adjusted_bias:+.2f} · "
            f"combined risk {combined}"
        ),
        "",
    ]

    # --- Snapshot ---
    body += [
        "## Snapshot",
        "",
        (
            f"Fundamentals bias **{fund_bias:+.2f}**, macro bias "
            f"**{macro_bias:+.2f}**, filings bias **{filings_bias:+.2f}**. "
            f"Combined risk **{combined}**. Post-dampening bias "
            f"**{adjusted_bias:+.2f}** lands in the *{price_direction}* "
            "bucket on the 5-bucket direction map, and the "
            f"*{outlook}* bucket on the 3-bucket outlook map."
        ),
        "",
    ]

    # --- Bull / Bear ---
    body += ["## Bull case", ""]
    for b in bull:
        body.append(f"- {b}")
    body += ["", "## Bear case", ""]
    for b in bear:
        body.append(f"- {b}")

    # --- Catalysts ---
    body += ["", "## Catalysts", ""]
    for c in catalysts:
        body.append(f"- {c}")

    # --- Risks ---
    body += ["", "## Risks", ""]
    if key_risks:
        for r in key_risks:
            body.append(f"- {r}")
    else:
        body.append("- No HIGH or MEDIUM risk axes flagged in current snapshot.")

    # --- Valuation summary ---
    body += ["", "## Valuation summary", ""]
    if valuation.get("implied_eps") is None:
        body.append(
            f"Valuation unavailable — missing: "
            f"{', '.join(valuation.get('missing_inputs') or []) or 'unknown'}."
        )
    else:
        body += [
            "| Component | Value |",
            "|---|---|",
            f"| Current price | ${valuation['current_price']:.2f} |",
            (
                f"| Trailing P/E | {valuation['trailing_pe']:.1f}x |"
                if valuation.get("trailing_pe") is not None
                else "| Trailing P/E | n/a |"
            ),
            (
                f"| Forward P/E | {valuation['forward_pe']:.1f}x |"
                if valuation.get("forward_pe") is not None
                else "| Forward P/E | n/a |"
            ),
            f"| Implied EPS | ${valuation['implied_eps']:.2f} |",
            f"| Target multiple (1Y) | {valuation['target_multiple']:.1f}x |",
            f"| Fair value (mid) | ${valuation['fair_value_mid']:.2f} |",
            (
                f"| Fair value range | "
                f"${valuation['fair_value_low']:.2f} – ${valuation['fair_value_high']:.2f} |"
            ),
            (
                f"| Upside to mid | {valuation['upside_to_mid_pct']:+.1f}% |"
                if valuation.get("upside_to_mid_pct") is not None
                else "| Upside to mid | n/a |"
            ),
            "",
            f"**Method.** {valuation['basis']}",
        ]

    # --- Monitoring plan ---
    body += ["", "## Monitoring plan", ""]
    body += [
        "Re-evaluate the thesis when **any** of the following triggers fire:",
        "",
        "| Trigger | Predicate |",
        "|---|---|",
        f"| Price move | `{monitoring['price']['predicate']}` |",
        f"| Earnings | {monitoring['earnings']['predicate']} |",
        f"| Risk escalation | `{monitoring['risk']['predicate']}` |",
        f"| Fundamentals | {monitoring['fundamentals']['predicate']} |",
        "",
        (
            "Status flips to *closed* when the position is exited; "
            "`portfolio_db.close_thesis` stamps `closed_at` and the "
            "learning loop re-scores against realized P&L on the next "
            "daily cycle."
        ),
    ]

    return "\n".join(yaml_lines + body)


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