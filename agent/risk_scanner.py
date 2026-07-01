"""
agent/risk_scanner.py
=====================

Rule-based risk-flag engine. **No LLM calls, no DB writes.**

This module produces the same top-level output shape as before
(``{"levels": {...}, "reasons": {...}}``) but the fundamental axis is
upgraded with:

1. **Sector-relative z-scores.** For metrics where a peer cohort is
   available, the absolute threshold is replaced with a z-score tier.
   See ``_score_metric_sector_aware``.
2. **Formal distress models.** Altman Z, Piotroski F, and Beneish M are
   computed when their inputs are present and used as *escalation-only*
   signals — they can raise the fundamental level but never lower it.
   See ``compute_altman_z``, ``compute_piotroski_f``, ``compute_beneish_m``.

Macro / market / agent scoring are unchanged. Sector context only enters
fundamentals; that's the axis where peer comparisons and distress models
actually carry information.

Inputs
------
- Normalised metric snapshot from
  :func:`agent.filing_analyzer.extract_key_metrics_for_agent`.
- Macro dict from ``pipeline.build_agent_context(...)["macro"]``.
- ``agent_risks``: list of free-form risk strings from the LLM.
- Optional ``peer_stats`` kwarg or ``metrics["peer_stats"]`` —
  ``{metric: {"mean": float, "std": float, "n": int}}``.

Output
------
Same as before::

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
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Canonical level ordering, used by the combiner. HIGH > MEDIUM > LOW.
_LEVEL_RANK: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_RANK_LEVEL: Dict[int, str] = {v: k for k, v in _LEVEL_RANK.items()}

# Fallback thresholds, used when ``config.RISK_THRESHOLDS`` is absent or
# missing a key. Keeps the module importable in isolation.
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


# Alias groups for commonly-misnamed metric keys.
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


# Z-score tier thresholds. Negative z = worse than peers in our convention,
# so the cut-offs are negative. See module docstring for the percentiles.
_Z_TIER_HIGH = -1.5    # ~7th percentile of a normal cohort
_Z_TIER_MEDIUM = -0.5  # ~31st percentile

# Minimum peer-cohort size before shrinkage stops penalising the z-score.
# For n < 5 the z is scaled by sqrt(n/5) to avoid escalating on noise.
_PEER_N_FLOOR = 5


# Metric direction registry. Each entry says which way is *risky*.
#   "lower_is_safer": high values are bad (debt, valuation multiples)
#   "higher_is_safer": high values are good (coverage, yield, returns)
# Metrics not listed here are not z-scored, even if peer stats are present.
_METRIC_DIRECTIONS: Dict[str, str] = {
    "debt_ebitda":       "lower_is_safer",
    "net_debt_ebitda":   "lower_is_safer",
    "interest_coverage": "higher_is_safer",
    "fcf_yield":         "higher_is_safer",
    "roic":              "higher_is_safer",
    "operating_margin":  "higher_is_safer",
    "gross_margin":      "higher_is_safer",
    "cash_conversion":   "higher_is_safer",
}


# Hard-coded sector default peer distributions. Coarse — the right answer
# long-term is to compute these from the actual universe and pass them in
# via ``peer_stats``. Values are broad-market averages for each sector.
# ``n`` is set to a representative cohort size so the shrinkage factor
# doesn't kick in for the defaults themselves.
#
# Sector names match yfinance's ``sector`` field. Lookup is case-insensitive.
_SECTOR_DEFAULTS: Dict[str, Dict[str, Dict[str, float]]] = {
    # REITs / real estate: structurally levered; FCF yield runs high
    # because depreciation is large vs. true economic capex.
    "real estate": {
        "debt_ebitda":       {"mean": 7.5, "std": 1.8, "n": 30},
        "net_debt_ebitda":   {"mean": 7.0, "std": 1.7, "n": 30},
        "interest_coverage": {"mean": 2.8, "std": 0.9, "n": 30},
        "fcf_yield":         {"mean": 0.06, "std": 0.025, "n": 30},
        "roic":              {"mean": 0.05, "std": 0.02, "n": 30},
        "operating_margin":  {"mean": 0.30, "std": 0.10, "n": 30},
    },
    # Utilities: capital-heavy, regulated revenue.
    "utilities": {
        "debt_ebitda":       {"mean": 4.5, "std": 1.2, "n": 30},
        "net_debt_ebitda":   {"mean": 4.2, "std": 1.1, "n": 30},
        "interest_coverage": {"mean": 3.2, "std": 1.0, "n": 30},
        "fcf_yield":         {"mean": 0.04, "std": 0.02, "n": 30},
        "roic":              {"mean": 0.06, "std": 0.02, "n": 30},
        "operating_margin":  {"mean": 0.18, "std": 0.06, "n": 30},
    },
    # Financial services: included so lookup hits, but debt/EBITDA isn't
    # really the right metric for banks.
    "financial services": {
        "debt_ebitda":       {"mean": 5.0, "std": 2.0, "n": 30},
        "interest_coverage": {"mean": 4.0, "std": 1.5, "n": 30},
        "roic":              {"mean": 0.10, "std": 0.04, "n": 30},
        "operating_margin":  {"mean": 0.30, "std": 0.10, "n": 30},
    },
    # Energy: cyclical; leverage varies wildly with the commodity cycle.
    "energy": {
        "debt_ebitda":       {"mean": 2.0, "std": 1.0, "n": 30},
        "interest_coverage": {"mean": 8.0, "std": 4.0, "n": 30},
        "fcf_yield":         {"mean": 0.08, "std": 0.04, "n": 30},
        "roic":              {"mean": 0.12, "std": 0.06, "n": 30},
        "operating_margin":  {"mean": 0.20, "std": 0.10, "n": 30},
    },
    # Software / technology: asset-light, low debt, high margins.
    "technology": {
        "debt_ebitda":       {"mean": 0.8, "std": 0.6, "n": 30},
        "interest_coverage": {"mean": 25.0, "std": 15.0, "n": 30},
        "fcf_yield":         {"mean": 0.04, "std": 0.02, "n": 30},
        "roic":              {"mean": 0.22, "std": 0.08, "n": 30},
        "operating_margin":  {"mean": 0.28, "std": 0.10, "n": 30},
        "gross_margin":      {"mean": 0.65, "std": 0.10, "n": 30},
    },
    "consumer cyclical": {
        "debt_ebitda":       {"mean": 2.2, "std": 1.0, "n": 30},
        "interest_coverage": {"mean": 8.0, "std": 4.0, "n": 30},
        "fcf_yield":         {"mean": 0.05, "std": 0.025, "n": 30},
        "roic":              {"mean": 0.14, "std": 0.06, "n": 30},
        "operating_margin":  {"mean": 0.12, "std": 0.06, "n": 30},
    },
    "consumer defensive": {
        "debt_ebitda":       {"mean": 2.5, "std": 0.9, "n": 30},
        "interest_coverage": {"mean": 10.0, "std": 4.0, "n": 30},
        "fcf_yield":         {"mean": 0.05, "std": 0.02, "n": 30},
        "roic":              {"mean": 0.15, "std": 0.05, "n": 30},
        "operating_margin":  {"mean": 0.14, "std": 0.05, "n": 30},
    },
    "industrials": {
        "debt_ebitda":       {"mean": 2.5, "std": 1.0, "n": 30},
        "interest_coverage": {"mean": 9.0, "std": 4.0, "n": 30},
        "fcf_yield":         {"mean": 0.05, "std": 0.025, "n": 30},
        "roic":              {"mean": 0.13, "std": 0.05, "n": 30},
        "operating_margin":  {"mean": 0.13, "std": 0.05, "n": 30},
    },
    "healthcare": {
        "debt_ebitda":       {"mean": 2.0, "std": 1.0, "n": 30},
        "interest_coverage": {"mean": 12.0, "std": 6.0, "n": 30},
        "fcf_yield":         {"mean": 0.05, "std": 0.025, "n": 30},
        "roic":              {"mean": 0.16, "std": 0.07, "n": 30},
        "operating_margin":  {"mean": 0.18, "std": 0.08, "n": 30},
    },
    "communication services": {
        "debt_ebitda":       {"mean": 2.3, "std": 1.1, "n": 30},
        "interest_coverage": {"mean": 10.0, "std": 5.0, "n": 30},
        "fcf_yield":         {"mean": 0.06, "std": 0.03, "n": 30},
        "roic":              {"mean": 0.15, "std": 0.07, "n": 30},
        "operating_margin":  {"mean": 0.20, "std": 0.08, "n": 30},
    },
    "basic materials": {
        "debt_ebitda":       {"mean": 2.0, "std": 1.0, "n": 30},
        "interest_coverage": {"mean": 9.0, "std": 5.0, "n": 30},
        "fcf_yield":         {"mean": 0.06, "std": 0.03, "n": 30},
        "roic":              {"mean": 0.12, "std": 0.06, "n": 30},
        "operating_margin":  {"mean": 0.13, "std": 0.06, "n": 30},
    },
}


# Distress-composite tier thresholds. Composite is max(altman_norm,
# piotroski_norm, beneish_norm) — each in [0, 1], 1 = fully distressed.
_DISTRESS_HIGH = 0.80
_DISTRESS_MEDIUM = 0.50

# Sectors that use Altman Z'' (revised, non-manufacturing) rather than Z.
_SECTORS_USE_ZDOUBLEPRIME = {
    "financial services", "real estate", "utilities",
    "communication services",
}


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_risk_flags(
    ticker: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    agent_risks: List[str],
    config: Any,
    *,
    peer_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Compute fundamental / macro / market / agent / combined risk levels.

    Parameters
    ----------
    ticker:
        Ticker symbol, used only for logging.
    metrics:
        Metric snapshot. Optional fields that unlock the new logic:

        - ``sector`` (str): used to look up default peer stats and pick the
          Altman variant.
        - ``peer_stats`` (dict): per-metric ``{"mean", "std", "n"}``.
          Overrides the ``peer_stats`` kwarg.
        - Balance-sheet inputs for Altman: ``total_assets``,
          ``total_liabilities``, ``working_capital``, ``retained_earnings``,
          ``ebit``, ``revenue``, ``market_cap``, ``book_equity``.
        - Prior-period inputs for Piotroski / Beneish: any field with
          ``..._prev`` suffix.

    macro:
        Macro dict from ``pipeline.build_agent_context``.
    agent_risks:
        Free-form risk strings from the LLM.
    config:
        The project's ``config`` module exposing ``RISK_THRESHOLDS``.
    peer_stats:
        Optional keyword-only fallback peer-stats dict. Used only if
        ``metrics`` doesn't supply its own.

    Returns
    -------
    dict
        Same shape as before — fully backward-compatible.
    """
    thresholds = _load_thresholds(config)
    m = metrics or {}
    mc = macro or {}
    ar = list(agent_risks or [])

    # Peer stats resolution: metric-embedded > kwarg > sector defaults > none.
    resolved_peer_stats = _resolve_peer_stats(m, peer_stats)

    fundamental_level, fundamental_reasons = _score_fundamental(
        m, thresholds, resolved_peer_stats
    )
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

    # Distinguish "evaluated, nothing elevated" (all clear) from a genuine
    # data gap. Empty reasons + inputs present -> all-clear note; empty reasons
    # + no inputs -> "no data". Levels are unaffected.
    def _had(d, keys):
        try:
            return any(d.get(k) is not None for k in keys)
        except Exception:  # noqa: BLE001
            return False

    _macro_had = _had(mc, ("vix", "recession_probability",
                           "yield_curve_spread", "yield_curve_inverted"))
    _market_had = _had(m, ("realized_vol", "volatility", "drawdown",
                           "max_drawdown", "rsi"))
    _fund_had = bool(m)
    _agent_had = bool(ar)

    def _reasons_for(scored, had, clear_msg):
        if scored:
            return scored
        return [clear_msg] if had else ["no data"]

    reasons: Dict[str, List[str]] = {
        "fundamental": _reasons_for(
            fundamental_reasons, _fund_had,
            "fundamentals within normal ranges vs peers"),
        "macro": _reasons_for(
            macro_reasons, _macro_had,
            "macro indicators within normal ranges (VIX, recession odds, "
            "yield curve)"),
        "market": _reasons_for(
            market_reasons, _market_had,
            "price/volatility metrics within normal ranges"),
        "agent": _reasons_for(
            agent_reasons, _agent_had,
            "no additional risks flagged by the agent"),
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
# PUBLIC: DISTRESS MODELS (exposed for unit testing and direct use)
# =============================================================================

def compute_altman_z(
    metrics: Dict[str, Any],
    *,
    variant: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Altman Z (1968) or Z'' (revised) bankruptcy score.

    Original Z (manufacturing / industrials):
        Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA)
            + 0.6*(MktCap/TL) + 1.0*(Sales/TA)
        Zones: < 1.81 = distress, 1.81-2.99 = grey, > 2.99 = safe.

    Z'' (revised, non-manufacturing / EM / services):
        Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA)
              + 1.05*(BookEq/TL)
        Zones: < 1.10 = distress, 1.10-2.60 = grey, > 2.60 = safe.

    Parameters
    ----------
    metrics:
        Snapshot dict. Required keys depend on variant.
    variant:
        ``"original"``, ``"zdoubleprime"`` or ``None``. ``None`` auto-picks
        based on ``metrics["sector"]``: financials/real-estate/utilities
        get Z'', everything else gets original Z. Also falls back to Z'' if
        ``revenue`` is missing.

    Returns
    -------
    dict or None
        ``{"z": float, "variant": str, "distress": float, "zone": str}``
        where ``distress`` is in [0, 1] (1 = fully distressed).
        Returns ``None`` if the required inputs are not all present.
    """
    sector = str(metrics.get("sector") or "").strip().lower()
    if variant is None:
        if sector in _SECTORS_USE_ZDOUBLEPRIME:
            variant = "zdoubleprime"
        elif _f(metrics.get("revenue")) is None:
            variant = "zdoubleprime"
        else:
            variant = "original"

    ta = _f(metrics.get("total_assets"))
    tl = _f(metrics.get("total_liabilities"))
    wc = _f(metrics.get("working_capital"))
    re = _f(metrics.get("retained_earnings"))
    ebit = _f(metrics.get("ebit"))

    if ta is None or ta <= 0 or tl is None or tl <= 0:
        return None
    if wc is None or re is None or ebit is None:
        return None

    if variant == "original":
        sales = _f(metrics.get("revenue") or metrics.get("sales"))
        mcap = _f(metrics.get("market_cap"))
        if sales is None or mcap is None:
            return None
        z = (
            1.2 * (wc / ta)
            + 1.4 * (re / ta)
            + 3.3 * (ebit / ta)
            + 0.6 * (mcap / tl)
            + 1.0 * (sales / ta)
        )
        distress_thr, safe_thr = 1.81, 2.99
    elif variant == "zdoubleprime":
        # Z'' uses book equity / TL instead of market cap / TL, and drops
        # the sales/TA term (so it works for services / financials).
        book_eq = _f(metrics.get("book_equity"))
        if book_eq is None:
            book_eq = ta - tl
        z = (
            6.56 * (wc / ta)
            + 3.26 * (re / ta)
            + 6.72 * (ebit / ta)
            + 1.05 * (book_eq / tl)
        )
        distress_thr, safe_thr = 1.10, 2.60
    else:
        return None

    distress = _clip01((safe_thr - z) / (safe_thr - distress_thr))
    if z < distress_thr:
        zone = "distress"
    elif z < safe_thr:
        zone = "grey"
    else:
        zone = "safe"

    return {"z": z, "variant": variant, "distress": distress, "zone": zone}


def compute_piotroski_f(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Piotroski F-score (2000) — 9 binary fundamental-strength tests.

    Requires both current and prior-year (suffix ``_prev``) values for
    several fields. Returns ``None`` if fewer than 5 of the 9 tests can
    be evaluated (too sparse to be meaningful).

    The 9 tests, each scoring 1 if passed:
        Profitability:
            1. Net income > 0
            2. Operating cash flow > 0
            3. ROA improvement (ROA > ROA_prev)
            4. Quality of earnings: OCF > Net income (no accrual inflation)
        Leverage / liquidity / dilution:
            5. Long-term debt / TA decreased
            6. Current ratio increased
            7. No new shares issued
        Operating efficiency:
            8. Gross margin increased
            9. Asset turnover increased

    Returns
    -------
    dict or None
        ``{"f": int, "n_evaluated": int, "distress": float, "tests": dict}``.
        ``distress = (9 - f_scaled) / 9`` is in [0, 1].
    """
    tests: Dict[str, Optional[bool]] = {}

    ni = _f(metrics.get("net_income"))
    ocf = _f(metrics.get("operating_cash_flow") or metrics.get("ocf"))
    ta = _f(metrics.get("total_assets"))
    ta_prev = _f(metrics.get("total_assets_prev"))
    ni_prev = _f(metrics.get("net_income_prev"))

    tests["ni_positive"] = (ni > 0) if ni is not None else None
    tests["ocf_positive"] = (ocf > 0) if ocf is not None else None

    if ni is not None and ta and ta_prev and ni_prev is not None:
        tests["roa_improving"] = (ni / ta) > (ni_prev / ta_prev)
    else:
        tests["roa_improving"] = None

    if ocf is not None and ni is not None:
        tests["accrual_quality"] = ocf > ni
    else:
        tests["accrual_quality"] = None

    ltd = _f(metrics.get("long_term_debt"))
    ltd_prev = _f(metrics.get("long_term_debt_prev"))
    if ltd is not None and ltd_prev is not None:
        if ta and ta_prev:
            tests["leverage_down"] = (ltd / ta) < (ltd_prev / ta_prev)
        else:
            tests["leverage_down"] = ltd < ltd_prev
    else:
        tests["leverage_down"] = None

    cr = _f(metrics.get("current_ratio"))
    cr_prev = _f(metrics.get("current_ratio_prev"))
    if cr is not None and cr_prev is not None:
        tests["liquidity_up"] = cr > cr_prev
    else:
        tests["liquidity_up"] = None

    sh = _f(metrics.get("shares_outstanding"))
    sh_prev = _f(metrics.get("shares_outstanding_prev"))
    if sh is not None and sh_prev is not None:
        tests["no_dilution"] = sh <= sh_prev * 1.001
    else:
        tests["no_dilution"] = None

    gm = _f(metrics.get("gross_margin"))
    gm_prev = _f(metrics.get("gross_margin_prev"))
    if gm is not None and gm_prev is not None:
        tests["gm_up"] = gm > gm_prev
    else:
        tests["gm_up"] = None

    rev = _f(metrics.get("revenue"))
    rev_prev = _f(metrics.get("revenue_prev"))
    if rev is not None and ta and rev_prev is not None and ta_prev:
        tests["turnover_up"] = (rev / ta) > (rev_prev / ta_prev)
    else:
        tests["turnover_up"] = None

    evaluated = [v for v in tests.values() if v is not None]
    if len(evaluated) < 5:
        return None

    f_count = sum(1 for v in evaluated if v)
    # Scale F to the full 9-point range when only a subset evaluated. If
    # we evaluated 6 and got 4, the implied F is 4 * (9/6) = 6.
    f_scaled = f_count * (9 / len(evaluated))
    distress = _clip01((9 - f_scaled) / 9)

    return {
        "f": f_count,
        "f_scaled": f_scaled,
        "n_evaluated": len(evaluated),
        "distress": distress,
        "tests": tests,
    }


def compute_beneish_m(metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Beneish M-score (1999) — earnings-manipulation detector.

    M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Threshold M = -1.78 (above => likely manipulator). Normalised to
    distress in [0, 1] linearly between -2.22 (clean) and -1.78 (suspect).

    Returns ``None`` if too many inputs are missing (we require at least
    5 of the 8 ratios).
    """
    sales = _f(metrics.get("revenue") or metrics.get("sales"))
    sales_prev = _f(metrics.get("revenue_prev") or metrics.get("sales_prev"))
    ar = _f(metrics.get("accounts_receivable"))
    ar_prev = _f(metrics.get("accounts_receivable_prev"))
    gm = _f(metrics.get("gross_margin"))
    gm_prev = _f(metrics.get("gross_margin_prev"))
    ca = _f(metrics.get("current_assets"))
    ca_prev = _f(metrics.get("current_assets_prev"))
    ppe = _f(metrics.get("ppe") or metrics.get("property_plant_equipment"))
    ppe_prev = _f(metrics.get("ppe_prev") or metrics.get("property_plant_equipment_prev"))
    ta = _f(metrics.get("total_assets"))
    ta_prev = _f(metrics.get("total_assets_prev"))
    dep = _f(metrics.get("depreciation"))
    dep_prev = _f(metrics.get("depreciation_prev"))
    sga = _f(metrics.get("sga") or metrics.get("sg_and_a"))
    sga_prev = _f(metrics.get("sga_prev") or metrics.get("sg_and_a_prev"))
    ni = _f(metrics.get("net_income"))
    ocf = _f(metrics.get("operating_cash_flow") or metrics.get("ocf"))
    ltd = _f(metrics.get("long_term_debt"))
    ltd_prev = _f(metrics.get("long_term_debt_prev"))
    cl = _f(metrics.get("current_liabilities"))
    cl_prev = _f(metrics.get("current_liabilities_prev"))

    ratios: Dict[str, Optional[float]] = {}

    # DSRI = (AR/Sales)_t / (AR/Sales)_{t-1}
    if ar is not None and sales and ar_prev is not None and sales_prev:
        ratios["DSRI"] = (ar / sales) / (ar_prev / sales_prev)
    else:
        ratios["DSRI"] = None

    # GMI = GM_{t-1} / GM_t
    if gm is not None and gm_prev is not None and gm != 0:
        ratios["GMI"] = gm_prev / gm
    else:
        ratios["GMI"] = None

    # AQI = (1 - (CA + PPE) / TA)_t / (1 - (CA + PPE) / TA)_{t-1}
    if (
        ca is not None and ppe is not None and ta
        and ca_prev is not None and ppe_prev is not None and ta_prev
    ):
        aq_t = 1 - (ca + ppe) / ta
        aq_p = 1 - (ca_prev + ppe_prev) / ta_prev
        if aq_p != 0:
            ratios["AQI"] = aq_t / aq_p
        else:
            ratios["AQI"] = None
    else:
        ratios["AQI"] = None

    # SGI = Sales_t / Sales_{t-1}
    if sales is not None and sales_prev:
        ratios["SGI"] = sales / sales_prev
    else:
        ratios["SGI"] = None

    # DEPI = (Dep / (Dep + PPE))_{t-1} / (Dep / (Dep + PPE))_t
    if (
        dep is not None and ppe is not None
        and dep_prev is not None and ppe_prev is not None
    ):
        denom_t = dep + ppe
        denom_p = dep_prev + ppe_prev
        if denom_t > 0 and denom_p > 0:
            rate_t = dep / denom_t
            rate_p = dep_prev / denom_p
            if rate_t > 0:
                ratios["DEPI"] = rate_p / rate_t
            else:
                ratios["DEPI"] = None
        else:
            ratios["DEPI"] = None
    else:
        ratios["DEPI"] = None

    # SGAI = (SGA/Sales)_t / (SGA/Sales)_{t-1}
    if (
        sga is not None and sales
        and sga_prev is not None and sales_prev
    ):
        ratios["SGAI"] = (sga / sales) / (sga_prev / sales_prev)
    else:
        ratios["SGAI"] = None

    # TATA = (NI - OCF) / TA
    if ni is not None and ocf is not None and ta:
        ratios["TATA"] = (ni - ocf) / ta
    else:
        ratios["TATA"] = None

    # LVGI = ((LTD + CL) / TA)_t / ((LTD + CL) / TA)_{t-1}
    if (
        ltd is not None and cl is not None and ta
        and ltd_prev is not None and cl_prev is not None and ta_prev
    ):
        lev_t = (ltd + cl) / ta
        lev_p = (ltd_prev + cl_prev) / ta_prev
        if lev_p != 0:
            ratios["LVGI"] = lev_t / lev_p
        else:
            ratios["LVGI"] = None
    else:
        ratios["LVGI"] = None

    available = {k: v for k, v in ratios.items() if v is not None}
    if len(available) < 5:
        return None

    # Substitute neutral 1.0 for missing ratios (1.0 = no change YoY).
    # TATA's neutral is 0.0 (zero accruals) per the formula.
    coeffs = {
        "DSRI": 0.920,
        "GMI": 0.528,
        "AQI": 0.404,
        "SGI": 0.892,
        "DEPI": 0.115,
        "SGAI": -0.172,
        "TATA": 4.679,
        "LVGI": -0.327,
    }
    neutral = {"TATA": 0.0}

    m_score = -4.84
    for name, coef in coeffs.items():
        val = ratios.get(name)
        if val is None:
            val = neutral.get(name, 1.0)
        m_score += coef * val

    distress = _clip01((m_score - (-2.22)) / ((-1.78) - (-2.22)))

    return {
        "m": m_score,
        "n_evaluated": len(available),
        "distress": distress,
        "ratios": ratios,
    }


# =============================================================================
# PUBLIC: PEER/Z-SCORE HELPERS (exposed for unit testing)
# =============================================================================

def zscore_risk_signed(
    value: float,
    mean: float,
    std: float,
    direction: str,
    n: Optional[int] = None,
) -> float:
    """Risk-signed z-score.

    By convention, **negative z = worse than peers** regardless of which
    side of the mean is "bad":

        direction == "higher_is_safer":  z = (x - mean) / std
        direction == "lower_is_safer":   z = (mean - x) / std

    Applies sample-size shrinkage when ``n`` is provided and < 5:
        z_adj = z * sqrt(n / 5)

    Returns 0.0 (neutral) if ``std`` is non-positive.
    """
    if std is None or std <= 0:
        return 0.0
    if direction == "higher_is_safer":
        z = (value - mean) / std
    elif direction == "lower_is_safer":
        z = (mean - value) / std
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    if n is not None and 0 < n < _PEER_N_FLOOR:
        z *= math.sqrt(n / _PEER_N_FLOOR)
    return z


def ztier(z: float) -> str:
    """Map a risk-signed z-score to LOW/MEDIUM/HIGH."""
    if z < _Z_TIER_HIGH:
        return "HIGH"
    if z < _Z_TIER_MEDIUM:
        return "MEDIUM"
    return "LOW"


# =============================================================================
# INTERNAL: SUB-SCORERS
# =============================================================================

def _score_fundamental(
    metrics: Dict[str, Any],
    t: Dict[str, float],
    peer_stats: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[str, List[str]]:
    """Fundamental risk: per-metric tiers + distress escalation.

    For each scorable metric:
        - If peer stats are available -> z-score tier (REPLACES absolute).
        - Else                        -> absolute threshold tier (legacy).

    Distress escalation:
        - composite = max(altman_distress, piotroski_distress, beneish_distress)
        - composite >= _DISTRESS_HIGH   -> contribute HIGH
        - composite >= _DISTRESS_MEDIUM -> contribute MEDIUM
        - else                          -> no contribution

    Final level is the most-severe tier across all contributions.
    Distress can only RAISE risk, never lower it.
    """
    reasons: List[str] = []
    level = "LOW"

    # --- Per-metric scoring ----------------------------------------------
    metric_tier, metric_reasons = _score_metrics_per_axis(
        metrics, t, peer_stats
    )
    level = _max_level(level, metric_tier)
    reasons.extend(metric_reasons)

    # --- 3-year negative FCF streak (legacy escalation, kept verbatim) ---
    flag = metrics.get("cash_flow_negative_3_years")
    if flag is True:
        level = _max_level(level, "HIGH")
        reasons.append("3+ consecutive years of negative free cash flow")
    elif flag is None:
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

    # --- Margin trend (legacy) ------------------------------------------
    if metrics.get("margin_trend") == "compressing":
        level = _max_level(level, "MEDIUM")
        reasons.append("operating margins compressing")

    # --- Distress escalation ---------------------------------------------
    distress_level, distress_reasons = _distress_escalation_tier(metrics)
    if distress_level != "LOW":
        level = _max_level(level, distress_level)
        reasons.extend(distress_reasons)

    return level, reasons


def _score_metrics_per_axis(
    metrics: Dict[str, Any],
    t: Dict[str, float],
    peer_stats: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[str, List[str]]:
    """Score each fundamental metric, choosing sector-z or absolute path."""
    reasons: List[str] = []
    level = "LOW"

    # debt/EBITDA
    de = _f(_read(metrics, "debt_ebitda"))
    if de is not None:
        m_level, m_reason = _score_metric_sector_aware(
            "debt_ebitda", de, peer_stats,
            absolute_fn=lambda v: _absolute_debt_ebitda(v, t),
        )
        level = _max_level(level, m_level)
        if m_reason:
            reasons.append(m_reason)

    # interest coverage
    ic = _f(metrics.get("interest_coverage"))
    if ic is not None:
        m_level, m_reason = _score_metric_sector_aware(
            "interest_coverage", ic, peer_stats,
            absolute_fn=lambda v: _absolute_interest_coverage(v, t),
        )
        level = _max_level(level, m_level)
        if m_reason:
            reasons.append(m_reason)

    # FCF yield
    fy = _f(_read(metrics, "fcf_yield"))
    if fy is not None:
        m_level, m_reason = _score_metric_sector_aware(
            "fcf_yield", fy, peer_stats,
            absolute_fn=lambda v: _absolute_fcf_yield(v, t),
        )
        level = _max_level(level, m_level)
        if m_reason:
            reasons.append(m_reason)

    return level, reasons


def _score_metric_sector_aware(
    metric_name: str,
    value: float,
    peer_stats: Optional[Dict[str, Dict[str, float]]],
    *,
    absolute_fn,
) -> Tuple[str, str]:
    """Pick sector-z path if peers available, else absolute threshold.

    Sector-z path REPLACES the absolute threshold for that metric — this
    is what lets a REIT at 8x debt/EBITDA land LOW when its peers also
    sit at 7-8x. Without peer stats we fall back to the legacy absolute
    threshold logic, preserving prior behaviour.
    """
    ps = (peer_stats or {}).get(metric_name) if peer_stats else None
    direction = _METRIC_DIRECTIONS.get(metric_name)

    if ps and direction and float(ps.get("std", 0) or 0) > 0:
        mean = float(ps["mean"])
        std = float(ps["std"])
        n = int(ps.get("n", _PEER_N_FLOOR))
        z = zscore_risk_signed(value, mean, std, direction, n=n)
        tier = ztier(z)
        if tier == "LOW":
            # Silent for LOW — no need to clutter reasons with healthy z's.
            return tier, ""
        reason = (
            f"{metric_name} {value:.2f} vs peers (mu={mean:.2f}, "
            f"sigma={std:.2f}), z={z:+.2f} -> {tier}"
        )
        return tier, reason

    # Fallback: legacy absolute threshold path.
    return absolute_fn(value)


def _absolute_debt_ebitda(v: float, t: Dict[str, float]) -> Tuple[str, str]:
    """Legacy absolute-threshold scoring for debt/EBITDA."""
    if v > t["debt_ebitda_high"]:
        return "HIGH", (
            f"debt/EBITDA {v:.2f}x exceeds {t['debt_ebitda_high']:.1f}x"
        )
    if v > t["debt_ebitda_medium"]:
        return "MEDIUM", (
            f"debt/EBITDA {v:.2f}x above {t['debt_ebitda_medium']:.1f}x"
        )
    return "LOW", ""


def _absolute_interest_coverage(
    v: float, t: Dict[str, float]
) -> Tuple[str, str]:
    if v < t["interest_coverage_low"]:
        return "HIGH", (
            f"interest coverage {v:.2f}x below {t['interest_coverage_low']:.1f}x"
        )
    return "LOW", ""


def _absolute_fcf_yield(v: float, t: Dict[str, float]) -> Tuple[str, str]:
    if v < t["fcf_yield_low"]:
        return "MEDIUM", (
            f"FCF yield {v * 100:.2f}% below {t['fcf_yield_low'] * 100:.1f}%"
        )
    return "LOW", ""


def _distress_escalation_tier(
    metrics: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Compute the three distress models and convert to an escalation tier.

    Returns
    -------
    (tier, reasons)
        ``tier`` is LOW (no escalation), MEDIUM, or HIGH.
        Each contributing distress model adds a reason line.
    """
    altman = compute_altman_z(metrics)
    piotroski = compute_piotroski_f(metrics)
    beneish = compute_beneish_m(metrics)

    contributions: List[Tuple[float, str]] = []
    if altman is not None:
        contributions.append(
            (
                altman["distress"],
                f"Altman {altman['variant']} Z={altman['z']:.2f} "
                f"({altman['zone']}, distress={altman['distress']:.2f})",
            )
        )
    if piotroski is not None:
        contributions.append(
            (
                piotroski["distress"],
                f"Piotroski F={piotroski['f']}/9 "
                f"({piotroski['n_evaluated']} tests evaluated, "
                f"distress={piotroski['distress']:.2f})",
            )
        )
    if beneish is not None:
        contributions.append(
            (
                beneish["distress"],
                f"Beneish M={beneish['m']:+.2f} "
                f"(distress={beneish['distress']:.2f})",
            )
        )

    if not contributions:
        return "LOW", []

    composite = max(d for d, _ in contributions)
    if composite >= _DISTRESS_HIGH:
        tier = "HIGH"
    elif composite >= _DISTRESS_MEDIUM:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    # Surface only models that contributed meaningfully (>= MEDIUM gate).
    reasons = [
        msg for d, msg in contributions if d >= _DISTRESS_MEDIUM
    ]
    return tier, reasons


def _score_macro(
    macro: Dict[str, Any],
    t: Dict[str, float],
) -> Tuple[str, List[str]]:
    """Assess recession probability, VIX regime, and yield-curve state.

    Unchanged from prior version.
    """
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

    Unchanged from prior version.
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

    Unchanged from prior version.
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

    Unchanged policy:
        - 2+ HIGH          -> HIGH
        - 1 HIGH + 1+ MED  -> HIGH
        - 1 HIGH alone     -> MEDIUM
        - 3+ MED           -> HIGH
        - 1-2 MED, no HIGH -> MEDIUM
        - Else             -> LOW
    """
    values = [levels.get(k, "LOW") for k in ("fundamental", "macro", "market", "agent")]
    high_count = sum(1 for v in values if v == "HIGH")
    medium_count = sum(1 for v in values if v == "MEDIUM")

    if high_count >= 2:
        return "HIGH"
    if high_count == 1 and medium_count >= 1:
        return "HIGH"
    if high_count == 1:
        return "MEDIUM"
    if medium_count >= 3:
        return "HIGH"
    if medium_count >= 1:
        return "MEDIUM"
    return "LOW"


def _max_level(a: str, b: str) -> str:
    """Return the more severe of two risk levels."""
    return _RANK_LEVEL[max(_LEVEL_RANK.get(a, 0), _LEVEL_RANK.get(b, 0))]


# =============================================================================
# INTERNAL: PEER-STATS RESOLUTION
# =============================================================================

def _resolve_peer_stats(
    metrics: Dict[str, Any],
    explicit: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Layered resolution: snapshot > kwarg > sector defaults > {}."""
    embedded = metrics.get("peer_stats") if isinstance(metrics, dict) else None
    if isinstance(embedded, dict) and embedded:
        return embedded

    if explicit:
        return explicit

    sector = metrics.get("sector") if isinstance(metrics, dict) else None
    if isinstance(sector, str):
        key = sector.strip().lower()
        if key in _SECTOR_DEFAULTS:
            return _SECTOR_DEFAULTS[key]

    return {}


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


def _read(d: Dict[str, Any], canonical: str) -> Any:
    """Read ``d[canonical]`` trying every known alias, in priority order."""
    for key in _METRIC_ALIASES.get(canonical, (canonical,)):
        v = d.get(key)
        if v is not None:
            return v
    return None


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


def _clip01(x: float) -> float:
    """Clamp ``x`` to the unit interval [0, 1]."""
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x