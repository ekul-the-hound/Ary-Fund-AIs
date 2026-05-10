"""
agent/data_point_analyzer.py
============================

Per-data-point investment analysis for the Streamlit dashboard.

The user checks a set of data points in the UI (e.g. "Price", "Trailing P/E",
"VIX", "Free Cash Flow") and this module produces:

  1. ONE overview paragraph (150-200 words) synthesizing the SELECTED points
     into a single investment thesis.
  2. ONE paragraph per selected point (150-200 words each) explaining what
     that specific number means for buying the stock.

Total paragraphs = 1 + N where N = number of selected data points.

Design contract
---------------
- Reuses the existing ``data.pipeline.build_agent_context()`` output. No
  extra API calls — the UI passes in the context it already has.
- Same graceful-degradation pattern as ``thesis_essay.py``: deterministic
  fallback when the LLM is unavailable.
- Pure function — no side effects beyond logging.

Usage
-----
::

    from agent.data_point_analyzer import (
        analyze_data_points, AVAILABLE_DATA_POINTS, get_data_point_value,
    )

    # The UI's checkboxes map to keys from AVAILABLE_DATA_POINTS.
    selected = ["prices.last", "metrics.trailing_pe", "macro.vix"]

    result = analyze_data_points(
        ticker=ticker,
        selected_keys=selected,
        context=ctx,           # output of build_agent_context()
        config=config,
    )
    # result["text"] is the full analysis (overview + N paragraphs)
    # result["paragraphs"] is a dict {key: paragraph_text} for per-point rendering
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.base_agent import AgentRequest, _resolve_model


logger = logging.getLogger(__name__)


# =============================================================================
# DATA POINT REGISTRY
# =============================================================================
#
# Each entry is: dotted-path-key -> (display_name, category, formatter)
# - dotted_path: how to look it up in the build_agent_context() dict
# - display_name: human-readable label for the UI checkbox
# - category: groups checkboxes in the UI (Price / Metrics / Macro)
# - formatter: callable(value) -> str for rendering the value in prompts/UI

def _fmt_dollar(v: Any) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_dollar_big(v: Any) -> str:
    """Format a large dollar value as $XB or $YT."""
    try:
        v = float(v)
        if abs(v) >= 1e12:
            return f"${v / 1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct(v: Any) -> str:
    """yfinance returns margins/growth as decimals (0.473 = 47.3%)."""
    try:
        return f"{float(v) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_pct_raw(v: Any) -> str:
    """For values that are already in percent units (e.g. fed_funds = 5.33)."""
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ratio(v: Any) -> str:
    try:
        return f"{float(v):.2f}x"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_str(v: Any) -> str:
    return "n/a" if v is None else str(v)


# Registry: key -> (display_name, category, formatter)
# Categories drive UI grouping.
#
# IMPORTANT — key naming:
#   The dotted paths below MUST exactly match the keys produced by
#   ``data.pipeline.build_agent_context``. That function flattens
#   ``market_data.get_fundamentals()`` sections (overview / valuation /
#   financials / growth / analyst) into ``ctx["metrics"]`` using the
#   snake_case keys those sections already use (e.g. ``free_cash_flow``,
#   ``revenue_growth``, ``debt_to_equity``). A mismatch — for example
#   looking up ``metrics.freeCashflow`` when the pipeline writes
#   ``free_cash_flow`` — silently resolves to None and the UI shows
#   "n/a", which was the original bug.
#
#   When you add a new entry: cross-check against the keys in
#   ``market_data.get_fundamentals()`` (or, for macro entries, against
#   ``macro_data.get_macro_dashboard()``). Run the smoke test in
#   ``test_data_point_analyzer.py`` to catch regressions.
AVAILABLE_DATA_POINTS: Dict[str, Tuple[str, str, Any]] = {
    # --- Price ---
    "prices.last": ("Last Price", "Price", _fmt_dollar),
    # MarketData.get_latest_price() already stores change_pct in percent
    # units (e.g. -4.63), not as a decimal — use _fmt_pct_raw, NOT
    # _fmt_pct (which would render -4.63 as "-463.00%").
    "prices.change_pct": ("Daily Change %", "Price", _fmt_pct_raw),
    "prices.market_cap": ("Market Cap", "Price", _fmt_dollar_big),
    "prices.fifty_two_week_high": ("52-Week High", "Price", _fmt_dollar),
    "prices.fifty_two_week_low": ("52-Week Low", "Price", _fmt_dollar),

    # --- Valuation Metrics ---
    "metrics.trailing_pe": ("Trailing P/E", "Valuation", _fmt_ratio),
    "metrics.forward_pe": ("Forward P/E", "Valuation", _fmt_ratio),
    "metrics.peg_ratio": ("PEG Ratio", "Valuation", _fmt_ratio),
    "metrics.price_to_sales": ("Price-to-Sales", "Valuation", _fmt_ratio),
    "metrics.ev_to_ebitda": ("EV / EBITDA", "Valuation", _fmt_ratio),

    # --- Margins ---
    "metrics.gross_margin": ("Gross Margin", "Margins", _fmt_pct),
    "metrics.operating_margin": ("Operating Margin", "Margins", _fmt_pct),
    "metrics.profit_margin": ("Profit Margin", "Margins", _fmt_pct),

    # --- Growth ---
    "metrics.revenue_growth": ("Revenue Growth (YoY)", "Growth", _fmt_pct),
    "metrics.earnings_growth": ("Earnings Growth (YoY)", "Growth", _fmt_pct),

    # --- Cash & Capital ---
    "metrics.free_cash_flow": ("Free Cash Flow", "Cash & Balance Sheet", _fmt_dollar_big),
    "metrics.operating_cash_flow": ("Operating Cash Flow", "Cash & Balance Sheet", _fmt_dollar_big),
    "metrics.total_cash": ("Total Cash", "Cash & Balance Sheet", _fmt_dollar_big),
    "metrics.total_debt": ("Total Debt", "Cash & Balance Sheet", _fmt_dollar_big),
    "metrics.debt_to_equity": ("Debt-to-Equity", "Cash & Balance Sheet", _fmt_num),
    "metrics.current_ratio": ("Current Ratio", "Cash & Balance Sheet", _fmt_ratio),

    # --- Returns ---
    "metrics.return_on_equity": ("Return on Equity", "Returns", _fmt_pct),
    "metrics.return_on_assets": ("Return on Assets", "Returns", _fmt_pct),

    # --- Analyst ---
    "metrics.target_mean": ("Analyst Target Price", "Analyst", _fmt_dollar),
    # Note: requires market_data.get_fundamentals() to capture
    # info.get("recommendationMean"). If that field is absent the
    # display will fall back to "n/a" gracefully.
    "metrics.recommendation_mean": ("Analyst Recommendation Score", "Analyst", _fmt_num),

    # --- Macro: Rates ---
    # NOTE: pipeline._flatten_macro lifts the nested macro_data sections
    # ("interest_rates", "inflation", etc.) to top-level keys. So the
    # context's macro dict is FLAT (e.g. ctx["macro"]["fed_funds"]), not
    # nested. The keys below reflect that flattening.
    "macro.fed_funds": ("Fed Funds Rate", "Macro / Rates", _fmt_pct_raw),
    "macro.treasury_10y": ("10-Year Treasury Yield", "Macro / Rates", _fmt_pct_raw),
    "macro.treasury_2y": ("2-Year Treasury Yield", "Macro / Rates", _fmt_pct_raw),
    "macro.yield_curve_spread": ("Yield Curve (10Y-2Y)", "Macro / Rates", _fmt_pct_raw),

    # --- Macro: Inflation & Growth ---
    "macro.cpi_yoy_pct": ("CPI YoY", "Macro / Inflation", _fmt_pct_raw),
    "macro.unemployment_rate": ("Unemployment Rate", "Macro / Employment", _fmt_pct_raw),
    "macro.gdp_growth": ("GDP Growth (Annualized)", "Macro / Growth", _fmt_pct_raw),

    # --- Macro: Conditions ---
    "macro.vix": ("VIX (Volatility Index)", "Macro / Conditions", _fmt_num),
    # FRED RECPROUSM156N is reported in percent units already (e.g. 23.5
    # means 23.5%) — use _fmt_pct_raw, NOT _fmt_pct (which would multiply
    # by 100 and produce absurd values like "+2350.00%").
    "macro.recession_probability": (
        "Recession Probability (12M)", "Macro / Conditions", _fmt_pct_raw,
    ),
}


# =============================================================================
# METRIC-SPECIFIC PROMPT GUIDANCE
# =============================================================================
#
# Each entry tells the LLM how to reason about that specific metric. The
# guidance is injected inline in the prompt next to the metric's value so
# the model has the right interpretive frame for each data point. Without
# this, the model produced vague output for less-common fields (e.g. it
# would treat "Recommendation Score 1.8" as a generic number rather than
# knowing 1=Strong Buy / 5=Strong Sell on the Wall Street scale).
#
# Format per entry: 2-4 sentences covering what the metric means, how to
# interpret high/low values, what historical/peer context matters, and
# how to phrase the conclusion.

METRIC_GUIDANCE: Dict[str, str] = {
    # --- Price ---
    "prices.last": (
        "The current share price is most useful when paired with valuation "
        "multiples and 52-week range. A price near the 52-week high suggests "
        "momentum but also limited margin of safety; a price near the low "
        "may indicate value or genuine deterioration — the metrics will tell "
        "you which. Reference the company's recent price action, not the "
        "absolute level."
    ),
    "prices.change_pct": (
        "Daily change is noise on its own; only meaningful in context. A "
        "+5% move on no news is suspect; a -5% move into earnings carries "
        "different information than -5% on a quiet macro day. Cite whether "
        "the move is consistent with sector or VIX behavior, and whether "
        "it aligns with or contradicts the longer-term trend."
    ),
    "prices.market_cap": (
        "Market cap defines the universe a stock plays in. <$2B = small "
        "cap (illiquid, higher volatility), $2-10B = mid cap, $10-200B = "
        "large cap, >$200B = mega cap. Compare to the leader of the same "
        "sector. Mega caps are typically less mispriced; smaller caps offer "
        "more alpha but more risk."
    ),
    "prices.fifty_two_week_high": (
        "Distance from 52-week high signals momentum and crowding. Within "
        "5% of the high = overheated/popular; 20%+ off = either a value "
        "opportunity or a falling knife. Pair with revenue growth and "
        "margin trend to decide which."
    ),
    "prices.fifty_two_week_low": (
        "Proximity to 52-week low is a contrarian signal. Within 10% of "
        "the low merits a value-vs-deterioration check: are fundamentals "
        "intact (FCF positive, margins stable) or impaired (FCF negative, "
        "leverage rising)? State which case applies."
    ),

    # --- Valuation ---
    "metrics.trailing_pe": (
        "Trailing P/E is past 12-month earnings yield. S&P 500 long-term "
        "average ~16x; current ~22x. <15x = cheap, 15-25x = fair for "
        "average growth, >30x = expensive unless growth is exceptional. "
        "ALWAYS pair with growth: a 30x P/E is reasonable at 25% growth, "
        "punitive at 5%. Sector matters: tech runs higher, financials lower."
    ),
    "metrics.forward_pe": (
        "Forward P/E uses next-year analyst estimates. Compare to trailing "
        "P/E: if forward < trailing, analysts expect earnings to grow; if "
        "forward > trailing, they expect contraction. A wide gap (forward "
        "much lower than trailing) signals either strong growth or analyst "
        "over-optimism — verify against revenue_growth and earnings_growth."
    ),
    "metrics.peg_ratio": (
        "PEG = P/E divided by earnings growth rate. PEG < 1 = arguably "
        "cheap relative to growth; PEG = 1 fair; PEG > 2 = expensive even "
        "for the growth. Caveat: PEG breaks down for low-growth (<5%) or "
        "negative-earnings companies — say so if either applies."
    ),
    "metrics.price_to_sales": (
        "P/S is most useful when earnings are negative or volatile (early-"
        "stage tech, cyclicals at trough). S&P average ~2-3x. >10x demands "
        "exceptional gross margins (>60%) or hypergrowth. For mature "
        "businesses with stable earnings, P/E is the better lens."
    ),
    "metrics.ev_to_ebitda": (
        "EV/EBITDA captures the whole-company value (debt + equity) per "
        "dollar of operating cash earnings. Industry-average ~10-12x. "
        "<8x = cheap; 12-18x = full; >20x = priced for perfection. "
        "Better than P/E for capital-heavy or leveraged businesses because "
        "it neutralizes capital structure differences."
    ),

    # --- Margins ---
    "metrics.gross_margin": (
        "Gross margin = (Revenue − COGS) / Revenue. Reflects pricing power "
        "and cost structure. Software/SaaS: >70%. Branded consumer: 30-50%. "
        "Retail: 20-35%. Commodities/distribution: 5-15%. Year-over-year "
        "compression of 200+ bps signals input-cost pressure or pricing "
        "loss; expansion signals scale or premium positioning."
    ),
    "metrics.operating_margin": (
        "Operating margin reflects core profitability after SG&A and R&D. "
        "Cite the gap between gross and operating: a wide gap means high "
        "fixed costs, scale matters; narrow gap means lean ops. <5% = "
        "thin/risky; 10-20% = healthy; >25% = exceptional (typically "
        "software, payment networks, or monopoly assets)."
    ),
    "metrics.profit_margin": (
        "Net profit margin includes tax and interest. Compare to operating "
        "margin: if much lower, the company is heavily taxed or interest-"
        "burdened. <3% = thin; 5-10% = average; >15% = high-quality. "
        "Trend matters more than absolute level — direction signals "
        "operating leverage."
    ),

    # --- Growth ---
    "metrics.revenue_growth": (
        "Year-over-year revenue growth. Critical to pair with the company's "
        "size: 30% growth on $200M is great, on $200B is extraordinary. "
        "S&P 500 long-term ~5-7%. <0% = contracting (concerning unless "
        "cyclical trough), 0-10% = mature, 10-20% = strong, >20% = high-"
        "growth. Compare to trailing P/E to assess if growth justifies "
        "the multiple."
    ),
    "metrics.earnings_growth": (
        "Earnings growth amplifies or dampens revenue growth based on "
        "operating leverage. EPS growth >> revenue growth means margin "
        "expansion (good); EPS growth << revenue growth means margin "
        "compression or higher share count (bad). Negative earnings "
        "growth on positive revenue growth is a yellow flag."
    ),

    # --- Cash & Capital ---
    "metrics.free_cash_flow": (
        "FCF = operating cash flow minus capex. The cleanest measure of "
        "cash returnable to shareholders. Compare to net income: if FCF "
        "approximates net income, earnings are real; if FCF lags, watch "
        "for working-capital strain or aggressive accruals. Negative FCF "
        "is acceptable for hypergrowth (Amazon early years) but a warning "
        "for mature businesses."
    ),
    "metrics.operating_cash_flow": (
        "Operating cash flow before capex. Direction matters most: rising "
        "OCF with stable margins = healthy; falling OCF on growing revenue "
        "= working capital deterioration. Always cite OCF / Net Income "
        "ratio: above 1.0 is high quality; below 0.7 deserves scrutiny."
    ),
    "metrics.total_cash": (
        "Cash and equivalents on the balance sheet. Useful as a dry-powder "
        "indicator and downside cushion. Compare to total debt and to "
        "annual operating cash flow: cash > debt = net cash position "
        "(strong); cash < 1 year of OCF and debt growing = potential "
        "liquidity issue."
    ),
    "metrics.total_debt": (
        "Total interest-bearing debt. The absolute number matters less "
        "than debt/EBITDA (covered separately). Year-over-year increase "
        "in debt without a corresponding revenue or asset increase is a "
        "yellow flag. Refinancing risk rises when rates are elevated — "
        "cite the current Fed Funds rate context if available."
    ),
    "metrics.debt_to_equity": (
        "Debt/Equity. <0.5 = conservative; 0.5-1.5 = typical; >2.0 = "
        "leveraged. Banks and utilities run higher by design. The metric "
        "can mislead with negative book equity (buybacks, persistent "
        "losses) — note if equity is negative or shrinking."
    ),
    "metrics.current_ratio": (
        "Current assets / current liabilities. >1.5 = liquid and safe; "
        "1.0-1.5 = adequate; <1.0 = potential short-term squeeze. "
        "Different industries normalize at different levels — software "
        "and subscription businesses often run high; retailers run lower "
        "because inventory turns fast."
    ),

    # --- Returns ---
    "metrics.return_on_equity": (
        "Net income / shareholders' equity. >15% = high-quality; 10-15% = "
        "average; <8% = capital-inefficient. Caveats: ROE is inflated by "
        "leverage (high debt makes ROE rise without underlying improvement) "
        "— always cross-check ROIC, which neutralizes capital structure."
    ),
    "metrics.return_on_assets": (
        "Net income / total assets. Less affected by leverage than ROE. "
        ">5% = solid; >10% = excellent. Asset-heavy businesses (retailers, "
        "industrials) run lower; asset-light (software, services) run "
        "higher. Compare to peers, not absolute thresholds."
    ),

    # --- Analyst ---
    "metrics.target_mean": (
        "Mean analyst price target for the next 12 months. Implied upside "
        "= (target / current price - 1). Anything >20% upside or >20% "
        "downside is meaningful; tighter ranges suggest consensus and "
        "less alpha opportunity. Caveat: analyst targets are anchored to "
        "current price and trend-follow — treat as one input, not a "
        "decision driver."
    ),
    "metrics.recommendation_mean": (
        "Mean analyst rating on the standard 1-5 scale: 1.0 = Strong Buy, "
        "2.0 = Buy, 3.0 = Hold, 4.0 = Sell, 5.0 = Strong Sell. Most stocks "
        "cluster between 1.8 and 2.8 (sell-side has buy-bias). Below 1.5 "
        "= unanimous buy (often crowded); above 3.0 = unloved (potential "
        "contrarian setup). State the rating and what it implies about "
        "consensus crowding."
    ),

    # --- Macro ---
    "macro.fed_funds": (
        "The Federal Reserve's benchmark short rate. Above 4-5% = restrictive "
        "(slows borrowing, equities). Below 2% = accommodative (supports "
        "risk assets). The transmission to equities depends on duration: "
        "long-duration assets (high-growth tech) feel rates more than "
        "short-duration (utilities, staples). Reference the rate level "
        "and direction of recent moves."
    ),
    "macro.treasury_10y": (
        "10-year Treasury yield. The discount rate baseline for equities. "
        ">5% = headwind for valuations; 3-5% = normal range; <3% = supportive. "
        "Direction matters: rising yields compress P/E multiples even if "
        "earnings hold. Cite the level AND whether it's risen/fallen recently."
    ),
    "macro.treasury_2y": (
        "2-year Treasury yield. Reflects near-term Fed expectations. "
        "Compare to the 10Y: if 2Y > 10Y, the curve is inverted (recession "
        "signal, ~85% historical hit rate within 18 months). If 2Y < 10Y, "
        "curve is normal."
    ),
    "macro.yield_curve_spread": (
        "10Y minus 2Y Treasury yield. Negative (inverted) historically "
        "precedes recessions by 6-18 months — though the lag and false-"
        "positive rate vary. Above +0.5% = normal; 0 to +0.5% = flat; "
        "below 0 = inverted. Cite the spread and what it implies for the "
        "growth backdrop the company operates in."
    ),
    "macro.cpi_yoy_pct": (
        "Year-over-year CPI inflation. Fed target = 2%. Above 4% = elevated "
        "(supports tighter policy and valuation pressure); 2-4% = manageable; "
        "<2% = soft (supportive of growth assets). Direction matters: "
        "decelerating inflation is bullish for risk assets even at high "
        "absolute levels."
    ),
    "macro.unemployment_rate": (
        "U-3 unemployment. <4% = tight labor market (supports consumer "
        "spending, wage pressure); 4-5% = healthy; >5% = soft and "
        "deteriorating. Rising 0.5%+ over six months has historically "
        "preceded recessions (Sahm rule). Cite the level and the trend."
    ),
    "macro.gdp_growth": (
        "Annualized real GDP growth. >3% = strong expansion; 1-3% = "
        "trend growth; <1% = stagnation; negative = contraction. "
        "Cyclical sectors (industrials, materials, consumer discretionary) "
        "benefit most from above-trend growth; defensive sectors (staples, "
        "utilities, healthcare) less sensitive."
    ),
    "macro.vix": (
        "CBOE Volatility Index — the market's 30-day forward expected "
        "volatility from S&P 500 options. <15 = complacent; 15-20 = "
        "normal; 20-30 = elevated stress; >30 = panic. High-beta names "
        "feel VIX more; low-beta defensives less. Cite the level and "
        "what it implies about position sizing."
    ),
    "macro.recession_probability": (
        "NY Fed model probability of recession in 12 months, derived from "
        "the yield curve. Reported in percent units (e.g. 23.5 = 23.5%). "
        "<15% = benign; 15-30% = elevated; 30-50% = high-risk; >50% = "
        "near-certain. Cyclical companies (industrials, financials, "
        "consumer discretionary) are most exposed; staples and healthcare "
        "less so."
    ),
}


# =============================================================================
# DERIVED-VALUE FALLBACKS
# =============================================================================
#
# When a key returns None (the metric isn't in context), some fields can be
# reconstructed from related fields. This is the second source of "n/a" the
# review flagged — values that look unavailable but can be derived. We only
# return a derived value when the math is clean and uncontroversial; when
# in doubt, return None and let the formatter show "n/a".

def _derive_value(context: Dict[str, Any], key: str) -> Any:
    """Try to compute ``key`` from related fields. Return None on failure."""
    metrics = context.get("metrics") or {}
    prices = context.get("prices") or {}

    # Trailing P/E from price and EPS, if both present.
    if key == "metrics.trailing_pe":
        last = prices.get("last") or context.get("price")
        eps = metrics.get("trailingEps") or metrics.get("eps_trailing")
        if last and eps:
            try:
                eps_f = float(eps)
                if eps_f != 0:
                    return float(last) / eps_f
            except (TypeError, ValueError):
                pass

    # Forward P/E from price and forward EPS.
    if key == "metrics.forward_pe":
        last = prices.get("last") or context.get("price")
        feps = metrics.get("forwardEps") or metrics.get("eps_forward")
        if last and feps:
            try:
                feps_f = float(feps)
                if feps_f != 0:
                    return float(last) / feps_f
            except (TypeError, ValueError):
                pass

    # PEG from forward P/E and earnings growth (if pegRatio absent).
    if key == "metrics.peg_ratio":
        fpe = metrics.get("forward_pe") or metrics.get("forwardPE")
        eg = metrics.get("earnings_growth") or metrics.get("earningsGrowth")
        if fpe and eg:
            try:
                eg_pct = float(eg) * 100  # decimal -> percent
                if eg_pct > 0:
                    return float(fpe) / eg_pct
            except (TypeError, ValueError):
                pass

    # Profit margin from net income / revenue.
    if key == "metrics.profit_margin":
        ni = metrics.get("net_income") or metrics.get("netIncomeToCommon")
        rev = metrics.get("revenue") or metrics.get("totalRevenue")
        if ni and rev:
            try:
                rev_f = float(rev)
                if rev_f != 0:
                    return float(ni) / rev_f
            except (TypeError, ValueError):
                pass

    # FCF yield from FCF and market cap.
    if key == "metrics.fcf_yield":
        fcf = metrics.get("free_cash_flow") or metrics.get("freeCashflow")
        mc = metrics.get("market_cap") or prices.get("market_cap") or metrics.get("marketCap")
        if fcf and mc:
            try:
                mc_f = float(mc)
                if mc_f > 0:
                    return float(fcf) / mc_f
            except (TypeError, ValueError):
                pass

    # Current ratio from current assets / current liabilities.
    if key == "metrics.current_ratio":
        ca = metrics.get("currentAssets") or metrics.get("current_assets")
        cl = metrics.get("currentLiabilities") or metrics.get("current_liabilities")
        if ca and cl:
            try:
                cl_f = float(cl)
                if cl_f > 0:
                    return float(ca) / cl_f
            except (TypeError, ValueError):
                pass

    return None


# =============================================================================
# CONSTANTS
# =============================================================================

_PARAGRAPH_TEMPERATURE: float = 0.30
_TARGET_WORDS_PER_PARA_MIN: int = 150
_TARGET_WORDS_PER_PARA_MAX: int = 200


# =============================================================================
# PUBLIC API
# =============================================================================

def get_data_point_value(context: Dict[str, Any], key: str) -> Any:
    """Look up a dotted-path key in the agent context.

    First tries direct lookup. If that returns None, tries to derive the
    value from related fields via :func:`_derive_value`. Only returns
    None when both lookup and derivation fail.

    Examples
    --------
    >>> get_data_point_value(ctx, "prices.last")
    215.43
    >>> get_data_point_value(ctx, "macro.fed_funds")
    5.33
    """
    parts = key.split(".")
    cursor: Any = context
    for p in parts:
        if isinstance(cursor, dict):
            cursor = cursor.get(p)
        else:
            cursor = None
            break
        if cursor is None:
            break

    if cursor is not None:
        return cursor

    # Direct lookup failed — try a derived fallback.
    return _derive_value(context, key)


def get_display_name(key: str) -> str:
    """Friendly label for a key."""
    entry = AVAILABLE_DATA_POINTS.get(key)
    return entry[0] if entry else key


def get_formatted_value(context: Dict[str, Any], key: str) -> str:
    """Look up + format a value for display."""
    raw = get_data_point_value(context, key)
    entry = AVAILABLE_DATA_POINTS.get(key)
    if entry is None:
        return _fmt_str(raw)
    return entry[2](raw)


def get_categories() -> Dict[str, List[str]]:
    """Return data points grouped by category for UI rendering.

    Returns
    -------
    dict
        {category_name: [list_of_keys]}
    """
    grouped: Dict[str, List[str]] = {}
    for key, (_, category, _) in AVAILABLE_DATA_POINTS.items():
        grouped.setdefault(category, []).append(key)
    return grouped


def analyze_data_points(
    ticker: str,
    selected_keys: List[str],
    context: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Generate overview + per-point analysis paragraphs.

    Parameters
    ----------
    ticker : str
        The ticker being analyzed (e.g. "AAPL").
    selected_keys : list of str
        Dotted-path keys from AVAILABLE_DATA_POINTS that the user checked.
    context : dict
        Output of ``data.pipeline.build_agent_context(ticker)``. Used to
        look up values for each selected key. No extra API calls are made.
    config : module or namespace
        The application config (for OLLAMA_BASE_URL, MAX_TOKENS, etc.).

    Returns
    -------
    dict
        {
          "text": str,                # full analysis (overview + paragraphs)
          "overview": str,            # just the overview paragraph
          "paragraphs": dict,         # {key: paragraph_text}
          "selected_keys": list,
          "model_used": str,
          "elapsed_ms": float,
          "fallback": bool,
          "word_count": int,
        }
    """
    started_at = time.perf_counter()

    if not selected_keys:
        return {
            "text": "(No data points selected.)",
            "overview": "",
            "paragraphs": {},
            "selected_keys": [],
            "model_used": "n/a",
            "elapsed_ms": 0.0,
            "fallback": True,
            "word_count": 0,
        }

    # Filter to only valid keys we know how to handle
    valid_keys = [k for k in selected_keys if k in AVAILABLE_DATA_POINTS]
    if not valid_keys:
        return {
            "text": "(No recognized data points selected.)",
            "overview": "",
            "paragraphs": {},
            "selected_keys": selected_keys,
            "model_used": "n/a",
            "elapsed_ms": 0.0,
            "fallback": True,
            "word_count": 0,
        }

    sector = context.get("sector") or context.get("metrics", {}).get("sector") or "n/a"
    industry = context.get("industry") or context.get("metrics", {}).get("industry") or "n/a"
    as_of = context.get("as_of") or time.strftime("%Y-%m-%d")

    prompt = _build_prompt(
        ticker=ticker,
        sector=sector,
        industry=industry,
        as_of=as_of,
        selected_keys=valid_keys,
        context=context,
    )

    pseudo_request = AgentRequest(prompt=prompt, context={}, model_tag=None)
    model_used = _resolve_model(pseudo_request, config)

    if model_used == "mock":
        logger.info("data_point_analyzer | %s | mock mode -> deterministic fallback", ticker)
        text = _deterministic_fallback(ticker, valid_keys, context)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        overview, paragraphs = _split_text(text, valid_keys)
        return {
            "text": text,
            "overview": overview,
            "paragraphs": paragraphs,
            "selected_keys": valid_keys,
            "model_used": "mock",
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": True,
            "word_count": len(text.split()),
        }

    try:
        text = _call_ollama_text(prompt, model_used, config, temperature=_PARAGRAPH_TEMPERATURE)
        # Acceptance check — replaces the previous rigid
        # ``len(text.split()) >= 150 * (1+N) // 2`` rule, which rejected
        # perfectly usable concise output from smaller models like
        # phi3:3.8b (a 350-word reply for 8 points would fail at the
        # 675-word threshold even though it parsed cleanly into
        # overview + 8 paragraphs).
        #
        # The new gate is structural: we parse the text first and accept
        # it as long as (a) it has at least 100 words total AND (b) it
        # contains either an Overview paragraph or at least one
        # recognizable per-point paragraph. Truly broken responses
        # ("I cannot help with that", empty string, prompt regurgitation)
        # still fail.
        word_count = len(text.split()) if text else 0
        parsed_overview, parsed_paragraphs = _split_text(text or "", valid_keys)
        has_structure = bool(parsed_overview) or len(parsed_paragraphs) >= 1
        if word_count < 100 or not has_structure:
            raise RuntimeError(
                f"Output unusable: {word_count} words, "
                f"overview={'yes' if parsed_overview else 'no'}, "
                f"paragraphs_parsed={len(parsed_paragraphs)}"
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        # Reuse the parse we already did.
        overview, paragraphs = parsed_overview, parsed_paragraphs
        logger.info(
            "data_point_analyzer | %s | model=%s | points=%d | words=%d | "
            "paragraphs_parsed=%d | elapsed_ms=%.0f",
            ticker, model_used, len(valid_keys), word_count,
            len(paragraphs), elapsed_ms,
        )
        return {
            "text": text,
            "overview": overview,
            "paragraphs": paragraphs,
            "selected_keys": valid_keys,
            "model_used": model_used,
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": False,
            "word_count": word_count,
        }
    except Exception as exc:
        logger.warning("data_point_analyzer | %s | LLM failed: %s -> fallback", ticker, exc)
        text = _deterministic_fallback(ticker, valid_keys, context)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        overview, paragraphs = _split_text(text, valid_keys)
        return {
            "text": text,
            "overview": overview,
            "paragraphs": paragraphs,
            "selected_keys": valid_keys,
            "model_used": f"{model_used} (failed)",
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": True,
            "word_count": len(text.split()),
        }


# =============================================================================
# PROMPT
# =============================================================================

def _build_prompt(
    ticker: str,
    sector: str,
    industry: str,
    as_of: str,
    selected_keys: List[str],
    context: Dict[str, Any],
) -> str:
    # Build a compact "selected points" section: name = value, plus
    # per-metric reasoning guidance so the model has the right interpretive
    # frame for each data point.
    selected_lines: List[str] = []
    for k in selected_keys:
        display = get_display_name(k)
        value = get_formatted_value(context, k)
        guidance = METRIC_GUIDANCE.get(k, "")
        selected_lines.append(f"- {display}: {value}")
        if guidance:
            selected_lines.append(f"  GUIDANCE: {guidance}")
    selected_block = "\n".join(selected_lines)

    # Brief supporting context (not the focus, just so the LLM has anchors
    # if it needs to reference benchmarks like "vs sector average")
    metrics = context.get("metrics") or {}
    macro = context.get("macro") or {}
    supporting_lines: List[str] = []
    # Add a few high-signal anchors that aren't necessarily selected.
    # Note: macro is now a FLAT dict (post-_flatten_macro), so we look
    # up "fed_funds" directly, not "interest_rates.fed_funds".
    if metrics.get("trailing_pe") is not None and "metrics.trailing_pe" not in selected_keys:
        supporting_lines.append(f"  - Trailing P/E (context): {_fmt_ratio(metrics.get('trailing_pe'))}")
    fed = macro.get("fed_funds")
    if fed is not None and "macro.fed_funds" not in selected_keys:
        supporting_lines.append(f"  - Fed Funds (context): {_fmt_pct_raw(fed)}")
    supporting_block = "\n".join(supporting_lines) if supporting_lines else "  (none)"

    # Resolve the company's full name for stock-specific framing. Without
    # this, briefings drift into generic sector commentary on less-famous
    # tickers (the COST -> "discount retailer" boilerplate problem).
    company_name = (
        metrics.get("name")
        or metrics.get("shortName")
        or metrics.get("longName")
        or ticker
    )

    n_points = len(selected_keys)
    total_paragraphs = 1 + n_points

    # Render the per-point output template — this drives the LLM toward the
    # "Display Name:" header format we'll parse later.
    output_template_lines = [
        "Overview:",
        "[150-200 word synthesis of all selected points]",
        "",
    ]
    for k in selected_keys:
        display = get_display_name(k)
        output_template_lines.append(f"{display}:")
        output_template_lines.append(f"[150-200 word analysis of {display}]")
        output_template_lines.append("")
    output_template = "\n".join(output_template_lines)

    return f"""You are writing an institutional-grade stock research note for a hedge fund analyst.

=== COMPANY ===
Ticker: {ticker}
Company name: {company_name}
Sector: {sector}
Industry: {industry}
As of: {as_of}

=== SELECTED DATA POINTS (the analyst checked these {n_points}) ===
Each item below has a value and a GUIDANCE line telling you how to reason
about that specific metric. Use the guidance — do not substitute generic
analyst boilerplate.

{selected_block}

=== SUPPORTING CONTEXT (available for benchmarking, not the focus) ===
{supporting_block}

=== TASK ===
Produce {total_paragraphs} paragraphs total: ONE overview paragraph plus one paragraph per selected data point. Every paragraph must be specifically about **{company_name} ({ticker})**, not its sector or industry in general.

1. OVERVIEW (150-200 words): Synthesize the selected points into a single investment thesis for {company_name} ({ticker}). State which selected metrics are most bullish, which are most bearish, and a clear BUY / HOLD / AVOID stance. Reference numbers from the selected points only — do not invent.

2. PER-POINT PARAGRAPHS (150-200 words each): For each selected point, write one paragraph that:
   - Quotes the exact value shown above.
   - Applies the GUIDANCE for that metric — interpret high/low correctly for that specific field.
   - Compares the value to a relevant benchmark (sector average, historical norm, peer, or macro context).
   - Explains the investment implication for {company_name} ({ticker}): how it affects cash flow, valuation, growth, or risk for THIS company.
   - Ends with an explicit stance for that single metric: "supports buying," "is a sell signal," or "is neutral."

=== RULES ===
- Use the EXACT display names above as paragraph headers (e.g. "Trailing P/E:", "Fed Funds Rate:").
- Every paragraph must include at least one number AND mention {company_name} or {ticker} by name.
- Apply the GUIDANCE attached to each metric — don't fall back to generic sector commentary.
- Do not invent metrics that are not in the data.
- No bullet points inside paragraphs — write in prose.
- No filler ("appears," "may," "could") unless paired with a specific number or condition.
- Do not summarize the prompt or describe the task. Just write the paragraphs.

=== OUTPUT FORMAT (use this exact structure) ===
{output_template}

Now write the analysis for {company_name} ({ticker})."""


# =============================================================================
# OUTPUT PARSING
# =============================================================================

# Regex for stripping markdown header decoration. Matches optional leading
# whitespace, then optional ``#`` characters (1-6), then optional whitespace.
_MD_HASH_PREFIX = re.compile(r"^\s*#{1,6}\s*")

# Heading length ceiling — anything longer is body text, not a header.
_MAX_HEADER_LEN = 80


def _normalize_header(raw: str) -> Optional[str]:
    """Normalize an LLM-emitted heading to its canonical lowercase label.

    Accepts every common LLM output style:

      ``Overview``                 -> ``"overview"``
      ``Overview:``                -> ``"overview"``
      ``# Overview``               -> ``"overview"``
      ``## Overview``              -> ``"overview"``
      ``**Overview**``             -> ``"overview"``
      ``**Overview:**``            -> ``"overview"``
      ``__Overview__``             -> ``"overview"``
      ``## **Overview**``          -> ``"overview"``
      ``  Overview  ``             -> ``"overview"`` (whitespace tolerated)

    Returns ``None`` for anything that's clearly body text:

      - empty / whitespace-only strings
      - lines longer than 80 chars (real headers are short)
      - lines ending in sentence punctuation (``.``, ``;``)
      - lines containing comma followed by a space (prose connector)
      - lines containing ``=`` or ``$`` or ``%`` (numeric content)
      - lines containing the word ``is`` between alphanumeric runs
        (e.g. "Trailing P/E is 40.7x")

    This rejection list is empirical: it matches the body-text examples
    in ``TestNormalizeHeaderRejects``.
    """
    if not raw or not raw.strip():
        return None

    s = raw.strip()
    if len(s) > _MAX_HEADER_LEN:
        return None

    # Reject body-text patterns BEFORE we start stripping decoration —
    # so that "Trailing P/E is 40.7x" doesn't get mistaken for the
    # heading "Trailing P/E" after we strip everything after a space.
    if "$" in s or "%" in s or "=" in s:
        return None
    if s.endswith((".", ";")):
        return None
    # Semicolons inside the line are sentence connectors, not part of any
    # heading label.
    if ";" in s:
        return None
    if ", " in s:
        return None
    # "X is Y" pattern (embedded value): the word "is" between word chars.
    if re.search(r"\b\w+\s+is\s+\w", s, re.IGNORECASE):
        return None

    # Strip leading markdown hashes
    s = _MD_HASH_PREFIX.sub("", s).strip()
    # Strip bold/italic markdown delimiters anywhere
    s = s.replace("**", "").replace("__", "")
    # Strip trailing colon
    if s.endswith(":"):
        s = s[:-1]
    # Re-strip whitespace after decoration removal
    s = s.strip().lower()

    if not s:
        return None

    return s


def _split_text(text: str, selected_keys: List[str]) -> Tuple[str, Dict[str, str]]:
    """Split the LLM output into overview + per-point paragraphs.

    The model is instructed to produce headers like "Display Name:" or
    "## Display Name". We split on any line that ``_normalize_header``
    recognizes as a known section heading.

    Returns
    -------
    (overview_text, {key: paragraph_text})
    """
    # Build a header -> key map so we can match the LLM output's headers
    # back to our canonical keys (case-insensitive, ignoring trailing colons).
    header_to_key: Dict[str, str] = {}
    for k in selected_keys:
        display = get_display_name(k)
        header_to_key[display.lower().strip()] = k

    overview = ""
    paragraphs: Dict[str, str] = {}
    current_header: Optional[str] = None  # "overview" or display name (lower)
    current_lines: List[str] = []

    def _flush() -> None:
        nonlocal overview
        if current_header is None:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        if current_header == "overview":
            overview = body
        else:
            key = header_to_key.get(current_header)
            if key:
                paragraphs[key] = body

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        # Try to normalize as a heading
        normalized = _normalize_header(stripped)
        if normalized == "overview":
            _flush()
            current_header = "overview"
            current_lines = []
            continue
        if normalized in header_to_key:
            _flush()
            current_header = normalized
            current_lines = []
            continue

        current_lines.append(line)

    _flush()
    return overview, paragraphs


# =============================================================================
# DETERMINISTIC FALLBACK
# =============================================================================

def _deterministic_fallback(
    ticker: str, selected_keys: List[str], context: Dict[str, Any]
) -> str:
    """Mechanical fallback when LLM is unavailable.

    Produces a value-by-value description with no interpretation. Honest,
    short, and clearly less useful than the LLM output — which is the
    point: the user should know they're seeing a fallback.
    """
    lines = ["Overview:"]
    n = len(selected_keys)
    lines.append(
        f"This is a deterministic fallback for {ticker}. The LLM was unavailable, "
        f"so the system is reporting the {n} selected data point{'s' if n != 1 else ''} "
        f"verbatim without analysis. Connect to Ollama (or set the agent model tag to "
        f"a real model) to generate institutional-quality interpretation. The values "
        f"below are pulled from the existing pipeline context — no extra data fetching "
        f"was performed."
    )
    lines.append("")

    for k in selected_keys:
        display = get_display_name(k)
        value = get_formatted_value(context, k)
        lines.append(f"{display}:")
        lines.append(
            f"The current value of {display} for {ticker} is {value}. No interpretation "
            f"is available in fallback mode — this would normally include a comparison "
            f"to sector or historical benchmarks and an explicit buy/hold/avoid signal."
        )
        lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# OLLAMA TEXT CALL
# =============================================================================

def _call_ollama_text(
    prompt: str, model_name: str, config: Any, temperature: float = 0.3,
) -> str:
    import urllib.request

    base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 360.0)
    max_tokens = int(getattr(config, "MAX_TOKENS", 4096))

    body = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    return (payload.get("response") or "").strip()