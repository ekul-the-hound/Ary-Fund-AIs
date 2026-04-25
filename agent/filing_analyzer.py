"""
agent/filing_analyzer.py
========================

Pre-agent data shaping for SEC filings and fundamental metrics.

This module is **pure CPU, no LLM calls**. Its job is to take the raw outputs
of ``data/sec_fetcher.py`` and ``data/market_data.py`` (as assembled by
``pipeline.build_agent_context``) and turn them into compact, normalised
structures that:

    1. ``main.py`` can drop straight into the agent prompt.
    2. ``risk_scanner.py`` and ``thesis_generator.py`` can consume without
       having to re-parse filing text or re-derive ratios.

Two public functions
--------------------

    - :func:`summarize_filings_by_year` -> structured summary of recent
      10-K / 10-Q / 8-K filings (risk phrases, management tone, red flags).
    - :func:`extract_key_metrics_for_agent` -> normalised 10-ish metric
      snapshot a hedge-fund analyst actually looks at (growth, margins,
      leverage, valuation, cash conversion, coverage).

Both functions are defensive: missing or malformed inputs produce empty /
``None`` fields rather than raising, so a single bad ticker does not abort a
batch run.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)


# =============================================================================
# TONE / RED-FLAG LEXICONS
# =============================================================================
# Small hand-built lexicons are fine for a first pass. They're intentionally
# conservative; swap for an LLM-based tone classifier later if needed.

_CONFIDENT_TERMS: frozenset = frozenset({
    "record", "strong", "robust", "accelerate", "accelerating", "expand",
    "expanding", "momentum", "outperform", "beat", "exceeded", "growth",
    "opportunity", "leadership", "best-in-class", "market-leading",
})

_CAUTIOUS_TERMS: frozenset = frozenset({
    "uncertain", "uncertainty", "challenging", "headwind", "headwinds",
    "softness", "softening", "pressure", "cautious", "mixed", "slowdown",
    "moderate", "weaker", "decline", "contraction",
})

_DEFENSIVE_TERMS: frozenset = frozenset({
    "allegation", "investigation", "restatement", "material weakness",
    "going concern", "litigation", "subpoena", "sec inquiry",
    "non-reliance", "adverse opinion", "impairment",
})

# Phrases that warrant a standalone "red flag" entry in the output.
_RED_FLAG_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"going concern", re.IGNORECASE),
    re.compile(r"material weakness", re.IGNORECASE),
    re.compile(r"restatement", re.IGNORECASE),
    re.compile(r"non-reliance", re.IGNORECASE),
    re.compile(r"sec (?:inquiry|investigation|subpoena)", re.IGNORECASE),
    re.compile(r"auditor\s+resign", re.IGNORECASE),
    re.compile(r"delist(?:ing)?", re.IGNORECASE),
    re.compile(r"covenant\s+(?:breach|violation|waiver)", re.IGNORECASE),
    re.compile(r"impair(?:ment|ed)\s+(?:goodwill|assets)", re.IGNORECASE),
    re.compile(r"adverse\s+opinion", re.IGNORECASE),
)

# Item 1A is where 10-K risk factors live. Match the header and grab what
# follows, stopping at the next Item header or end of text.
_RISK_SECTION_RE = re.compile(
    r"item\s*1a[\.\s]*risk\s*factors(.+?)(?=item\s*1b|item\s*2|item\s*7|$)",
    re.IGNORECASE | re.DOTALL,
)

# Coarse sentence splitter. Good enough for risk-bullet extraction; no need
# for nltk/spacy here.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# =============================================================================
# PUBLIC API
# =============================================================================

def summarize_filings_by_year(
    ticker: str,
    filings: List[dict],
    max_filings: int = 10,
) -> Dict[str, Any]:
    """Compress a list of SEC filings into a structured, agent-ready summary.

    Parameters
    ----------
    ticker:
        Ticker symbol, used only for logging.
    filings:
        List of filing dicts as returned by ``data/sec_fetcher.py``. Each
        entry is expected to contain at least ``form_type``, ``filing_date``,
        and ``text``; ``html_url`` is preserved when present. Missing fields
        are tolerated.
    max_filings:
        Cap on how many of the most recent filings to process. Keeps runtime
        bounded on tickers with long filing histories.

    Returns
    -------
    dict
        ``{
            "ticker":             str,
            "filings_considered": int,
            "summary":            str,   # short human-readable prose
            "risk_factors":       List[str],
            "management_tone":    str,   # "confident"|"cautious"|"defensive"|"neutral"
            "red_flags":          List[str],
            "by_year":            Dict[str, Dict[str, Any]],
        }``
    """
    safe_filings = list(filings or [])[:max_filings]
    if not safe_filings:
        logger.info("filing_analyzer.summarize | %s | no filings", ticker)
        return _empty_summary(ticker)

    # Bucket by fiscal year (parsed from filing_date YYYY-MM-DD).
    by_year: Dict[str, List[dict]] = defaultdict(list)
    for f in safe_filings:
        year = _extract_year(f.get("filing_date"))
        if year is not None:
            by_year[year].append(f)

    # Pool all text once for tone inference and red-flag scanning. Cheaper
    # than doing it per-filing and tone is inherently a corpus-level signal.
    all_text = " ".join(str(f.get("text") or "") for f in safe_filings)

    risk_factors: List[str] = []
    for f in safe_filings:
        risk_factors.extend(_extract_risk_sentences(str(f.get("text") or "")))
    # Dedup while preserving order; cap length for prompt-budget reasons.
    risk_factors = _dedup_preserve_order(risk_factors)[:20]

    management_tone = _infer_tone(all_text)
    red_flags = _find_red_flags(all_text)

    by_year_out: Dict[str, Dict[str, Any]] = {}
    for year, group in sorted(by_year.items(), reverse=True):
        form_counts = Counter(str(f.get("form_type") or "UNKNOWN") for f in group)
        by_year_out[year] = {
            "filing_count": len(group),
            "form_types": dict(form_counts),
            "latest_date": max(
                (str(f.get("filing_date") or "") for f in group),
                default="",
            ),
            "latest_url": next(
                (f.get("html_url") for f in group if f.get("html_url")),
                None,
            ),
        }

    summary = _build_prose_summary(
        ticker=ticker,
        filings_considered=len(safe_filings),
        by_year=by_year_out,
        tone=management_tone,
        red_flag_count=len(red_flags),
        risk_count=len(risk_factors),
    )

    logger.info(
        "filing_analyzer.summarize | %s | filings=%d | risks=%d | "
        "red_flags=%d | tone=%s",
        ticker,
        len(safe_filings),
        len(risk_factors),
        len(red_flags),
        management_tone,
    )

    return {
        "ticker": ticker,
        "filings_considered": len(safe_filings),
        "summary": summary,
        "risk_factors": risk_factors,
        "management_tone": management_tone,
        "red_flags": red_flags,
        "by_year": by_year_out,
    }


def extract_key_metrics_for_agent(
    ticker: str,
    metrics: Dict[str, Any],
    price: float,
) -> Dict[str, Any]:
    """Normalise raw fundamentals into a compact, agent-ready metric snapshot.

    The input ``metrics`` dict is whatever ``data/market_data.py`` and XBRL
    extraction place in ``pipeline.build_agent_context()["metrics"]``. This
    function tolerates any of the common field aliases (``revenue``,
    ``total_revenue``, etc.) and returns ``None`` for anything it can't derive.

    Parameters
    ----------
    ticker:
        Ticker symbol, used only for logging.
    metrics:
        Flat or nested dict of fundamentals. Expected-ish keys include:
        ``revenue``, ``revenue_history`` (list of {year, value}),
        ``gross_margin``, ``operating_margin``, ``net_margin``,
        ``ebitda``, ``total_debt``, ``cash``, ``shares_outstanding``,
        ``net_income``, ``operating_cash_flow``, ``capex``,
        ``interest_expense``, ``invested_capital``, ``ebit``, ``fcf``,
        ``eps``, ``enterprise_value``.
    price:
        Latest closing price. Used for P/E and market-cap derivations.

    Returns
    -------
    dict
        Snapshot of 10 core fields. Missing inputs yield ``None`` values;
        shape is always stable.
    """
    m = metrics or {}

    revenue_history = _get_revenue_history(m)
    revenue_growth_3y = _cagr_from_history(revenue_history, years=3)
    revenue_growth_5y = _cagr_from_history(revenue_history, years=5)

    margin_trend = _margin_trend(m)

    ebitda = _first_number(m, "ebitda")
    total_debt = _first_number(m, "total_debt", "debt_total", "long_term_debt", "totalDebt")
    cash = _first_number(m, "cash", "cash_and_equivalents", "total_cash", "totalCash")
    net_debt = _safe_sub(total_debt, cash)
    debt_ebitda = _safe_ratio(total_debt, ebitda)
    net_debt_ebitda = _safe_ratio(net_debt, ebitda)

    # --- P/E -----------------------------------------------------------------
    # Prefer pre-computed trailing P/E from the data source (yfinance etc.).
    # Only fall back to price/EPS derivation if the pre-computed value is absent.
    eps = _first_number(m, "eps", "earnings_per_share")
    p_e = _first_number(m, "trailing_pe", "trailingPE", "pe_ratio")
    if p_e is None and price and eps and eps > 0:
        p_e = _safe_ratio(price, eps)
    # Also grab forward P/E for the prompt if available.
    forward_pe = _first_number(m, "forward_pe", "forwardPE")

    # --- EV/EBITDA -----------------------------------------------------------
    # Prefer pre-computed from yfinance; fall back to derivation.
    ev_ebitda = _first_number(m, "ev_to_ebitda", "enterpriseToEbitda")
    if ev_ebitda is None:
        enterprise_value = _first_number(m, "enterprise_value", "ev", "enterpriseValue")
        ev_ebitda = _safe_ratio(enterprise_value, ebitda)
    else:
        enterprise_value = _first_number(m, "enterprise_value", "ev", "enterpriseValue")

    op_cf = _first_number(m, "operating_cash_flow", "cfo", "operatingCashflow")
    net_income = _first_number(m, "net_income", "earnings", "netIncomeToCommon")
    cash_conversion = _safe_ratio(op_cf, net_income)

    capex = _first_number(m, "capex", "capital_expenditures")
    fcf = _first_number(m, "fcf", "free_cash_flow", "freeCashflow")
    if fcf is None and op_cf is not None and capex is not None:
        # capex is typically reported negative in cash-flow statements; handle
        # both conventions so we don't double-subtract.
        fcf = op_cf - abs(capex)

    shares = _first_number(m, "shares_outstanding", "shares", "sharesOutstanding")
    market_cap = _first_number(m, "market_cap", "marketCap")
    if market_cap is None and price and shares:
        market_cap = price * shares
    fcf_yield = _safe_ratio(fcf, market_cap)

    ebit = _first_number(m, "ebit", "operating_income")
    interest_expense = _first_number(m, "interest_expense", "interestExpense")
    interest_coverage = _safe_ratio(ebit, interest_expense)

    invested_capital = _first_number(m, "invested_capital")
    # ROIC using NOPAT proxy = EBIT * (1 - 0.21 assumed tax). Coarse; fine
    # for screening. Tax rate is overridable later via config if needed.
    # Also accept yfinance returnOnEquity as a fallback signal.
    nopat = ebit * (1 - 0.21) if ebit is not None else None
    roic = _safe_ratio(nopat, invested_capital)
    if roic is None:
        roic = _first_number(m, "return_on_equity", "returnOnEquity")

    # Streak of negative FCF years, capped by available history.
    fcf_history = _get_numeric_history(m, "fcf_history", "free_cash_flow_history")
    cash_flow_negative_3_years = _is_negative_streak(fcf_history, min_streak=3)

    # Pass-through fields: standard fundamentals that come in from the
    # market-data layer. Not recomputed here, just surfaced in the snapshot
    # so downstream consumers (risk scanner, thesis generator, tests) can
    # read them under their industry-standard names.
    # yfinance returns these as 'revenue_growth' / 'grossMargins' / etc.
    revenue_growth_yoy = _first_number(m, "revenue_growth_yoy", "revenue_growth", "revenueGrowth")
    gross_margin = _first_number(m, "gross_margin", "grossMargins", "gross_margins")
    operating_margin = _first_number(m, "operating_margin", "operatingMargins", "operating_margins")
    profit_margin = _first_number(m, "profit_margin", "profitMargins", "net_margin")
    # ``fcf`` above already prefers explicit input (``fcf`` / ``free_cash_flow``)
    # over the op_cf − capex derivation, so we can surface it directly.
    free_cash_flow_out = fcf

    snapshot: Dict[str, Any] = {
        "ticker": ticker,
        "price": _safe_float(price),
        "revenue_growth_3y": revenue_growth_3y,
        "revenue_growth_5y": revenue_growth_5y,
        "revenue_growth_yoy": revenue_growth_yoy,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "margin_trend": margin_trend,        # "expanding" | "flat" | "compressing" | None
        "free_cash_flow": free_cash_flow_out,
        "debt_ebitda": debt_ebitda,
        "debt_to_ebitda": debt_ebitda,       # alias, industry-standard spelling
        "net_debt_ebitda": net_debt_ebitda,
        "net_debt_to_ebitda": net_debt_ebitda,  # alias
        "p_e": p_e,
        "pe_ratio": p_e,                     # alias
        "forward_pe": forward_pe,
        "ev_ebitda": ev_ebitda,
        "ev_to_ebitda": ev_ebitda,           # alias
        "cash_conversion": cash_conversion,  # OCF / NI
        "fcf_yield": fcf_yield,              # FCF / market cap
        "roic": roic,
        "interest_coverage": interest_coverage,
        "cash_flow_negative_3_years": cash_flow_negative_3_years,
        "market_cap": market_cap,
    }

    logger.info(
        "filing_analyzer.metrics | %s | rev_g_3y=%s | debt_ebitda=%s | "
        "p_e=%s | fcf_yield=%s",
        ticker,
        _fmt(revenue_growth_3y),
        _fmt(debt_ebitda),
        _fmt(p_e),
        _fmt(fcf_yield),
    )

    return snapshot


# =============================================================================
# INTERNAL: FILING HELPERS
# =============================================================================

def _empty_summary(ticker: str) -> Dict[str, Any]:
    """Stable empty shape for tickers with no filings."""
    return {
        "ticker": ticker,
        "filings_considered": 0,
        "summary": f"No filings available for {ticker}.",
        "risk_factors": [],
        "management_tone": "neutral",
        "red_flags": [],
        "by_year": {},
    }


def _extract_year(date_str: Optional[str]) -> Optional[str]:
    """Return the 4-digit year from 'YYYY-MM-DD' or similar, or None."""
    if not date_str:
        return None
    m = re.match(r"(\d{4})", str(date_str))
    return m.group(1) if m else None


def _extract_risk_sentences(text: str, limit: int = 20) -> List[str]:
    """Pull sentences from the Item 1A risk-factors section of a 10-K.

    Falls back to scanning the whole document for sentences containing risk
    cue words if no Item 1A header is found (common for 10-Q / 8-K filings,
    which don't have a formal risk-factors section).
    """
    if not text:
        return []

    match = _RISK_SECTION_RE.search(text)
    region = match.group(1) if match else text

    sentences = _SENT_SPLIT_RE.split(region)
    picked: List[str] = []
    cue_words = ("risk", "could adversely", "may adversely", "uncertain",
                 "depend on", "subject to")

    for s in sentences:
        s_clean = re.sub(r"\s+", " ", s).strip()
        if not (40 <= len(s_clean) <= 400):
            continue
        lower = s_clean.lower()
        if any(cue in lower for cue in cue_words):
            picked.append(s_clean)
            if len(picked) >= limit:
                break

    return picked


def _infer_tone(text: str) -> str:
    """Classify overall tone by lexicon frequency.

    Returns one of: ``"confident"``, ``"cautious"``, ``"defensive"``,
    ``"neutral"``. "defensive" wins when any defensive/legal term appears,
    since those signals dominate the others.
    """
    if not text:
        return "neutral"

    lower = text.lower()
    # Defensive terms trump everything — presence of a going-concern or
    # restatement phrase is not balanced out by enthusiastic marketing copy.
    for term in _DEFENSIVE_TERMS:
        if term in lower:
            return "defensive"

    confident = sum(lower.count(t) for t in _CONFIDENT_TERMS)
    cautious = sum(lower.count(t) for t in _CAUTIOUS_TERMS)

    if confident == 0 and cautious == 0:
        return "neutral"
    if confident >= 2 * cautious:
        return "confident"
    if cautious >= 2 * confident:
        return "cautious"
    return "neutral"


def _find_red_flags(text: str) -> List[str]:
    """Return distinct red-flag phrases, each with a short context snippet."""
    if not text:
        return []
    hits: List[str] = []
    seen: set = set()
    for pat in _RED_FLAG_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        key = pat.pattern
        if key in seen:
            continue
        seen.add(key)
        # ~80-char context window for human review.
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        hits.append(snippet)
    return hits


def _dedup_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x.strip())
    return out


def _build_prose_summary(
    ticker: str,
    filings_considered: int,
    by_year: Dict[str, Dict[str, Any]],
    tone: str,
    red_flag_count: int,
    risk_count: int,
) -> str:
    """One-paragraph summary suitable for inclusion in an agent prompt."""
    if not by_year:
        return f"{ticker}: {filings_considered} filings processed; no year breakdown available."

    years_sorted = sorted(by_year.keys(), reverse=True)
    span = f"{years_sorted[-1]}–{years_sorted[0]}" if len(years_sorted) > 1 else years_sorted[0]
    form_totals: Counter = Counter()
    for y in years_sorted:
        form_totals.update(by_year[y]["form_types"])
    forms_str = ", ".join(f"{k}:{v}" for k, v in form_totals.most_common())

    red_flag_clause = (
        f" {red_flag_count} red-flag phrase(s) detected."
        if red_flag_count else
        " No red-flag phrases detected."
    )
    return (
        f"{ticker}: processed {filings_considered} filings ({span}) — {forms_str}. "
        f"Management tone reads as {tone}. "
        f"Extracted {risk_count} risk-factor sentence(s).{red_flag_clause}"
    )


# =============================================================================
# INTERNAL: METRIC HELPERS
# =============================================================================

def _first_number(m: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return first numeric value found under any of ``keys``; else None."""
    for k in keys:
        if k in m and m[k] is not None:
            v = _safe_float(m[k])
            if v is not None:
                return v
    return None


def _safe_float(x: Any) -> Optional[float]:
    """Coerce to float, returning None on failure or NaN."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    # Guard against NaN sneaking in from pandas / numpy.
    if v != v:  # NaN check without importing math
        return None
    return v


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """num / den with None propagation and zero-denominator protection."""
    if num is None or den is None:
        return None
    if den == 0:
        return None
    return num / den


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """a - b with None propagation."""
    if a is None or b is None:
        return None
    return a - b


def _get_revenue_history(m: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Extract a sorted list of (year, revenue) tuples from varied shapes.

    Accepts:
        - m["revenue_history"] = [{"year": 2023, "value": 100.0}, ...]
        - m["revenue_history"] = [(2023, 100.0), ...]
        - m["revenue_by_year"]  = {"2023": 100.0, ...}
    Returns oldest-first.
    """
    hist = m.get("revenue_history") or m.get("revenue_by_year")
    if not hist:
        return []

    pairs: List[Tuple[int, float]] = []

    if isinstance(hist, dict):
        for y, v in hist.items():
            y_int = _safe_int(y)
            v_f = _safe_float(v)
            if y_int is not None and v_f is not None:
                pairs.append((y_int, v_f))
    elif isinstance(hist, list):
        for item in hist:
            if isinstance(item, dict):
                y_int = _safe_int(item.get("year") or item.get("fy"))
                v_f = _safe_float(item.get("value") or item.get("revenue"))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                y_int = _safe_int(item[0])
                v_f = _safe_float(item[1])
            else:
                continue
            if y_int is not None and v_f is not None:
                pairs.append((y_int, v_f))

    pairs.sort(key=lambda t: t[0])
    return pairs


def _get_numeric_history(m: Dict[str, Any], *keys: str) -> List[float]:
    """Return a flat list of floats from any history-like field."""
    for k in keys:
        raw = m.get(k)
        if not raw:
            continue
        out: List[float] = []
        if isinstance(raw, dict):
            # Sort by year ascending so "last 3" means last 3 years.
            for _, v in sorted(raw.items(), key=lambda kv: str(kv[0])):
                vf = _safe_float(v)
                if vf is not None:
                    out.append(vf)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    vf = _safe_float(item.get("value") or item.get("fcf"))
                else:
                    vf = _safe_float(item)
                if vf is not None:
                    out.append(vf)
        if out:
            return out
    return []


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(str(x)[:4]) if x is not None else None
    except (TypeError, ValueError):
        return None


def _cagr_from_history(
    history: List[Tuple[int, float]],
    years: int,
) -> Optional[float]:
    """Compute CAGR over the last ``years`` points; None if insufficient.

    Returns a decimal (e.g. 0.08 for 8%).
    """
    if len(history) < years + 1:
        return None
    start_val = history[-(years + 1)][1]
    end_val = history[-1][1]
    if start_val is None or start_val <= 0 or end_val is None or end_val <= 0:
        return None
    try:
        return (end_val / start_val) ** (1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError):
        return None


def _margin_trend(m: Dict[str, Any]) -> Optional[str]:
    """Classify operating-margin trajectory as expanding/flat/compressing.

    Prefers an explicit ``operating_margin_history`` list; falls back to
    comparing current operating margin with a prior-period value if given.
    """
    hist = _get_numeric_history(m, "operating_margin_history", "op_margin_history")
    if len(hist) >= 2:
        change = hist[-1] - hist[0]
        if change > 0.02:
            return "expanding"
        if change < -0.02:
            return "compressing"
        return "flat"

    cur = _first_number(m, "operating_margin")
    prev = _first_number(m, "operating_margin_prior", "operating_margin_prev")
    if cur is None or prev is None:
        return None
    diff = cur - prev
    if diff > 0.02:
        return "expanding"
    if diff < -0.02:
        return "compressing"
    return "flat"


def _is_negative_streak(values: List[float], min_streak: int) -> bool:
    """True if the last ``min_streak`` values are all negative."""
    if len(values) < min_streak:
        return False
    return all(v < 0 for v in values[-min_streak:])


def _fmt(x: Optional[float]) -> str:
    """Compact float formatter for log lines."""
    if x is None:
        return "None"
    return f"{x:.3f}"