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

# Severe signals: rare, and genuinely indicative of trouble. Presence of
# any one of these legitimately flips tone to "defensive" — a going-concern
# or restatement phrase is not balanced out by upbeat language elsewhere.
_SEVERE_TERMS: frozenset = frozenset({
    "restatement", "material weakness", "going concern",
    "non-reliance", "adverse opinion", "sec inquiry", "subpoena",
})

# Routine legal/accounting terms that appear in essentially EVERY large-cap
# 10-K as standard disclosure (a company this size always has some
# litigation, some investigation, some impairment somewhere in 1.5M chars
# of filing text). On a full-filing corpus these are NOT a defensive signal
# on mere presence — treating them as such made tone=defensive nearly
# deterministic for every real filing. They contribute only as cautious
# signal when they appear at meaningful frequency (handled in _infer_tone).
_ROUTINE_NEGATIVE_TERMS: frozenset = frozenset({
    "litigation", "investigation", "allegation", "impairment",
})

# Kept for backwards compatibility (some callers/tests import this name).
_DEFENSIVE_TERMS: frozenset = _SEVERE_TERMS

# Phrases that warrant a standalone "red flag" entry in the output.
#
# Split into two tiers because the severe conditions (going concern,
# material weakness, restatement, non-reliance, adverse opinion, SEC
# enforcement) appear in DEFINITIONAL and HYPOTHETICAL form in essentially
# every 10-K ("a material weakness exists when ..."; "if we were to
# restate ..."). Matching them as bare substrings flagged healthy filings.
# For those we reuse _AFFIRMATIVE_SEVERE_PATTERNS (declaration-context
# only). The patterns below are already specific enough that a bare match
# is a genuine signal — they don't have a common definitional form.
_RED_FLAG_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"auditor\s+resign", re.IGNORECASE),
    re.compile(r"delist(?:ing)?", re.IGNORECASE),
    re.compile(r"covenant\s+(?:breach|violation|waiver)", re.IGNORECASE),
    re.compile(r"impair(?:ment|ed)\s+(?:goodwill|assets)", re.IGNORECASE),
)

# Locating the Item 1A risk-factors section is format-dependent: real
# filings write the header inconsistently — "Item 1A. Risk Factors",
# "Item 1A.Risk Factors" (no space), "Item 1ARisk Factors" (no space, no
# period) — and some (e.g. MSFT) have no usable header adjacent to the
# actual prose at all. We therefore use a HYBRID strategy (see
# _extract_risk_sentences): try to isolate a valid Item 1A section, and if
# that fails, fall back to a selective whole-document cue scan.

# Header: "Item 1A" then optional punctuation/space then "Risk Factors",
# tolerating missing spaces. \W* allows ".", "", "-", whitespace between.
_RISK_HEADER_RE = re.compile(r"item\s*1a\W{0,3}risk\s*factors", re.IGNORECASE)

# Section END boundary: the next genuine item header that reliably follows
# Item 1A. We use 1B / 1C / 2 only — Item 7 / 8 appear as stray
# cross-references ("refer to Item 7") that truncate the section early.
_RISK_BOUNDARY_RE = re.compile(r"item\s*(?:1b|1c|2)\b", re.IGNORECASE)

# A captured region must be at least this large to be the real section
# rather than a table-of-contents line ("Item 1A. Risk Factors 16").
_MIN_RISK_REGION_CHARS = 1200

# Cue phrases that mark a sentence as an actual risk factor (forward-looking
# adverse-consequence language). Deliberately stricter than the bare word
# "risk" so the whole-doc fallback doesn't grab cover-page boilerplate
# ("indicate by check mark ...") or cross-references.
_RISK_CUE_RE = re.compile(
    r"(?:could|may|might|would)\s+(?:adversely|materially|negatively|"
    r"seriously)?\s*(?:affect|harm|impact|reduce|impair|result|"
    r"disrupt|damage)|adversely\s+affect|subject\s+to\s+(?:various|"
    r"numerous|significant|certain)?\s*risks|depend(?:s|ent)?\s+(?:on|"
    r"upon)|materially\s+adversely",
    re.IGNORECASE,
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

    # Bucket by fiscal year.
    # sec_fetcher.get_filings() writes "filed_date" and "filing_type";
    # older callers and manual test fixtures may use "filing_date" and
    # "form_type". Accept both so neither side needs to change.
    def _get_date(f: dict) -> Optional[str]:
        return f.get("filing_date") or f.get("filed_date")

    def _get_form(f: dict) -> str:
        return str(f.get("form_type") or f.get("filing_type") or "UNKNOWN")

    by_year: Dict[str, List[dict]] = defaultdict(list)
    for f in safe_filings:
        year = _extract_year(_get_date(f))
        if year is not None:
            by_year[year].append(f)

    # Pool all text once for tone inference and red-flag scanning. Cheaper
    # than doing it per-filing and tone is inherently a corpus-level signal.
    all_text = " ".join(str(f.get("text") or "") for f in safe_filings)

    risk_factors: List[str] = []
    for f in safe_filings:
        # Pass a high limit so the TRUE count survives (the default limit of
        # 20 would pre-cap each filing before we can measure real magnitude).
        risk_factors.extend(
            _extract_risk_sentences(str(f.get("text") or ""), limit=200)
        )
    # Dedup while preserving order. Keep the TRUE count (uncapped) for
    # scoring — the displayed sentence list is capped at 20 for prompt
    # budget, but every large-cap 10-K saturates that cap, which destroys
    # the variance any count-based signal needs. The true count is stored
    # separately as ``risk_factor_count`` so a future sector-relative
    # penalty can compare real magnitudes across tickers.
    risk_factors = _dedup_preserve_order(risk_factors)
    risk_factor_count = len(risk_factors)
    risk_factors = risk_factors[:20]

    management_tone = _infer_tone(all_text)
    red_flags = _find_red_flags(all_text)

    by_year_out: Dict[str, Dict[str, Any]] = {}
    for year, group in sorted(by_year.items(), reverse=True):
        form_counts = Counter(_get_form(f) for f in group)
        by_year_out[year] = {
            "filing_count": len(group),
            "form_types": dict(form_counts),
            "latest_date": max(
                (_get_date(f) or "" for f in group),
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
        "filing_analyzer.summarize | %s | filings=%d | risks=%d "
        "(shown %d) | red_flags=%d | tone=%s",
        ticker,
        len(safe_filings),
        risk_factor_count,
        len(risk_factors),
        len(red_flags),
        management_tone,
    )

    return {
        "ticker": ticker,
        "filings_considered": len(safe_filings),
        "summary": summary,
        "risk_factors": risk_factors,
        "risk_factor_count": risk_factor_count,
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

    # --- Company identity & extra valuation multiples -------------------
    # The essay prompt needs the company NAME (otherwise it falls back to
    # the literal "Unknown Company"), and the review kept *inventing*
    # price-to-book / price-to-sales because they weren't supplied. Pull
    # all of them from the input metrics dict, tolerating both the flat
    # shape (build_agent_context output) and the nested shape that
    # market_data.get_fundamentals() returns (name/sector at root,
    # multiples under a "valuation" sub-dict).
    _val = m.get("valuation") if isinstance(m.get("valuation"), dict) else {}

    def _pick_str(*keys: str) -> Optional[str]:
        for src in (m, _val):
            for k in keys:
                v = src.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    def _pick_num(*keys: str) -> Optional[float]:
        # Top-level first, then nested "valuation".
        v = _first_number(m, *keys)
        if v is not None:
            return v
        return _first_number(_val, *keys)

    company_name = _pick_str("name", "shortName", "longName", "company_name")
    sector = _pick_str("sector")
    industry = _pick_str("industry")
    price_to_book = _pick_num("price_to_book", "priceToBook", "pb_ratio")
    price_to_sales = _pick_num(
        "price_to_sales", "priceToSalesTrailing12Months", "ps_ratio"
    )

    snapshot: Dict[str, Any] = {
        "ticker": ticker,
        "name": company_name,                # company identity for the essay header
        "sector": sector,
        "industry": industry,
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
        "price_to_book": price_to_book,      # supplied so the review stops inventing it
        "price_to_sales": price_to_sales,    # supplied so the review stops inventing it
        "cash_conversion": cash_conversion,  # OCF / NI
        "fcf_yield": fcf_yield,              # FCF / market cap
        "roic": roic,
        "interest_coverage": interest_coverage,
        "cash_flow_negative_3_years": cash_flow_negative_3_years,
        "market_cap": market_cap,
        "profit_margin": profit_margin,         # net profit margin (was computed but missing)
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
        "risk_factor_count": 0,
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


def _select_risk_region(text: str) -> Optional[str]:
    """Isolate the real Item 1A section, or return None if not locatable.

    For each "Item 1A ... Risk Factors" header occurrence, slice from the
    header to the next 1B/1C/2 boundary and keep the largest such slice.
    Returns the region only if it clears ``_MIN_RISK_REGION_CHARS`` AND
    contains real risk-cue language (guards against a table-of-contents
    slice). Returns None when no valid section is found, signalling the
    caller to fall back to a whole-document scan.
    """
    best = ""
    for h in _RISK_HEADER_RE.finditer(text):
        start = h.end()
        nb = _RISK_BOUNDARY_RE.search(text, start)
        end = nb.start() if nb else len(text)
        region = text[start:end]
        if len(region) > len(best):
            best = region
    if len(best) >= _MIN_RISK_REGION_CHARS and _RISK_CUE_RE.search(best):
        return best
    return None


def _scan_for_risk_sentences(region: str, limit: int) -> List[str]:
    """Pull risk-factor-like sentences from a text region.

    Splits on sentence punctuation and newlines (filing text often uses
    line breaks instead of ". "), then keeps reasonable-length sentences
    matching the strict ``_RISK_CUE_RE`` (forward-looking adverse
    consequences), de-duplicated in order.
    """
    pieces: List[str] = []
    for chunk in _SENT_SPLIT_RE.split(region):
        pieces.extend(chunk.split("\n"))

    picked: List[str] = []
    seen: set = set()
    for s in pieces:
        s_clean = re.sub(r"\s+", " ", s).strip()
        if not (40 <= len(s_clean) <= 400):
            continue
        if not _RISK_CUE_RE.search(s_clean):
            continue
        key = s_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(s_clean)
        if len(picked) >= limit:
            break
    return picked


def _extract_risk_sentences(text: str, limit: int = 20) -> List[str]:
    """Pull risk-factor sentences from a filing — hybrid strategy.

    1. Try to isolate a valid Item 1A section (``_select_risk_region``),
       handling the header-spacing variants seen in real filings
       ("Item 1A. Risk Factors", "Item 1A.Risk Factors", "Item 1ARisk
       Factors").
    2. If no usable section is found (e.g. a filing whose risk prose has no
       adjacent Item 1A header, or a 10-Q/8-K with no formal section), fall
       back to scanning the WHOLE document.

    Either way, only sentences matching the strict risk-cue pattern are
    kept, so the whole-doc fallback does not pick up cover-page boilerplate
    or cross-references.
    """
    if not text:
        return []
    region = _select_risk_region(text)
    if region is not None:
        hits = _scan_for_risk_sentences(region, limit)
        if hits:
            return hits
    # Fallback: scan the entire document with the same strict cue filter.
    return _scan_for_risk_sentences(text, limit)


def _has_affirmative_severe_signal(lower: str) -> bool:
    """True only when a severe condition is AFFIRMATIVELY DECLARED.

    Distinguishes "we identified a material weakness" (real) from "if we
    identify a material weakness ... investors could lose confidence"
    (hypothetical risk-factor boilerplate present in nearly every 10-K).

    Strategy: for each severe condition, require declaration phrasing —
    a first-person/possessive or past-tense verb asserting the condition
    actually occurred — rather than the bare term, which appears in
    hypothetical form in healthy filings.
    """
    return any(p.search(lower) for p in _AFFIRMATIVE_SEVERE_PATTERNS)


# Patterns that indicate a severe condition is actually present, not merely
# described as a hypothetical risk. Tuned to avoid the "if/could/would/may"
# hedged constructions that dominate risk-factor sections.
_AFFIRMATIVE_SEVERE_PATTERNS: Tuple[re.Pattern, ...] = (
    # AFFIRMATIVE material weakness: the company declaring one was found.
    # Must NOT match (a) the standard definition every 10-K includes
    # ("a material weakness is a deficiency ..."; "a material weakness
    # exists when ..."), nor (b) negated conclusions ("no material
    # weakness was identified"; "did not identify any material weakness").
    # We therefore require a first-person/possessive subject paired with an
    # affirmative finding verb, and forbid an immediately-preceding "no/not/
    # any/without" within the short window.
    re.compile(
        r"(?:we|management|the\s+company)\s+"
        r"(?:identified|concluded\s+(?:that\s+)?(?:there\s+(?:was|is|were)\s+)?"
        r"a?|determined\s+(?:that\s+)?(?:there\s+(?:was|is|were)\s+)?a?|"
        r"reported)\b"
        r"(?:(?!\b(?:no|not|any|without|effective|did\s+not)\b)[^.]){0,50}?"
        r"material\s+weakness",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:identified|reported|disclosed)\s+(?:a|one\s+or\s+more)\s+"
        r"material\s+weakness",
        re.IGNORECASE,
    ),
    # Actual restatement (past tense / completed action), not "a restatement
    # could be required". Only the past-tense verb forms are unambiguous;
    # "restatement of our financial statements" appears in hypothetical
    # risk-factor language too, so we do NOT match the noun phrase.
    re.compile(
        r"(?:we|the\s+company)\s+(?:have\s+)?restated\b",
        re.IGNORECASE,
    ),
    # Going concern actually raised, not hypothetical.
    re.compile(
        r"substantial\s+doubt\b[^.]{0,40}going\s+concern",
        re.IGNORECASE,
    ),
    # Non-reliance / Item 4.02 (companies file this only when restating).
    re.compile(r"non-reliance", re.IGNORECASE),
    # Adverse audit opinion actually expressed.
    re.compile(
        r"(?:expressed|issued|rendered)\b[^.]{0,40}adverse\s+opinion",
        re.IGNORECASE,
    ),
    # Active SEC enforcement actually disclosed (not "if the SEC were to...").
    re.compile(
        r"(?:received|are\s+subject\s+to|is\s+subject\s+to|served\s+with)"
        r"\b[^.]{0,40}(?:subpoena|sec\s+(?:inquiry|investigation))",
        re.IGNORECASE,
    ),
)


def _infer_tone(text: str) -> str:
    """Classify overall tone by lexicon frequency, length-normalized.

    Returns one of ``"confident"``, ``"cautious"``, ``"defensive"``,
    ``"neutral"``.

    Design notes (important on full-filing text):
      * Only SEVERE terms (going concern, restatement, material weakness,
        ...) flip tone to "defensive" on presence. They are rare and real.
      * Routine legal/accounting terms (litigation, impairment, ...) appear
        in every large 10-K, so they are NOT defensive triggers. They add
        to the cautious side, but only frequency above a small floor counts
        — one stray "litigation" in 1.5M chars is noise.
      * Confident vs cautious is decided on RATE per 100k chars with a
        margin requirement, so a long filing doesn't trivially win one side
        just by having more words.
    """
    if not text:
        return "neutral"

    lower = text.lower()

    # Severe signals trump everything — BUT only when AFFIRMATIVELY DECLARED,
    # not merely described as a hypothetical risk. Every large-cap 10-K's
    # risk-factor section mentions "material weakness", "restatement", etc.
    # in the abstract ("if we were to identify a material weakness, investors
    # could lose confidence"). Those hypothetical mentions are not a defensive
    # signal — only an actual declaration is ("we identified a material
    # weakness", "we restated our financial statements"). Matching bare
    # substrings made every healthy filing read as defensive.
    if _has_affirmative_severe_signal(lower):
        return "defensive"

    n = len(lower)
    if n == 0:
        return "neutral"
    per100k = 100_000.0 / n  # scale raw counts to a per-100k-char rate

    confident = sum(lower.count(t) for t in _CONFIDENT_TERMS) * per100k
    cautious = sum(lower.count(t) for t in _CAUTIOUS_TERMS) * per100k
    # Routine negatives count toward caution, but only the portion above a
    # small floor (they're partly boilerplate). Half-weighted.
    routine = sum(lower.count(t) for t in _ROUTINE_NEGATIVE_TERMS) * per100k
    cautious += 0.5 * max(0.0, routine - 2.0)

    # Require a meaningful signal and a clear margin before leaving neutral.
    if confident < 1.0 and cautious < 1.0:
        return "neutral"
    if confident >= 1.5 * cautious and confident >= 1.0:
        return "confident"
    if cautious >= 1.5 * confident and cautious >= 1.0:
        return "cautious"
    return "neutral"


def _find_red_flags(text: str) -> List[str]:
    """Return distinct red-flag phrases, each with a short context snippet.

    Two sources:
      * ``_AFFIRMATIVE_SEVERE_PATTERNS`` — severe conditions, but only when
        affirmatively declared (not the definitional/hypothetical mentions
        that appear in every 10-K). Shared with tone classification so the
        two stay consistent.
      * ``_RED_FLAG_PATTERNS`` — already-specific phrases (auditor
        resignation, delisting, covenant breach, goodwill/asset impairment)
        with no common definitional form, so a bare match is a real signal.
    """
    if not text:
        return []
    hits: List[str] = []
    seen: set = set()
    flagged_spans: List[tuple] = []  # (start, end) of matched cores
    for pat in (*_AFFIRMATIVE_SEVERE_PATTERNS, *_RED_FLAG_PATTERNS):
        m = pat.search(text)
        if not m:
            continue
        key = pat.pattern
        if key in seen:
            continue
        # Positional dedup: if this match overlaps a span already
        # flagged, it is the SAME underlying event picked up by a second
        # pattern (e.g. "material weakness" and "controls not effective"
        # in one sentence). Count the event once.
        ms, me = m.start(), m.end()
        if any(ms < fe and me > fs for fs, fe in flagged_spans):
            continue
        seen.add(key)
        flagged_spans.append((ms, me))
        # ~80-char context window for human review.
        start = max(0, ms - 40)
        end = min(len(text), me + 40)
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