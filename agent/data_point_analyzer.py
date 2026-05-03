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
    "macro.interest_rates.fed_funds": ("Fed Funds Rate", "Macro / Rates", _fmt_pct_raw),
    "macro.interest_rates.treasury_10y": ("10-Year Treasury Yield", "Macro / Rates", _fmt_pct_raw),
    "macro.interest_rates.treasury_2y": ("2-Year Treasury Yield", "Macro / Rates", _fmt_pct_raw),
    "macro.interest_rates.yield_spread_10y2y": ("Yield Curve (10Y-2Y)", "Macro / Rates", _fmt_pct_raw),

    # --- Macro: Inflation & Growth ---
    "macro.inflation.cpi_yoy_pct": ("CPI YoY", "Macro / Inflation", _fmt_pct_raw),
    "macro.employment.unemployment_rate": ("Unemployment Rate", "Macro / Employment", _fmt_pct_raw),
    "macro.growth.gdp_growth_annualized": ("GDP Growth (Annualized)", "Macro / Growth", _fmt_pct_raw),

    # --- Macro: Conditions ---
    "macro.financial_conditions.vix": ("VIX (Volatility Index)", "Macro / Conditions", _fmt_num),
    # FRED RECPROUSM156N is reported in percent units already (e.g. 23.5
    # means 23.5%) — use _fmt_pct_raw, NOT _fmt_pct (which would multiply
    # by 100 and produce absurd values like "+2350.00%").
    "macro.recession_signals.recession_probability": (
        "Recession Probability (12M)", "Macro / Conditions", _fmt_pct_raw,
    ),
}


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

    Examples
    --------
    >>> get_data_point_value(ctx, "prices.last")
    215.43
    >>> get_data_point_value(ctx, "macro.interest_rates.fed_funds")
    5.33
    """
    parts = key.split(".")
    cursor: Any = context
    for p in parts:
        if isinstance(cursor, dict):
            cursor = cursor.get(p)
        else:
            return None
        if cursor is None:
            return None
    return cursor


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
        text = _call_ollama_text(
            prompt, model_used, config,
            temperature=_PARAGRAPH_TEMPERATURE,
            system=_SYSTEM_PROMPT,
        )
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

# System instruction sent separately from the user prompt.
# phi3:3.8b follows system-level format rules far more reliably than
# instructions buried inside a long user prompt.
_SYSTEM_PROMPT = (
    "You are a hedge fund equity analyst. "
    "You write structured research notes. "
    "You ALWAYS use the exact section headers given to you, each on its own line followed by a colon. "
    "You NEVER write continuous prose without section headers. "
    "You NEVER skip a section. "
    "You NEVER add sections that were not requested."
)


def _build_prompt(
    ticker: str,
    sector: str,
    industry: str,
    as_of: str,
    selected_keys: List[str],
    context: Dict[str, Any],
) -> str:
    # Build the data block
    selected_lines: List[str] = []
    for k in selected_keys:
        display = get_display_name(k)
        value = get_formatted_value(context, k)
        selected_lines.append(f"  {display}: {value}")
    data_block = "\n".join(selected_lines)

    # Build the required sections list (just the names, no template filler)
    section_names = ["Overview"] + [get_display_name(k) for k in selected_keys]
    sections_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(section_names))

    # Few-shot example using a DIFFERENT ticker so phi3 doesn't copy
    # the values verbatim instead of using the real data block.
    example_output = (
        "Overview:\n"
        "AAPL trades at a slight premium to peers but strong FCF and brand moat justify "
        "the multiple. The 28.5x P/E is above the sector median but covered by 12% "
        "earnings growth. Stance: HOLD with bullish bias.\n\n"
        "Last Price:\n"
        "At $171.21, AAPL sits 4% below its 52-week high of $178.19. "
        "The recent pullback appears technical rather than fundamental. "
        "This level supports buying.\n\n"
        "Trailing P/E:\n"
        "A 28.5x trailing multiple is elevated versus the S&P 500 median of 18x but "
        "is in line with mega-cap tech peers. Given 12% EPS growth, the PEG ratio of "
        "2.4x is acceptable. Neutral at current levels."
    )

    return f"""Data for {ticker} ({sector} / {industry}) as of {as_of}:
{data_block}

Required sections ({len(section_names)} total):
{sections_list}

Here is an example of the EXACT output format you must produce (this example uses AAPL, not {ticker}):

{example_output}

Now write the same format for {ticker}. Use ONLY the section headers listed above. Each header must appear on its own line followed by a colon. Write 3-5 sentences per section. Use the {ticker} data provided above — do not use the AAPL example values."""


# =============================================================================
# OUTPUT PARSING
# =============================================================================

def _normalize_header(line: str) -> Optional[str]:
    """Normalize a candidate header line to its lowercase core text.

    Accepts plain, colon-terminated, markdown (#/##/###), bold (**),
    and any combination. Returns None for lines that are clearly prose
    (too long, sentence punctuation, embedded values with $ or %).
    """
    if not line:
        return None
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None

    # Iteratively strip markdown markers, bold wrappers, and trailing
    # colons until stable — handles interleaved orderings like
    # "**Overview**:", "## **Trailing P/E:**", etc.
    core = stripped
    while True:
        before = core
        core = re.sub(r"^#{1,6}\s+", "", core)
        core = re.sub(r"^\*\*(.+?)\*\*$", r"\1", core)
        core = re.sub(r"^\*(.+?)\*$",     r"\1", core)
        core = re.sub(r"^__(.+?)__$",     r"\1", core)
        core = core.strip()
        if core.endswith(":"):
            core = core[:-1].strip()
        if core == before:
            break

    if not core:
        return None

    # Reject prose lines: sentence punctuation or embedded value chars.
    if any(ch in core for ch in (".", ",", ";", "$", "%")):
        return None

    return core.lower()


def _split_text(text: str, selected_keys: List[str]) -> Tuple[str, Dict[str, str]]:
    """Split the LLM output into overview + per-point paragraphs.

    Accepts headers in any format recognised by _normalize_header:
    plain, colon-terminated, markdown (#/##/###), bold, or combined.
    """
    header_to_key: Dict[str, str] = {}
    for k in selected_keys:
        display = get_display_name(k)
        header_to_key[display.lower().strip()] = k

    overview = ""
    paragraphs: Dict[str, str] = {}
    current_header: Optional[str] = None
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

        normalized = _normalize_header(stripped)
        if normalized is not None:
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
    system: str = "",
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
    if system:
        body["system"] = system

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    return (payload.get("response") or "").strip()