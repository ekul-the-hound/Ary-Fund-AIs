"""
agent/thesis_essay.py
=====================

LLM-powered institutional-memo essay generator.

This module is **additive** to ``agent/thesis_generator.py``. The existing
heuristic still produces the structured verdict (outlook, direction,
confidence, risks, opportunities, bias scores). This module takes that
structured verdict, plus the raw filings / metrics / macro context, and
asks the agent LLM (via ``base_agent.ask_agent``) to write a 2+ page
analytical essay that *interprets* the data rather than just summarising
it.

Design contract
---------------
- Never regresses the heuristic. If the LLM is unavailable, in mock mode,
  or fails for any reason, ``generate_thesis_essay`` returns a deterministic
  fallback essay assembled from the heuristic numbers — so the pipeline
  always produces usable prose.
- Stays within ``config.MAX_TOKENS``. The prompt is compact and the output
  is free-form text (not JSON), so we bypass ``format="json"`` by using
  a dedicated helper that calls Ollama for text completion directly.
- Pure function. No DB writes, no side effects beyond logging.

Usage
-----
::

    from agent.thesis_generator import generate_thesis
    from agent.thesis_essay import generate_thesis_essay

    thesis = generate_thesis(ticker, filings_summary, metrics, macro, risk_flags)
    essay  = generate_thesis_essay(
        ticker=ticker,
        thesis=thesis,
        filings_summary=filings_summary,
        metrics=metrics,
        macro=macro,
        risk_flags=risk_flags,
        config=config,
    )
    thesis["essay"] = essay["text"]
    thesis["essay_meta"] = {
        "model": essay["model_used"],
        "elapsed_ms": essay["elapsed_ms"],
        "fallback": essay["fallback"],
        "word_count": essay["word_count"],
    }
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.base_agent import AgentRequest, _estimate_tokens, _resolve_model


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Target essay length. 2 pages of 12pt prose ~ 800-1000 words. We ask the
# model for at least 900 and cap softly via num_predict.
_TARGET_WORDS_MIN: int = 900
_TARGET_WORDS_SOFT_MAX: int = 1400

# Temperature for essay writing. Slightly higher than the structured-JSON
# pipeline (0.2) because we want fluent prose, not deterministic tokens.
_ESSAY_TEMPERATURE: float = 0.35

# Cap on how many risk_factors / red_flags we paste into the prompt. The
# prompt budget matters more than completeness — the LLM needs room to
# write, not just read.
_MAX_RISK_FACTORS_IN_PROMPT: int = 12
_MAX_RED_FLAGS_IN_PROMPT: int = 8


# =============================================================================
# PUBLIC API
# =============================================================================

def generate_thesis_essay(
    ticker: str,
    thesis: Dict[str, Any],
    filings_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Produce a 2+ page institutional-memo essay for a single ticker.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    thesis:
        Output of :func:`agent.thesis_generator.generate_thesis`.
    filings_summary:
        Output of :func:`agent.filing_analyzer.summarize_filings_by_year`.
    metrics:
        Output of :func:`agent.filing_analyzer.extract_key_metrics_for_agent`.
    macro:
        Macro dashboard dict.
    risk_flags:
        Output of :func:`agent.risk_scanner.compute_risk_flags`.
    config:
        Project config module (see ``agent.base_agent.ask_agent``).

    Returns
    -------
    dict
        ``{"text": str, "model_used": str, "elapsed_ms": float,
            "fallback": bool, "word_count": int}``
        ``fallback`` is ``True`` when the deterministic back-up was used
        (mock mode, LLM failure, or Ollama unreachable).
    """
    started_at = time.perf_counter()

    prompt = _build_essay_prompt(
        ticker=ticker,
        thesis=thesis,
        filings_summary=filings_summary,
        metrics=metrics,
        macro=macro,
        risk_flags=risk_flags,
    )

    # Resolve which model we'd actually hit so we can short-circuit mock
    # mode without building an AgentRequest and round-tripping through
    # base_agent's JSON-formatted path.
    pseudo_request = AgentRequest(prompt=prompt, context={}, model_tag=None)
    model_used = _resolve_model(pseudo_request, config)

    if model_used == "mock":
        logger.info("thesis_essay | %s | mock mode -> deterministic fallback", ticker)
        text = _deterministic_fallback_essay(
            ticker=ticker,
            thesis=thesis,
            filings_summary=filings_summary,
            metrics=metrics,
            macro=macro,
            risk_flags=risk_flags,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return _wrap(text, model_used="mock", elapsed_ms=elapsed_ms, fallback=True)

    # Real Ollama path. Text-mode, not JSON-mode.
    try:
        text = _call_ollama_text(prompt=prompt, model_name=model_used, config=config)
        if not text or len(text.split()) < 150:
            # Model returned nothing useful (empty, truncated, or refused).
            raise RuntimeError(
                f"LLM returned too-short essay ({len(text.split()) if text else 0} words)"
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "thesis_essay | %s | model=%s | words=%d | elapsed_ms=%.0f",
            ticker,
            model_used,
            len(text.split()),
            elapsed_ms,
        )
        return _wrap(text, model_used=model_used, elapsed_ms=elapsed_ms, fallback=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "thesis_essay | %s | LLM call failed: %s -> deterministic fallback",
            ticker,
            exc,
        )
        text = _deterministic_fallback_essay(
            ticker=ticker,
            thesis=thesis,
            filings_summary=filings_summary,
            metrics=metrics,
            macro=macro,
            risk_flags=risk_flags,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return _wrap(
            text,
            model_used=f"{model_used} (failed)",
            elapsed_ms=elapsed_ms,
            fallback=True,
        )


# =============================================================================
# INTERNAL: PROMPT CONSTRUCTION
# =============================================================================

def _build_essay_prompt(
    ticker: str,
    thesis: Dict[str, Any],
    filings_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
) -> str:
    """Construct the full prompt for the LLM essay writer.

    The prompt is built in six clearly-labelled blocks so the model can
    ground every claim against named evidence. Order matters: verdict and
    metrics go first (so they anchor the model's reasoning), followed by
    filings context, then macro, then risks, then writing instructions.
    """
    metrics_block = _format_metrics_block(metrics)
    filings_block = _format_filings_block(filings_summary)
    macro_block = _format_macro_block(macro)
    risk_block = _format_risk_block(risk_flags)
    thesis_block = _format_thesis_block(thesis)

    return f"""You are a senior equity analyst writing a formal investment memo for a hedge-fund investment committee. Your job is to interpret the provided financial data, not summarize it. Every major claim must be grounded in a specific number, filing, or macro datapoint from the context below.

=== TICKER ===
{ticker}

=== HEURISTIC VERDICT (from quant model) ===
{thesis_block}

=== KEY METRICS (from 10-K / 10-Q filings) ===
{metrics_block}

=== FILINGS CONTEXT ===
{filings_block}

=== MACRO BACKDROP ===
{macro_block}

=== RISK FLAGS ===
{risk_block}

=== WRITING INSTRUCTIONS ===
Write an institutional-quality investment memo of {_TARGET_WORDS_MIN}-{_TARGET_WORDS_SOFT_MAX} words, structured as follows:

1. **Executive Summary** (1 short paragraph). State the direction of the thesis, the core reason, and the single biggest risk.

2. **Business & Financial Performance** (2-3 paragraphs). Interpret the metrics. Discuss revenue growth, margin trajectory, cash conversion, ROIC, and leverage. Connect numbers to what they imply about the business — e.g. rising debt/EBITDA combined with shrinking margins suggests operating leverage is breaking down; high OCF/NI ratio indicates accounting quality. Reference specific values.

3. **Valuation & Capital Allocation** (1-2 paragraphs). Interpret P/E, EV/EBITDA, and FCF yield relative to typical ranges. Is the stock cheap or expensive given the growth and quality profile? Reference specific values.

4. **Filings & Management Signal** (1 paragraph). Comment on management tone, red flags, and any notable risk-factor disclosures from the 10-K / 10-Q / 8-K. If tone diverges from numbers, flag the tension.

5. **Macro & Industry Context** (1 paragraph). How do current recession probability, VIX level, yield curve, and rate backdrop affect the setup? Be specific about which macro variables matter most for this business.

6. **Risks & Counter-Arguments** (1 paragraph). Spell out the key risks. If any risk flag contradicts the heuristic verdict, explain which evidence should carry more weight and why.

7. **Conclusion & Outlook** (1 short paragraph). Restate the 1-year outlook. Identify the one or two catalysts or data points that would change the thesis.

STYLE RULES:
- Write in fluent prose paragraphs. No bullet points inside sections (section headers are fine).
- Every quantitative claim must cite a number from the context above. If a metric is marked "(not available in data)", say so in the essay — never infer, estimate, or reconstruct unavailable metrics from other numbers. P/E and EV/EBITDA cannot be derived from margins or FCF.
- If a data section is marked with ⚠ DATA GAP, acknowledge the gap in the corresponding essay section. Do not write credible-sounding analysis from a data gap.
- Discuss both positives and negatives. A memo that only says good things or only says bad things is not credible.
- If the heuristic verdict shows a signal tension (e.g. bullish outlook + flat direction), address it directly: explain what is causing it and which signal carries more weight for the investor.
- Do not repeat the heuristic bias scores verbatim — translate them into plain analytical language.
- Do not use generic filler like "as we move forward" or "in today's market". Every sentence must say something specific about {ticker}.
- Bullish and bearish are return expectations. If the thesis is bullish but price direction is flat, you MUST reconcile these — "bullish quality but constrained near-term return" is a valid position; silently calling both true without explanation is not.

Now write the memo."""


def _format_thesis_block(thesis: Dict[str, Any]) -> str:
    """Compact rendering of the heuristic verdict.

    Explicitly flags internal contradictions (e.g. bullish outlook but flat
    price direction) so the LLM is forced to address the tension rather than
    blindly repeating both labels.
    """
    if not thesis:
        return "(no heuristic verdict available)"

    outlook = thesis.get("outlook", "neutral")
    direction = thesis.get("price_direction", "flat")
    confidence = thesis.get("confidence", 0.5)
    bias = thesis.get("bias_score", 0.0)

    parts = [
        f"Outlook: {outlook}",
        f"Price direction (1Y): {direction}",
        f"Confidence: {confidence:.2f}",
        f"Bias score (composite, [-1,+1]): {bias:+.2f}",
    ]

    # Detect the outlook/direction mismatch that plagued the AAPL essay.
    # Bullish outlook should imply at least moderate_up; bearish should imply
    # at least moderate_down.  If there is a gap, flag it explicitly so the
    # model cannot pretend it isn't there.
    mismatch = False
    if outlook == "bullish" and direction in ("flat", "moderate_down", "strong_down"):
        parts.append(
            "⚠ INTERNAL SIGNAL TENSION: outlook is bullish but price direction "
            f"is '{direction}'. This often means the fundamental score is positive "
            "but a HIGH risk dampener compressed the final bias below the "
            "moderate_up threshold. You MUST address this tension explicitly in "
            "the Risks section — do not simultaneously call the thesis bullish "
            "and the return expectation flat without explaining why."
        )
        mismatch = True
    elif outlook == "bearish" and direction in ("flat", "moderate_up", "strong_up"):
        parts.append(
            "⚠ INTERNAL SIGNAL TENSION: outlook is bearish but price direction "
            f"is '{direction}'. Address this contradiction explicitly."
        )
        mismatch = True

    if not mismatch:
        parts.append("Outlook and direction are internally consistent.")

    risks = thesis.get("key_risks") or []
    opps = thesis.get("key_opportunities") or []
    if risks:
        parts.append("Heuristic key risks: " + "; ".join(risks))
    if opps:
        parts.append("Heuristic key opportunities: " + "; ".join(opps))
    return "\n".join(parts)


def _format_metrics_block(metrics: Dict[str, Any]) -> str:
    """Render metrics as labelled lines, omitting missing values.

    Crucially: missing valuation multiples (P/E, EV/EBITDA) are rendered
    as explicit "(not available in data)" lines rather than simply being
    omitted.  This prevents the LLM from 'inferring' multiples from margins
    and FCF — an analytically unsound move the AAPL essay was caught doing.
    """
    if not metrics:
        return "(no metrics available)"

    def _pct(x: Any) -> Optional[str]:
        v = _as_float(x)
        return f"{v * 100:+.2f}%" if v is not None else None

    def _num(x: Any, fmt: str = "{:,.2f}") -> Optional[str]:
        v = _as_float(x)
        return fmt.format(v) if v is not None else None

    def _ratio(x: Any) -> Optional[str]:
        v = _as_float(x)
        return f"{v:.2f}x" if v is not None else None

    def _ratio_required(x: Any, label: str) -> str:
        """Valuation multiples: render value OR explicit not-available warning."""
        v = _as_float(x)
        if v is not None:
            return f"{v:.2f}x"
        return f"(not available in data — do NOT infer or estimate {label})"

    rows: List[tuple] = [
        ("Revenue growth (3Y CAGR)", _pct(metrics.get("revenue_growth_3y"))),
        ("Revenue growth (5Y CAGR)", _pct(metrics.get("revenue_growth_5y"))),
        ("Revenue growth (YoY)", _pct(metrics.get("revenue_growth_yoy"))),
        ("Gross margin", _pct(metrics.get("gross_margin"))),
        ("Operating margin", _pct(metrics.get("operating_margin"))),
        ("Margin trend", metrics.get("margin_trend")),
        ("Debt / EBITDA", _ratio(metrics.get("debt_ebitda"))),
        ("Net debt / EBITDA", _ratio(metrics.get("net_debt_ebitda"))),
        ("Interest coverage (EBIT / Interest)", _ratio(metrics.get("interest_coverage"))),
        # Valuation multiples — always rendered, missing = explicit warning
        ("P / E", _ratio_required(metrics.get("p_e"), "P/E")),
        ("EV / EBITDA", _ratio_required(metrics.get("ev_ebitda"), "EV/EBITDA")),
        ("FCF yield", _pct(metrics.get("fcf_yield"))),
        ("Cash conversion (OCF / NI)", _ratio(metrics.get("cash_conversion"))),
        ("ROIC", _pct(metrics.get("roic"))),
        ("Free cash flow ($)", _num(metrics.get("free_cash_flow"), "${:,.0f}")),
        ("Market cap ($)", _num(metrics.get("market_cap"), "${:,.0f}")),
        ("Price ($)", _num(metrics.get("price"))),
    ]
    streak = metrics.get("cash_flow_negative_3_years")
    if streak is True:
        rows.append(("WARNING", "FCF negative for 3+ consecutive years"))

    lines = [f"- {label}: {value}" for label, value in rows if value is not None]
    return "\n".join(lines) if lines else "(no metrics available)"


def _format_filings_block(filings_summary: Dict[str, Any]) -> str:
    """Render the filings summary: count, tone, years covered, red flags, risks.

    When filings_considered == 0, emits an explicit data-gap warning so the
    LLM cannot write a credible-sounding filings paragraph from nothing.
    """
    if not filings_summary:
        return (
            "⚠ DATA GAP: No filings data available. "
            "You MUST state in the Filings section that no filings were "
            "retrieved and that any management-tone or disclosure claims "
            "are unsupported."
        )

    considered = filings_summary.get("filings_considered", 0)
    tone = filings_summary.get("management_tone", "neutral")
    red_flags = filings_summary.get("red_flags") or []
    risk_factors = filings_summary.get("risk_factors") or []
    by_year = filings_summary.get("by_year") or {}
    prose_summary = filings_summary.get("summary") or ""

    if considered == 0:
        return (
            "⚠ DATA GAP: filings_considered = 0. The pipeline fetched no "
            "SEC filings for this ticker. The Filings & Management Signal "
            "section of the memo MUST acknowledge this gap explicitly — "
            "do not invent tone assessments or disclosure commentary."
        )

    lines: List[str] = [
        f"Filings considered: {considered}",
        f"Management tone (inferred from filing text): {tone}",
    ]

    if by_year:
        year_lines: List[str] = []
        for year in sorted(by_year.keys(), reverse=True):
            info = by_year[year] or {}
            form_counts = info.get("form_types") or {}
            if form_counts:
                forms_str = ", ".join(
                    f"{n}x {form}" for form, n in form_counts.items()
                )
            else:
                forms_str = f"{info.get('filing_count', 0)} filings"
            year_lines.append(f"  {year}: {forms_str}")
        lines.append("Filings by year:")
        lines.extend(year_lines)

    if red_flags:
        lines.append("Disclosure red flags:")
        for rf in red_flags[:_MAX_RED_FLAGS_IN_PROMPT]:
            lines.append(f"  - {rf}")
    else:
        lines.append("Disclosure red flags: none detected")

    if risk_factors:
        lines.append("Notable risk-factor language from filings:")
        for rf in risk_factors[:_MAX_RISK_FACTORS_IN_PROMPT]:
            lines.append(f"  - {rf}")

    if prose_summary:
        lines.append(f"Prose summary: {prose_summary}")

    return "\n".join(lines)


def _format_macro_block(macro: Dict[str, Any]) -> str:
    """Render the macro dashboard.

    When macro data is entirely absent, emits a hard warning so the LLM
    writes a placeholder-free macro paragraph instead of inventing signals.
    """
    if not macro:
        return (
            "⚠ DATA GAP: No macro data available. "
            "The Macro section MUST acknowledge this gap rather than "
            "discussing hypothetical macro conditions."
        )

    def _pct(x: Any) -> Optional[str]:
        v = _as_float(x)
        return f"{v * 100:.2f}%" if v is not None else None

    def _num(x: Any, fmt: str = "{:.2f}") -> Optional[str]:
        v = _as_float(x)
        return fmt.format(v) if v is not None else None

    rows: List[tuple] = [
        ("Recession probability (12M)", _pct(macro.get("recession_probability"))),
        ("VIX", _num(macro.get("vix"))),
        ("10Y-2Y yield spread", _pct(macro.get("yield_curve_spread"))),
        ("Yield curve inverted", macro.get("yield_curve_inverted")),
        ("Fed funds rate", _pct(macro.get("fed_funds_rate"))),
        ("CPI YoY", _pct(macro.get("cpi_yoy"))),
        ("Unemployment rate", _pct(macro.get("unemployment_rate"))),
        ("10Y Treasury yield", _pct(macro.get("treasury_10y"))),
        ("USD index (DXY)", _num(macro.get("dxy"))),
    ]
    lines = [f"- {label}: {value}" for label, value in rows if value is not None]

    if not lines:
        return (
            "⚠ DATA GAP: Macro dict present but all fields are None. "
            "The Macro section MUST acknowledge the absence of macro data."
        )

    return "\n".join(lines)


def _format_risk_block(risk_flags: Dict[str, Any]) -> str:
    """Render the risk_flags dict: combined level, per-component, reasons."""
    if not risk_flags:
        return "(no risk flags)"
    levels = risk_flags.get("levels") or {}
    reasons = risk_flags.get("reasons") or {}

    combined = levels.get("combined", "MEDIUM")
    lines = [f"Combined risk: {combined}"]

    for component in ("fundamental", "macro", "market", "agent"):
        lvl = levels.get(component)
        if lvl is not None:
            lines.append(f"{component.capitalize()} risk: {lvl}")

    # Reasons may be dict (keyed by component) or flat list
    if isinstance(reasons, dict):
        for component, items in reasons.items():
            if not items:
                continue
            lines.append(f"{component.capitalize()} reasons:")
            for r in items:
                lines.append(f"  - {r}")
    elif isinstance(reasons, list):
        if reasons:
            lines.append("Reasons:")
            for r in reasons:
                lines.append(f"  - {r}")

    return "\n".join(lines)


# =============================================================================
# INTERNAL: OLLAMA TEXT-MODE CALL
# =============================================================================

def _call_ollama_text(prompt: str, model_name: str, config: Any) -> str:
    """Call Ollama for a free-text completion (no JSON formatting).

    Kept separate from ``base_agent._call_ollama`` because that helper forces
    ``format="json"``, which makes the model wrap prose in a JSON envelope
    and cuts into the output-token budget. For the essay we want raw text.

    Raises
    ------
    Exception
        On any network / HTTP / decoding error. Caller is responsible for
        catching and falling back.
    """
    import urllib.error
    import urllib.request

    base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(getattr(config, "AGENT_TIMEOUT", 30))
    # Essays need a longer timeout than structured JSON calls. Override the
    # default if the user hasn't set a specifically long one. 180s is generous
    # for a 30B model running partially on CPU.
    if timeout < 180:
        timeout = 180.0
    max_tokens = int(getattr(config, "MAX_TOKENS", 4096))

    body = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": _ESSAY_TEMPERATURE,
        },
        # NOTE: no "format": "json" — we want prose.
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


# =============================================================================
# INTERNAL: DETERMINISTIC FALLBACK ESSAY
# =============================================================================

def _deterministic_fallback_essay(
    ticker: str,
    thesis: Dict[str, Any],
    filings_summary: Dict[str, Any],
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
) -> str:
    """Assemble a template-driven essay from the heuristic outputs.

    Used when the LLM is unavailable. Does not invent numbers — only
    references fields that are actually present in the inputs. Reads
    noticeably more mechanical than an LLM essay, but satisfies the
    "2+ pages grounded in the data" contract as best it can without a model.
    """
    outlook = thesis.get("outlook", "neutral")
    direction = thesis.get("price_direction", "flat")
    confidence = _as_float(thesis.get("confidence")) or 0.5
    bias = _as_float(thesis.get("bias_score")) or 0.0
    combined_risk = ((risk_flags or {}).get("levels") or {}).get("combined", "MEDIUM")

    paras: List[str] = []

    # --- Executive summary -------------------------------------------------
    paras.append(
        f"**Executive Summary.** The systematic model assigns {ticker} a "
        f"{outlook.upper()} one-year outlook with a projected price direction "
        f"of {direction.replace('_', ' ')} and model confidence of "
        f"{confidence:.0%}. The composite bias score is {bias:+.2f} on a "
        f"[-1, +1] scale, with combined risk classified as {combined_risk}. "
        f"This memo is generated by a deterministic fallback writer because "
        f"the language model was unavailable; its claims are drawn verbatim "
        f"from the underlying metric and filing inputs."
    )

    # --- Business & financial performance ---------------------------------
    biz_parts: List[str] = []
    g3 = _as_float(metrics.get("revenue_growth_3y"))
    gy = _as_float(metrics.get("revenue_growth_yoy"))
    gm = _as_float(metrics.get("gross_margin"))
    om = _as_float(metrics.get("operating_margin"))
    if g3 is not None:
        biz_parts.append(
            f"Three-year revenue CAGR of {g3 * 100:+.1f}% "
            + _growth_commentary(g3)
        )
    if gy is not None:
        biz_parts.append(f"Year-over-year revenue growth is {gy * 100:+.1f}%.")
    if gm is not None:
        biz_parts.append(f"Gross margin stands at {gm * 100:.1f}%.")
    if om is not None:
        biz_parts.append(f"Operating margin is {om * 100:.1f}%.")
    trend = metrics.get("margin_trend")
    if trend:
        biz_parts.append(f"The trailing margin trend is {trend}.")

    cc = _as_float(metrics.get("cash_conversion"))
    if cc is not None:
        biz_parts.append(
            f"Operating cash flow covers reported net income at {cc:.2f}x "
            + _cc_commentary(cc)
        )
    roic = _as_float(metrics.get("roic"))
    if roic is not None:
        biz_parts.append(
            f"ROIC of {roic * 100:.1f}% " + _roic_commentary(roic)
        )
    de = _as_float(metrics.get("debt_ebitda"))
    if de is not None:
        biz_parts.append(
            f"Debt-to-EBITDA sits at {de:.2f}x " + _leverage_commentary(de)
        )
    ic = _as_float(metrics.get("interest_coverage"))
    if ic is not None:
        biz_parts.append(f"Interest coverage is {ic:.2f}x.")

    if biz_parts:
        paras.append("**Business & Financial Performance.** " + " ".join(biz_parts))

    # --- Valuation --------------------------------------------------------
    val_parts: List[str] = []
    pe = _as_float(metrics.get("p_e"))
    ev = _as_float(metrics.get("ev_ebitda"))
    fy = _as_float(metrics.get("fcf_yield"))
    if pe is not None:
        val_parts.append(f"P/E of {pe:.1f}x")
    if ev is not None:
        val_parts.append(f"EV/EBITDA of {ev:.1f}x")
    if fy is not None:
        val_parts.append(f"FCF yield of {fy * 100:.2f}%")
    if val_parts:
        paras.append(
            "**Valuation.** Current valuation multiples are "
            + ", ".join(val_parts)
            + ". A full relative-value comparison requires industry and "
            + "historical benchmarks not present in this context."
        )

    # --- Filings & management signal --------------------------------------
    tone = (filings_summary or {}).get("management_tone", "neutral")
    considered = (filings_summary or {}).get("filings_considered", 0)
    red_flags = (filings_summary or {}).get("red_flags") or []
    filing_parts: List[str] = [
        f"The analysis incorporates {considered} recent filings, with "
        f"management tone inferred as {tone}."
    ]
    if red_flags:
        flag_sample = "; ".join(list(red_flags)[:3])
        filing_parts.append(
            f"Disclosure red flags surfaced include: {flag_sample}."
        )
    else:
        filing_parts.append("No disclosure red flags were surfaced.")
    paras.append("**Filings & Management Signal.** " + " ".join(filing_parts))

    # --- Macro ------------------------------------------------------------
    macro_parts: List[str] = []
    rp = _as_float((macro or {}).get("recession_probability"))
    vix = _as_float((macro or {}).get("vix"))
    curve = _as_float((macro or {}).get("yield_curve_spread"))
    inverted = (macro or {}).get("yield_curve_inverted")
    if rp is not None:
        macro_parts.append(
            f"implied 12-month recession probability of {rp * 100:.0f}%"
        )
    if vix is not None:
        macro_parts.append(f"VIX at {vix:.1f}")
    if inverted is True:
        macro_parts.append("an inverted 10Y-2Y yield curve")
    elif curve is not None:
        macro_parts.append(f"a 10Y-2Y spread of {curve * 100:.0f} basis points")
    if macro_parts:
        paras.append(
            "**Macro Backdrop.** The macro context at the time of analysis "
            "features " + ", ".join(macro_parts) + "."
        )

    # --- Risks ------------------------------------------------------------
    risks = thesis.get("key_risks") or []
    if risks:
        risk_sentences = "; ".join(risks[:5])
        paras.append(
            "**Risks & Counter-Arguments.** The primary risks flagged by the "
            f"systematic layer are: {risk_sentences}. These should be weighed "
            "against the opportunity set before finalizing position sizing."
        )

    # --- Conclusion -------------------------------------------------------
    paras.append(
        f"**Conclusion & Outlook.** The one-year outlook is {outlook.upper()} "
        f"with confidence {confidence:.0%}. A meaningful re-rating of this "
        f"thesis would require either a material change in the combined risk "
        f"level (currently {combined_risk}) or a shift in the three component "
        f"biases that feed the composite score. This memo was produced by "
        f"the deterministic fallback writer; for a full analytical "
        f"interpretation, re-run with the language model available."
    )

    return "\n\n".join(paras)


# =============================================================================
# INTERNAL: HELPERS
# =============================================================================

def _wrap(
    text: str, model_used: str, elapsed_ms: float, fallback: bool
) -> Dict[str, Any]:
    """Normalise the essay return shape."""
    return {
        "text": text,
        "model_used": model_used,
        "elapsed_ms": round(elapsed_ms, 1),
        "fallback": fallback,
        "word_count": len(text.split()),
    }


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _growth_commentary(g: float) -> str:
    if g >= 0.15:
        return "indicates strong top-line momentum."
    if g >= 0.08:
        return "indicates healthy but not exceptional growth."
    if g >= 0.0:
        return "indicates modest growth that may lag inflation."
    return "indicates top-line contraction, a material concern."


def _cc_commentary(cc: float) -> str:
    if cc >= 1.1:
        return "— a quality signal indicating reported earnings are well-supported by cash generation."
    if cc >= 0.9:
        return "— broadly in line with reported earnings quality."
    if cc >= 0.7:
        return "— modestly below expectations, worth monitoring for accruals build-up."
    return "— a meaningful gap suggesting aggressive revenue recognition or working-capital drag."


def _roic_commentary(r: float) -> str:
    if r >= 0.20:
        return "indicates capital-efficient compounding well above the typical cost of capital."
    if r >= 0.10:
        return "indicates returns above a reasonable cost-of-capital estimate."
    if r >= 0.05:
        return "indicates returns roughly matching cost of capital — a neutral signal."
    return "indicates returns below typical cost of capital, eroding economic value."


def _leverage_commentary(de: float) -> str:
    if de < 1.0:
        return "— a conservative capital structure providing strategic flexibility."
    if de < 2.0:
        return "— a moderate, widely acceptable leverage level."
    if de < 3.0:
        return "— approaching the upper bound of comfort for most industries."
    if de < 4.5:
        return "— elevated leverage that materially limits balance-sheet flexibility."
    return "— a stressed capital structure warranting close monitoring."
