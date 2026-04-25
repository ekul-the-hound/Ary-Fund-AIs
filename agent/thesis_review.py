"""
agent/thesis_review.py
======================

LLM-powered quality-gate for investment memos.

This module implements a two-pass review loop:

  1. ``review_essay(essay, metrics, macro, ...)`` → structured critique
     with section-level scores (1-10) and specific revision instructions.
  2. ``revise_essay(essay, review, metrics, macro, ...)`` → an improved
     essay that applies the critique.

Both passes are optional — the pipeline can run essay-only (1 LLM call),
essay + review (2 calls), or essay + review + revision (3 calls). Each
additional pass costs ~30-90s on a 30B model with CPU offload, so the
caller decides the trade-off.

Design contract
---------------
- Same graceful-degradation pattern as ``thesis_essay.py``: if the LLM
  is unavailable or in mock mode, both functions return deterministic
  fallback outputs rather than crashing.
- Pure functions, no side effects beyond logging.
- The review prompt is tuned to produce *actionable numeric critique*,
  not vague "make it better" feedback.

Usage
-----
::

    from agent.thesis_essay import generate_thesis_essay
    from agent.thesis_review import review_essay, revise_essay

    essay = generate_thesis_essay(ticker, thesis, filings, metrics, macro, risk_flags, config)

    review = review_essay(
        ticker=ticker,
        essay_text=essay["text"],
        metrics=metrics,
        macro=macro,
        risk_flags=risk_flags,
        config=config,
    )
    # review["scores"] = {"executive_summary": 7, "valuation": 4, ...}
    # review["overall_score"] = 6.2
    # review["text"] = "1. Executive Summary\n\nMain weakness: ..."

    if review["overall_score"] < 8.0:
        revised = revise_essay(
            ticker=ticker,
            original_essay=essay["text"],
            review_text=review["text"],
            metrics=metrics,
            macro=macro,
            risk_flags=risk_flags,
            config=config,
        )
        thesis["essay"] = revised["text"]
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.base_agent import AgentRequest, _resolve_model


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_REVIEW_TEMPERATURE: float = 0.15   # lower than the essay; we want precise critique
_REVISION_TEMPERATURE: float = 0.30

_SECTION_NAMES: List[str] = [
    "executive_summary",
    "business_performance",
    "valuation",
    "filings_signal",
    "macro_context",
    "risks",
    "catalysts",
    "verdict",
]

_REVIEW_MIN_WORDS: int = 1200


# =============================================================================
# PUBLIC API
# =============================================================================

def review_essay(
    ticker: str,
    essay_text: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Score and critique an investment memo.

    Returns
    -------
    dict
        {
          "text": str,             # full review text
          "scores": dict,          # section name -> 1-10 score
          "overall_score": float,  # weighted average
          "model_used": str,
          "elapsed_ms": float,
          "fallback": bool,
          "word_count": int,
        }
    """
    started_at = time.perf_counter()
    prompt = _build_review_prompt(ticker, essay_text, metrics, macro, risk_flags)

    pseudo_request = AgentRequest(prompt=prompt, context={}, model_tag=None)
    model_used = _resolve_model(pseudo_request, config)

    if model_used == "mock":
        logger.info("thesis_review | %s | mock mode -> deterministic fallback", ticker)
        text = _deterministic_review(ticker, essay_text, metrics, macro)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        scores = _extract_scores(text)
        return _wrap(text, scores, model_used="mock", elapsed_ms=elapsed_ms, fallback=True)

    try:
        text = _call_ollama_text(prompt, model_used, config, temperature=_REVIEW_TEMPERATURE)
        if not text or len(text.split()) < 200:
            raise RuntimeError(f"Review too short ({len(text.split()) if text else 0} words)")
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        scores = _extract_scores(text)
        logger.info(
            "thesis_review | %s | model=%s | overall=%.1f | words=%d | elapsed_ms=%.0f",
            ticker, model_used, scores.get("overall", 0), len(text.split()), elapsed_ms,
        )
        return _wrap(text, scores, model_used=model_used, elapsed_ms=elapsed_ms, fallback=False)
    except Exception as exc:
        logger.warning("thesis_review | %s | LLM failed: %s -> fallback", ticker, exc)
        text = _deterministic_review(ticker, essay_text, metrics, macro)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        scores = _extract_scores(text)
        return _wrap(text, scores, model_used=f"{model_used} (failed)", elapsed_ms=elapsed_ms, fallback=True)


def revise_essay(
    ticker: str,
    original_essay: str,
    review_text: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Rewrite an essay applying the critique from ``review_essay``.

    Returns the same shape as ``generate_thesis_essay`` so it can be a
    drop-in replacement for ``thesis["essay"]``.
    """
    started_at = time.perf_counter()
    prompt = _build_revision_prompt(ticker, original_essay, review_text, metrics, macro, risk_flags)

    pseudo_request = AgentRequest(prompt=prompt, context={}, model_tag=None)
    model_used = _resolve_model(pseudo_request, config)

    if model_used == "mock":
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "text": original_essay,
            "model_used": "mock",
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": True,
            "word_count": len(original_essay.split()),
            "revision_applied": False,
        }

    try:
        text = _call_ollama_text(
            prompt, model_used, config, temperature=_REVISION_TEMPERATURE,
            min_output_tokens=6144,  # revision needs ~2100 tokens for 1500 words
        )
        if not text or len(text.split()) < 1400:
            raise RuntimeError(f"Revision too short ({len(text.split()) if text else 0} words)")
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "thesis_review | %s | revision | model=%s | words=%d | elapsed_ms=%.0f",
            ticker, model_used, len(text.split()), elapsed_ms,
        )
        return {
            "text": text,
            "model_used": model_used,
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": False,
            "word_count": len(text.split()),
            "revision_applied": True,
        }
    except Exception as exc:
        logger.warning("thesis_review | %s | revision failed: %s -> keeping original", ticker, exc)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "text": original_essay,
            "model_used": f"{model_used} (failed)",
            "elapsed_ms": round(elapsed_ms, 1),
            "fallback": True,
            "word_count": len(original_essay.split()),
            "revision_applied": False,
        }


# =============================================================================
# REVIEW PROMPT
# =============================================================================

def _build_review_prompt(
    ticker: str,
    essay_text: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
) -> str:
    """Build the critic prompt.

    The prompt gives the reviewer access to the RAW metrics so it can
    check whether the essay cited them correctly, fabricated numbers,
    or missed available data.
    """
    metrics_summary = _compact_metrics(metrics)
    macro_summary = _compact_macro(macro)
    risk_summary = _compact_risks(risk_flags)

    return f"""You are a buy-side portfolio manager. A junior analyst has submitted the investment memo below. Your job is to tell them exactly what to fix before it goes to committee.

=== TICKER ===
{ticker}

=== RAW DATA (ground truth) ===
{metrics_summary}

{macro_summary}

{risk_summary}

=== MEMO TO REVIEW ===
{essay_text}

=== INSTRUCTIONS ===
Your review must be at least {_REVIEW_MIN_WORDS} words. Be direct, use numbers, and prioritize judgment over coverage.

For each of the 8 sections below, do exactly this:
1. Score from 1 to 10.
2. State the single biggest weakness in 1-2 sentences.
3. Give the single most important fix. This fix must include a specific number, threshold, or comparison. Write one sentence explaining WHY this fix matters — specifically, how it changes estimates, multiples, or the stock.
4. Give 1-2 additional fixes if needed, same format.

Do NOT list every possible improvement. Pick the ones that matter most.

=== SECTION-SPECIFIC RULES ===

VALUATION: Choose the ONE most important missing valuation metric for this stock. Do not mention all of P/E, EV/EBITDA, FCF yield, and price-to-sales — pick the one that matters most and explain why. If the RAW DATA contains a valuation metric the memo ignores, flag it. State the threshold where valuation changes the verdict.

MACRO: Discuss only the 3-5 most material macro variables for THIS stock. Do not dump every indicator. For each one, explain the transmission mechanism: how does it affect revenue, margins, or the multiple? Check the RAW DATA — if a macro number exists but the memo ignores it, flag it and explain what it means.

RISKS: Rank the top 3 risks. For each: state the trigger (a specific number), explain how it flows through to earnings or the multiple, and say why it changes the stock price. Format:
1. [Risk] — trigger: [number] — because: [how it hits earnings/multiple/stock]

CATALYSTS: List the top 3 catalysts. For each: state the threshold, the time frame (quarter or 6-12 months), and explain why it moves the stock. Format:
- [Catalyst] — threshold: [number] — time frame: [when] — because: [why it moves the stock]

VERDICT: The conclusion must include:
- Rating: BUY (expected upside >15%), HOLD (upside 0-15%), or SELL (downside >15%)
- The 2-3 numbers that justify this rating
- Change-of-view triggers: what specific numbers would flip the rating
- One sentence separating business quality from stock attractiveness

=== QUALITY FLAGS ===
Flag any of these if present (one sentence each, do not belabor):
- "Bullish" and "stretched" used together without reconciliation
- Valuation inferred from margins/FCF instead of market multiples
- Management tone cited without filing evidence
- "Positive surprise" used without defining it numerically
- Filler words ("appears," "likely," "suggests") without a backing number

=== OUTPUT FORMAT ===
Machine-parsed. Use EXACTLY this structure:

## 1. Executive Summary [SCORE: X/10]

Main weakness: ...
Fix 1: ... This matters because ...
Fix 2: ...

## 2. Business & Financial Performance [SCORE: X/10]

Main weakness: ...
Fix 1: ... This matters because ...
Fix 2: ...

## 3. Valuation [SCORE: X/10]

Main weakness: ...
Most important missing metric: [name] at [value] — this matters because ...
Fix 1: ...
Threshold that changes the verdict: ...

## 4. Filings & Management Signal [SCORE: X/10]

Main weakness: ...
Fix 1: ...
Keep, rewrite, or remove this section: [decision] because ...

## 5. Macro & Industry Context [SCORE: X/10]

Main weakness: ...
Top 3 macro variables for this stock:
1. [Variable] at [value] — matters because [transmission mechanism]
2. [Variable] at [value] — matters because [transmission mechanism]
3. [Variable] at [value] — matters because [transmission mechanism]

## 6. Risks [SCORE: X/10]

Main weakness: ...
Top 3 risks ranked:
1. [Risk] — trigger: [number] — because: [how it hits earnings/multiple/stock]
2. [Risk] — trigger: [number] — because: [how it hits earnings/multiple/stock]
3. [Risk] — trigger: [number] — because: [how it hits earnings/multiple/stock]

## 7. Catalysts [SCORE: X/10]

Main weakness: ...
Top 3 catalysts:
- [Catalyst] — threshold: [number] — time frame: [when] — because: [why it moves the stock]
- [Catalyst] — threshold: [number] — time frame: [when] — because: [why it moves the stock]
- [Catalyst] — threshold: [number] — time frame: [when] — because: [why it moves the stock]

## 8. Verdict [SCORE: X/10]

Main weakness: ...
Fix 1: ...
Rating: [BUY/HOLD/SELL] — expected return: [X-Y%] over [time frame]
Key numbers: ...
Change-of-view triggers: ...
Business quality vs stock attractiveness: ...

## Overall Assessment [SCORE: X/10]

Top 3 weaknesses across the entire memo:
1. ...
2. ...
3. ...

Verdict: rewrite as [bullish/neutral/bearish]. The rating flips if: [2-3 numeric thresholds].

Now write the review."""


# =============================================================================
# REVISION PROMPT
# =============================================================================

def _build_revision_prompt(
    ticker: str,
    original_essay: str,
    review_text: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
    risk_flags: Dict[str, Any],
) -> str:
    metrics_summary = _compact_metrics(metrics)
    macro_summary = _compact_macro(macro)

    return f"""You are a senior equity research analyst, investment memo editor, and buy-side committee reviewer. Your job is to revise the investment memorandum below into a much stronger, committee-ready version that is precise, quantitative, and decision-useful.

=== TICKER ===
{ticker}

=== RAW DATA (ground truth — every number you cite must come from here) ===
{metrics_summary}

{macro_summary}

=== ORIGINAL ESSAY ===
{original_essay}

=== REVIEWER CRITIQUE ===
{review_text}

=== REVISION INSTRUCTIONS ===
Rewrite the memo so it is analytically sharper, more specific, more quantitative, and fully aligned with the supplied data. Apply every fix from the reviewer critique.

HARD REQUIREMENTS:
- The revised memo must be at least 1500 words.
- Keep the same 8-section structure.
- Every section must be materially expanded, not compressed.
- Every section must include at least: 1 specific metric, 1 threshold, 1 direct comparison, 1 causal explanation.
- Use numbers throughout.
- Do not use vague filler like "appears," "likely," "suggests," "could," or "may" unless immediately paired with a number or hard condition.
- Do not invent metrics. If a metric is not available, say "not available" once and move on.
- Do not include the review text or any meta-commentary about the revision process.

SECTION WORD-COUNT TARGETS (write enough to roughly meet these):
- Executive Summary: 150-220 words
- Business & Financial Performance: 220-300 words
- Valuation: 200-260 words
- Filings & Management Signal: 100-160 words
- Macro & Industry Context: 180-260 words
- Risks: 250-320 words
- Catalysts: 180-240 words
- Verdict: 120-180 words

WHAT TO IMPROVE:
1. VALUATION: Use the most relevant valuation metric from the data, explain why it matters, and compare it to a benchmark or historical norm. Prioritize the metric that best fits this company's business model.
2. MACRO: Only discuss the 3-5 most relevant macro factors. For each, explain the transmission mechanism (how it affects revenue, margins, or multiples).
3. RISKS: Rank the top 3 risks. For each: state the trigger, a numeric threshold, the mechanism of damage, and the stock-level impact.
4. CATALYSTS: List the top 3 catalysts. For each: state the threshold, the time frame, and why it would move the stock.
5. VERDICT: State BUY (upside >15%), HOLD (0-15%), or SELL (downside >15%). Include 2-3 supporting numbers and numeric change-of-view triggers. Separate business quality from stock attractiveness.

STYLE: Write like a disciplined buy-side analyst. Short, concrete sentences. No filler transitions. No repeated points across sections.

Now write the revised memo."""


# =============================================================================
# SCORE EXTRACTION
# =============================================================================

def _extract_scores(review_text: str) -> Dict[str, Any]:
    """Parse [SCORE: X/10] markers from the review.

    Returns a dict with per-section scores and an overall weighted average.
    """
    scores: Dict[str, float] = {}

    # Pattern: ## N. Section Name [SCORE: X/10]
    pattern = re.compile(
        r"##\s*\d+\.\s*(.+?)\s*\[SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\]",
        re.IGNORECASE,
    )
    for match in pattern.finditer(review_text):
        section_name = match.group(1).strip().lower()
        score = float(match.group(2))
        # Normalize to a stable key
        key = _normalize_section_key(section_name)
        scores[key] = min(max(score, 0.0), 10.0)

    # Overall: look for a separate overall marker
    overall_pattern = re.compile(
        r"##\s*Overall\s+Assessment\s*\[SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\]",
        re.IGNORECASE,
    )
    overall_match = overall_pattern.search(review_text)
    if overall_match:
        scores["overall"] = float(overall_match.group(1))
    elif scores:
        # Compute from section scores if overall not found
        section_scores = [v for k, v in scores.items() if k != "overall"]
        scores["overall"] = round(sum(section_scores) / len(section_scores), 1) if section_scores else 5.0
    else:
        scores["overall"] = 5.0

    return scores


def _normalize_section_key(raw: str) -> str:
    """Map free-form section names to canonical keys."""
    raw_lower = raw.lower()
    mapping = {
        "executive": "executive_summary",
        "business": "business_performance",
        "valuation": "valuation",
        "filing": "filings_signal",
        "macro": "macro_context",
        "risk": "risks",
        "catalyst": "catalysts",
        "verdict": "verdict",
        "conclusion": "verdict",
    }
    for keyword, key in mapping.items():
        if keyword in raw_lower:
            return key
    return raw_lower.replace(" ", "_").replace("&", "and")


# =============================================================================
# COMPACT DATA FORMATTERS
# =============================================================================

def _compact_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "Metrics: (none available)"
    lines = ["Metrics:"]
    for key in (
        "price", "market_cap", "p_e", "forward_pe", "ev_ebitda",
        "gross_margin", "operating_margin", "profit_margin",
        "revenue_growth_yoy", "revenue_growth_3y", "revenue_growth_5y",
        "free_cash_flow", "fcf_yield", "cash_conversion", "roic",
        "debt_ebitda", "net_debt_ebitda", "interest_coverage",
    ):
        v = metrics.get(key)
        if v is not None:
            lines.append(f"  {key}: {v}")
    return "\n".join(lines)


def _compact_macro(macro: Dict[str, Any]) -> str:
    if not macro:
        return "Macro: (none available)"
    lines = ["Macro:"]
    for key in (
        "recession_probability", "vix", "yield_curve_spread",
        "yield_curve_inverted", "fed_funds_rate", "cpi_yoy",
        "unemployment_rate", "treasury_10y", "dxy",
    ):
        v = macro.get(key)
        if v is not None:
            lines.append(f"  {key}: {v}")
    return "\n".join(lines)


def _compact_risks(risk_flags: Dict[str, Any]) -> str:
    if not risk_flags:
        return "Risk flags: (none)"
    levels = risk_flags.get("levels") or {}
    lines = ["Risk flags:"]
    for comp in ("combined", "fundamental", "macro", "market", "agent"):
        lvl = levels.get(comp)
        if lvl:
            lines.append(f"  {comp}: {lvl}")
    reasons = risk_flags.get("reasons") or {}
    if isinstance(reasons, dict):
        for comp, items in reasons.items():
            for r in (items or []):
                lines.append(f"  reason ({comp}): {r}")
    return "\n".join(lines)


# =============================================================================
# OLLAMA TEXT CALL (reusable, same pattern as thesis_essay)
# =============================================================================

def _call_ollama_text(
    prompt: str, model_name: str, config: Any, temperature: float = 0.2,
    min_output_tokens: int = 4096,
) -> str:
    import urllib.request

    base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    # Revision calls need more time than essay calls.  On an RTX 2080 with a
    # 30B model partially offloaded to CPU, generation runs at ~10 tok/s.
    # 1500 words ≈ 2025 tokens → needs ~200s minimum, plus prompt processing.
    # 360s is generous but safe; the model will stop at EOS well before this
    # if it finishes normally.
    timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 360.0)
    # The caller can override min_output_tokens for longer outputs (e.g.
    # revision pass needs ~2100 tokens for 1500 words, so we use 6144 to be
    # safe).  Default remains 4096 for reviews.
    max_tokens = max(int(getattr(config, "MAX_TOKENS", 4096)), min_output_tokens)

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


# =============================================================================
# DETERMINISTIC FALLBACK REVIEW
# =============================================================================

def _deterministic_review(
    ticker: str,
    essay_text: str,
    metrics: Dict[str, Any],
    macro: Dict[str, Any],
) -> str:
    """Rule-based review when LLM is unavailable.

    Checks for common failure modes:
    - Missing valuation multiples (P/E, EV/EBITDA) when data exists
    - Missing macro references when data exists
    - Data-gap over-repetition
    - Missing BUY/HOLD/AVOID verdict
    - Word count adequacy
    """
    m = metrics or {}
    mc = macro or {}
    issues: List[str] = []
    scores: Dict[str, int] = {}

    essay_lower = essay_text.lower()
    word_count = len(essay_text.split())

    # --- Executive Summary ---
    has_thesis_sentence = any(
        phrase in essay_lower
        for phrase in ("buy", "hold", "avoid", "is a high-quality", "is overvalued", "is undervalued")
    )
    exec_score = 7 if has_thesis_sentence else 4
    exec_issues = []
    if not has_thesis_sentence:
        exec_issues.append("Fix 1: Open with a single thesis sentence: '[TICKER] is [quality], but at [valuation condition] [return expectation].'")
    exec_issues.append("Fix 2: Include the biggest risk and strongest catalyst in the first paragraph.")
    scores["executive_summary"] = exec_score

    # --- Valuation ---
    pe_val = m.get("p_e")
    ev_val = m.get("ev_ebitda")
    pe_in_essay = any(s in essay_lower for s in ("p/e", "pe ratio", "trailing pe", "price-to-earnings"))
    ev_in_essay = any(s in essay_lower for s in ("ev/ebitda", "enterprise value"))

    val_score = 8
    val_issues = []
    if pe_val is not None and not pe_in_essay:
        val_score -= 3
        val_issues.append(f"Fix 1: The data contains trailing P/E of {pe_val:.1f}x — this must appear in the Valuation section with a comparison to S&P 500 average (~20-22x).")
    if ev_val is not None and not ev_in_essay:
        val_score -= 2
        val_issues.append(f"Fix 2: EV/EBITDA of {ev_val:.1f}x is available — compare against sector median (~14-16x for large-cap tech).")
    if pe_val is None and ev_val is None:
        val_issues.append("Fix 1: No valuation multiples in data. State this once, then assess based on FCF yield alone.")
    fwd = m.get("forward_pe")
    if fwd is not None:
        val_issues.append(f"Fix 3: Forward P/E of {fwd:.1f}x should be cited to show whether the market expects earnings growth.")
    scores["valuation"] = max(val_score, 2)

    # --- Data-gap overuse ---
    gap_count = essay_lower.count("data gap") + essay_lower.count("not available") + essay_lower.count("unavailable")
    gap_issue = ""
    if gap_count > 3:
        gap_issue = f"The essay mentions data gaps {gap_count} times. Reduce to at most 1-2 brief acknowledgments."

    # --- Macro ---
    macro_score = 7
    macro_issues = []
    vix = mc.get("vix")
    fed = mc.get("fed_funds_rate")
    recession = mc.get("recession_probability")
    if vix is not None and "vix" not in essay_lower:
        macro_score -= 2
        macro_issues.append(f"Fix 1: VIX at {vix:.1f} should be cited — {'elevated risk aversion' if vix > 20 else 'moderate complacency'} affects positioning.")
    if fed is not None and "fed funds" not in essay_lower and "interest rate" not in essay_lower:
        macro_score -= 2
        macro_issues.append(f"Fix 2: Fed funds rate at {fed*100:.2f}% directly affects discount rates and consumer credit conditions.")
    if recession is not None:
        macro_issues.append(f"Fix 3: 12-month recession probability of {recession*100:.0f}% should frame the cyclical risk discussion.")
    if not mc:
        macro_score = 5
        macro_issues.append("Fix 1: No macro data available. Acknowledge briefly and move on.")
    scores["macro_context"] = max(macro_score, 2)

    # --- Verdict ---
    has_verdict = any(v in essay_lower for v in ("buy", "hold", "avoid"))
    verdict_score = 8 if has_verdict else 3
    verdict_issues = []
    if not has_verdict:
        verdict_issues.append("Fix 1: End with a clear BUY, HOLD, or AVOID recommendation.")
        verdict_issues.append("Fix 2: State the single data point that would change the verdict.")
    scores["verdict"] = verdict_score

    # --- Business Performance ---
    scores["business_performance"] = 7
    biz_issues = [
        "Fix 1: For every margin figure cited, explain what it implies about the business model.",
        "Fix 2: Connect revenue growth to margin trajectory — is the company growing into or out of profitability?",
    ]
    gm = m.get("gross_margin")
    om = m.get("operating_margin")
    if gm is not None and om is not None:
        opex_pct = (gm - om) * 100
        biz_issues.append(f"Fix 3: Note that the {opex_pct:.0f} percentage-point gap between gross margin ({gm*100:.1f}%) and operating margin ({om*100:.1f}%) represents operating expense intensity.")

    # --- Filings ---
    scores["filings_signal"] = 6
    filing_issues = ["Fix 1: If no filings were analyzed, state that in one sentence and move on — do not write speculative management commentary."]

    # --- Risks ---
    scores["risks"] = 6
    risk_issues = [
        "Fix 1: Rank risks from most to least impactful with specific transmission mechanisms.",
        "Fix 2: For each risk, name the leading indicator you would watch (e.g. 'China iPhone shipment data' not 'demand weakness').",
    ]

    # --- Catalysts ---
    scores["catalysts"] = 6
    catalyst_issues = [
        "Fix 1: Include 2-3 catalysts with rough timing (next quarter, next 6 months, next 12 months).",
        "Fix 2: Quantify the catalyst impact where possible (e.g. 'Services growing at 15% could add $X to revenue').",
    ]

    # --- Overall ---
    all_scores = [v for k, v in scores.items()]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 5.0

    # Build the review text
    sections = []
    section_data = [
        ("1. Executive Summary", exec_score, exec_issues),
        ("2. Business & Financial Performance", scores["business_performance"], biz_issues),
        ("3. Valuation", scores["valuation"], val_issues),
        ("4. Filings & Management Signal", scores["filings_signal"], filing_issues),
        ("5. Macro & Industry Context", scores["macro_context"], macro_issues),
        ("6. Risks", scores["risks"], risk_issues),
        ("7. Catalysts", scores["catalysts"], catalyst_issues),
        ("8. Verdict", scores["verdict"], verdict_issues),
    ]

    for title, score, fixes in section_data:
        section = f"## {title} [SCORE: {score}/10]\n"
        if fixes:
            section += "\n".join(fixes)
        sections.append(section)

    if gap_issue:
        sections.append(f"## Style Note\n{gap_issue}")

    sections.append(
        f"## Overall Assessment [SCORE: {overall}/10]\n\n"
        f"Word count: {word_count}. "
        f"{'Adequate length.' if word_count >= 800 else f'Too short at {word_count} words — target 900-1400.'}"
    )

    return "\n\n".join(sections)


# =============================================================================
# HELPERS
# =============================================================================

def _wrap(
    text: str,
    scores: Dict[str, Any],
    model_used: str,
    elapsed_ms: float,
    fallback: bool,
) -> Dict[str, Any]:
    return {
        "text": text,
        "scores": scores,
        "overall_score": scores.get("overall", 5.0),
        "model_used": model_used,
        "elapsed_ms": round(elapsed_ms, 1),
        "fallback": fallback,
        "word_count": len(text.split()),
    }