"""
agent/base_agent.py
===================

Model-agnostic agent core for the hedge-fund AI research system.

This module is the **only** place that talks to an LLM backend. Every other
agent module (``filing_analyzer``, ``risk_scanner``, ``thesis_generator``,
``main``) speaks to the LLM through ``ask_agent`` and never imports Ollama,
HTTP clients, or a specific model tag directly.

Swapping backends
-----------------
The backend is selected in three steps:

    1. ``AgentRequest.model_tag`` — per-call override (e.g. force ``"mock"``
       in unit tests).
    2. ``config.DEFAULT_AGENT_MODEL`` — project-wide default.
    3. ``config.AGENT_MODELS`` — maps a tag ("mock", "dev", "prod") to an
       actual Ollama model string.

If the resolved tag is ``"mock"``, or Ollama is unreachable, ``ask_agent``
returns a deterministic JSON response so the full pipeline can run offline.

Response contract
-----------------
Every call returns an :class:`AgentResponse` whose ``generated_json`` contains
at least these keys:

    - ``risks``            : List[str]
    - ``thesis``           : str   (e.g. "NEUTRAL 1Y")
    - ``price_direction``  : str   (e.g. "slight_up", "neutral")
    - ``confidence``       : float in [0.0, 1.0]

Callers downstream (``risk_scanner``, ``thesis_generator``) rely on those
four keys being present. Mock mode, real Ollama success, and Ollama failure
all produce the same shape.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentRequest:
    """A single inference request to the agent layer.

    Attributes
    ----------
    prompt:
        The fully-rendered prompt string. ``main.py`` constructs this via
        ``build_agent_prompt(...)`` from the pipeline context.
    context:
        The structured dict returned by ``pipeline.build_agent_context(...)``.
        Kept alongside the prompt so the backend (or future tool-calling
        agents) can access raw data without re-parsing the prompt.
    tools:
        Optional list of tool names this request is permitted to use
        (e.g. ``["filings", "prices", "macro"]``). Not enforced by the mock
        path; reserved for a future tool-calling implementation.
    model_tag:
        Per-call override. If ``None``, ``config.DEFAULT_AGENT_MODEL`` wins.
        Set to ``"mock"`` to force deterministic output regardless of config.
    """

    prompt: str
    context: Dict[str, Any]
    tools: Optional[List[str]] = field(default_factory=list)
    model_tag: Optional[str] = None


@dataclass
class AgentResponse:
    """Structured response from the agent layer.

    ``generated_json`` is the parsed, validated payload downstream consumers
    should use. ``content`` and ``raw_output`` are kept for logging and
    debugging — they may be identical for simple backends.
    """

    content: str
    raw_output: str
    generated_json: Dict[str, Any]
    model_used: str
    tokens_in: int
    tokens_out: int
    elapsed_ms: float


# =============================================================================
# CONSTANTS
# =============================================================================

# Default payload returned when the backend fails or JSON parsing breaks.
# Downstream code (risk_scanner, thesis_generator) depends on these keys
# existing, so every code path must return at least this shape.
_SAFE_DEFAULT_JSON: Dict[str, Any] = {
    "risks": [],
    "thesis": "NEUTRAL 1Y",
    "price_direction": "neutral",
    "confidence": 0.5,
}

# Deterministic mock payload. Matches the contract in the project spec.
_MOCK_JSON: Dict[str, Any] = {
    "risks": ["HIGH: debt", "MEDIUM: macro"],
    "thesis": "NEUTRAL 1Y",
    "price_direction": "slight_up",
    "confidence": 0.6,
    # Extra keys that downstream consumers (thesis_generator) look for.
    "outlook": "neutral",
    "time_horizon": "1Y",
    "key_risks": ["HIGH: debt", "MEDIUM: macro"],
    "key_opportunities": ["margin expansion"],
    "summary": "Mock response for deterministic pipeline runs.",
}


# =============================================================================
# PUBLIC API
# =============================================================================

def ask_agent(request: AgentRequest, config: Any) -> AgentResponse:
    """Dispatch an agent request to mock or real backend and return a response.

    Parameters
    ----------
    request:
        The :class:`AgentRequest` to process.
    config:
        The project's ``config`` module (or any object exposing the same
        attributes: ``AGENT_MODELS``, ``DEFAULT_AGENT_MODEL``, ``AGENT_TIMEOUT``,
        ``MAX_TOKENS``, ``OLLAMA_BASE_URL``).

    Returns
    -------
    AgentResponse
        Always returns a valid response. On backend failure or parse error,
        ``generated_json`` falls back to :data:`_SAFE_DEFAULT_JSON` and the
        caller can continue processing.
    """
    started_at = time.perf_counter()
    model_used = _resolve_model(request, config)

    logger.info(
        "agent.ask | model=%s | prompt_len=%d | tools=%s",
        model_used,
        len(request.prompt),
        request.tools or [],
    )

    # Mock path: deterministic, no I/O, no external deps.
    if model_used == "mock":
        response = _mock_response(request, model_used, started_at)
        logger.info(
            "agent.done | mock=True | model=%s | elapsed_ms=%.1f",
            model_used,
            response.elapsed_ms,
        )
        return response

    # Real path: Ollama. Wrapped so any failure degrades to the safe default
    # rather than taking down the whole pipeline.
    try:
        response = _call_ollama(request, model_used, config, started_at)
        logger.info(
            "agent.done | mock=False | model=%s | elapsed_ms=%.1f | "
            "tokens_in=%d | tokens_out=%d",
            model_used,
            response.elapsed_ms,
            response.tokens_in,
            response.tokens_out,
        )
        return response
    except Exception as exc:  # noqa: BLE001 — we intentionally catch all
        logger.warning(
            "agent.fail | model=%s | err=%s | falling back to safe default",
            model_used,
            exc,
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return AgentResponse(
            content="",
            raw_output="",
            generated_json=dict(_SAFE_DEFAULT_JSON),
            model_used=f"{model_used} (failed)",
            tokens_in=_estimate_tokens(request.prompt),
            tokens_out=0,
            elapsed_ms=elapsed_ms,
        )


# =============================================================================
# INTERNAL: MODEL RESOLUTION
# =============================================================================

def _resolve_model(request: AgentRequest, config: Any) -> str:
    """Resolve ``request.model_tag`` -> actual Ollama model name.

    Resolution order:
        1. ``request.model_tag`` if supplied.
        2. ``config.DEFAULT_AGENT_MODEL`` otherwise.

    The chosen tag is then looked up in ``config.AGENT_MODELS``. Unknown tags
    and any tag resolving to "mock" return ``"mock"``.
    """
    tag = request.model_tag or getattr(config, "DEFAULT_AGENT_MODEL", "mock")
    models = getattr(config, "AGENT_MODELS", {}) or {}

    if tag not in models:
        logger.warning(
            "agent.resolve | unknown tag=%r; defaulting to mock. "
            "Known tags: %s",
            tag,
            list(models.keys()),
        )
        return "mock"

    resolved = models[tag]
    # "mock" tag or a model string literally equal to "mock" both short-circuit.
    if resolved == "mock":
        return "mock"
    return resolved


# =============================================================================
# INTERNAL: MOCK BACKEND
# =============================================================================

def _mock_response(
    request: AgentRequest,
    model_used: str,
    started_at: float,
) -> AgentResponse:
    """Build a deterministic mock :class:`AgentResponse`.

    The payload is fixed (see :data:`_MOCK_JSON`) so tests and pipeline runs
    are reproducible. Timing fields reflect actual wall-clock elapsed so
    logging still works the same way.
    """
    payload = dict(_MOCK_JSON)  # defensive copy so callers can mutate freely
    raw = json.dumps(payload, indent=2, sort_keys=True)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    return AgentResponse(
        content=raw,
        raw_output=raw,
        generated_json=payload,
        model_used="mock",
        tokens_in=_estimate_tokens(request.prompt),
        tokens_out=_estimate_tokens(raw),
        elapsed_ms=elapsed_ms,
    )


# =============================================================================
# INTERNAL: OLLAMA BACKEND
# =============================================================================

def _call_ollama(
    request: AgentRequest,
    model_name: str,
    config: Any,
    started_at: float,
) -> AgentResponse:
    """Call a local Ollama server via its ``/api/generate`` endpoint.

    This is kept as a thin, synchronous shim so it's easy to unit-test and
    easy to swap (e.g. for ``ollama-python`` or a remote inference server).

    ``urllib`` is used instead of ``requests`` to avoid adding a dependency;
    swap it in if the project already depends on ``requests``.

    Raises
    ------
    Exception
        Any network, HTTP, or decoding error bubbles up to ``ask_agent``,
        which downgrades it to the safe default response.
    """
    # Imports are local so test environments without these deps can still
    # import the module and use mock mode.
    import urllib.error
    import urllib.request

    base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = float(getattr(config, "AGENT_TIMEOUT", 30))
    max_tokens = int(getattr(config, "MAX_TOKENS", 4096))

    body = {
        "model": model_name,
        "prompt": request.prompt,
        "stream": False,
        # Ollama passes options through to the underlying runtime. ``num_predict``
        # caps generated tokens; matches our MAX_TOKENS budget.
        "options": {
            "num_predict": max_tokens,
            # Deterministic-ish output; raise temperature if you want variety.
            "temperature": 0.2,
        },
        # Ask Ollama to format as JSON where the model supports it. The model
        # may still wrap output in prose — _safe_parse_json handles that.
        "format": "json",
    }

    req = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    raw_output: str = payload.get("response", "") or ""
    generated_json = _safe_parse_json(raw_output)

    # Ollama returns eval counts when available; fall back to estimates.
    tokens_in = int(payload.get("prompt_eval_count") or _estimate_tokens(request.prompt))
    tokens_out = int(payload.get("eval_count") or _estimate_tokens(raw_output))
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    return AgentResponse(
        content=raw_output,
        raw_output=raw_output,
        generated_json=generated_json,
        model_used=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        elapsed_ms=elapsed_ms,
    )


# =============================================================================
# INTERNAL: PARSING HELPERS
# =============================================================================

# Strips ```json ... ``` or ``` ... ``` fences the model may add even when
# we ask for raw JSON.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    """Parse an LLM JSON blob defensively.

    Strategy, in order:
        1. Direct ``json.loads``.
        2. Strip markdown code fences, retry.
        3. Extract the first ``{...}`` substring, retry.
        4. Return :data:`_SAFE_DEFAULT_JSON`.

    The returned dict is then **shape-enforced**: any missing required key
    (``risks``, ``thesis``, ``price_direction``, ``confidence``) is filled
    from the safe default so downstream consumers never KeyError.
    """
    result: Optional[Dict[str, Any]] = None
    text = (raw or "").strip()

    if not text:
        return dict(_SAFE_DEFAULT_JSON)

    # Attempt 1: straight parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            result = parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip code fences.
    if result is None:
        stripped = _FENCE_RE.sub("", text).strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                result = parsed
        except json.JSONDecodeError:
            pass

    # Attempt 3: first brace-balanced substring.
    if result is None:
        match = _extract_first_json_object(text)
        if match is not None:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    result = parsed
            except json.JSONDecodeError:
                pass

    if result is None:
        logger.warning("agent.parse | could not parse JSON; using safe default")
        return dict(_SAFE_DEFAULT_JSON)

    # Shape enforcement: fill any missing required keys.
    for key, default in _SAFE_DEFAULT_JSON.items():
        result.setdefault(key, default)

    return result


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` substring in ``text``, or ``None``.

    Naive brace-matcher — adequate for LLM output wrapped in prose. Does not
    handle strings containing literal unescaped braces, but the LLM's ``format``
    option plus the other parse attempts make that edge case rare.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token.

    Used only when the backend does not report real token counts. Good enough
    for logging and budget checks; not a replacement for a real tokenizer.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)