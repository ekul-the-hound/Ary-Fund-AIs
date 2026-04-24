"""ARY QUANT chat agent — conversational wrapper around the local LLM.

This module provides the streaming text-mode call path used by the chat
tab in the Streamlit dashboard. It is deliberately separate from
``agent.base_agent`` (which is JSON-mode only) and from
``agent.thesis_essay._call_ollama_text`` (which is non-streaming).

Design choices
--------------
- *Streaming*: the UI is much more responsive if tokens arrive as the
  model generates them. A 30B model on a partial-GPU setup can take
  30–90s for a chat turn; a progress-less spinner makes that feel
  broken, streaming makes it feel alive.
- *Grounded*: every turn's system prompt carries the full current
  context (essay + risk + thesis + key metrics + macro snapshot)
  so ARY QUANT stays consistent across turns without us needing to
  implement retrieval or tool-calling yet.
- *Per-ticker history*: switching tickers should feel like opening a
  fresh conversation. History lives keyed by ticker in session_state.
- *Fallback path*: when Ollama is unreachable or the resolved model
  is "mock", we return a canned response so the UI never dead-ends.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

ARY_QUANT_SYSTEM_PROMPT = """You are ARY QUANT, an AI research analyst built for \
a single-user hedge-fund-style research workstation. You speak in the voice of a \
thoughtful buy-side analyst: concise, evidence-anchored, and skeptical of \
consensus narratives. You never invent financials. When the user's question \
cannot be answered from the briefing or your general training, you say so \
explicitly rather than guess.

Conventions:
- Prefer short paragraphs and, when listing, use plain bullets (•) or numbered
  items. Avoid heavy Markdown headers mid-answer.
- Refer to figures in the briefing by name, not by fabricated precision.
- When the user asks "what if" or "should I", structure the answer as: what the
  data currently says, what would change that read, and what you'd want to see
  to update the view.
- You are discussing research, not providing personalized investment advice.
"""


def build_grounded_system_prompt(
    ticker: str,
    essay_text: str | None,
    context: dict[str, Any] | None,
) -> str:
    """Fold the current briefing into the system prompt so ARY QUANT answers
    are grounded in the same material the user is reading.

    The essay is the primary anchor. The risk/thesis/metrics/macro blocks
    are included as a structured appendix so ARY QUANT can quote exact
    figures without re-fetching data.
    """
    parts: list[str] = [ARY_QUANT_SYSTEM_PROMPT]

    parts.append(f"\n--- CURRENT TICKER: {ticker} ---")

    if essay_text:
        parts.append(
            "\n--- RESEARCH BRIEFING (this is the canonical written analysis the "
            "user is reading; anchor answers here) ---\n"
            + essay_text.strip()
        )

    if context:
        thesis = context.get("thesis") or {}
        risk = context.get("risk") or {}
        metrics = context.get("metrics") or context.get("key_metrics") or {}
        macro = context.get("macro") or {}
        filings = context.get("filings") or context.get("recent_events") or []

        structured = {
            "thesis": thesis,
            "risk": risk,
            "key_metrics": metrics,
            "macro": macro,
            "recent_filings": filings[:5] if isinstance(filings, list) else filings,
        }
        # Only include keys that have content.
        structured = {k: v for k, v in structured.items() if v}
        if structured:
            parts.append(
                "\n--- STRUCTURED CONTEXT (JSON; use for exact numbers) ---\n"
                + json.dumps(structured, default=str, indent=2)[:8000]
            )

    parts.append(
        "\n--- Now answer the user's questions as ARY QUANT, referring to the "
        "briefing above when relevant."
    )
    return "\n".join(parts)


def build_chat_prompt(
    system_prompt: str,
    history: list[dict[str, str]],
    user_message: str,
    max_turns: int = 10,
) -> str:
    """Render a conversation into a single prompt string for Ollama's
    ``/api/generate`` endpoint.

    We intentionally use ``/api/generate`` rather than ``/api/chat`` to
    stay consistent with the rest of the codebase (thesis_essay also
    uses /api/generate). The tradeoff is we do the turn-formatting
    ourselves, which is fine because Qwen3 handles plain role-labelled
    transcripts well.

    ``max_turns`` caps the tail of history included in the prompt. Older
    turns are dropped to keep the KV cache bounded — a ChatGPT-style
    implicit truncation.
    """
    lines: list[str] = [system_prompt, ""]

    # Keep only the most recent max_turns exchanges.
    trimmed = history[-(max_turns * 2):] if history else []
    for turn in trimmed:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"ARY QUANT: {content}")

    lines.append(f"User: {user_message}")
    lines.append("ARY QUANT:")
    return "\n".join(lines)


# =============================================================================
# STREAMING OLLAMA CLIENT
# =============================================================================

def stream_chat_response(
    prompt: str,
    config: Any,
    temperature: float = 0.4,
) -> Iterable[str]:
    """Yield response tokens from Ollama as they arrive.

    Mirrors the configuration choices in ``thesis_essay._call_ollama_text``
    but with ``stream=True``. Yields strings (token chunks). Raises on
    network / HTTP / decoding failure; the caller should wrap to fall
    back to a non-streaming error message.

    The temperature default (0.4) is lower than essay-mode (~0.7) because
    chat answers should be less florid and more anchored.
    """
    import urllib.request

    # Resolve the model the same way base_agent does so users get consistent
    # behavior across the JSON agent, essay writer, and chat.
    model_name = _resolve_chat_model(config)

    if model_name == "mock":
        # Short-circuit: yield a canned response so the UI still works in
        # environments where Ollama isn't running.
        yield (
            "[ARY QUANT is in mock mode — no LLM backend is reachable. "
            "Start Ollama and set DEFAULT_AGENT_MODEL in config.py to "
            "enable live responses.]"
        )
        return

    base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    # Chat turns should not inherit the essay's 180s minimum; 300s gives
    # Qwen3-30B room for thoughtful answers on a partial-GPU setup without
    # hanging the UI indefinitely on a hung call.
    timeout = float(getattr(config, "AGENT_TIMEOUT", 120))
    if timeout < 300:
        timeout = 300.0
    max_tokens = int(getattr(config, "MAX_TOKENS", 4096))

    body = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
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
        # Ollama's streaming endpoint returns newline-delimited JSON objects.
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Defensive: skip malformed partials rather than dying mid-stream.
                continue
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


def _resolve_chat_model(config: Any) -> str:
    """Pick which Ollama model to use for chat.

    Mirrors base_agent._resolve_model's precedence without importing it
    (that module is JSON-mode focused and we want independence). Checks:
        1. config.CHAT_MODEL — explicit chat override if present
        2. config.DEFAULT_AGENT_MODEL — main chain default
        3. "mock" — safe dead end
    """
    for attr in ("CHAT_MODEL", "DEFAULT_AGENT_MODEL"):
        model = getattr(config, attr, None)
        if model:
            return str(model)
    return "mock"
