"""
Agent Metrics
=============
Non-invasive observability wrapper around ``base_agent.ask_agent``.

Design philosophy: **measure without touching.**
---------------------------------------------------
``ask_agent`` is the system's single LLM chokepoint and the most
heavily-tested function in the agent layer. Rather than thread
``agent_name`` / ``ticker`` / ``run_id`` kwargs *into* it — which would
mean changing the ``AgentRequest`` contract and the 7-field positional
``AgentResponse`` constructors that existing tests rely on — this module
wraps it from the outside.

``ask_agent`` already returns everything telemetry needs:
``model_used``, ``tokens_in``, ``tokens_out``, ``elapsed_ms``, and a
failure signal (a ``" (failed)"`` suffix on ``model_used`` when the
backend errored and the safe default was returned). So the wrapper
reads those fields off the real response, computes cost, and persists a
row — adding zero risk to the call path itself.

Two entry points:

* ``instrumented_ask(request, config, *, agent_name=, ticker=, run_id=)``
  — drop-in replacement for ``ask_agent``. Returns the *identical*
  ``AgentResponse`` ``ask_agent`` produced. Callers that want telemetry
  swap ``base_agent.ask_agent(...)`` → ``metrics.instrumented_ask(...)``.

* ``record_metrics(call_metadata)`` — low-level escape hatch for paths
  that don't go through ``ask_agent`` (e.g. ``thesis_essay``'s separate
  ``_call_ollama_text``, or a future tool-calling loop). You hand it a
  dict; it computes ``total_tokens`` / ``cost_usd`` if absent and
  persists.

Both are best-effort: a telemetry failure logs a warning and is
swallowed. The measured call's result is never affected.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agent import base_agent
from agent.base_agent import AgentRequest, AgentResponse
from data import metrics_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
# Local Ollama inference has no per-token API fee, so "cost" here is a
# notional figure for capacity planning / cloud-equivalent comparison.
# Default: $0.0001 per 1K tokens (i.e. $1e-7 per token), overridable via
# config.METRICS_COST_PER_1K_TOKENS. Set it to a real provider's price
# to model what this workload *would* cost on a hosted API.

_DEFAULT_COST_PER_1K = 0.0001


def _cost_per_1k(config: Any) -> float:
    if config is None:
        return _DEFAULT_COST_PER_1K
    return float(getattr(config, "METRICS_COST_PER_1K_TOKENS", _DEFAULT_COST_PER_1K))


def compute_cost(total_tokens: int, config: Any = None) -> float:
    """Notional USD cost for ``total_tokens`` at the configured rate."""
    if not total_tokens:
        return 0.0
    return (float(total_tokens) / 1000.0) * _cost_per_1k(config)


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------
# base_agent signals a degraded call by suffixing model_used with
# " (failed)" and returning the safe-default JSON. We treat that as the
# canonical success/failure signal so we don't have to change ask_agent
# to return an explicit flag.

def _infer_success(response: AgentResponse) -> tuple[bool, Optional[str]]:
    model = response.model_used or ""
    if model.endswith("(failed)"):
        return False, "backend error; safe default returned"
    return True, None


# ---------------------------------------------------------------------------
# Low-level recorder
# ---------------------------------------------------------------------------

def record_metrics(
    call_metadata: dict,
    config: Any = None,
    db_path: Optional[str] = None,
) -> int:
    """Persist one telemetry record. Returns the row id (or -1 on failure).

    ``call_metadata`` may contain any subset of the metric columns. This
    function fills two derived fields when absent:

    * ``total_tokens`` = prompt_tokens + completion_tokens
    * ``cost_usd``     = compute_cost(total_tokens, config)

    Never raises — telemetry is best-effort.
    """
    md = dict(call_metadata)

    pt = md.get("prompt_tokens") or 0
    ct = md.get("completion_tokens") or 0
    if md.get("total_tokens") is None:
        md["total_tokens"] = int(pt) + int(ct)
    if md.get("cost_usd") is None:
        md["cost_usd"] = compute_cost(md["total_tokens"], config)

    return metrics_db.insert_metric(md, db_path=db_path)


# ---------------------------------------------------------------------------
# Drop-in instrumented wrapper
# ---------------------------------------------------------------------------

def instrumented_ask(
    request: AgentRequest,
    config: Any,
    *,
    agent_name: Optional[str] = None,
    ticker: Optional[str] = None,
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> AgentResponse:
    """Call ``ask_agent`` and record telemetry. Returns its exact response.

    This is the function call sites should use when they want a labeled,
    persisted metric for the call. It is a pure superset of
    ``ask_agent``: same positional args, same return value, plus
    keyword-only labels that default to ``None`` (so an unlabeled call
    still records a row, just bucketed as "(unlabeled)").

    The telemetry write happens *after* the response is in hand and is
    wrapped so it can never disturb the returned result.
    """
    response = base_agent.ask_agent(request, config)

    try:
        success, err = _infer_success(response)
        record_metrics(
            {
                "agent_name": agent_name,
                "model": response.model_used,
                "ticker": ticker,
                "prompt_tokens": response.tokens_in,
                "completion_tokens": response.tokens_out,
                "latency_ms": response.elapsed_ms,
                "success": success,
                "error_message": err,
                "run_id": run_id,
            },
            config=config,
            db_path=db_path,
        )
    except Exception as e:  # noqa: BLE001 — never let telemetry break the call
        logger.warning("instrumented_ask telemetry failed (non-fatal): %s", e)

    return response
