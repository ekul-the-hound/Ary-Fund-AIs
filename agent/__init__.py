"""
agent: LLM-powered research agents.

Exposes the core agent contract (`AgentRequest`, `AgentResponse`, `ask_agent`)
so callers can do:

    from agent import ask_agent, AgentRequest

Re-exports are wrapped in a try/except so the package remains importable
during partial refactors or when optional dependencies are missing.
"""

from __future__ import annotations

__all__ = ["AgentRequest", "AgentResponse", "ask_agent"]

try:
    from agent.base_agent import AgentRequest, AgentResponse, ask_agent
except ImportError:  # pragma: no cover - defensive for partial builds
    AgentRequest = None  # type: ignore[assignment]
    AgentResponse = None  # type: ignore[assignment]
    ask_agent = None  # type: ignore[assignment]
