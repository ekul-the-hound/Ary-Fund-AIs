"""
End-to-end smoke test.

Runs main() with every external boundary mocked, verifies the final
saved payload is structurally valid, and verifies determinism across
two identical runs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class SavedOpinionStore:
    """In-memory replacement for portfolio_db.save_agent_opinion."""

    def __init__(self):
        self.rows: list[tuple[str, dict]] = []

    def save(self, ticker, payload):
        # tolerate either positional or unexpected shapes
        if isinstance(ticker, dict) and not isinstance(payload, dict):
            payload, ticker = ticker, payload
        self.rows.append((ticker, dict(payload) if isinstance(payload, dict) else {"raw": payload}))


def _install_smoke_harness(
    monkeypatch,
    safe_import,
    sample_context,
    deterministic_agent_response,
) -> SavedOpinionStore:
    store = SavedOpinionStore()

    # 1. pipeline boundaries
    pipeline = safe_import("data.pipeline")
    if pipeline is not None:
        if hasattr(pipeline, "run_daily_refresh"):
            monkeypatch.setattr(
                pipeline, "run_daily_refresh", lambda *a, **k: None
            )
        if hasattr(pipeline, "build_agent_context"):
            monkeypatch.setattr(
                pipeline,
                "build_agent_context",
                lambda ticker, *a, **k: {**sample_context, "ticker": ticker},
            )

    # 2. main's local namespace, if any
    main_mod = safe_import("main")
    if main_mod is not None:
        if hasattr(main_mod, "run_daily_refresh"):
            monkeypatch.setattr(
                main_mod, "run_daily_refresh", lambda *a, **k: None
            )
        if hasattr(main_mod, "build_agent_context"):
            monkeypatch.setattr(
                main_mod,
                "build_agent_context",
                lambda ticker, *a, **k: {**sample_context, "ticker": ticker},
            )
        if hasattr(main_mod, "save_agent_opinion"):
            monkeypatch.setattr(main_mod, "save_agent_opinion", store.save)

    # 3. portfolio_db save boundary
    portfolio_db = safe_import("data.portfolio_db")
    if portfolio_db is not None and hasattr(portfolio_db, "save_agent_opinion"):
        monkeypatch.setattr(portfolio_db, "save_agent_opinion", store.save)

    # 4. LLM boundary
    base_agent = safe_import("agent.base_agent")
    if base_agent is not None and hasattr(base_agent, "ask_agent"):
        AgentResponse = getattr(base_agent, "AgentResponse", None)

        def fake_ask(request, config):
            payload = dict(deterministic_agent_response)
            if AgentResponse is not None:
                try:
                    return AgentResponse(
                        content=str(payload),
                        raw_output=str(payload),
                        generated_json=payload,
                        model_used="mock",
                        tokens_in=50,
                        tokens_out=120,
                        elapsed_ms=0.5,
                    )
                except TypeError:
                    pass
            return SimpleNamespace(
                content=str(payload),
                raw_output=str(payload),
                generated_json=payload,
                model_used="mock",
                tokens_in=50,
                tokens_out=120,
                elapsed_ms=0.5,
            )

        monkeypatch.setattr(base_agent, "ask_agent", fake_ask)
        if main_mod is not None and hasattr(main_mod, "ask_agent"):
            monkeypatch.setattr(main_mod, "ask_agent", fake_ask)

    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_full_workflow_runs_and_saves_valid_opinion(
    monkeypatch,
    safe_import,
    require_module,
    sample_context,
    sample_config,
    deterministic_agent_response,
    mock_db,
    thesis_contract,
):
    main_mod = require_module("main", "main")
    store = _install_smoke_harness(
        monkeypatch, safe_import, sample_context, deterministic_agent_response
    )

    main_mod.main(["SMOKE"], mock_db, sample_config)

    assert len(store.rows) == 1, (
        f"expected exactly one saved opinion, got {len(store.rows)}"
    )
    ticker, payload = store.rows[0]
    assert ticker == "SMOKE"
    assert isinstance(payload, dict)

    # The saved payload should carry the thesis contract either directly
    # or nested under a recognizable key.
    def _locate_thesis(p: dict) -> dict:
        if thesis_contract["required_keys"].issubset(p.keys()):
            return p
        for v in p.values():
            if isinstance(v, dict) and thesis_contract["required_keys"].issubset(v.keys()):
                return v
        return {}

    thesis = _locate_thesis(payload)
    assert thesis, (
        f"could not find thesis contract in saved payload: {payload!r}"
    )
    assert thesis["time_horizon"] == "1Y"
    assert 0.0 <= float(thesis["confidence"]) <= 1.0
    assert thesis["outlook"] in thesis_contract["valid_outlooks"]
    assert thesis["price_direction"] in thesis_contract["valid_price_directions"]


def test_workflow_is_deterministic_across_runs(
    monkeypatch,
    safe_import,
    require_module,
    sample_context,
    sample_config,
    deterministic_agent_response,
    mock_db,
):
    main_mod = require_module("main", "main")
    store = _install_smoke_harness(
        monkeypatch, safe_import, sample_context, deterministic_agent_response
    )

    main_mod.main(["SMOKE"], mock_db, sample_config)
    main_mod.main(["SMOKE"], mock_db, sample_config)

    assert len(store.rows) == 2
    (t1, p1), (t2, p2) = store.rows
    assert t1 == t2 == "SMOKE"

    # Normalize any volatile fields before comparison.
    def _normalize(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if k in {"timestamp", "created_at", "elapsed_ms", "run_id", "as_of"}:
                continue
            if isinstance(v, dict):
                out[k] = _normalize(v)
            else:
                out[k] = v
        return out

    assert _normalize(p1) == _normalize(p2), (
        "two runs with identical inputs produced different saved opinions"
    )


def test_workflow_survives_multiple_tickers(
    monkeypatch,
    safe_import,
    require_module,
    sample_context,
    sample_config,
    deterministic_agent_response,
    mock_db,
):
    main_mod = require_module("main", "main")
    store = _install_smoke_harness(
        monkeypatch, safe_import, sample_context, deterministic_agent_response
    )

    tickers = ["AAA", "BBB", "CCC"]
    main_mod.main(tickers, mock_db, sample_config)

    assert len(store.rows) == len(tickers)
    assert [t for t, _ in store.rows] == tickers
    for _, payload in store.rows:
        assert isinstance(payload, dict) and payload