"""
Integration tests for main.main.

We monkeypatch every boundary so no real data is fetched, no real model
is invoked, and no real DB is written to.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def main_mod(require_module):
    return require_module("main", "main")


# ---------------------------------------------------------------------------
# Recorder fixtures
# ---------------------------------------------------------------------------
class CallRecorder:
    """Simple callable that records its invocations."""

    def __init__(self, return_value=None, side_effect=None):
        self.calls: list[tuple[tuple, dict]] = []
        self._return = return_value
        self._side_effect = side_effect

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._side_effect is not None:
            return self._side_effect(*args, **kwargs)
        return self._return

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def patch_boundaries(
    monkeypatch,
    safe_import,
    sample_context,
    deterministic_agent_response,
):
    """
    Patch every external boundary main.py touches and hand the test a
    handle to inspect what happened.
    """
    handles = SimpleNamespace(
        refresh=CallRecorder(return_value=None),
        build_context=CallRecorder(return_value=sample_context),
        ask_agent=None,
        save_opinion=CallRecorder(return_value=None),
    )

    # data.pipeline.run_daily_refresh + build_agent_context
    pipeline = safe_import("data.pipeline")
    if pipeline is not None:
        if hasattr(pipeline, "run_daily_refresh"):
            monkeypatch.setattr(pipeline, "run_daily_refresh", handles.refresh)
        if hasattr(pipeline, "build_agent_context"):
            monkeypatch.setattr(
                pipeline, "build_agent_context", handles.build_context
            )

    # Also patch the names as they may have been imported into main's namespace.
    main_mod = safe_import("main")
    if main_mod is not None:
        for name, target in (
            ("run_daily_refresh", handles.refresh),
            ("build_agent_context", handles.build_context),
            ("save_agent_opinion", handles.save_opinion),
        ):
            if hasattr(main_mod, name):
                monkeypatch.setattr(main_mod, name, target)

    # portfolio_db.save_agent_opinion
    portfolio_db = safe_import("data.portfolio_db")
    if portfolio_db is not None and hasattr(portfolio_db, "save_agent_opinion"):
        monkeypatch.setattr(
            portfolio_db, "save_agent_opinion", handles.save_opinion
        )

    # agent.base_agent.ask_agent → return a fake AgentResponse-like object
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
                        tokens_in=100,
                        tokens_out=200,
                        elapsed_ms=1.0,
                    )
                except TypeError:
                    # dataclass signature differs — fall through to namespace
                    pass
            return SimpleNamespace(
                content=str(payload),
                raw_output=str(payload),
                generated_json=payload,
                model_used="mock",
                tokens_in=100,
                tokens_out=200,
                elapsed_ms=1.0,
            )

        handles.ask_agent = CallRecorder(side_effect=fake_ask)
        monkeypatch.setattr(base_agent, "ask_agent", handles.ask_agent)
        # And in main's namespace if imported there.
        if main_mod is not None and hasattr(main_mod, "ask_agent"):
            monkeypatch.setattr(main_mod, "ask_agent", handles.ask_agent)

    return handles


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMainOrchestration:
    def test_main_calls_daily_refresh(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        main_mod.main(["TEST"], mock_db, sample_config)
        assert patch_boundaries.refresh.call_count >= 1, (
            "main() should call run_daily_refresh at least once"
        )

    def test_main_builds_context_once_per_ticker(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        tickers = ["AAA", "BBB", "CCC"]
        main_mod.main(tickers, mock_db, sample_config)
        assert patch_boundaries.build_context.call_count == len(tickers), (
            f"expected build_agent_context called {len(tickers)} times, "
            f"saw {patch_boundaries.build_context.call_count}"
        )

    def test_main_invokes_agent_per_ticker(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        tickers = ["AAA", "BBB"]
        main_mod.main(tickers, mock_db, sample_config)
        if patch_boundaries.ask_agent is None:
            pytest.skip("ask_agent boundary not patchable in this build")
        assert patch_boundaries.ask_agent.call_count == len(tickers)

    def test_main_saves_opinion_per_ticker(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        tickers = ["AAA", "BBB"]
        main_mod.main(tickers, mock_db, sample_config)
        assert patch_boundaries.save_opinion.call_count == len(tickers)

    def test_main_handles_single_ticker(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        main_mod.main(["ONLY"], mock_db, sample_config)
        assert patch_boundaries.build_context.call_count == 1
        assert patch_boundaries.save_opinion.call_count == 1

    def test_main_does_not_crash_with_empty_list(
        self, main_mod, patch_boundaries, sample_config, mock_db
    ):
        # Empty ticker list: refresh may or may not be called, but no agent
        # work should be dispatched and no exception should propagate.
        main_mod.main([], mock_db, sample_config)
        assert patch_boundaries.build_context.call_count == 0
        assert patch_boundaries.save_opinion.call_count == 0

    def test_main_passes_context_through_to_save(
        self,
        main_mod,
        patch_boundaries,
        sample_config,
        mock_db,
    ):
        main_mod.main(["TEST"], mock_db, sample_config)
        assert patch_boundaries.save_opinion.call_count == 1
        args, kwargs = patch_boundaries.save_opinion.calls[0]
        # signature is (ticker, agent_output) per the spec
        combined = list(args) + list(kwargs.values())
        assert any(a == "TEST" for a in combined), (
            f"ticker 'TEST' not passed to save_agent_opinion; got {combined!r}"
        )
        # the saved payload should be a dict containing thesis-ish keys
        dict_args = [a for a in combined if isinstance(a, dict)]
        assert dict_args, "save_agent_opinion received no dict payload"
        payload = dict_args[0]
        assert any(
            k in payload for k in ("outlook", "thesis", "price_direction")
        ), f"saved payload lacks thesis fields: {payload!r}"
