"""
Unit tests for agent.base_agent.

Scope
-----
- Mock mode is deterministic and self-contained.
- `generated_json` carries the expected thesis keys.
- The real-model code path (e.g. Ollama/HTTP) can be monkeypatched so no
  network, subprocess, or GPU is required.
- Malformed JSON from the model is degraded gracefully rather than crashing.
- `AgentResponse` carries valid timing and model metadata.
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------
@pytest.fixture
def base_agent(require_module):
    """Import base_agent and verify the public surface we rely on."""
    return require_module(
        "agent.base_agent",
        "AgentRequest",
        "AgentResponse",
        "ask_agent",
    )


@pytest.fixture
def basic_request(base_agent, sample_context):
    """Construct a minimal AgentRequest from the shared sample context."""
    return base_agent.AgentRequest(
        prompt="Produce a 1Y thesis for TEST.",
        context=sample_context,
        tools=None,
        model_tag="mock",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMockMode:
    def test_mock_mode_returns_agent_response_type(
        self, base_agent, basic_request, sample_config
    ):
        resp = base_agent.ask_agent(basic_request, sample_config)
        assert isinstance(resp, base_agent.AgentResponse)

    def test_mock_mode_returns_deterministic_json(
        self, base_agent, basic_request, sample_config
    ):
        a = base_agent.ask_agent(basic_request, sample_config)
        b = base_agent.ask_agent(basic_request, sample_config)
        assert a.generated_json == b.generated_json, (
            "Mock mode must produce identical JSON for identical inputs."
        )

    def test_mock_mode_generated_json_contains_required_keys(
        self, base_agent, basic_request, sample_config, thesis_contract
    ):
        resp = base_agent.ask_agent(basic_request, sample_config)
        missing = thesis_contract["required_keys"] - set(resp.generated_json.keys())
        assert not missing, (
            f"Mock JSON is missing required thesis keys: {missing}. "
            f"Got keys: {sorted(resp.generated_json.keys())}"
        )

    def test_mock_mode_confidence_in_unit_interval(
        self, base_agent, basic_request, sample_config
    ):
        resp = base_agent.ask_agent(basic_request, sample_config)
        conf = resp.generated_json.get("confidence")
        assert isinstance(conf, (int, float))
        assert 0.0 <= float(conf) <= 1.0


class TestResponseMetadata:
    def test_response_has_valid_timing(self, base_agent, basic_request, sample_config):
        resp = base_agent.ask_agent(basic_request, sample_config)
        assert isinstance(resp.elapsed_ms, (int, float))
        assert resp.elapsed_ms >= 0

    def test_response_has_model_identifier(
        self, base_agent, basic_request, sample_config
    ):
        resp = base_agent.ask_agent(basic_request, sample_config)
        assert isinstance(resp.model_used, str) and resp.model_used.strip()

    def test_response_token_counts_non_negative(
        self, base_agent, basic_request, sample_config
    ):
        resp = base_agent.ask_agent(basic_request, sample_config)
        assert isinstance(resp.tokens_in, int) and resp.tokens_in >= 0
        assert isinstance(resp.tokens_out, int) and resp.tokens_out >= 0


class TestRealModelBoundary:
    """
    The real-model path typically hits Ollama (HTTP) or a subprocess. We
    monkeypatch whichever low-level callable the module exposes so no
    network or model is ever touched.
    """

    def _candidate_patch_targets(self, base_agent):
        """Return names of likely low-level call points we can stub."""
        candidates = [
            "_call_ollama",
            "call_ollama",
            "_ollama_generate",
            "ollama_generate",
            "_call_model",
            "call_model",
            "_http_post",
        ]
        return [name for name in candidates if hasattr(base_agent, name)]

    def test_real_branch_can_be_monkeypatched(
        self,
        base_agent,
        basic_request,
        sample_config,
        monkeypatch,
        deterministic_agent_response,
    ):
        targets = self._candidate_patch_targets(base_agent)
        if not targets:
            pytest.skip(
                "base_agent exposes no known model-call hook to patch; "
                "real-branch test is a no-op here."
            )

        fake_raw = json.dumps(deterministic_agent_response)

        def fake_call(*_args, **_kwargs):
            return fake_raw

        for name in targets:
            monkeypatch.setattr(base_agent, name, fake_call)

        # Force the non-mock branch if the module honours a flag.
        live_cfg = sample_config
        for flag in ("USE_MOCK_AGENT", "MOCK_AGENT"):
            if hasattr(live_cfg, flag):
                setattr(live_cfg, flag, False)
        live_req = base_agent.AgentRequest(
            prompt=basic_request.prompt,
            context=basic_request.context,
            tools=None,
            model_tag="qwen3:30b",
        )

        resp = base_agent.ask_agent(live_req, live_cfg)
        assert isinstance(resp, base_agent.AgentResponse)
        assert resp.generated_json.get("outlook") in {"bullish", "neutral", "bearish"}


class TestMalformedJsonHandling:
    def test_handles_non_json_model_output(
        self, base_agent, basic_request, sample_config, monkeypatch
    ):
        """
        If the underlying model returns junk, ask_agent must not raise;
        it should return an AgentResponse with an empty-ish generated_json
        or a safe default shape.
        """
        candidates = [
            "_call_ollama",
            "call_ollama",
            "_ollama_generate",
            "ollama_generate",
            "_call_model",
            "call_model",
        ]
        patched_any = False
        for name in candidates:
            if hasattr(base_agent, name):
                monkeypatch.setattr(
                    base_agent,
                    name,
                    lambda *a, **k: "this is not JSON at all <<<>>>",
                )
                patched_any = True

        if not patched_any:
            pytest.skip("no real-model hook to patch for malformed-JSON test")

        # Force real branch if possible.
        for flag in ("USE_MOCK_AGENT", "MOCK_AGENT"):
            if hasattr(sample_config, flag):
                setattr(sample_config, flag, False)

        resp = base_agent.ask_agent(basic_request, sample_config)
        assert isinstance(resp, base_agent.AgentResponse)
        assert isinstance(resp.generated_json, dict)
        # raw_output should still capture what the model actually said
        assert isinstance(resp.raw_output, str)
