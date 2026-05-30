"""
tests/test_essay_metrics.py
===========================
Verifies that ``thesis_essay.generate_thesis_essay`` records a telemetry
row through the observability layer for every essay it produces — mock,
live success, and live failure — without changing essay behaviour or
breaking when telemetry itself fails.

All tests point the metrics writer at a temp DB and stub the Ollama text
call, so no server and no real inference are needed.
"""
from __future__ import annotations

import os
import tempfile

import pytest

import config
from data import metrics_db
import agent.metrics as agent_metrics
import agent.thesis_essay as thesis_essay


@pytest.fixture
def tmp_metrics_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(path)
    # Route every record_metrics write to this temp DB.
    real_insert = metrics_db.insert_metric
    monkeypatch.setattr(
        agent_metrics.metrics_db, "insert_metric",
        lambda rec, db_path=None: real_insert(rec, db_path=path),
    )
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except (PermissionError, OSError):
        pass


def _args():
    return dict(
        ticker="AAPL",
        thesis={"outlook": "neutral", "direction": "flat"},
        filings_summary={},
        metrics={"name": "Apple Inc.", "ticker": "AAPL", "p_e": 37.7},
        macro={},
        risk_flags={},
    )


class _MockCfg:
    DEFAULT_AGENT_MODEL = "mock"
    AGENT_MODELS = {"mock": "mock"}


class TestEssayMetricRecording:
    def test_mock_mode_records_row(self, tmp_metrics_db):
        out = thesis_essay.generate_thesis_essay(config=_MockCfg, **_args())
        assert out["text"]  # essay still produced
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "thesis_essay"
        assert rows[0]["ticker"] == "AAPL"

    def test_live_success_records_success(self, tmp_metrics_db, monkeypatch):
        class Cfg:
            DEFAULT_AGENT_MODEL = "dev"
            AGENT_MODELS = {"dev": "phi3:3.8b"}
        # Stub the Ollama text call with a long-enough essay.
        monkeypatch.setattr(
            thesis_essay, "_call_ollama_text",
            lambda prompt, model_name, config: "word " * 400,
        )
        out = thesis_essay.generate_thesis_essay(config=Cfg, **_args())
        assert not out["fallback"]
        row = metrics_db.get_metrics(db_path=tmp_metrics_db)[0]
        assert row["agent_name"] == "thesis_essay"
        assert row["success"] == 1
        assert row["model"] == "phi3:3.8b"
        # token counts are estimated, so just assert they're populated
        assert row["prompt_tokens"] and row["completion_tokens"]

    def test_live_failure_records_failure(self, tmp_metrics_db, monkeypatch):
        class Cfg:
            DEFAULT_AGENT_MODEL = "dev"
            AGENT_MODELS = {"dev": "phi3:3.8b"}

        def _boom(prompt, model_name, config):
            raise ConnectionError("no ollama")

        monkeypatch.setattr(thesis_essay, "_call_ollama_text", _boom)
        out = thesis_essay.generate_thesis_essay(config=Cfg, **_args())
        assert out["fallback"]  # deterministic fallback used
        row = metrics_db.get_metrics(db_path=tmp_metrics_db)[0]
        assert row["agent_name"] == "thesis_essay"
        assert row["success"] == 0
        assert row["model"].endswith("(failed)")

    def test_telemetry_failure_does_not_break_essay(self, tmp_metrics_db, monkeypatch):
        # If record_metrics raises, the essay must still be returned.
        monkeypatch.setattr(
            agent_metrics, "record_metrics",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
        )
        out = thesis_essay.generate_thesis_essay(config=_MockCfg, **_args())
        assert out["text"]
        assert "word_count" in out
