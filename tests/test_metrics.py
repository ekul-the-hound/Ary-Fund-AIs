"""
tests/test_metrics.py
=====================
Tests for the additive observability layer:

* data/metrics_db.py  — schema, insert, reads, percentile math
* agent/metrics.py     — cost model, success inference, record_metrics,
                         instrumented_ask wrapper

All tests use a temp metrics DB (via the ``db_path`` kwarg) so nothing
touches the real metrics.db. The instrumented_ask tests monkeypatch the
backend shim so no Ollama server is required.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from data import metrics_db
from agent import metrics as agent_metrics
from agent.base_agent import AgentRequest, AgentResponse
import agent.base_agent as base_agent_mod


@pytest.fixture
def tmp_metrics_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(path)
    yield path
    # On Windows, SQLite can leave a brief lock on the file even after
    # connections close, so os.unlink raises PermissionError (WinError 32)
    # during teardown. The temp file is harmless and the OS reaps it
    # later; a cleanup failure must not fail an otherwise-passing test.
    try:
        if os.path.exists(path):
            os.unlink(path)
    except (PermissionError, OSError):
        pass


def _iso_days_ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(
        timespec="seconds"
    )


# ---------------------------------------------------------------------------
# metrics_db: schema + insert
# ---------------------------------------------------------------------------

class TestInsertAndGet:
    def test_insert_returns_id(self, tmp_metrics_db):
        rid = metrics_db.insert_metric(
            {
                "agent_name": "thesis_generator",
                "model": "qwen3",
                "ticker": "NVDA",
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "latency_ms": 1234.5,
                "success": True,
                "cost_usd": 0.00003,
            },
            db_path=tmp_metrics_db,
        )
        assert rid >= 1

    def test_get_metrics_roundtrip(self, tmp_metrics_db):
        metrics_db.insert_metric(
            {"agent_name": "a", "ticker": "AAPL", "total_tokens": 10,
             "latency_ms": 5.0, "success": True},
            db_path=tmp_metrics_db,
        )
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "a"
        assert rows[0]["ticker"] == "AAPL"
        # success stored as int 1
        assert rows[0]["success"] == 1

    def test_unknown_keys_ignored(self, tmp_metrics_db):
        # Extra keys must not crash the insert.
        rid = metrics_db.insert_metric(
            {"agent_name": "a", "total_tokens": 1, "bogus_field": "x"},
            db_path=tmp_metrics_db,
        )
        assert rid >= 1

    def test_timestamp_autofilled(self, tmp_metrics_db):
        metrics_db.insert_metric({"agent_name": "a"}, db_path=tmp_metrics_db)
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert rows[0]["timestamp"]  # non-empty

    def test_filter_by_agent(self, tmp_metrics_db):
        metrics_db.insert_metric({"agent_name": "a", "total_tokens": 1},
                                 db_path=tmp_metrics_db)
        metrics_db.insert_metric({"agent_name": "b", "total_tokens": 1},
                                 db_path=tmp_metrics_db)
        rows = metrics_db.get_metrics(agent_name="b", db_path=tmp_metrics_db)
        assert len(rows) == 1 and rows[0]["agent_name"] == "b"

    def test_since_days_window_excludes_old(self, tmp_metrics_db):
        # One fresh, one 30 days old; default window is 7 days.
        metrics_db.insert_metric({"agent_name": "fresh", "total_tokens": 1},
                                 db_path=tmp_metrics_db)
        metrics_db.insert_metric(
            {"agent_name": "old", "total_tokens": 1,
             "timestamp": _iso_days_ago(30)},
            db_path=tmp_metrics_db,
        )
        rows = metrics_db.get_metrics(since_days=7, db_path=tmp_metrics_db)
        names = {r["agent_name"] for r in rows}
        assert "fresh" in names and "old" not in names


# ---------------------------------------------------------------------------
# metrics_db: aggregates
# ---------------------------------------------------------------------------

class TestAggregates:
    def test_token_spent_sums(self, tmp_metrics_db):
        for t in (100, 250, 50):
            metrics_db.insert_metric({"total_tokens": t}, db_path=tmp_metrics_db)
        assert metrics_db.get_token_spent_since_days(7, tmp_metrics_db) == 400.0

    def test_token_spent_empty_is_zero(self, tmp_metrics_db):
        assert metrics_db.get_token_spent_since_days(7, tmp_metrics_db) == 0.0

    def test_cost_spent_sums(self, tmp_metrics_db):
        metrics_db.insert_metric({"cost_usd": 0.01}, db_path=tmp_metrics_db)
        metrics_db.insert_metric({"cost_usd": 0.02}, db_path=tmp_metrics_db)
        assert metrics_db.get_cost_spent_since_days(7, tmp_metrics_db) == pytest.approx(0.03)

    def test_success_rate_by_agent(self, tmp_metrics_db):
        metrics_db.insert_metric({"agent_name": "x", "success": True},
                                 db_path=tmp_metrics_db)
        metrics_db.insert_metric({"agent_name": "x", "success": False},
                                 db_path=tmp_metrics_db)
        metrics_db.insert_metric({"agent_name": "x", "success": True},
                                 db_path=tmp_metrics_db)
        stats = metrics_db.get_success_rate_by_agent(7, tmp_metrics_db)
        assert stats["x"]["total"] == 3
        assert stats["x"]["ok"] == 2
        assert stats["x"]["rate"] == pytest.approx(2 / 3)

    def test_unlabeled_bucketed(self, tmp_metrics_db):
        metrics_db.insert_metric({"success": True}, db_path=tmp_metrics_db)
        stats = metrics_db.get_success_rate_by_agent(7, tmp_metrics_db)
        assert "(unlabeled)" in stats


# ---------------------------------------------------------------------------
# metrics_db: latency percentiles
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_empty_returns_nones(self, tmp_metrics_db):
        s = metrics_db.get_latency_stats(db_path=tmp_metrics_db)
        assert s["count"] == 0
        assert s["p50"] is None and s["p99"] is None

    def test_percentiles_nearest_rank(self, tmp_metrics_db):
        # 1..100 ms; nearest-rank percentiles should land on real values.
        for ms in range(1, 101):
            metrics_db.insert_metric({"latency_ms": float(ms)},
                                     db_path=tmp_metrics_db)
        s = metrics_db.get_latency_stats(db_path=tmp_metrics_db)
        assert s["count"] == 100
        assert s["p50"] == pytest.approx(50.0)
        assert s["p90"] == pytest.approx(90.0)
        assert s["p99"] == pytest.approx(99.0)
        assert s["mean"] == pytest.approx(50.5)

    def test_single_value(self, tmp_metrics_db):
        metrics_db.insert_metric({"latency_ms": 42.0}, db_path=tmp_metrics_db)
        s = metrics_db.get_latency_stats(db_path=tmp_metrics_db)
        assert s["p50"] == 42.0 and s["p99"] == 42.0


# ---------------------------------------------------------------------------
# agent/metrics.py: cost model + success inference
# ---------------------------------------------------------------------------

class TestCostModel:
    def test_default_rate(self):
        # 1000 tokens at $0.0001/1K = $0.0001
        assert agent_metrics.compute_cost(1000, None) == pytest.approx(0.0001)

    def test_zero_tokens(self):
        assert agent_metrics.compute_cost(0, None) == 0.0

    def test_config_override(self):
        class Cfg:
            METRICS_COST_PER_1K_TOKENS = 0.01
        # 2000 tokens at $0.01/1K = $0.02
        assert agent_metrics.compute_cost(2000, Cfg) == pytest.approx(0.02)


class TestSuccessInference:
    def _resp(self, model):
        return AgentResponse(
            content="", raw_output="", generated_json={},
            model_used=model, tokens_in=1, tokens_out=1, elapsed_ms=1.0,
        )

    def test_ok(self):
        ok, err = agent_metrics._infer_success(self._resp("qwen3"))
        assert ok is True and err is None

    def test_failed_suffix(self):
        ok, err = agent_metrics._infer_success(self._resp("qwen3 (failed)"))
        assert ok is False and err


# ---------------------------------------------------------------------------
# agent/metrics.py: record_metrics derives totals + cost
# ---------------------------------------------------------------------------

class TestRecordMetrics:
    def test_derives_total_and_cost(self, tmp_metrics_db):
        rid = agent_metrics.record_metrics(
            {"agent_name": "z", "prompt_tokens": 400, "completion_tokens": 600},
            config=None,
            db_path=tmp_metrics_db,
        )
        assert rid >= 1
        row = metrics_db.get_metrics(db_path=tmp_metrics_db)[0]
        assert row["total_tokens"] == 1000
        # default cost: 1000 tokens → $0.0001
        assert row["cost_usd"] == pytest.approx(0.0001)

    def test_records_all_fields(self, tmp_metrics_db):
        agent_metrics.record_metrics(
            {
                "agent_name": "thesis_generator", "model": "qwen3",
                "ticker": "NVDA", "prompt_tokens": 10, "completion_tokens": 20,
                "latency_ms": 99.9, "success": True, "run_id": "run-1",
            },
            db_path=tmp_metrics_db,
        )
        row = metrics_db.get_metrics(db_path=tmp_metrics_db)[0]
        assert row["agent_name"] == "thesis_generator"
        assert row["model"] == "qwen3"
        assert row["ticker"] == "NVDA"
        assert row["latency_ms"] == pytest.approx(99.9)
        assert row["run_id"] == "run-1"
        assert row["success"] == 1


# ---------------------------------------------------------------------------
# agent/metrics.py: instrumented_ask wrapper
# ---------------------------------------------------------------------------

class TestInstrumentedAsk:
    def _req(self):
        return AgentRequest(prompt="analyze NVDA", context={}, tools=[])

    def test_returns_identical_response_mock(self, tmp_metrics_db, monkeypatch):
        # Force mock path so no Ollama is needed.
        class Cfg:
            DEFAULT_AGENT_MODEL = "mock"
            AGENT_MODELS = {"mock": "mock"}
        resp = agent_metrics.instrumented_ask(
            self._req(), Cfg, agent_name="thesis_generator", ticker="NVDA",
            db_path=tmp_metrics_db,
        )
        # Response is a normal mock AgentResponse
        assert isinstance(resp, AgentResponse)
        assert resp.model_used == "mock"
        assert resp.generated_json  # mock payload present

    def test_records_a_row(self, tmp_metrics_db):
        class Cfg:
            DEFAULT_AGENT_MODEL = "mock"
            AGENT_MODELS = {"mock": "mock"}
        agent_metrics.instrumented_ask(
            self._req(), Cfg, agent_name="thesis_generator", ticker="NVDA",
            db_path=tmp_metrics_db,
        )
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert len(rows) == 1
        assert rows[0]["agent_name"] == "thesis_generator"
        assert rows[0]["ticker"] == "NVDA"
        assert rows[0]["success"] == 1
        assert rows[0]["total_tokens"] >= 0

    def test_unlabeled_call_still_records(self, tmp_metrics_db):
        class Cfg:
            DEFAULT_AGENT_MODEL = "mock"
            AGENT_MODELS = {"mock": "mock"}
        agent_metrics.instrumented_ask(
            self._req(), Cfg, db_path=tmp_metrics_db,
        )
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert len(rows) == 1
        assert rows[0]["agent_name"] is None

    def test_failed_backend_recorded_as_failure(self, tmp_metrics_db, monkeypatch):
        # Real path, but _call_ollama raises → base_agent returns the
        # "(failed)" safe-default response → wrapper records success=0.
        class Cfg:
            DEFAULT_AGENT_MODEL = "dev"
            AGENT_MODELS = {"dev": "phi3:3.8b"}
            AGENT_TIMEOUT = 5
            MAX_TOKENS = 256
            OLLAMA_BASE_URL = "http://localhost:11434"

        def _boom(*a, **kw):
            raise ConnectionError("no ollama")

        monkeypatch.setattr(base_agent_mod, "_call_ollama", _boom)
        resp = agent_metrics.instrumented_ask(
            self._req(), Cfg, agent_name="thesis_generator", ticker="NVDA",
            db_path=tmp_metrics_db,
        )
        assert resp.model_used.endswith("(failed)")
        row = metrics_db.get_metrics(db_path=tmp_metrics_db)[0]
        assert row["success"] == 0
        assert row["error_message"]

    def test_telemetry_failure_does_not_break_call(self, tmp_metrics_db, monkeypatch):
        # If the DB write blows up, instrumented_ask must still return the
        # response. Point insert at a path that can't be created.
        class Cfg:
            DEFAULT_AGENT_MODEL = "mock"
            AGENT_MODELS = {"mock": "mock"}

        def _explode(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(metrics_db, "insert_metric", _explode)
        resp = agent_metrics.instrumented_ask(
            self._req(), Cfg, agent_name="x", db_path=tmp_metrics_db,
        )
        assert isinstance(resp, AgentResponse)
        assert resp.model_used == "mock"


# ---------------------------------------------------------------------------
# Regression: existing ask_agent path is untouched
# ---------------------------------------------------------------------------

class TestNoRegression:
    def test_plain_ask_agent_still_works(self):
        from agent.base_agent import ask_agent

        class Cfg:
            DEFAULT_AGENT_MODEL = "mock"
            AGENT_MODELS = {"mock": "mock"}
        resp = ask_agent(
            AgentRequest(prompt="p", context={}, tools=[]), Cfg,
        )
        # Unchanged 7-field shape, all present.
        assert isinstance(resp, AgentResponse)
        assert isinstance(resp.tokens_in, int)
        assert isinstance(resp.tokens_out, int)
        assert isinstance(resp.elapsed_ms, float)
        assert resp.model_used == "mock"