"""
tests/test_portfolio_db_phase4.py
==================================
Coverage for the three Phase 4 RAG-learning-loop hooks added to
data.portfolio_db.PortfolioDB:

    - get_recently_closed_theses(since_days, limit=None)
    - get_thesis_by_id(thesis_id)
    - get_pnl_for_thesis(thesis)

Plus the two new write-side helpers that back them:

    - record_thesis(...)
    - close_thesis(thesis_id, exit_price, ...)

Plus the bonus accessor that fills the latent gap in
rag/document_loaders/theses.py:ThesesLoader:

    - get_thesis_history(ticker, limit)

Plus an integration test that pipes the hook output into the *actual*
rag.learning.scorer.score_thesis (pure Python, no LLM/ChromaDB
dependency) — proves the canonical contract works end-to-end.

Tests use a temporary on-disk SQLite DB per test (tmp_path fixture)
rather than ``:memory:`` because PortfolioDB opens new connections
per method call, which would each see a fresh empty DB on
``:memory:``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from data.portfolio_db import PortfolioDB


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def db(tmp_path: Path) -> PortfolioDB:
    """Fresh PortfolioDB on a temp file. New DB per test → no
    cross-test pollution."""
    return PortfolioDB(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def db_with_benchmark(tmp_path: Path) -> PortfolioDB:
    """Same, but with a benchmark lookup hook wired in. Returns a fixed
    +4% to make assertions deterministic."""
    def _bench(ticker, start, end):
        return 0.04
    return PortfolioDB(
        db_path=str(tmp_path / "test_bench.db"),
        benchmark_lookup_fn=_bench,
    )


def _iso_days_ago(n: int) -> str:
    """Return an ISO timestamp N days ago (UTC, no tz suffix — matches
    sqlite's datetime() default format)."""
    return (datetime.now() - timedelta(days=n)).isoformat(timespec="seconds")


# ======================================================================
# get_recently_closed_theses
# ======================================================================
class TestGetRecentlyClosedTheses:

    def test_returns_empty_list_when_no_theses(self, db: PortfolioDB):
        out = db.get_recently_closed_theses(since_days=7)
        assert out == []
        assert isinstance(out, list)

    def test_returns_empty_list_when_no_closed_theses(self, db: PortfolioDB):
        # Record three open theses — none closed.
        db.record_thesis("AAPL", thesis_text="t1", score=0.8)
        db.record_thesis("MSFT", thesis_text="t2", score=0.7)
        db.record_thesis("NVDA", thesis_text="t3", score=0.9)
        assert db.get_recently_closed_theses(since_days=30) == []

    def test_returns_only_closed_theses(self, db: PortfolioDB):
        t1 = db.record_thesis("AAPL", thesis_text="open", score=0.8,
                              entry_price=170.0, shares=10)
        t2 = db.record_thesis("MSFT", thesis_text="closed", score=0.7,
                              entry_price=380.0, shares=5)
        db.close_thesis(t2, exit_price=420.0)
        out = db.get_recently_closed_theses(since_days=30)
        assert len(out) == 1
        assert out[0]["id"] == t2
        assert out[0]["ticker"] == "MSFT"
        # Untouched open thesis is invisible
        assert all(r["id"] != t1 for r in out)

    def test_filters_by_since_days_window(self, db: PortfolioDB):
        # 3 theses, closed at varying ages.
        for ticker, age in [("OLD", 30), ("MID", 5), ("NEW", 1)]:
            tid = db.record_thesis(
                ticker, thesis_text=f"{ticker} thesis",
                entry_price=100.0, shares=1,
            )
            db.close_thesis(
                tid, exit_price=110.0,
                closed_at=_iso_days_ago(age),
            )
        # 7-day window catches MID + NEW only
        out = db.get_recently_closed_theses(since_days=7)
        tickers = {r["ticker"] for r in out}
        assert tickers == {"MID", "NEW"}

        # 60-day window catches all 3
        out = db.get_recently_closed_theses(since_days=60)
        assert {r["ticker"] for r in out} == {"OLD", "MID", "NEW"}

        # 0-day window catches nothing (cutoff = now)
        out = db.get_recently_closed_theses(since_days=0)
        assert out == []

    def test_ordered_by_closed_at_desc(self, db: PortfolioDB):
        # Insert in non-chronological order; close in known order.
        ages = [3, 1, 5, 2]  # NEW=1 should come first, OLDEST=5 last
        for i, age in enumerate(ages):
            tid = db.record_thesis(
                f"T{i}", thesis_text=f"t{i}",
                entry_price=100.0, shares=1,
            )
            db.close_thesis(tid, exit_price=105.0,
                            closed_at=_iso_days_ago(age))
        out = db.get_recently_closed_theses(since_days=30)
        closed_ats = [r["closed_at"] for r in out]
        # Strictly descending
        assert closed_ats == sorted(closed_ats, reverse=True)

    def test_limit_caps_result_count(self, db: PortfolioDB):
        # Create 5 closed theses.
        for i in range(5):
            tid = db.record_thesis(
                f"T{i}", thesis_text=f"t{i}",
                entry_price=100.0, shares=1,
            )
            db.close_thesis(tid, exit_price=105.0,
                            closed_at=_iso_days_ago(i))
        assert len(db.get_recently_closed_theses(since_days=30, limit=2)) == 2
        assert len(db.get_recently_closed_theses(since_days=30, limit=10)) == 5
        assert len(db.get_recently_closed_theses(since_days=30, limit=0)) == 0

    def test_returns_canonical_shape(self, db: PortfolioDB):
        tid = db.record_thesis(
            "AAPL",
            thesis_text="long aapl on services growth",
            essay_text="...full essay...",
            score=0.82,
            stance="bull",
            author="thesis_generator",
            model="qwen3:30b-a3b",
            entry_price=170.0,
            shares=10,
            thesis_note_path="data/fund_notes/aapl_2026q1.md",
            metadata={"sector": "Technology"},
        )
        db.close_thesis(tid, exit_price=195.0)

        out = db.get_recently_closed_theses(since_days=30)
        assert len(out) == 1
        row = out[0]
        # Canonical fields the integration_snippets.py + scorer.py + loop.py expect
        required = {
            "id", "ticker", "author", "model", "stance", "score",
            "thesis_text", "essay_text", "created_at", "closed_at",
            "outcome", "thesis_note_path",
        }
        missing = required - set(row.keys())
        assert not missing, f"missing canonical fields: {missing}"
        # Spot-check values
        assert row["ticker"] == "AAPL"
        assert row["stance"] == "bull"
        assert row["score"] == pytest.approx(0.82)
        assert row["author"] == "thesis_generator"
        assert row["model"] == "qwen3:30b-a3b"
        assert row["outcome"] == "win"  # exit > entry, long stance
        assert row["metadata"] == {"sector": "Technology"}

    def test_rejects_negative_since_days(self, db: PortfolioDB):
        with pytest.raises(ValueError):
            db.get_recently_closed_theses(since_days=-1)

    def test_rejects_negative_limit(self, db: PortfolioDB):
        with pytest.raises(ValueError):
            db.get_recently_closed_theses(since_days=7, limit=-1)


# ======================================================================
# get_thesis_by_id
# ======================================================================
class TestGetThesisById:

    def test_returns_none_for_missing_id(self, db: PortfolioDB):
        assert db.get_thesis_by_id(999) is None

    def test_returns_none_for_none_input(self, db: PortfolioDB):
        assert db.get_thesis_by_id(None) is None

    def test_returns_none_for_unparseable_id(self, db: PortfolioDB):
        assert db.get_thesis_by_id("not_a_number") is None

    def test_returns_row_for_valid_int_id(self, db: PortfolioDB):
        tid = db.record_thesis("AAPL", thesis_text="t", score=0.8)
        out = db.get_thesis_by_id(tid)
        assert out is not None
        assert out["id"] == tid
        assert out["ticker"] == "AAPL"

    def test_accepts_str_id(self, db: PortfolioDB):
        # loop.py and curator.py sometimes pass str(thesis['id'])
        tid = db.record_thesis("AAPL", thesis_text="t", score=0.8)
        out = db.get_thesis_by_id(str(tid))
        assert out is not None
        assert out["id"] == tid

    def test_returns_full_shape_including_metadata(self, db: PortfolioDB):
        tid = db.record_thesis(
            "AAPL",
            thesis_text="short thesis",
            essay_text="long essay",
            score=0.75,
            stance="neutral",
            metadata={"sector": "Tech", "confidence": "HIGH"},
        )
        out = db.get_thesis_by_id(tid)
        assert out["thesis_text"] == "short thesis"
        assert out["essay_text"] == "long essay"
        assert out["stance"] == "neutral"
        assert out["metadata"] == {"sector": "Tech", "confidence": "HIGH"}


# ======================================================================
# get_pnl_for_thesis
# ======================================================================
class TestGetPnlForThesis:

    def test_returns_none_for_open_thesis(self, db: PortfolioDB):
        tid = db.record_thesis("AAPL", thesis_text="t",
                               entry_price=170.0, shares=10)
        thesis = db.get_thesis_by_id(tid)
        # Not closed yet
        assert db.get_pnl_for_thesis(thesis) is None

    def test_returns_none_when_thesis_is_none(self, db: PortfolioDB):
        assert db.get_pnl_for_thesis(None) is None

    def test_returns_none_when_id_missing(self, db: PortfolioDB):
        assert db.get_pnl_for_thesis({"ticker": "AAPL"}) is None

    def test_returns_none_when_thesis_not_found(self, db: PortfolioDB):
        assert db.get_pnl_for_thesis({"id": 99999, "ticker": "X"}) is None

    def test_returns_none_when_prices_missing(self, db: PortfolioDB):
        # Closed but no entry/exit price.
        tid = db.record_thesis("AAPL", thesis_text="t")
        db.close_thesis(tid)  # no exit_price given
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert out is None

    def test_long_thesis_winning_return_pct(self, db: PortfolioDB):
        # Entry 100 → Exit 120 → +20% for a long
        tid = db.record_thesis(
            "AAPL", thesis_text="long",
            entry_price=100.0, shares=10, stance="bull",
            created_at=_iso_days_ago(180),
        )
        db.close_thesis(tid, exit_price=120.0, closed_at=_iso_days_ago(0))
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert out is not None
        assert out["return_pct"] == pytest.approx(0.20)
        assert out["days_held"] == 180
        assert out["benchmark_return_pct"] is None  # no hook wired

    def test_short_thesis_inverts_sign(self, db: PortfolioDB):
        # Bear stance: entry 100 → exit 80 → short made +20%
        tid = db.record_thesis(
            "BADCO", thesis_text="short",
            entry_price=100.0, shares=10, stance="bear",
            created_at=_iso_days_ago(90),
        )
        db.close_thesis(tid, exit_price=80.0, closed_at=_iso_days_ago(0))
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert out["return_pct"] == pytest.approx(0.20)
        assert out["days_held"] == 90

    def test_loss_returns_negative_pct(self, db: PortfolioDB):
        tid = db.record_thesis(
            "BADTRADE", thesis_text="loser",
            entry_price=100.0, shares=10, stance="bull",
            created_at=_iso_days_ago(60),
        )
        db.close_thesis(tid, exit_price=85.0, closed_at=_iso_days_ago(0))
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert out["return_pct"] == pytest.approx(-0.15)
        assert out["days_held"] == 60

    def test_days_held_floored_at_1(self, db: PortfolioDB):
        # Same-day close — scorer's annualizer needs days_held >= 1
        now = datetime.now().isoformat(timespec="seconds")
        tid = db.record_thesis(
            "DAYTRADE", thesis_text="fast",
            entry_price=100.0, shares=1, stance="bull",
            created_at=now,
        )
        db.close_thesis(tid, exit_price=101.0, closed_at=now)
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert out["days_held"] >= 1

    def test_benchmark_lookup_populates_field(self, db_with_benchmark):
        # Hook returns 0.04 always
        tid = db_with_benchmark.record_thesis(
            "AAPL", thesis_text="t",
            entry_price=100.0, shares=10, stance="bull",
            created_at=_iso_days_ago(30),
        )
        db_with_benchmark.close_thesis(
            tid, exit_price=110.0, closed_at=_iso_days_ago(0)
        )
        out = db_with_benchmark.get_pnl_for_thesis(
            db_with_benchmark.get_thesis_by_id(tid)
        )
        assert out["benchmark_return_pct"] == pytest.approx(0.04)

    def test_benchmark_lookup_exception_falls_back_to_none(
        self, tmp_path: Path
    ):
        def _bad_bench(*_args, **_kw):
            raise RuntimeError("market data down")
        db = PortfolioDB(
            db_path=str(tmp_path / "test.db"),
            benchmark_lookup_fn=_bad_bench,
        )
        tid = db.record_thesis("AAPL", thesis_text="t",
                               entry_price=100.0, shares=1, stance="bull")
        db.close_thesis(tid, exit_price=110.0)
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        # The thesis hook should still succeed; benchmark gracefully None
        assert out is not None
        assert out["benchmark_return_pct"] is None

    def test_canonical_shape_keys_exact(self, db: PortfolioDB):
        # Shape must EXACTLY match the scorer.py contract or it
        # silently misreads on the .get(...) lines
        tid = db.record_thesis(
            "AAPL", thesis_text="t",
            entry_price=100.0, shares=1, stance="bull",
        )
        db.close_thesis(tid, exit_price=110.0)
        out = db.get_pnl_for_thesis(db.get_thesis_by_id(tid))
        assert set(out.keys()) == {
            "return_pct", "days_held", "benchmark_return_pct"
        }
        assert isinstance(out["return_pct"], float)
        assert isinstance(out["days_held"], int)
        assert out["benchmark_return_pct"] is None or \
            isinstance(out["benchmark_return_pct"], float)


# ======================================================================
# record_thesis + close_thesis (the write side)
# ======================================================================
class TestThesisWriteOps:

    def test_record_thesis_returns_int_id(self, db: PortfolioDB):
        tid = db.record_thesis("AAPL", thesis_text="t")
        assert isinstance(tid, int)
        assert tid >= 1

    def test_record_thesis_normalises_ticker_case(self, db: PortfolioDB):
        tid = db.record_thesis("aapl", thesis_text="t")
        assert db.get_thesis_by_id(tid)["ticker"] == "AAPL"

    def test_record_thesis_rejects_bad_stance(self, db: PortfolioDB):
        with pytest.raises(ValueError):
            db.record_thesis("AAPL", thesis_text="t", stance="screaming-buy")

    def test_record_thesis_rejects_out_of_range_score(self, db: PortfolioDB):
        with pytest.raises(ValueError):
            db.record_thesis("AAPL", thesis_text="t", score=1.5)
        with pytest.raises(ValueError):
            db.record_thesis("AAPL", thesis_text="t", score=-0.01)

    def test_close_thesis_raises_for_unknown_id(self, db: PortfolioDB):
        with pytest.raises(ValueError):
            db.close_thesis(99999, exit_price=100.0)

    def test_close_thesis_auto_computes_outcome(self, db: PortfolioDB):
        # Long thesis, exit > entry → win
        tid = db.record_thesis(
            "AAPL", thesis_text="t",
            entry_price=100.0, shares=1, stance="bull",
        )
        row = db.close_thesis(tid, exit_price=110.0)
        assert row["outcome"] == "win"

        # Long thesis, exit < entry → loss
        tid2 = db.record_thesis(
            "BAD", thesis_text="t",
            entry_price=100.0, shares=1, stance="bull",
        )
        row2 = db.close_thesis(tid2, exit_price=90.0)
        assert row2["outcome"] == "loss"

        # Short thesis, exit < entry → win (sign flipped)
        tid3 = db.record_thesis(
            "BEAR", thesis_text="t",
            entry_price=100.0, shares=1, stance="bear",
        )
        row3 = db.close_thesis(tid3, exit_price=80.0)
        assert row3["outcome"] == "win"

    def test_close_thesis_is_idempotent(self, db: PortfolioDB):
        tid = db.record_thesis(
            "AAPL", thesis_text="t",
            entry_price=100.0, shares=1, stance="bull",
            created_at=_iso_days_ago(30),
        )
        original = db.close_thesis(
            tid, exit_price=110.0, closed_at=_iso_days_ago(7)
        )
        # Re-close with a different timestamp — closed_at stays put
        second = db.close_thesis(tid, exit_price=115.0)
        assert second["closed_at"] == original["closed_at"]
        # But exit_price updates (idempotent on close timestamp, not on data)
        assert second["exit_price"] == pytest.approx(115.0)


# ======================================================================
# get_thesis_history (bonus accessor for ThesesLoader)
# ======================================================================
class TestGetThesisHistory:

    def test_returns_empty_for_unknown_ticker(self, db: PortfolioDB):
        assert db.get_thesis_history("UNKNOWN") == []

    def test_returns_all_rows_for_ticker(self, db: PortfolioDB):
        db.record_thesis("AAPL", thesis_text="t1")
        db.record_thesis("AAPL", thesis_text="t2")
        db.record_thesis("MSFT", thesis_text="other")
        out = db.get_thesis_history("AAPL")
        assert len(out) == 2
        assert all(r["ticker"] == "AAPL" for r in out)

    def test_ordered_by_created_at_desc(self, db: PortfolioDB):
        db.record_thesis("AAPL", thesis_text="oldest",
                         created_at=_iso_days_ago(30))
        db.record_thesis("AAPL", thesis_text="newest",
                         created_at=_iso_days_ago(1))
        db.record_thesis("AAPL", thesis_text="middle",
                         created_at=_iso_days_ago(10))
        out = db.get_thesis_history("AAPL")
        assert [r["thesis_text"] for r in out] == ["newest", "middle", "oldest"]

    def test_normalises_ticker_case(self, db: PortfolioDB):
        db.record_thesis("AAPL", thesis_text="t")
        assert len(db.get_thesis_history("aapl")) == 1

    def test_limit_caps_count(self, db: PortfolioDB):
        for i in range(5):
            db.record_thesis("AAPL", thesis_text=f"t{i}")
        assert len(db.get_thesis_history("AAPL", limit=2)) == 2


# ======================================================================
# Backward compatibility
# ======================================================================
class TestBackwardCompat:
    """Smoke-test that existing PortfolioDB API still works after the
    upgrade."""

    def test_default_construction_still_works(self, tmp_path: Path):
        # Single-arg constructor must still work
        db = PortfolioDB(db_path=str(tmp_path / "compat.db"))
        assert db.get_cash() > 0  # default 100k seeded

    def test_existing_methods_untouched(self, db: PortfolioDB):
        # NOTE: add_position has a pre-existing nested-connection bug
        # (it opens a sqlite3.connect then calls record_trade which
        # opens another → "database is locked"). That bug exists on
        # the original portfolio_db.py and is OUT OF SCOPE for Phase 4.
        # We verify backward compat through methods not affected by it.
        db.record_trade("AAPL", "BUY", shares=10, price=170.0)
        trades = db.get_trade_history(limit=10)
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"
        assert trades[0]["ticker"] == "AAPL"

        # Watchlist round-trip
        db.add_to_watchlist("NVDA", target_entry=850.0, priority="HIGH")
        watch = db.get_watchlist()
        assert len(watch) == 1
        assert watch[0]["ticker"] == "NVDA"

        # Cash management still works
        original_cash = db.get_cash()
        db.deposit(5000.0)
        assert db.get_cash() == original_cash + 5000.0

    def test_module_level_save_agent_opinion_still_works(
        self, tmp_path: Path
    ):
        from data.portfolio_db import save_agent_opinion
        path = str(tmp_path / "opinions.db")
        rid = save_agent_opinion("AAPL", {"stance": "bull"}, db_path=path)
        assert rid >= 1


# ======================================================================
# Integration test — pipes the hook output through the *real* scorer
# ======================================================================
class TestLearningLoopIntegration:
    """The crucial regression: the three Phase 4 hooks must produce
    output that ``rag.learning.scorer.score_thesis`` accepts without
    error. That function is the first consumer in the learning loop
    chain (loop.py:108 → curator.py:256 → scorer.score_thesis), so if
    its contract holds, the AttributeError cascade documented in
    integration_snippets.py is fixed.

    We use the *actual* scorer module (pure Python, no LLM deps),
    not a mock. If the contract breaks, this test breaks.
    """

    def test_closed_thesis_scores_with_outcome(self, db: PortfolioDB):
        from rag.learning.scorer import score_thesis, QualityScore

        # Set up: one closed thesis with a known winning P&L
        tid = db.record_thesis(
            "AAPL",
            thesis_text="long aapl on services growth",
            essay_text="full essay body here",
            score=0.85,
            stance="bull",
            author="thesis_generator",
            entry_price=100.0,
            shares=10,
            created_at=_iso_days_ago(180),
        )
        db.close_thesis(tid, exit_price=125.0, closed_at=_iso_days_ago(0))

        # Walk the same call path the scheduler does
        closed = db.get_recently_closed_theses(since_days=30)
        assert len(closed) == 1
        thesis = closed[0]

        pnl = db.get_pnl_for_thesis(thesis)
        assert pnl is not None
        assert pnl["return_pct"] == pytest.approx(0.25)

        # The actual scorer must accept this without error
        score = score_thesis(thesis, realized_pnl=pnl)
        assert isinstance(score, QualityScore)
        assert score.outcome is not None  # P&L was provided → outcome populated
        # Big win on a thesis with good review score → high composite
        assert score.composite > 0.5
        # No "no_realized_pnl" warning when P&L was supplied
        assert "no_realized_pnl" not in score.warnings

    def test_open_thesis_path_routes_to_no_pnl_warning(
        self, db: PortfolioDB
    ):
        """get_pnl_for_thesis returns None for open positions; the
        scorer then attaches no_realized_pnl warning; the curator
        then blocks indexing. Verify the wiring."""
        from rag.learning.scorer import score_thesis

        tid = db.record_thesis(
            "OPEN", thesis_text="t", essay_text="e",
            score=0.9, entry_price=100.0, shares=1,
            created_at=_iso_days_ago(30),
        )
        # Not closed — so this thesis won't appear in
        # get_recently_closed_theses. Fetch it directly via get_thesis_by_id.
        thesis = db.get_thesis_by_id(tid)
        pnl = db.get_pnl_for_thesis(thesis)
        assert pnl is None  # open position → no P&L

        score = score_thesis(thesis, realized_pnl=pnl)
        assert "no_realized_pnl" in score.warnings
        assert score.outcome is None

    def test_full_loop_walk_no_attribute_errors(self, db: PortfolioDB):
        """Simulate the body of refresh_scheduler._run_learning_loop
        (integration_snippets.py:349-357) — just the DB-touching
        portion. The whole point of these hooks is that this walks
        without AttributeError."""
        # Seed two closed theses
        for i in range(2):
            tid = db.record_thesis(
                f"T{i}", thesis_text=f"t{i}",
                entry_price=100.0, shares=1, stance="bull",
                created_at=_iso_days_ago(60),
            )
            db.close_thesis(tid, exit_price=105.0,
                            closed_at=_iso_days_ago(1))

        # This is the body of _run_learning_loop, minus the LLM-side
        # indexer. If any of these three calls raised AttributeError
        # before this patch, the scheduler would have crashed.
        closed = db.get_recently_closed_theses(since_days=7)
        for thesis in closed:
            pnl = db.get_pnl_for_thesis(thesis)
            # Re-fetch via the auditor's path too
            roundtrip = db.get_thesis_by_id(thesis["id"])
            assert roundtrip["id"] == thesis["id"]
            assert pnl is not None
            # Confirm shape one more time, end-to-end
            assert set(pnl.keys()) == {
                "return_pct", "days_held", "benchmark_return_pct"
            }
