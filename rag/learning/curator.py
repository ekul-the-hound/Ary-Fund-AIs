"""
rag/learning/curator.py
=======================
Gatekeeper for what gets indexed back into the RAG corpus.

Why a separate curator
----------------------
Phase 1-3 ingested external prose (filings, transcripts, fund notes)
where the assumption was "we trust the source, just index it."
Phase 4 ingests the agent's OWN prior reasoning, which doesn't have
that trust property. A thesis is only worth indexing if:

    1. It scored well on the composite quality metric
    2. Its outcome confirmed the review (no review-only signal)
    3. It doesn't push the corpus toward author or topic monoculture
    4. It hasn't aged past relevance

The curator enforces all four. ``decide_indexable`` is the single
question a caller asks: "should I index this thesis?"

The four guards in detail
-------------------------

**Guard 1: composite score threshold.** Default 0.7. Filters out
roughly 90% of generated theses. The right threshold depends on
how stringent your reviewer is — measure it empirically and tune.

**Guard 2: P&L confirmation.** Theses on open positions are
excluded entirely. The "no_realized_pnl" warning from the scorer
blocks indexing. This is the most important guard because it
breaks the reinforcement loop where review-only signals could
compound into corpus drift.

**Guard 3: diversity quotas.** Per-author and per-ticker caps on
total indexed thesis count. If the cap is 30% per author, then
even if author X writes the 10 best theses, only the top 3-of-10
(rounded down) get indexed. The cap is enforced at index time;
re-evaluation can demote already-indexed items when newer ones
push the author over quota.

**Guard 4: re-evaluation.** Every N days, walk all indexed theses,
re-compute composite scores (age has decayed), demote anything
below the demotion threshold. Demotion removes the chunks from
the vector store but keeps the tracking row marked
``demoted_at`` so we don't re-evaluate it endlessly.

A note on the demotion threshold
--------------------------------
Demote at composite < 0.5, not at <0.7. Hysteresis prevents
flickering — a thesis right at the 0.7 boundary shouldn't get
indexed today and demoted tomorrow because age decay shaved 0.001.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rag.learning.scorer import QualityScore, score_thesis

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tunable thresholds
# ----------------------------------------------------------------------

# Composite score required to be indexed (or remain indexed after
# re-evaluation). High enough to filter most theses; low enough
# that a meaningful number pass.
INDEX_THRESHOLD = 0.7

# Below this, currently-indexed theses get demoted. Gap to
# INDEX_THRESHOLD provides hysteresis against flickering.
DEMOTE_THRESHOLD = 0.5

# Diversity quotas. ``max_per_author_pct=0.3`` means no single
# author may contribute >30% of indexed theses.
MAX_PER_AUTHOR_PCT = 0.3
MAX_PER_TICKER_PCT = 0.25

# Below this number of total indexed theses, quotas don't apply
# (otherwise the first thesis hits 100% per-author by definition).
QUOTA_MIN_CORPUS = 10


# ----------------------------------------------------------------------
# Decision dataclass
# ----------------------------------------------------------------------

@dataclass
class CurationDecision:
    """Output of curator.decide_indexable for one thesis."""
    thesis_id: int | str
    should_index: bool
    quality: QualityScore
    block_reasons: list[str]   # populated only when should_index=False

    @property
    def explanation(self) -> str:
        if self.should_index:
            return f"INDEX (composite={self.quality.composite:.3f})"
        return f"BLOCK ({', '.join(self.block_reasons)})"


# ----------------------------------------------------------------------
# Curator class
# ----------------------------------------------------------------------

class Curator:
    """Gatekeeper for self-indexing.

    Parameters
    ----------
    tracking_db_path:
        Same SQLite as the indexer uses. We add a small table
        ``learning_indexed_theses`` here to track per-thesis state
        across re-evaluation cycles.
    index_threshold, demote_threshold:
        See module docstring.
    max_per_author_pct, max_per_ticker_pct:
        Diversity caps in [0, 1]. Set to 1.0 to disable a quota.
    """

    def __init__(
        self,
        tracking_db_path: str = "data/rag_tracking.db",
        index_threshold: float = INDEX_THRESHOLD,
        demote_threshold: float = DEMOTE_THRESHOLD,
        max_per_author_pct: float = MAX_PER_AUTHOR_PCT,
        max_per_ticker_pct: float = MAX_PER_TICKER_PCT,
        quota_min_corpus: int = QUOTA_MIN_CORPUS,
    ):
        self.tracking_db_path = tracking_db_path
        self.index_threshold = index_threshold
        self.demote_threshold = demote_threshold
        self.max_per_author_pct = max_per_author_pct
        self.max_per_ticker_pct = max_per_ticker_pct
        self.quota_min_corpus = quota_min_corpus
        Path(tracking_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # State table
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_indexed_theses (
                    thesis_id        TEXT PRIMARY KEY,
                    doc_id           TEXT NOT NULL,
                    ticker           TEXT,
                    author           TEXT,
                    composite_score  REAL NOT NULL,
                    review_score     REAL,
                    outcome_score    REAL,
                    indexed_at       TEXT NOT NULL,
                    last_evaluated_at TEXT NOT NULL,
                    demoted_at       TEXT,
                    demoted_reason   TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learn_ticker "
                "ON learning_indexed_theses(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learn_author "
                "ON learning_indexed_theses(author)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_learn_demoted "
                "ON learning_indexed_theses(demoted_at)"
            )

    # ------------------------------------------------------------------
    # Snapshot of current state — used by quota checks
    # ------------------------------------------------------------------
    def _current_distribution(self) -> tuple[int, Counter, Counter]:
        """Return (n_active, by_author, by_ticker) for active theses."""
        with sqlite3.connect(self.tracking_db_path) as conn:
            rows = conn.execute("""
                SELECT author, ticker FROM learning_indexed_theses
                WHERE demoted_at IS NULL
            """).fetchall()
        n = len(rows)
        by_author: Counter[str] = Counter()
        by_ticker: Counter[str] = Counter()
        for author, ticker in rows:
            if author:
                by_author[author] += 1
            if ticker:
                by_ticker[ticker] += 1
        return n, by_author, by_ticker

    def _quota_blocks(
        self,
        thesis: dict,
    ) -> list[str]:
        """Check whether indexing this thesis would breach quotas.

        Returns a list of block reasons (empty if no breaches).
        """
        n, by_author, by_ticker = self._current_distribution()

        # Quotas only kick in once the corpus is large enough that
        # percentages are meaningful.
        if n < self.quota_min_corpus:
            return []

        author = thesis.get("author") or thesis.get("model")
        ticker = (thesis.get("ticker") or "").upper()
        blocks: list[str] = []

        # Projected share AFTER adding this thesis
        new_n = n + 1
        if author and self.max_per_author_pct < 1.0:
            projected = (by_author.get(author, 0) + 1) / new_n
            if projected > self.max_per_author_pct:
                blocks.append(
                    f"author_quota({author}:{projected:.0%}>"
                    f"{self.max_per_author_pct:.0%})"
                )
        if ticker and self.max_per_ticker_pct < 1.0:
            projected = (by_ticker.get(ticker, 0) + 1) / new_n
            if projected > self.max_per_ticker_pct:
                blocks.append(
                    f"ticker_quota({ticker}:{projected:.0%}>"
                    f"{self.max_per_ticker_pct:.0%})"
                )
        return blocks

    # ------------------------------------------------------------------
    # The main decision function
    # ------------------------------------------------------------------
    def decide_indexable(
        self,
        thesis: dict,
        realized_pnl: Optional[dict] = None,
        as_of: Optional[datetime] = None,
    ) -> CurationDecision:
        """Decide whether to index ``thesis``.

        The four guards run in order. The first that fails blocks
        indexing. We report ALL failed guards (not just the first)
        so users can see whether a thesis is close on multiple
        axes.

        Parameters mirror ``score_thesis``.
        """
        quality = score_thesis(thesis, realized_pnl=realized_pnl, as_of=as_of)
        block_reasons: list[str] = []

        # Guard 1: composite score
        if quality.composite < self.index_threshold:
            block_reasons.append(
                f"low_composite({quality.composite:.2f}<{self.index_threshold:.2f})"
            )

        # Guard 2: warnings (most importantly "no_realized_pnl")
        if quality.warnings:
            block_reasons.append(f"warnings({','.join(quality.warnings)})")

        # Guard 3: diversity quotas (only check if guards 1-2 passed
        # — no point reporting quota breach for a low-scoring thesis)
        if not block_reasons:
            block_reasons.extend(self._quota_blocks(thesis))

        return CurationDecision(
            thesis_id=thesis.get("id") or thesis.get("doc_id") or "?",
            should_index=not block_reasons,
            quality=quality,
            block_reasons=block_reasons,
        )

    # ------------------------------------------------------------------
    # State management — record what got indexed, what got demoted
    # ------------------------------------------------------------------
    def record_indexed(
        self,
        thesis: dict,
        doc_id: str,
        quality: QualityScore,
    ) -> None:
        """Persist that this thesis is now in the corpus.

        Called by the loop AFTER the indexer successfully writes
        chunks. The record is what re-evaluation uses to find and
        re-score theses later.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        thesis_id = str(thesis.get("id") or doc_id)
        author = thesis.get("author") or thesis.get("model")
        ticker = (thesis.get("ticker") or "").upper() or None
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO learning_indexed_theses
                    (thesis_id, doc_id, ticker, author, composite_score,
                     review_score, outcome_score, indexed_at, last_evaluated_at,
                     demoted_at, demoted_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """, (
                thesis_id, doc_id, ticker, author,
                quality.composite, quality.review, quality.outcome,
                now, now,
            ))

    def mark_demoted(
        self,
        thesis_id: str,
        reason: str,
    ) -> None:
        """Mark a previously-indexed thesis as demoted.

        Doesn't delete from the vector store — that's the loop's
        responsibility. We just record that this thesis no longer
        belongs in the active set.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                UPDATE learning_indexed_theses
                SET demoted_at = ?, demoted_reason = ?
                WHERE thesis_id = ?
            """, (now, reason, thesis_id))

    def update_evaluation(
        self,
        thesis_id: str,
        quality: QualityScore,
    ) -> None:
        """Update last_evaluated_at and current scores without changing
        demotion status. Used after re-scoring during periodic audit."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                UPDATE learning_indexed_theses
                SET composite_score = ?, review_score = ?, outcome_score = ?,
                    last_evaluated_at = ?
                WHERE thesis_id = ?
            """, (quality.composite, quality.review, quality.outcome,
                  now, thesis_id))

    # ------------------------------------------------------------------
    # Audit queries — used by the loop and the CLI
    # ------------------------------------------------------------------
    def active_theses(self) -> list[dict]:
        """All currently-indexed thesis records (demoted_at IS NULL)."""
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM learning_indexed_theses
                WHERE demoted_at IS NULL
                ORDER BY indexed_at DESC
            """).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Snapshot for the CLI status command."""
        with sqlite3.connect(self.tracking_db_path) as conn:
            n_active = conn.execute(
                "SELECT COUNT(*) FROM learning_indexed_theses WHERE demoted_at IS NULL"
            ).fetchone()[0]
            n_demoted = conn.execute(
                "SELECT COUNT(*) FROM learning_indexed_theses WHERE demoted_at IS NOT NULL"
            ).fetchone()[0]
            by_author = dict(conn.execute("""
                SELECT author, COUNT(*) FROM learning_indexed_theses
                WHERE demoted_at IS NULL GROUP BY author
            """).fetchall())
            by_ticker = dict(conn.execute("""
                SELECT ticker, COUNT(*) FROM learning_indexed_theses
                WHERE demoted_at IS NULL GROUP BY ticker
            """).fetchall())
        return {
            "active": n_active,
            "demoted": n_demoted,
            "by_author": by_author,
            "by_ticker": by_ticker,
            "quota_active": n_active >= self.quota_min_corpus,
        }
