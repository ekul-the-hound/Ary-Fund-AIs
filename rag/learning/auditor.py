"""
rag/learning/auditor.py
=======================
Periodic audit of the self-indexed corpus.

What the auditor does
---------------------
Two distinct jobs that share infrastructure:

**Job 1: re-evaluation.** Walk every active thesis in the
``learning_indexed_theses`` table. Re-compute its composite score
against current ``as_of``. If the new score drops below the
demotion threshold (default 0.5), demote it and remove its chunks
from the vector store.

Most demotions happen because age decay has eaten the score.
Occasionally a thesis was indexed when the position was closed and
profitable, then the position got reopened and lost — in which case
the outcome score changes too. Either way, re-evaluation catches it
without anyone having to think about it.

**Job 2: sampling for human review.** Pick N random active theses
and surface them to a human-readable audit log. The auditor doesn't
judge whether a thesis is actually correct (it can't), but it
ensures you SEE the corpus periodically. Rot spreads when nobody
looks.

Sampling strategy
-----------------
Pure random sampling under-weights edge cases. We use stratified
sampling:

    * 1/3 from theses closest to the demotion threshold (most likely
      to be problems)
    * 1/3 from highest-scoring theses (most influential — if one of
      these is wrong, it affects many future generations)
    * 1/3 truly random (catches things the other two strata miss)

The result is a small but bias-toward-interesting sample list.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from rag.learning.curator import Curator
from rag.learning.scorer import score_thesis

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------

@dataclass
class ReevaluationResult:
    """Summary of one re-evaluation pass."""
    n_evaluated: int
    n_demoted: int
    demoted_thesis_ids: list[str]
    n_score_changed: int       # had >0.05 absolute score change
    errors: int


@dataclass
class AuditSample:
    """One thesis surfaced for human review."""
    thesis_id: str
    doc_id: str
    ticker: Optional[str]
    author: Optional[str]
    composite_score: float
    indexed_at: str
    last_evaluated_at: str
    sample_reason: str  # 'near_demotion' | 'high_influence' | 'random'


# ----------------------------------------------------------------------
# Auditor class
# ----------------------------------------------------------------------

class Auditor:
    """Periodic re-evaluation and sampling.

    Parameters
    ----------
    curator:
        The curator instance whose state we're auditing. We share its
        thresholds and tracking DB so the auditor enforces the same
        rules the curator does.
    chunk_delete_fn:
        Callable ``(doc_id: str) -> int`` that removes chunks for a
        document from the vector store. The auditor uses this when
        demoting. Typically pass ``vector_store.delete_document``.

        Why injection instead of just calling the vector store
        directly? So the auditor can be tested without a vector
        store, and so callers can wire in a wrapped version that
        emits audit events.
    thesis_loader_fn:
        Callable ``(thesis_id: str) -> Optional[dict]`` that fetches
        the original thesis row by ID. Used to re-score with fresh
        ``created_at``, ``score``, etc.
    pnl_lookup_fn:
        Callable ``(thesis: dict) -> Optional[dict]`` returning the
        realized P&L for the thesis's position, or None if still
        open. Same shape as ``score_thesis`` expects.
    """

    def __init__(
        self,
        curator: Curator,
        chunk_delete_fn: Callable[[str], int],
        thesis_loader_fn: Callable[[str], Optional[dict]],
        pnl_lookup_fn: Callable[[dict], Optional[dict]],
    ):
        self.curator = curator
        self.delete_chunks = chunk_delete_fn
        self.load_thesis = thesis_loader_fn
        self.lookup_pnl = pnl_lookup_fn

    # ------------------------------------------------------------------
    # Re-evaluation
    # ------------------------------------------------------------------
    def reevaluate(self, as_of: Optional[datetime] = None) -> ReevaluationResult:
        """Re-score every active thesis. Demote those that fail."""
        as_of = as_of or datetime.now(timezone.utc)
        active = self.curator.active_theses()

        n_evaluated = 0
        n_demoted = 0
        demoted_ids: list[str] = []
        n_score_changed = 0
        errors = 0

        for row in active:
            thesis_id = row["thesis_id"]
            try:
                thesis = self.load_thesis(thesis_id)
                if thesis is None:
                    # The original thesis is gone (perhaps deleted from
                    # portfolio_db). Demote so we stop carrying its
                    # chunks in retrieval.
                    self.curator.mark_demoted(thesis_id, "source_missing")
                    self.delete_chunks(row["doc_id"])
                    demoted_ids.append(thesis_id)
                    n_demoted += 1
                    n_evaluated += 1
                    continue

                pnl = self.lookup_pnl(thesis)
                quality = score_thesis(thesis, realized_pnl=pnl, as_of=as_of)
                n_evaluated += 1

                # Score change tracking
                prior = row.get("composite_score") or 0.0
                if abs(quality.composite - prior) > 0.05:
                    n_score_changed += 1

                # Demotion check
                if quality.composite < self.curator.demote_threshold:
                    self.curator.mark_demoted(
                        thesis_id,
                        reason=f"reevaluation_below_threshold({quality.composite:.2f})",
                    )
                    try:
                        self.delete_chunks(row["doc_id"])
                    except Exception as e:  # noqa: BLE001
                        logger.warning("auditor | delete chunks failed for %s: %s",
                                       row["doc_id"], e)
                    demoted_ids.append(thesis_id)
                    n_demoted += 1
                else:
                    # Still active — update last_evaluated_at and current
                    # scores so the audit trail stays fresh.
                    self.curator.update_evaluation(thesis_id, quality)
            except Exception as e:  # noqa: BLE001
                logger.warning("auditor | reeval failed for %s: %s", thesis_id, e)
                errors += 1

        logger.info(
            "audit reeval | evaluated=%d demoted=%d changed=%d errors=%d",
            n_evaluated, n_demoted, n_score_changed, errors,
        )
        return ReevaluationResult(
            n_evaluated=n_evaluated,
            n_demoted=n_demoted,
            demoted_thesis_ids=demoted_ids,
            n_score_changed=n_score_changed,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Sampling for human review
    # ------------------------------------------------------------------
    def sample_for_review(
        self,
        n: int = 6,
        rng: Optional[random.Random] = None,
    ) -> list[AuditSample]:
        """Pick N theses stratified across risk profiles.

        Splits N roughly into thirds:
            * Near-demotion: lowest current scores (most likely to be problems)
            * High-influence: highest current scores (most heavily retrieved)
            * Random: catches everything else

        Returns up to N samples. May return fewer if the corpus is small.
        """
        rng = rng or random.Random()
        active = self.curator.active_theses()
        if not active:
            return []

        # Allocate slots
        n_near = max(1, n // 3)
        n_high = max(1, n // 3)
        n_random = n - n_near - n_high

        # Sort by composite score
        by_score = sorted(active, key=lambda r: r.get("composite_score") or 0.0)

        near_picks = by_score[:n_near]
        high_picks = by_score[-n_high:] if len(by_score) >= n_high else []

        # Random from the remainder (avoid duplicates with strata)
        picked_ids = {r["thesis_id"] for r in near_picks + high_picks}
        pool = [r for r in active if r["thesis_id"] not in picked_ids]
        random_picks = rng.sample(pool, min(n_random, len(pool))) if pool else []

        samples: list[AuditSample] = []
        for r in near_picks:
            samples.append(self._to_sample(r, "near_demotion"))
        for r in high_picks:
            samples.append(self._to_sample(r, "high_influence"))
        for r in random_picks:
            samples.append(self._to_sample(r, "random"))
        return samples

    @staticmethod
    def _to_sample(row: dict, reason: str) -> AuditSample:
        return AuditSample(
            thesis_id=row["thesis_id"],
            doc_id=row["doc_id"],
            ticker=row.get("ticker"),
            author=row.get("author"),
            composite_score=row.get("composite_score") or 0.0,
            indexed_at=row.get("indexed_at") or "",
            last_evaluated_at=row.get("last_evaluated_at") or "",
            sample_reason=reason,
        )

    # ------------------------------------------------------------------
    # Combined: full audit pass
    # ------------------------------------------------------------------
    def full_audit(
        self,
        n_samples: int = 6,
        as_of: Optional[datetime] = None,
    ) -> dict:
        """Run re-evaluation and produce sampling. Returns a dict
        suitable for logging or pretty-printing in the CLI.
        """
        reeval = self.reevaluate(as_of=as_of)
        samples = self.sample_for_review(n=n_samples)
        return {
            "reevaluation": {
                "evaluated": reeval.n_evaluated,
                "demoted": reeval.n_demoted,
                "demoted_thesis_ids": reeval.demoted_thesis_ids,
                "score_changes_significant": reeval.n_score_changed,
                "errors": reeval.errors,
            },
            "samples_for_review": [
                {
                    "thesis_id": s.thesis_id,
                    "doc_id": s.doc_id,
                    "ticker": s.ticker,
                    "author": s.author,
                    "composite_score": round(s.composite_score, 3),
                    "indexed_at": s.indexed_at,
                    "reason": s.sample_reason,
                }
                for s in samples
            ],
            "corpus_stats": self.curator.stats(),
        }
