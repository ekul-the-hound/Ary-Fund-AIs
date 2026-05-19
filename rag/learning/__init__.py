"""
rag.learning — Phase 4 self-improvement.

The package implements Phase 4: indexing the agent's own prior
reasoning back into the RAG corpus, with guards against poisoning.

Quick start
-----------
::

    from rag.indexer import Indexer
    from rag.learning import Curator, Auditor, LearningLoop

    curator = Curator()  # uses default thresholds
    auditor = Auditor(
        curator=curator,
        chunk_delete_fn=vector_store.delete_document,
        thesis_loader_fn=portfolio_db.get_thesis_by_id,
        pnl_lookup_fn=portfolio_db.get_pnl_for_thesis,
    )
    loop = LearningLoop(
        curator=curator, auditor=auditor,
        indexer=Indexer(),
        pnl_lookup_fn=portfolio_db.get_pnl_for_thesis,
    )

    # On position close:
    loop.process_closed_theses(recently_closed_theses)

    # Weekly:
    loop.scheduled_audit()

Public API
----------
* :class:`Curator` — gatekeeper. Decides what gets indexed.
* :class:`Auditor` — periodic re-evaluation and sampling.
* :class:`LearningLoop` — orchestrator the scheduler calls.
* :func:`score_thesis` — composite quality scoring.

The four guards
---------------
1. Composite score threshold (review + outcome + age)
2. P&L confirmation (no review-only indexing)
3. Per-author / per-ticker diversity quotas
4. Periodic re-evaluation with demotion

See module docstrings in scorer.py, curator.py, auditor.py for the
reasoning behind each.
"""

from rag.learning.auditor import Auditor, AuditSample, ReevaluationResult
from rag.learning.curator import CurationDecision, Curator
from rag.learning.loop import LearningLoop
from rag.learning.scorer import QualityScore, score_thesis

__all__ = [
    "Auditor",
    "AuditSample",
    "CurationDecision",
    "Curator",
    "LearningLoop",
    "QualityScore",
    "ReevaluationResult",
    "score_thesis",
]
