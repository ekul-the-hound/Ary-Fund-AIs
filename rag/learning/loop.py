"""
rag/learning/loop.py
====================
The Phase 4 orchestrator. Ties scorer + curator + auditor into
two entry points the scheduler invokes.

The two operations
------------------
**process_closed_theses(thesis_rows, pnl_lookup_fn)**
    Called whenever positions close. Walks the rows, applies the
    curator's gates, indexes the ones that pass. Idempotent — a
    re-run on the same rows skips already-indexed theses.

**scheduled_audit()**
    Called weekly (or whatever cadence the scheduler likes). Walks
    the active corpus, re-scores each thesis, demotes stragglers.
    Also produces a sample list for human review and writes it to
    an audit log file.

What this module is NOT
-----------------------
This module doesn't decide WHEN to run. The scheduler does. The
loop just exposes "run this when called."

This module also doesn't directly call the LLM-side indexer; it
delegates to a callable injected at construction. That keeps it
testable without spinning up Ollama + ChromaDB.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from rag.document_loaders import RawDocument
from rag.learning.auditor import Auditor
from rag.learning.curator import Curator

logger = logging.getLogger(__name__)


class LearningLoop:
    """Orchestrates self-indexing of high-quality theses.

    Parameters
    ----------
    curator:
        Configured Curator instance.
    auditor:
        Configured Auditor instance. Wires into the same curator.
    indexer:
        Your project's Indexer. We call ``indexer.index_document``
        to push thesis chunks into the vector store.
    pnl_lookup_fn:
        ``(thesis: dict) -> Optional[dict]`` returning realized P&L
        for a thesis's position. Used by curator and auditor.
    audit_log_dir:
        Directory where ``scheduled_audit`` writes JSON audit logs.
        Defaults to ``data/learning_audits/``.
    """

    def __init__(
        self,
        curator: Curator,
        auditor: Auditor,
        indexer: Any,
        pnl_lookup_fn: Callable[[dict], Optional[dict]],
        audit_log_dir: str = "data/learning_audits",
    ):
        self.curator = curator
        self.auditor = auditor
        self.indexer = indexer
        self.lookup_pnl = pnl_lookup_fn
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # process_closed_theses
    # ------------------------------------------------------------------
    def process_closed_theses(
        self,
        thesis_rows: Iterable[dict],
    ) -> dict:
        """Walk closed-position theses; index the ones that pass curation.

        Parameters
        ----------
        thesis_rows:
            Iterable of dicts. Each must have at minimum ``id``,
            ``ticker``, ``created_at``, ``score``, ``thesis_text`` or
            ``essay_text``. Author/model fields optional.

        Returns
        -------
        Dict with counts: ``indexed``, ``blocked``, ``error``, plus
        per-thesis decisions for the audit log.
        """
        indexed = 0
        blocked = 0
        errors = 0
        decisions: list[dict] = []

        for thesis in thesis_rows:
            try:
                pnl = self.lookup_pnl(thesis)
                decision = self.curator.decide_indexable(thesis, realized_pnl=pnl)
                if decision.should_index:
                    # Convert to RawDocument and pass to indexer
                    doc = self._thesis_to_doc(thesis, decision)
                    result = self.indexer.index_document(doc, force=False)
                    if result.get("status") == "indexed":
                        self.curator.record_indexed(
                            thesis, doc_id=doc.doc_id, quality=decision.quality,
                        )
                        indexed += 1
                    decisions.append({
                        "thesis_id": str(thesis.get("id")),
                        "decision": "indexed",
                        "composite": decision.quality.composite,
                    })
                else:
                    blocked += 1
                    decisions.append({
                        "thesis_id": str(thesis.get("id")),
                        "decision": "blocked",
                        "reasons": decision.block_reasons,
                        "composite": decision.quality.composite,
                    })
            except Exception as e:  # noqa: BLE001
                logger.warning("loop | thesis %s | %s", thesis.get("id"), e)
                errors += 1

        return {
            "indexed": indexed,
            "blocked": blocked,
            "errors": errors,
            "decisions": decisions,
            "corpus_stats": self.curator.stats(),
        }

    @staticmethod
    def _thesis_to_doc(thesis: dict, decision) -> RawDocument:
        """Build a RawDocument from a thesis row.

        Same shape as ``rag.document_loaders.theses.ThesesLoader``
        produces, so the indexer treats self-indexed and externally-
        loaded theses identically.

        We carry the composite score into metadata so that retrieval
        can surface it ("this came from a thesis with quality 0.85"
        is useful provenance in the agent prompt).
        """
        text = (thesis.get("essay_text") or thesis.get("thesis_text") or "").strip()
        ticker = (thesis.get("ticker") or "").upper() or None
        created = thesis.get("created_at") or ""
        date_part = str(created)[:10] if created else "unknown"
        # Deterministic doc_id. Format mirrors the existing theses loader
        # so that re-running won't create duplicates.
        doc_id = f"{ticker or 'X'}_thesis_{date_part}_{thesis.get('id')}"
        return RawDocument(
            doc_id=doc_id,
            doc_type="thesis",
            text=text,
            title=f"{ticker or '?'} thesis essay ({date_part})",
            ticker=ticker,
            as_of=date_part,
            metadata={
                "thesis_id": thesis.get("id"),
                "author": thesis.get("author") or thesis.get("model"),
                "review_score": thesis.get("score"),
                "composite_quality": decision.quality.composite,
                "outcome_score": decision.quality.outcome,
                "stance": thesis.get("stance"),
                "self_indexed": True,  # flag for retrieval-time filtering
            },
        )

    # ------------------------------------------------------------------
    # scheduled_audit
    # ------------------------------------------------------------------
    def scheduled_audit(self, n_samples: int = 6) -> dict:
        """Run the full audit pass. Writes a JSON log file too.

        The log file is timestamped and saved under
        ``audit_log_dir/audit_YYYY-MM-DD_HHMMSS.json``. Humans can
        review it during scheduled review windows.
        """
        result = self.auditor.full_audit(n_samples=n_samples)

        # Write the log
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        log_path = self.audit_log_dir / f"audit_{now}.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                }, f, indent=2, default=str)
            result["log_path"] = str(log_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("loop | failed to write audit log: %s", e)
            result["log_path"] = None

        return result
