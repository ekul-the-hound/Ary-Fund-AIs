"""
rag/retriever.py
================
Query-time entry point: takes a natural-language question, returns the
top-K most relevant chunks from the vector store.

This module is intentionally thin. It composes the embedder and vector
store, translates Python kwargs into ChromaDB filter dicts, and
optionally merges results from both collections. There's no business
logic here — every decision (which collections, what filters, K) is
driven by the caller.

Responsibilities
----------------
1. Embed the query with ``role="query"`` so the right model prefix
   is applied. Embedding a query as a document is one of the most
   common silent-failure bugs in RAG.

2. Translate caller-friendly kwargs (``ticker``, ``doc_types``,
   ``as_of_after``) to ChromaDB's ``where=`` dict syntax. This is
   syntactic sugar but it pays off — every call site is more
   readable, and changing vendors only updates this file.

3. Optionally merge results from ``research_docs`` and
   ``structured_signals``. Long chunks and short facts have
   different similarity distributions, so we don't naively merge by
   score; we interleave by rank.

What this module does NOT do
----------------------------
* Re-ranking (Phase 2 — see ``reranker.py``)
* Query expansion / multi-query (Phase 3)
* Hybrid retrieval with BM25 (Phase 3)
* Diversification with MMR (Phase 3)

Phase 1 is just dense vector search with metadata filters. The
later phases plug in here without changing the caller-facing API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rag.embedder import Embedder, get_default_embedder
from rag.vector_store import (
    COLLECTION_RESEARCH,
    COLLECTION_SIGNALS,
    RetrievalResult,
    VectorStore,
    get_default_store,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Filter translation
# ----------------------------------------------------------------------

def _build_where_clause(
    ticker: Optional[str] = None,
    doc_types: Optional[list[str]] = None,
    as_of_after: Optional[str] = None,
    as_of_before: Optional[str] = None,
    section: Optional[str] = None,
    speaker_role: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Optional[dict]:
    """Translate caller kwargs to ChromaDB's `where=` dict.

    ChromaDB's filter language uses MongoDB-style operators:
        ``$eq``, ``$in``, ``$gte``, ``$lte``, ``$and``, ``$or``
    For exact matches you can also write the value directly:
        ``{"ticker": "AAPL"}``  is shorthand for  ``{"ticker": {"$eq": "AAPL"}}``

    A subtlety: when there's only one filter clause, Chroma accepts
    ``{"ticker": "AAPL"}``. When there are multiple, you must wrap
    them in ``{"$and": [...]}`` — otherwise the second clause
    silently overwrites the first in the dict literal.

    Returns None when no filters are set, so the caller can pass it
    directly to ``store.query(where=...)`` without conditionals.
    """
    clauses: list[dict] = []
    if ticker:
        clauses.append({"ticker": ticker})
    if doc_types:
        clauses.append({"doc_type": {"$in": list(doc_types)}})
    if as_of_after:
        clauses.append({"as_of": {"$gte": as_of_after}})
    if as_of_before:
        clauses.append({"as_of": {"$lte": as_of_before}})
    if section:
        clauses.append({"section": section})
    if speaker_role:
        clauses.append({"speaker_role": speaker_role})
    if extra:
        for k, v in extra.items():
            clauses.append({k: v})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ----------------------------------------------------------------------
# Retriever class
# ----------------------------------------------------------------------

class Retriever:
    """Compose embedder + vector store into a query-time interface."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or get_default_embedder()
        self.store = store or get_default_store(embedding_dim=self.embedder.dimension)

    # ------------------------------------------------------------------
    # Single-collection retrieval (the common case)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = 8,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        as_of_after: Optional[str] = None,
        as_of_before: Optional[str] = None,
        section: Optional[str] = None,
        speaker_role: Optional[str] = None,
        collection: str = COLLECTION_RESEARCH,
        extra_filters: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Run a single vector search with optional metadata filters.

        Parameters
        ----------
        query:
            The user's question. Embedded with ``role="query"``.
        k:
            Maximum results to return.
        ticker, doc_types, as_of_after, as_of_before, section,
        speaker_role:
            Common filter knobs. None = no filter on that field.
        collection:
            Which collection to search. Defaults to research prose.
        extra_filters:
            Escape hatch for any filter the named kwargs don't cover.
            Merged into the ``where=`` clause with ``$and``.

        Returns
        -------
        List of RetrievalResult, sorted by similarity descending.
        Empty list if the store has no matching chunks.
        """
        # Empty query → empty result. Saves an embed round-trip and
        # avoids weird behavior on whitespace-only inputs.
        if not query or not query.strip():
            return []

        query_vec = self.embedder.embed_one(query, role="query")
        where = _build_where_clause(
            ticker=ticker, doc_types=doc_types,
            as_of_after=as_of_after, as_of_before=as_of_before,
            section=section, speaker_role=speaker_role,
            extra=extra_filters,
        )

        try:
            results = self.store.query(
                query_embedding=query_vec, k=k, where=where,
                collection=collection,
            )
        except Exception as e:  # noqa: BLE001 — chroma can fail many ways
            logger.warning("retrieve | %s | %s", collection, e)
            return []

        logger.debug(
            "retrieve | q=%r | k=%d | where=%s | got %d",
            query[:50], k, where, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Multi-collection retrieval
    # ------------------------------------------------------------------
    def retrieve_combined(
        self,
        query: str,
        k: int = 8,
        prose_weight: float = 0.7,
        **filters,
    ) -> list[RetrievalResult]:
        """Search both ``research_docs`` and ``structured_signals``,
        merge results by interleaved rank.

        Why not merge by raw score? Because score distributions differ
        between collections. Long prose chunks tend to score lower in
        absolute terms (more dilution), short fact sentences score
        higher. Naive max-by-score would always pick fact sentences.

        Instead we take the top from each and interleave: 70% prose,
        30% signals by default (knob via ``prose_weight``).

        Phase 1 note: ``structured_signals`` is empty until Phase 2.
        This method works either way — empty collection yields zero
        results from that side and the prose results flow through.
        """
        n_prose = max(1, int(round(k * prose_weight)))
        n_signals = k - n_prose

        prose = self.retrieve(
            query, k=n_prose * 2, collection=COLLECTION_RESEARCH, **filters,
        )
        signals = self.retrieve(
            query, k=max(1, n_signals * 2), collection=COLLECTION_SIGNALS, **filters,
        )

        # Interleave: prose[0], signals[0], prose[1], signals[1], ...
        # but stop once we hit k. Skip exhausted side.
        merged: list[RetrievalResult] = []
        i = j = 0
        while len(merged) < k and (i < len(prose) or j < len(signals)):
            # Decide whose turn it is based on the ratio achieved so far
            n_p = sum(1 for r in merged if r.metadata.get("doc_type") != "signal")
            current_prose_ratio = (n_p / len(merged)) if merged else 0
            take_prose = (current_prose_ratio < prose_weight) and i < len(prose)
            if take_prose and i < len(prose):
                merged.append(prose[i]); i += 1
            elif j < len(signals):
                merged.append(signals[j]); j += 1
            elif i < len(prose):
                merged.append(prose[i]); i += 1
            else:
                break
        return merged[:k]


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

_default_retriever: Optional[Retriever] = None


def get_default_retriever() -> Retriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever
