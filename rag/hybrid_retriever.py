"""
rag/hybrid_retriever.py
=======================
Phase 3 retriever: hybrid search with optional diversification and
query expansion.

This module composes Phase 1+2 components (Retriever, BM25Index,
Reranker) with Phase 3 additions (MMR, QueryExpander) into a single
configurable retrieval pipeline.

The full Phase 3 retrieval flow
-------------------------------
::

    user question
        │
        ├──► [query_expander]      (optional, +500ms)
        │       │
        │       ▼
        │   list of N sub-queries
        │
        ▼
    for each sub-query:
        ├──► vector_retriever.retrieve(...)        (~50ms)
        ├──► bm25_index.query(...)                  (~5ms)
        └──► reciprocal_rank_fusion([vec, bm25])    (~1ms)
            │
            ▼
        candidate chunks for this sub-query
            │
            ▼ merge across sub-queries (RRF)
            │
            ├──► [reranker]                         (optional, +300ms)
            │       cross-encoder rescores
            │
            ├──► [mmr_select]                       (optional, +1ms)
            │       diversification re-ordering
            │
            ▼
        final top-K

Every stage is opt-in. With everything off, behavior matches Phase 1.
With everything on, you get the best retrieval quality at ~1s latency.

Design choice: composition, not config explosion
------------------------------------------------
Each Phase 3 capability is its own class, passed in via the
constructor. The retriever's role is to orchestrate them, not to
own them. Two benefits:

1. Each capability is independently testable and replaceable.
   Want a different BM25 implementation? Pass a different
   ``BM25Index`` subclass. Want re-ranking with a different model?
   Pass a different ``Reranker``.

2. Combinations are explicit. Reading the constructor of a
   ``HybridRetriever`` tells you exactly which Phase 3 features
   are active for this instance.

The non-default settings
------------------------
``MMR=False`` by default. MMR helps coverage on broad queries and
hurts on narrow factual queries; the right setting depends on what
the agent typically asks. Start with it off and turn on only after
you've measured the impact on your eval set.

``query_expansion=False`` by default. It costs an LLM call (~500ms).
Most queries don't need it. Turn on for open-ended user questions.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rag.bm25_index import BM25Index, reciprocal_rank_fusion
from rag.embedder import Embedder, get_default_embedder
from rag.mmr import mmr_select
from rag.query_expander import QueryExpander, make_disabled_expander
from rag.reranker import Reranker
from rag.retriever import Retriever, _build_where_clause
from rag.vector_store import (
    COLLECTION_RESEARCH,
    RetrievalResult,
    VectorStore,
    get_default_store,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Vector + BM25 hybrid with optional reranking, MMR, and query expansion.

    Parameters
    ----------
    embedder:
        For embedding the query (and for re-embedding candidates if
        MMR is used). If None, uses the global default.
    store:
        The vector store. If None, uses the global default.
    bm25_index:
        The lexical index. Must be built before use — call
        ``bm25_index.ensure_built(store)`` once at startup.
    reranker:
        Cross-encoder reranker. None disables reranking.
    query_expander:
        LLM-driven query expander. Pass a disabled one (default)
        to skip expansion.
    use_mmr:
        Enable Maximum Marginal Relevance diversification.
    mmr_lambda:
        See ``rag.mmr.mmr_select`` docstring. 0.7 is a good default.
    k_initial:
        How many candidates to retrieve from EACH retriever (vector
        and BM25) before fusion. 30 is the standard. Higher = more
        chances to find the right chunk, more work for the reranker.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[Reranker] = None,
        query_expander: Optional[QueryExpander] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.7,
        k_initial: int = 30,
    ):
        self.embedder = embedder or get_default_embedder()
        self.store = store or get_default_store(embedding_dim=self.embedder.dimension)
        self.bm25 = bm25_index  # may be None → vector-only
        self.reranker = reranker
        self.query_expander = query_expander or make_disabled_expander()
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.k_initial = k_initial

        # The "inner" vector retriever — we delegate to it for the
        # actual vector search step.
        self._vector_retriever = Retriever(embedder=self.embedder, store=self.store)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int = 8,
        ticker: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
        as_of_after: Optional[str] = None,
        as_of_before: Optional[str] = None,
        collection: str = COLLECTION_RESEARCH,
        extra_filters: Optional[dict] = None,
    ) -> list[RetrievalResult]:
        """Run the full Phase 3 retrieval pipeline."""
        if not query or not query.strip():
            return []

        filter_kwargs = dict(
            ticker=ticker, doc_types=doc_types,
            as_of_after=as_of_after, as_of_before=as_of_before,
            extra_filters=extra_filters,
        )

        # Step 1: Query expansion
        sub_queries = self.query_expander.expand(query)

        # Step 2: Retrieve candidates for each sub-query, fuse
        candidates_by_id: dict[str, RetrievalResult] = {}
        ranked_lists_for_rrf: list[list[str]] = []

        for sq in sub_queries:
            # Vector retrieval
            vec_results = self._vector_retriever.retrieve(
                sq, k=self.k_initial, collection=collection, **filter_kwargs,
            )
            ranked_lists_for_rrf.append([r.chunk_id for r in vec_results])
            for r in vec_results:
                if r.chunk_id not in candidates_by_id:
                    candidates_by_id[r.chunk_id] = r

            # BM25 retrieval (if available)
            if self.bm25 and self.bm25.ready:
                bm25_pairs = self.bm25.query(sq, k=self.k_initial)
                bm25_ids = [cid for cid, _ in bm25_pairs]
                ranked_lists_for_rrf.append(bm25_ids)
                # Hydrate any BM25 hits we don't already have. We
                # need their text + metadata for the downstream
                # rerank/MMR stages. Pull from vector store by ID.
                missing = [cid for cid in bm25_ids if cid not in candidates_by_id]
                if missing:
                    hydrated = self._hydrate_chunks(missing, collection)
                    candidates_by_id.update(hydrated)

        # Step 3: Fuse all ranked lists into one
        fused = reciprocal_rank_fusion(ranked_lists_for_rrf)
        # Truncate the merged pool to a reasonable size for downstream
        # stages. ~50 is enough for cross-encoder reranking; more would
        # just slow things down.
        merged_ids = [cid for cid, _ in fused[: max(50, k * 5)]]
        merged: list[RetrievalResult] = []
        for cid in merged_ids:
            if cid in candidates_by_id:
                merged.append(candidates_by_id[cid])

        # Step 4: Reranking (optional)
        if self.reranker is not None and merged:
            try:
                merged = self.reranker.rerank(query, merged, top_k=None)
            except Exception as e:  # noqa: BLE001
                logger.warning("rerank failed, skipping: %s", e)

        # Step 5: MMR diversification (optional)
        if self.use_mmr and merged and len(merged) > k:
            try:
                # MMR needs candidate embeddings. Re-embed the chunk
                # texts. (Alternative: store embeddings on the
                # RetrievalResult and pass through; would require
                # changes to the Phase 1 retriever path.)
                cand_texts = [r.text for r in merged]
                cand_embeds = self.embedder.embed(cand_texts, role="document")
                query_embed = self.embedder.embed_one(query, role="query")
                merged = mmr_select(
                    query_embed, merged, cand_embeds,
                    k=k, lambda_=self.mmr_lambda,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("mmr failed, skipping: %s", e)

        return merged[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hydrate_chunks(
        self,
        chunk_ids: list[str],
        collection: str,
    ) -> dict[str, RetrievalResult]:
        """Fetch chunks by ID from the vector store.

        Used when BM25 returns IDs we haven't already seen from
        vector retrieval. We need the text + metadata to pass through
        downstream stages.

        Returns a dict ``{chunk_id: RetrievalResult}`` (no scores —
        BM25 scores aren't directly comparable to vector scores, and
        the downstream RRF doesn't need them).
        """
        if not chunk_ids:
            return {}
        col = self.store._collection(collection)
        try:
            res = col.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception as e:  # noqa: BLE001
            logger.warning("hydrate chunks failed: %s", e)
            return {}
        out: dict[str, RetrievalResult] = {}
        for cid, doc, md in zip(
            res.get("ids", []),
            res.get("documents", []),
            res.get("metadatas", []),
        ):
            out[cid] = RetrievalResult(
                chunk_id=cid, text=doc, score=0.0,
                metadata=dict(md or {}),
            )
        return out
