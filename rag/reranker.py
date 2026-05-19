"""
rag/reranker.py
===============
Cross-encoder reranker. Second-stage scoring over an over-retrieved
candidate pool.

The retrieval pipeline becomes:

    1. Embedder + VectorStore: retrieve top K_INITIAL (~30) chunks
       using ANN search. Fast, pre-indexed.
    2. Reranker (this file): score each of those 30 chunks against
       the query using a cross-encoder. Slow, but only 30 forward
       passes, not 100,000.
    3. Take top K_FINAL (~8) by cross-encoder score.

Why this works
--------------
Bi-encoders embed query and chunk independently. They never see the
two together. A cross-encoder feeds them through one transformer
forward pass with attention between query and chunk tokens — much
more accurate, far slower.

The two-stage trick keeps query latency reasonable while getting most
of the cross-encoder accuracy benefit.

Quality vs latency knob
-----------------------
``K_initial`` controls how many candidates the cross-encoder sees.
Higher = more chances to find the right chunk if the bi-encoder
ranked it lowly. Lower = faster query. Default 30 is a reasonable
midpoint.

Model choice
------------
Default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``. 6-layer MiniLM,
trained on MS MARCO. CPU-friendly (~10ms per pair).

For higher quality: ``cross-encoder/ms-marco-MiniLM-L-12-v2`` (twice
the layers, ~2x slower, marginally better).

For non-English: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``.

Failure modes
-------------
If ``sentence-transformers`` isn't installed or the model can't
download, ``Reranker.rerank`` falls back to returning the input list
unchanged. We log a warning once at construction time so the user
knows quality is degraded.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


# Default cross-encoder. MS MARCO MiniLM L-6 is the standard
# CPU-friendly choice and what most production RAG systems start with.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker over an existing list of retrieval results.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for a ``CrossEncoder``-compatible model.
        Defaults to the MS MARCO MiniLM L-6.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model = None
        # Touch the model lazily — first ``rerank`` call triggers
        # download (one-time, ~80MB) and load. Constructor stays fast.

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                logger.info("Reranker loaded | %s", self.model_name)
            except ImportError as e:
                raise RuntimeError(
                    "Reranker requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    @property
    def available(self) -> bool:
        """True iff the model can be loaded. Useful to gate calls
        without forcing model load just to check."""
        try:
            _ = self.model
            return True
        except Exception:  # noqa: BLE001
            return False

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Re-score results with the cross-encoder, return new ordering.

        Parameters
        ----------
        query:
            The original query string. The cross-encoder scores
            (query, chunk_text) pairs together.
        results:
            Output of ``Retriever.retrieve``. Typically 25-50 items
            (over-retrieved to give the reranker headroom).
        top_k:
            If set, truncate output to this many. Often 8 for the
            final agent prompt.

        Returns
        -------
        Same list, re-sorted by cross-encoder relevance descending.
        Each result's ``score`` field is REPLACED with the
        cross-encoder score (the original bi-encoder score is moved
        to ``metadata["bi_encoder_score"]`` for debugging).

        On any failure (no model, model crash), returns the input
        list unchanged.
        """
        if not results:
            return results

        try:
            model = self.model
        except Exception as e:  # noqa: BLE001
            logger.warning("Reranker unavailable, returning input order: %s", e)
            return results[:top_k] if top_k else results

        # Build the pair list: cross-encoder takes (query, doc) tuples
        # and returns a single score per pair. The model decides
        # similarity via cross-attention; the score is unbounded but
        # higher = more relevant (typically -5 to +15 for MS MARCO).
        pairs = [(query, r.text) for r in results]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("Reranker prediction failed: %s", e)
            return results[:top_k] if top_k else results

        # Build new results with cross-encoder scores. We keep the
        # bi-encoder score in metadata so the eval harness can compare
        # the two rankings.
        reranked: list[RetrievalResult] = []
        for r, ce_score in zip(results, scores):
            new_md = dict(r.metadata)
            new_md["bi_encoder_score"] = r.score
            new_md["ce_score"] = float(ce_score)
            reranked.append(RetrievalResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=float(ce_score),
                metadata=new_md,
            ))

        reranked.sort(key=lambda r: r.score, reverse=True)
        if top_k:
            reranked = reranked[:top_k]
        return reranked


# ----------------------------------------------------------------------
# Convenience: a retriever that does both stages automatically
# ----------------------------------------------------------------------

class TwoStageRetriever:
    """Wraps ``Retriever`` + ``Reranker`` into one call.

    Use this in pipeline.build_agent_context() when you want the
    quality lift without managing the two stages yourself.

    Example
    -------
        >>> from rag.retriever import Retriever
        >>> from rag.reranker import Reranker, TwoStageRetriever
        >>> r = TwoStageRetriever(Retriever(), Reranker())
        >>> chunks = r.retrieve("supply chain risk", ticker="AAPL", k=8)
    """

    def __init__(
        self,
        base_retriever,
        reranker: Reranker,
        k_initial: int = 30,
    ):
        self.base = base_retriever
        self.reranker = reranker
        self.k_initial = k_initial

    def retrieve(self, query: str, k: int = 8, **filters) -> list[RetrievalResult]:
        # Over-retrieve, then rerank, then truncate.
        candidates = self.base.retrieve(query, k=self.k_initial, **filters)
        return self.reranker.rerank(query, candidates, top_k=k)
