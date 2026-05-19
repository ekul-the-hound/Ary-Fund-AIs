"""
rag/mmr.py
==========
Maximum Marginal Relevance diversification.

The problem
-----------
Top-K retrieval has a redundancy failure mode. When several chunks in
the corpus say roughly the same thing, all of them score similarly
high. The top K becomes K copies of the same content.

For broad investment-research queries — "risks for Apple," "AI
strategy for Microsoft" — this is wasteful. Five chunks all
restating the same risk factor consume prompt budget without
adding signal.

The trick
---------
Pick chunks greedily, but each pick is scored as:

    MMR(d) = λ · sim(d, query) - (1 - λ) · max sim(d, already_picked)

* λ=1.0  → pure relevance (no diversification, ignores redundancy)
* λ=0.5  → balance — typical default
* λ=0.0  → pure diversification (ignores query, just spreads out)

We pick K chunks one at a time. Each pick maximizes MMR given what's
already been picked. Result: a coverage-balanced top-K instead of
five copies of the most-relevant chunk.

Reference: Carbonell & Goldstein (1998),
"The Use of MMR, Diversity-Based Reranking for Reordering Documents
and Producing Summaries."

When to enable
--------------
* Broad queries: yes. "Tell me about X" benefits from coverage.
* Narrow factual queries: no. "What was Q4 revenue?" wants the
  single best chunk; diversification could push it out for an
  unrelated chunk that happens to be less redundant.

The retriever exposes diversification as an opt-in flag.

Performance
-----------
Pairwise similarity over K_initial candidates is O(K_initial²) — for
K_initial=30 that's 900 dot products in 32-768 dim, ~0.5ms total.
Negligible compared to the cross-encoder reranker (300ms).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from rag.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[RetrievalResult],
    candidate_embeddings: np.ndarray,
    k: int,
    lambda_: float = 0.7,
) -> list[RetrievalResult]:
    """Select K chunks from candidates using Maximum Marginal Relevance.

    Parameters
    ----------
    query_embedding:
        Shape ``(dim,)``. The original query vector. Used to score
        each candidate's relevance to the query.
    candidates:
        The over-retrieved pool to pick from. Typically 20-50 items.
    candidate_embeddings:
        Shape ``(len(candidates), dim)``. Embedding for each
        candidate, parallel order. NB: these are the embeddings the
        candidates were stored with (i.e. the contextualized version
        if Contextualizer was on at indexing time).
    k:
        How many to pick.
    lambda_:
        Trade-off between query relevance (high lambda) and
        diversification (low lambda). 0.7 is a good default for
        most RAG applications — leans toward relevance but still
        spreads coverage.

        Tuning note: MMR's effective behavior depends on the
        relevance distribution in your candidate pool. If your
        top candidates all have similar relevance scores, λ=0.7
        produces visible diversification. If there's a wide spread
        (e.g., the top candidate is much more relevant than the
        rest), λ as low as 0.3 may be needed for diversification
        to activate. Measure with the eval harness if it matters.

    Returns
    -------
    Up to K items from candidates, re-ordered for diversity. The
    score field is replaced with the final MMR score (relevance
    minus redundancy penalty).

    Pre-normalization
    -----------------
    We assume incoming vectors are already L2-normalized (which they
    are for both nomic and MiniLM — these models output unit-length
    vectors by design). With normalization, cosine similarity equals
    the dot product. We normalize defensively anyway in case a
    caller passes unnormalized vectors.
    """
    if not candidates or k <= 0:
        return []
    if k >= len(candidates):
        return candidates  # nothing to diversify

    # Normalize for safe dot-product = cosine
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    q = _norm(query_embedding)
    C = _norm(candidate_embeddings)

    # Pre-compute relevance: cosine(candidate_i, query) for all i
    # Shape: (n_candidates,)
    relevance = C @ q

    # Greedy selection
    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))

    # First pick: just the most relevant
    first = int(np.argmax(relevance))
    selected_idx.append(first)
    remaining.discard(first)

    for _ in range(k - 1):
        if not remaining:
            break
        # For each remaining candidate, compute MMR score
        best_score = -np.inf
        best_idx = -1
        for i in remaining:
            # Max similarity to any already-selected
            sims_to_selected = C[i] @ C[selected_idx].T  # shape (len(selected),)
            redundancy = float(np.max(sims_to_selected))
            mmr_score = lambda_ * float(relevance[i]) - (1 - lambda_) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx < 0:
            break
        selected_idx.append(best_idx)
        remaining.discard(best_idx)

    # Build output. Use MMR score as the visible score so downstream
    # ranking respects the diversification choice. Keep originals
    # in metadata for diagnostics.
    out: list[RetrievalResult] = []
    for rank, i in enumerate(selected_idx, start=1):
        r = candidates[i]
        new_md = dict(r.metadata)
        new_md["mmr_rank"] = rank
        new_md["pre_mmr_score"] = r.score
        new_md["mmr_relevance"] = float(relevance[i])
        # Compute final MMR score for display
        if rank == 1:
            final = float(relevance[i])  # first pick is unpenalized
        else:
            prior = selected_idx[:rank - 1]
            redundancy = float(np.max(C[i] @ C[prior].T))
            final = lambda_ * float(relevance[i]) - (1 - lambda_) * redundancy
        out.append(RetrievalResult(
            chunk_id=r.chunk_id,
            text=r.text,
            score=final,
            metadata=new_md,
        ))
    return out
