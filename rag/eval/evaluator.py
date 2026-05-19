"""
rag/eval/evaluator.py
=====================
Retrieval quality metrics.

Why measure
-----------
Without measurement, every change to the RAG system is a guess. Was
the new chunker better than the old one? Did contextualization help?
Did the cross-encoder reranker pull its weight given the latency cost?
You can't answer any of those without a fixed eval set and a fixed
metric.

The three metrics
-----------------

**Recall@K**: of the chunks we know to be relevant for a query, what
fraction did the system put in the top K positions?

    Recall@K = |expected ∩ retrieved[:K]| / |expected|

Bounded in [0, 1]. 1.0 = found every expected chunk. Robust to
small eval sets. Doesn't care about ordering WITHIN the top K.

**MRR (Mean Reciprocal Rank)**: averaged over queries, the
reciprocal of the rank of the FIRST relevant chunk.

    RR = 1 / rank_of_first_relevant
    MRR = mean(RR over queries)

Bounded in [0, 1]. Rewards getting at least one relevant chunk to
the very top — what matters when the LLM weighs early chunks
heavier.

**NDCG@K (Normalized Discounted Cumulative Gain)**: a single
number summarizing both how many relevant chunks made it AND how
high they ranked.

    DCG@K = sum over i=1..K of rel(i) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

where IDCG is the ideal (maximum possible) DCG given the labels.
Bounded in [0, 1]. The log discount means rank 1 matters much more
than rank 10.

Binary vs graded relevance
--------------------------
We use binary relevance ("chunk is or isn't in the expected set")
because that's what your manually-labeled eval cases will look like.
Graded relevance (e.g., 0/1/2 for "irrelevant/partial/exact") is more
expressive but requires more annotation labor. Start binary; upgrade
to graded only if you find queries where the system retrieves
"close but not quite right" chunks and you need to penalize those.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Callable, Optional

from rag.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class EvalCase:
    """One (query, expected) pair for retrieval evaluation.

    Attributes
    ----------
    query:
        Natural-language question. Will be passed to the retriever.
    expected_chunk_ids:
        List of chunk IDs that should appear in the retrieved set.
        Built by reading documents in your corpus and noting which
        chunk(s) actually answer the query.
    filters:
        Filter kwargs passed to the retriever. E.g.
        ``{"ticker": "AAPL"}``. Empty dict = no filters.
    notes:
        Human-written explanation of why these chunks are the
        answers. Useful when revisiting eval failures.
    """
    query: str
    expected_chunk_ids: list[str]
    filters: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class EvalSet:
    """A collection of EvalCases. Loads from / saves to JSON."""
    cases: list[EvalCase]

    @classmethod
    def load(cls, path: str) -> "EvalSet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(cases=[EvalCase(**c) for c in data])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [{
                    "query": c.query,
                    "expected_chunk_ids": c.expected_chunk_ids,
                    "filters": c.filters,
                    "notes": c.notes,
                } for c in self.cases],
                f, indent=2,
            )

    def __len__(self) -> int:
        return len(self.cases)


@dataclass
class EvalResult:
    """Aggregate metrics over an EvalSet."""
    n_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    mean_latency_ms: float
    per_query: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------
# Metric implementations
# ----------------------------------------------------------------------

def recall_at_k(
    expected_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """Fraction of expected chunks in the top K retrieved."""
    if not expected_ids:
        return 1.0  # vacuously perfect — nothing to retrieve
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k)
    return hits / len(expected_ids)


def reciprocal_rank(
    expected_ids: list[str],
    retrieved_ids: list[str],
) -> float:
    """1 / (rank of first relevant chunk). 0 if none retrieved.

    Note: rank is 1-indexed. The chunk at position 0 in the list is
    rank 1. So if the first expected chunk is at position 2, RR = 1/3.
    """
    expected_set = set(expected_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    expected_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """Normalized DCG at K with binary relevance.

    Each retrieved chunk has relevance 1 if it's in expected_ids, 0
    otherwise. The DCG sums those relevances with logarithmic
    discount. NDCG normalizes by the ideal DCG (which is what you'd
    get if all relevant chunks ranked first).
    """
    expected_set = set(expected_ids)
    if not expected_set:
        return 1.0

    # DCG of the actual ranking
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if rid in expected_set else 0.0
        dcg += rel / log2(i + 1)

    # Ideal DCG: all relevant chunks first
    n_rel = min(len(expected_set), k)
    idcg = sum(1.0 / log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------
# Evaluation runner
# ----------------------------------------------------------------------

# The retrieve_fn signature: takes a query and filter kwargs, returns
# a list of RetrievalResult (chunk_id is what we score on).
RetrieveFn = Callable[..., list[RetrievalResult]]


def evaluate(
    retrieve_fn: RetrieveFn,
    eval_set: EvalSet,
    k_for_recall: int = 10,
) -> EvalResult:
    """Run retrieve_fn over every case in eval_set, return aggregates.

    Parameters
    ----------
    retrieve_fn:
        Callable ``(query: str, k: int, **filters) -> list[RetrievalResult]``.
        Usually ``Retriever.retrieve`` or ``TwoStageRetriever.retrieve``.
    eval_set:
        The held-out (query, expected) pairs.
    k_for_recall:
        ``k`` passed to retrieve_fn. Use a generous K (20-30) so the
        downstream metrics @5 and @10 are computed from the same
        result list, and so MRR can find expected chunks ranked
        beyond top 10.

    Returns
    -------
    EvalResult with aggregated metrics and per-query breakdown.
    """
    per_query = []
    latencies: list[float] = []

    sum_r5 = sum_r10 = sum_mrr = sum_ndcg = 0.0

    for case in eval_set.cases:
        t0 = time.perf_counter()
        try:
            results = retrieve_fn(case.query, k=k_for_recall, **case.filters)
        except Exception as e:  # noqa: BLE001
            logger.warning("eval | %r failed: %s", case.query[:50], e)
            results = []
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_ids = [r.chunk_id for r in results]
        r5 = recall_at_k(case.expected_chunk_ids, retrieved_ids, 5)
        r10 = recall_at_k(case.expected_chunk_ids, retrieved_ids, 10)
        rr = reciprocal_rank(case.expected_chunk_ids, retrieved_ids)
        ndcg = ndcg_at_k(case.expected_chunk_ids, retrieved_ids, 10)

        sum_r5 += r5; sum_r10 += r10; sum_mrr += rr; sum_ndcg += ndcg

        per_query.append({
            "query": case.query,
            "expected": case.expected_chunk_ids,
            "retrieved_top10": retrieved_ids[:10],
            "recall@5": r5, "recall@10": r10,
            "rr": rr, "ndcg@10": ndcg,
            "latency_ms": latency_ms,
        })

    n = len(eval_set)
    return EvalResult(
        n_queries=n,
        recall_at_5=sum_r5 / n if n else 0.0,
        recall_at_10=sum_r10 / n if n else 0.0,
        mrr=sum_mrr / n if n else 0.0,
        ndcg_at_10=sum_ndcg / n if n else 0.0,
        mean_latency_ms=sum(latencies) / n if n else 0.0,
        per_query=per_query,
    )
