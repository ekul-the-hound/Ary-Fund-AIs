"""
rag/eval — retrieval quality evaluation.

Three pieces:

* ``evaluator.py``    — the metric implementations and the EvalSet
                        dataclass for held-out (query, expected) pairs.
* ``benchmark.py``    — run a configuration over an EvalSet and print
                        a comparison table.
* ``test_queries.json`` — starter set of 5 queries. Replace with your
                          own once you've built a corpus.
"""

from rag.eval.evaluator import (
    EvalCase,
    EvalResult,
    EvalSet,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalSet",
    "evaluate",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
