"""
rag/eval/benchmark.py
=====================
Compare retrieval configurations on the same eval set.

Usage
-----
::

    python -m rag.eval.benchmark --eval-set rag/eval/test_queries.json

Configurations are defined inline below. Add a new one by appending
to ``CONFIGS``. Each is a function that returns a ``retrieve_fn`` —
the signature ``(query, k, **filters) -> list[RetrievalResult]``.

Output is a side-by-side table so the impact of each change is
visible at a glance.

Reading the table
-----------------
* **Recall@K**: higher = better.
* **MRR**: higher = better (first relevant chunk ranks higher).
* **NDCG@10**: higher = better (relevance + ordering combined).
* **Latency**: lower = better. Watch this when adding the reranker.

Sample output::

                       Recall@5  Recall@10  MRR    NDCG@10  Latency
    baseline           0.62      0.78       0.51   0.69     45ms
    + contextualize    0.78      0.86       0.65   0.78     46ms
    + rerank           0.81      0.89       0.71   0.83     320ms

Use this output to defend every change with a number.
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable

from rag.eval.evaluator import EvalResult, EvalSet, evaluate


# ----------------------------------------------------------------------
# Table formatting
# ----------------------------------------------------------------------

def format_table(rows: list[tuple[str, EvalResult]]) -> str:
    """Render a list of (config_name, EvalResult) as a fixed-width table."""
    headers = ["Config", "Recall@5", "Recall@10", "MRR", "NDCG@10", "Latency"]
    # Determine column widths
    widths = [max(len(headers[0]), max(len(r[0]) for r in rows) if rows else 0)]
    widths += [max(len(h), 8) for h in headers[1:]]

    def fmt_row(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    out = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for name, r in rows:
        out.append(fmt_row([
            name,
            f"{r.recall_at_5:.3f}",
            f"{r.recall_at_10:.3f}",
            f"{r.mrr:.3f}",
            f"{r.ndcg_at_10:.3f}",
            f"{r.mean_latency_ms:.0f}ms",
        ]))
    return "\n".join(out)


# ----------------------------------------------------------------------
# Config builders
# ----------------------------------------------------------------------
#
# Each function returns a retrieve_fn ready to pass to evaluate().
# They live behind callables so we can construct them lazily (heavy
# imports only happen when that config is actually run).

def config_baseline():
    """Phase 1 retrieval: bi-encoder only."""
    from rag.retriever import Retriever
    r = Retriever()
    return r.retrieve


def config_with_rerank():
    """Phase 2: bi-encoder + cross-encoder rerank."""
    from rag.reranker import Reranker, TwoStageRetriever
    from rag.retriever import Retriever
    rr = TwoStageRetriever(
        base_retriever=Retriever(),
        reranker=Reranker(),
        k_initial=30,
    )
    return rr.retrieve


def config_hybrid():
    """Phase 3: bi-encoder + BM25 + RRF (no rerank, no MMR, no expand)."""
    from rag.bm25_index import BM25Index
    from rag.hybrid_retriever import HybridRetriever
    from rag.vector_store import get_default_store
    bm25 = BM25Index()
    bm25.ensure_built(get_default_store())
    return HybridRetriever(bm25_index=bm25).retrieve


def config_hybrid_rerank():
    """Phase 3: hybrid + cross-encoder rerank."""
    from rag.bm25_index import BM25Index
    from rag.hybrid_retriever import HybridRetriever
    from rag.reranker import Reranker
    from rag.vector_store import get_default_store
    bm25 = BM25Index()
    bm25.ensure_built(get_default_store())
    return HybridRetriever(bm25_index=bm25, reranker=Reranker()).retrieve


def config_hybrid_rerank_mmr():
    """Phase 3: hybrid + rerank + MMR diversification."""
    from rag.bm25_index import BM25Index
    from rag.hybrid_retriever import HybridRetriever
    from rag.reranker import Reranker
    from rag.vector_store import get_default_store
    bm25 = BM25Index()
    bm25.ensure_built(get_default_store())
    return HybridRetriever(
        bm25_index=bm25, reranker=Reranker(),
        use_mmr=True, mmr_lambda=0.7,
    ).retrieve


CONFIGS = {
    "baseline": config_baseline,
    "with_rerank": config_with_rerank,
    "hybrid": config_hybrid,
    "hybrid_rerank": config_hybrid_rerank,
    "hybrid_rerank_mmr": config_hybrid_rerank_mmr,
}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-set", default="rag/eval/test_queries.json",
        help="Path to the eval set JSON.",
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["baseline", "with_rerank"],
        help="Which configurations to run. See CONFIGS dict.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-query results.",
    )
    parser.add_argument(
        "-k", type=int, default=20,
        help="K passed to retrieve. Recall@5/@10 and NDCG@10 are "
             "still measured at their named K.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)

    eval_set = EvalSet.load(args.eval_set)
    print(f"Loaded {len(eval_set)} eval cases from {args.eval_set}\n")

    rows: list[tuple[str, EvalResult]] = []
    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            print(f"warning: unknown config {cfg_name}, skipping")
            continue
        print(f"Running {cfg_name}...")
        retrieve_fn = CONFIGS[cfg_name]()
        result = evaluate(retrieve_fn, eval_set, k_for_recall=args.k)
        rows.append((cfg_name, result))

    print("\n" + format_table(rows) + "\n")

    if args.verbose:
        for name, r in rows:
            print(f"\n=== {name} per-query ===")
            for pq in r.per_query:
                print(f"  {pq['query'][:60]!r}")
                print(f"    expected: {pq['expected']}")
                print(f"    top 5:    {pq['retrieved_top10'][:5]}")
                print(f"    r@5={pq['recall@5']:.2f}  r@10={pq['recall@10']:.2f}  "
                      f"rr={pq['rr']:.3f}  ndcg={pq['ndcg@10']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
