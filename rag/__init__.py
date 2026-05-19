"""
rag — Retrieval-Augmented Generation for Ary Quant.

The package indexes prose documents (SEC filings, earnings transcripts,
generated theses, hand-written notes), embeds them, stores them in
ChromaDB, and lets the agent layer retrieve relevant chunks at query
time.

Quick start
-----------
::

    from rag.indexer import Indexer
    from rag.retriever import Retriever

    # Index everything for a set of tickers
    Indexer().run_tickers(["AAPL", "MSFT"], sec_fetcher=..., portfolio_db=...)

    # Query
    chunks = Retriever().retrieve(
        "What are Apple's supply chain risks?",
        ticker="AAPL", k=8,
    )
    for c in chunks:
        print(c.score, c.text[:100])

Public API
----------
* :class:`rag.indexer.Indexer`    — ingest + chunk + embed + store
* :class:`rag.retriever.Retriever` — query → top-K chunks
* :class:`rag.embedder.Embedder`   — text → vector
* :class:`rag.chunker.Chunk`       — chunked document unit
* :class:`rag.vector_store.VectorStore` — ChromaDB wrapper

Run ``python -m rag --help`` for the CLI.
"""

from rag.bm25_index import BM25Index, reciprocal_rank_fusion, tokenize
from rag.chunker import Chunk
from rag.contextualizer import (
    Contextualizer,
    make_disabled_contextualizer,
    make_ollama_contextualizer,
)
from rag.embedder import Embedder, get_default_embedder
from rag.hybrid_retriever import HybridRetriever
from rag.indexer import Indexer
from rag.learning import (
    Auditor,
    Curator,
    LearningLoop,
    QualityScore,
    score_thesis,
)
from rag.mmr import mmr_select
from rag.query_expander import (
    QueryExpander,
    make_disabled_expander,
    make_ollama_expander,
)
from rag.reranker import Reranker, TwoStageRetriever
from rag.retriever import Retriever, get_default_retriever
from rag.vector_store import RetrievalResult, VectorStore, get_default_store

__all__ = [
    "Auditor",
    "BM25Index",
    "Chunk",
    "Contextualizer",
    "Curator",
    "Embedder",
    "HybridRetriever",
    "Indexer",
    "LearningLoop",
    "QualityScore",
    "QueryExpander",
    "Reranker",
    "RetrievalResult",
    "Retriever",
    "TwoStageRetriever",
    "VectorStore",
    "get_default_embedder",
    "get_default_retriever",
    "get_default_store",
    "make_disabled_contextualizer",
    "make_disabled_expander",
    "make_ollama_contextualizer",
    "make_ollama_expander",
    "mmr_select",
    "reciprocal_rank_fusion",
    "score_thesis",
    "tokenize",
]
