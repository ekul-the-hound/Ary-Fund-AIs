"""
rag/integration_snippets.py
===========================
Reference code for integrating RAG (Phase 1+2+3) into your project.
NOT meant to be imported.

Four integration points:
    1. config.py            — RAG knobs
    2. pipeline.py          — retriever wiring in build_agent_context
    3. main.py              — prompt rendering of retrieved chunks
    4. refresh_scheduler.py — daily incremental index + BM25 rebuild

Phase 3 changes
---------------
* HybridRetriever replaces TwoStageRetriever as the recommended path
* BM25Index must be built at startup (or after re-indexes)
* archive_stale_documents() lets you clean up the corpus
* QueryExpander provides LLM-driven multi-query
* MMR for diversification on broad queries
"""

# ====================================================================
# config.py
# ====================================================================
#
# # ---- RAG ----
# RAG_ENABLED: bool = True
#
# # Storage
# RAG_VECTOR_STORE_PATH: str = "data/chroma"
# RAG_TRACKING_DB: str = "data/rag_tracking.db"
# RAG_EMBEDDING_CACHE_DB: str = "data/rag_embeddings.db"
# RAG_BM25_INDEX_PATH: str = "data/chroma/bm25_index.pkl"
# RAG_FUND_NOTES_DIR: str = "data/fund_notes"
#
# # Embedding & chunking
# RAG_EMBEDDING_MODEL: str = "nomic-embed-text"
# RAG_CHUNK_TOKENS: int = 500
# RAG_OVERLAP_TOKENS: int = 50
#
# # Retrieval — Phase 3 knobs
# RAG_DEFAULT_K: int = 8
# RAG_K_INITIAL: int = 30
# RAG_HYBRID: bool = True            # use BM25 + vector hybrid
# RAG_RERANK: bool = True            # cross-encoder rerank
# RAG_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# RAG_MMR: bool = False              # diversification (broad queries only)
# RAG_MMR_LAMBDA: float = 0.7
# RAG_QUERY_EXPAND: bool = True
# RAG_QUERY_EXPAND_MODEL: str = "phi3:3.8b"
#
# # Indexing — Phase 2 knobs
# RAG_CONTEXTUALIZE: bool = False
# RAG_CONTEXTUALIZE_MODEL: str = "phi3:3.8b"


# ====================================================================
# pipeline.py — build_agent_context
# ====================================================================
#
# Build a HybridRetriever once at module load (BM25 build is expensive)
# and reuse it for every context build.
#
# from functools import lru_cache
#
# @lru_cache(maxsize=1)
# def _get_rag_retriever(cfg):
#     """Construct and cache the configured Phase 3 retriever."""
#     if not getattr(cfg, "RAG_ENABLED", False):
#         return None
#
#     from rag.bm25_index import BM25Index
#     from rag.hybrid_retriever import HybridRetriever
#     from rag.query_expander import (
#         make_ollama_expander, make_disabled_expander,
#     )
#     from rag.reranker import Reranker
#     from rag.retriever import Retriever
#     from rag.vector_store import get_default_store
#
#     store = get_default_store()
#
#     # BM25 — only build if hybrid is enabled
#     bm25 = None
#     if cfg.RAG_HYBRID:
#         bm25 = BM25Index(persist_path=cfg.RAG_BM25_INDEX_PATH)
#         bm25.ensure_built(store, tracking_db_path=cfg.RAG_TRACKING_DB)
#
#     # Reranker — load on demand
#     reranker = Reranker(cfg.RAG_RERANK_MODEL) if cfg.RAG_RERANK else None
#
#     # Query expander
#     if cfg.RAG_QUERY_EXPAND:
#         expander = make_ollama_expander(model=cfg.RAG_QUERY_EXPAND_MODEL)
#     else:
#         expander = make_disabled_expander()
#
#     return HybridRetriever(
#         store=store, bm25_index=bm25, reranker=reranker,
#         query_expander=expander,
#         use_mmr=cfg.RAG_MMR, mmr_lambda=cfg.RAG_MMR_LAMBDA,
#         k_initial=cfg.RAG_K_INITIAL,
#     )
#
#
# def build_agent_context(ticker: str, db_path: str, cfg) -> dict:
#     ctx = { ... existing fields ... }
#     ctx["retrieved_context"] = _rag_retrieve(ticker, cfg)
#     return ctx
#
#
# def _rag_retrieve(ticker: str, cfg) -> list[dict]:
#     retriever = _get_rag_retriever(cfg)
#     if retriever is None:
#         return []
#
#     try:
#         # With Phase 3, a single intent-driven query is usually
#         # enough — the query expander handles multi-angle coverage
#         # internally. We can still issue 1-3 targeted queries for
#         # belt-and-suspenders.
#         query = f"investment analysis: thesis, risks, and recent moves for {ticker}"
#         results = retriever.retrieve(
#             query=query, k=cfg.RAG_DEFAULT_K, ticker=ticker,
#             doc_types=["filing", "transcript", "thesis", "note"],
#         )
#         return [
#             {
#                 "source": r.metadata.get("doc_type", "?"),
#                 "doc_id": r.metadata.get("doc_id"),
#                 "ticker": r.metadata.get("ticker"),
#                 "as_of": r.metadata.get("as_of"),
#                 "section": r.metadata.get("section") or r.metadata.get("speaker"),
#                 "score": round(r.score, 3),
#                 "text": r.text,
#             }
#             for r in results
#         ]
#     except Exception as e:
#         import logging
#         logging.getLogger(__name__).warning("rag retrieval failed: %s", e)
#         return []


# ====================================================================
# refresh_scheduler.py — daily incremental + BM25 rebuild
# ====================================================================
#
# def run_daily(self, force: bool = False) -> list[TaskResult]:
#     results = super().run_daily(force=force)
#
#     try:
#         from rag.indexer import Indexer
#         from rag.contextualizer import (
#             make_ollama_contextualizer, make_disabled_contextualizer,
#         )
#         from rag.bm25_index import BM25Index
#         from rag.vector_store import get_default_store
#
#         cx = (make_ollama_contextualizer(model=self.cfg.RAG_CONTEXTUALIZE_MODEL)
#               if self.cfg.RAG_CONTEXTUALIZE else make_disabled_contextualizer())
#
#         indexer = Indexer(
#             contextualizer=cx,
#             tracking_db_path=self.cfg.RAG_TRACKING_DB,
#             chunk_tokens=self.cfg.RAG_CHUNK_TOKENS,
#             overlap_tokens=self.cfg.RAG_OVERLAP_TOKENS,
#         )
#
#         # Index new/changed docs
#         stats = indexer.run_tickers(
#             tickers=self.tickers, sec_fetcher=self.sec,
#             portfolio_db=self.portfolio_db, force=force,
#         )
#         total_new = sum(s.get("indexed", 0) for s in stats.values())
#         results.append(TaskResult(name="daily_rag_index", rows_written=total_new))
#
#         # Rebuild BM25 if anything changed
#         if total_new > 0:
#             bm25 = BM25Index(persist_path=self.cfg.RAG_BM25_INDEX_PATH)
#             bm25.build_from_store(get_default_store())
#             results.append(TaskResult(
#                 name="daily_rag_bm25_rebuild",
#                 rows_written=bm25._chunk_count,
#             ))
#
#     except Exception as e:
#         results.append(TaskResult(name="daily_rag_index", error=repr(e)[:200]))
#
#     return results
#
#
# def run_weekly(self, force: bool = False) -> list[TaskResult]:
#     """Phase 3 addition: clean up stale documents weekly."""
#     results = super().run_weekly(force=force)
#
#     try:
#         from rag.document_loaders.filings import FilingsLoader
#         from rag.document_loaders.theses import ThesesLoader
#         from rag.document_loaders.notes import NotesLoader
#         from rag.indexer import Indexer
#
#         indexer = Indexer(tracking_db_path=self.cfg.RAG_TRACKING_DB)
#
#         # Build the canonical "what should be in the corpus" set
#         keep: set[str] = set()
#         fl = FilingsLoader(self.sec)
#         tl = ThesesLoader(self.portfolio_db)
#         for tk in self.tickers:
#             keep.update(d.doc_id for d in fl.load_for_ticker(tk))
#             keep.update(d.doc_id for d in tl.load_for_ticker(tk))
#         keep.update(d.doc_id for d in NotesLoader(self.cfg.RAG_FUND_NOTES_DIR).load_all())
#
#         result = indexer.archive_stale_documents(keep, dry_run=False)
#         results.append(TaskResult(
#             name="weekly_rag_archive_stale",
#             rows_written=result["chunks_deleted"],
#         ))
#     except Exception as e:
#         results.append(TaskResult(name="weekly_rag_archive_stale", error=repr(e)[:200]))
#
#     return results


# ====================================================================
# CLI examples
# ====================================================================
#
# # Initial backfill
# python -m rag index --tickers AAPL MSFT NVDA GOOGL AMZN
#
# # Then build BM25 index manually:
# python -c "
# from rag.bm25_index import BM25Index
# from rag.vector_store import get_default_store
# bm25 = BM25Index()
# bm25.build_from_store(get_default_store())
# "
#
# # Run the benchmark comparing all 5 configs
# python -m rag.eval.benchmark --eval-set rag/eval/test_queries.json \
#     --configs baseline with_rerank hybrid hybrid_rerank hybrid_rerank_mmr


# ====================================================================
# Phase 4 wiring — config.py additions
# ====================================================================
#
# # ---- Phase 4: self-improving RAG ----
# RAG_LEARNING_ENABLED: bool = True
# RAG_LEARNING_INDEX_THRESHOLD: float = 0.7
# RAG_LEARNING_DEMOTE_THRESHOLD: float = 0.5
# RAG_LEARNING_MAX_AUTHOR_PCT: float = 0.30
# RAG_LEARNING_MAX_TICKER_PCT: float = 0.25
# RAG_LEARNING_AUDIT_SAMPLES: int = 6
# RAG_LEARNING_AUDIT_LOG_DIR: str = "data/learning_audits"


# ====================================================================
# Phase 4 wiring — portfolio_db hooks needed
# ====================================================================
#
# Your portfolio_db needs three accessor methods that the Learning
# Loop calls. If you don't have them, add these:
#
# def get_recently_closed_theses(self, since_days: int = 7) -> list[dict]:
#     """Return thesis rows for positions closed in the last N days.
#     Each row should have: id, ticker, author/model, score (review),
#     created_at, thesis_text, essay_text, stance."""
#     ...
#
# def get_thesis_by_id(self, thesis_id: str) -> Optional[dict]:
#     """Used by the auditor when re-scoring an indexed thesis."""
#     ...
#
# def get_pnl_for_thesis(self, thesis: dict) -> Optional[dict]:
#     """Return realized P&L info or None if position is still open.
#     Shape: {'return_pct': float, 'days_held': int,
#             'benchmark_return_pct': float}"""
#     ...


# ====================================================================
# Phase 4 wiring — refresh_scheduler.py
# ====================================================================
#
# def run_daily(self, force: bool = False) -> list[TaskResult]:
#     results = super().run_daily(force=force)
#     # ... existing RAG daily refresh ...
#
#     # Phase 4: process recently-closed positions
#     if getattr(self.cfg, "RAG_LEARNING_ENABLED", False):
#         try:
#             results.append(self._run_learning_loop())
#         except Exception as e:
#             results.append(TaskResult(name="daily_rag_learning",
#                                       error=repr(e)[:200]))
#     return results
#
#
# def run_weekly(self, force: bool = False) -> list[TaskResult]:
#     results = super().run_weekly(force=force)
#
#     # Phase 4: weekly audit of the self-indexed corpus
#     if getattr(self.cfg, "RAG_LEARNING_ENABLED", False):
#         try:
#             loop = self._build_learning_loop()
#             audit = loop.scheduled_audit(
#                 n_samples=self.cfg.RAG_LEARNING_AUDIT_SAMPLES
#             )
#             results.append(TaskResult(
#                 name="weekly_rag_audit",
#                 rows_written=audit["reevaluation"]["demoted"],
#             ))
#         except Exception as e:
#             results.append(TaskResult(name="weekly_rag_audit",
#                                       error=repr(e)[:200]))
#     return results
#
#
# def _build_learning_loop(self):
#     """Construct the Phase 4 stack with the right hooks."""
#     from rag.learning import Curator, Auditor, LearningLoop
#     from rag.indexer import Indexer
#     from rag.vector_store import get_default_store
#
#     store = get_default_store()
#     indexer = Indexer()
#     curator = Curator(
#         tracking_db_path=self.cfg.RAG_TRACKING_DB,
#         index_threshold=self.cfg.RAG_LEARNING_INDEX_THRESHOLD,
#         demote_threshold=self.cfg.RAG_LEARNING_DEMOTE_THRESHOLD,
#         max_per_author_pct=self.cfg.RAG_LEARNING_MAX_AUTHOR_PCT,
#         max_per_ticker_pct=self.cfg.RAG_LEARNING_MAX_TICKER_PCT,
#     )
#     auditor = Auditor(
#         curator=curator,
#         chunk_delete_fn=store.delete_document,
#         thesis_loader_fn=self.portfolio_db.get_thesis_by_id,
#         pnl_lookup_fn=self.portfolio_db.get_pnl_for_thesis,
#     )
#     return LearningLoop(
#         curator=curator, auditor=auditor, indexer=indexer,
#         pnl_lookup_fn=self.portfolio_db.get_pnl_for_thesis,
#         audit_log_dir=self.cfg.RAG_LEARNING_AUDIT_LOG_DIR,
#     )
#
#
# def _run_learning_loop(self) -> TaskResult:
#     loop = self._build_learning_loop()
#     # Look at theses from positions closed in the last week
#     closed = self.portfolio_db.get_recently_closed_theses(since_days=7)
#     result = loop.process_closed_theses(closed)
#     return TaskResult(
#         name="daily_rag_learning",
#         rows_written=result["indexed"],
#     )
#
#
# # ====================================================================
# # Phase 4 wiring — retrieval should filter by self_indexed flag
# # ====================================================================
# #
# # When the agent retrieves context, you may want to weight or filter
# # self-indexed theses differently from external sources. Add this to
# # the retrieve call in pipeline._rag_retrieve:
# #
# # results = retriever.retrieve(
# #     query=query, ticker=ticker, k=cfg.RAG_DEFAULT_K,
# #     # Only retrieve self-indexed theses if they have composite >= 0.75
# #     # — gives them less weight than external prose for safety
# #     extra_filters={"$or": [
# #         {"self_indexed": {"$ne": True}},
# #         {"composite_quality": {"$gte": 0.75}},
# #     ]},
# # )
