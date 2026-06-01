"""
config.py
=========

Central configuration for the hedge-fund AI research system.

This module is imported by every layer:

    - ``data/``  uses ``FRED_API_KEY``, DB paths, and log level.
    - ``agent/`` uses the AGENT LAYER CONFIG block (bottom of file).
    - ``rag/``   uses the RAG CONFIG block (bottom of file).
    - ``main.py`` passes this module through as the ``config`` argument
      to ``pipeline.run_daily_refresh`` and ``base_agent.ask_agent``.

All downstream modules should read values from here rather than hard-coding
paths, keys, or model tags. To swap the LLM backend (e.g. from mock to a real
Ollama model), change ``DEFAULT_AGENT_MODEL`` below - no other code needs to
be touched.
"""

from __future__ import annotations

# stdlib
import logging
import os
from pathlib import Path
from typing import Dict, List

# third-party
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# PATHS
# =============================================================================
# Resolve project root as the directory containing this file. All data/DB
# paths are expressed relative to it so the project is portable.

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# SQLite databases used by the data layer.
PORTFOLIO_DB_PATH: str = str(DATA_DIR / "portfolio.db")
SEC_CACHE_DB_PATH: str = str(CACHE_DIR / "sec_cache.db")
MARKET_CACHE_DB_PATH: str = str(CACHE_DIR / "market_cache.db")
MACRO_CACHE_DB_PATH: str = str(CACHE_DIR / "macro_cache.db")

# Operational telemetry (agent token usage / latency / cost). Kept in a
# SEPARATE database from portfolio.db: it's high-write-rate (one row per
# LLM call) and never joins portfolio state, so isolating it keeps the
# portfolio DB small and lets telemetry be rotated/wiped independently.
METRICS_DB_PATH: str = str(DATA_DIR / "metrics.db")

# Notional cost per 1K tokens used by agent.metrics.compute_cost. Local
# Ollama inference is free, so this is a planning figure — set it to a
# hosted provider's price to model what this workload would cost on a
# cloud API. Default $0.0001 / 1K tokens.
METRICS_COST_PER_1K_TOKENS: float = 0.0001

# ----------------------------------------------------------------------
# Telemetry alerter thresholds (agent/metrics_alerts.py)
# ----------------------------------------------------------------------
# These tune the three built-in alert rules. Set any to 0 / negative to
# disable that rule. Cooldown prevents Slack spam when a rule keeps
# tripping — the alerter records each firing and suppresses repeats
# within the cooldown window.
ALERT_CONSECUTIVE_FAILURES: int = 5         # N consecutive failures per agent
ALERT_LATENCY_P95_MS: float = 15_000.0       # qwen3:30b runs ~3.5s, 15s = degraded
ALERT_LATENCY_WINDOW_MIN: int = 60           # latency rule looks at last hour
ALERT_MIN_SAMPLES_FOR_LATENCY: int = 10      # don't alert on tiny samples
ALERT_COST_BUDGET_USD: float = 1.0           # notional daily cost budget
ALERT_COST_WINDOW_HOURS: int = 24
ALERT_COOLDOWN_MINUTES: int = 60             # dedup window per (rule, scope)

# Ensure required directories exist on import. Cheap and idempotent.
for _d in (DATA_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# API KEYS
# =============================================================================
# Prefer environment variables; fall back to empty string so missing keys fail
# loudly at the call site rather than silently at import time.

FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# SEC EDGAR requires a descriptive User-Agent with a contact email.
SEC_USER_AGENT: str = os.environ.get(
    "SEC_USER_AGENT",
    "hedgefund-ai-research research@example.com",
)


# =============================================================================
# UNIVERSE / WATCHLIST
# =============================================================================
# Default tickers used by ``main.py`` when no --tickers flag is provided.
# Override at runtime by editing this list or passing CLI args.

WATCHLIST: List[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
]


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_FILE: str = str(LOG_DIR / "hedgefund_ai.log")


# =============================================================================
# DATA PIPELINE TUNING
# =============================================================================

# Max lookback for price history (used by ``data/market_data.py``).
PRICE_HISTORY_YEARS: int = 5

# How many most-recent filings per ticker to pull from SEC EDGAR.
MAX_FILINGS_PER_TICKER: int = 10

# HTTP retry/backoff for data fetchers.
HTTP_TIMEOUT: int = 20
HTTP_MAX_RETRIES: int = 3


# =============================================================================
# AGENT LAYER CONFIG
# =============================================================================
# Everything below is consumed by ``agent/base_agent.py`` and the master
# ``main.py`` loop. The agent layer is intentionally model-agnostic: callers
# never reference a specific model tag - they go through ``AGENT_MODELS``
# and ``DEFAULT_AGENT_MODEL``.
#
# Swap models by editing ``DEFAULT_AGENT_MODEL`` only. For example:
#   - "mock" -> no Ollama required, returns deterministic JSON. Use for tests.
#   - "dev"  -> Phi-3-mini-3.8B via Ollama. Current filler while we validate
#               the wiring end-to-end on an RTX 2080.
#   - "prod" -> Qwen 14B/30B instruct - swap in once the 3.8B model proves
#               the pipeline works.
# -----------------------------------------------------------------------------

AGENT_MODELS: Dict[str, str] = {
    # Deterministic stub - no model server required. Safe default.
    "mock": "mock",

    # Current filler model. Phi-3-mini (~3.8B) runs comfortably on the
    # RTX 2080 (8GB) via Ollama. Used to validate the agent pipeline before
    # scaling up to Qwen.
    "dev": "phi3:3.8b",

    # ---- 8B grounding-quality test models (validation pass) ----
    # phi3:3.8b validated the PIPELINE but hallucinated financial figures
    # in generated prose (invented P/E, margins, revenue growth that didn't
    # match the real key_metrics). These ~8B models are large enough to
    # ground far better and both fit the RTX 2080's ~6.9 GB free VRAM at
    # Q4. Test with:  python main.py --tickers AAPL --model dev_llama8
    "dev_llama8": "llama3.1:8b",         # 8.0B, Q4_K_M (~4.9 GB)
    "dev_qwen8": "qwen3-vl:latest",      # 8.8B, Q4_K_M (~6.1 GB)

    # Production target. Will be flipped to "qwen3:14b-instruct" or
    # "qwen3:30b" once hardware/quant thresholds are confirmed.
    # Kept as a placeholder; not active until DEFAULT_AGENT_MODEL == "prod".
    "prod": "qwen3:14b-instruct",
}

# Which model to use by default when an ``AgentRequest`` does not specify
# a ``model_tag``. "mock" means the system runs fully offline with no LLM.
# Switched from "dev" (phi3:3.8b) to "dev_llama8" (llama3.1:8b): phi3
# validated the pipeline but hallucinated financial figures in generated
# prose. The 8B model grounds far better and fits the RTX 2080's VRAM.
DEFAULT_AGENT_MODEL: str = "dev_llama8"

# Hard wall-clock limit for a single agent call, in seconds. Ollama calls
# that exceed this are aborted and fall back to a safe default response.
# Raised 30 -> 120 when moving to an 8B model: llama3.1:8b generations run
# ~12-16s each on the RTX 2080, and multi-step agent calls need headroom
# under the limit so they don't spuriously trip the safe-default fallback.
AGENT_TIMEOUT: int = 120

# Maximum tokens the agent is allowed to generate per call. Keeps prompts
# and outputs within a predictable budget for logging and DB storage.
MAX_TOKENS: int = 4096

# Where prompt templates will live once we start loading them from disk.
# The directory does not need to exist yet - ``base_agent`` currently builds
# prompts in code. This is reserved for the next iteration.
AGENT_PROMPT_TEMPLATES_DIR: str = str(DATA_DIR / "prompts" / "agent")

# Ollama HTTP endpoint. Override via env var when running against a remote
# machine (e.g. a GPU workstation on the LAN).
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# =============================================================================
# RISK-SCANNER THRESHOLDS
# =============================================================================
# Centralised so ``agent/risk_scanner.py`` stays a pure rule engine and
# thresholds can be tuned here without touching logic.

RISK_THRESHOLDS: Dict[str, float] = {
    # Fundamental
    "debt_ebitda_high": 3.0,
    "debt_ebitda_medium": 2.0,
    "interest_coverage_low": 2.0,
    "fcf_yield_low": 0.02,  # 2%

    # Macro
    "recession_prob_high": 0.6,
    "recession_prob_medium": 0.35,
    "vix_high": 28.0,
    "vix_medium": 20.0,

    # Market
    "drawdown_high": 0.25,       # 25% peak-to-trough
    "drawdown_medium": 0.15,
    "realized_vol_high": 0.45,   # 45% annualised
    "realized_vol_medium": 0.30,
}


# =============================================================================
# RAG (Retrieval-Augmented Generation)
# =============================================================================
# All knobs for the rag/ package. Read-once at import; module-level retriever
# instances cache the resulting config (see pipeline._get_rag_retriever).
#
# Disable the whole subsystem by setting RAG_ENABLED = False — the agent
# context builder will short-circuit and return [] for retrieved_context
# rather than failing.

RAG_ENABLED: bool = True

# ---- Storage paths ----------------------------------------------------------
RAG_VECTOR_STORE_PATH: str = str(DATA_DIR / "chroma")
RAG_TRACKING_DB: str = str(DATA_DIR / "rag_tracking.db")
RAG_EMBEDDING_CACHE_DB: str = str(DATA_DIR / "rag_embeddings.db")
RAG_BM25_INDEX_PATH: str = str(DATA_DIR / "chroma" / "bm25_index.pkl")
RAG_FUND_NOTES_DIR: str = str(DATA_DIR / "fund_notes")

# ---- Embedding & chunking ---------------------------------------------------
# Embedder runs through Ollama when available; falls back to MiniLM
# sentence-transformers. Backend selection is inside rag/embedder.py.
RAG_EMBEDDING_MODEL: str = "nomic-embed-text"
RAG_CHUNK_TOKENS: int = 500
RAG_OVERLAP_TOKENS: int = 50

# ---- Retrieval (Phase 3) ----------------------------------------------------
# Default K returned to the agent. K_INITIAL is the over-retrieve size for
# the reranker; it must be >= RAG_DEFAULT_K with comfortable headroom.
RAG_DEFAULT_K: int = 8
RAG_K_INITIAL: int = 30

# Hybrid (vector + BM25 + RRF) tends to beat vector-only on filings/proxy
# language where exact-term match matters (e.g. "DEF 14A", "Section 162(m)").
RAG_HYBRID: bool = True

# Cross-encoder rerank costs ~10ms per (query, chunk) pair on CPU. With
# K_INITIAL=30 that's ~300ms — usually worth it on an 8GB VRAM box where
# we can't run a bigger primary embedder anyway.
RAG_RERANK: bool = True
RAG_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# MMR diversifies the top-K. Enabled: in practice retrieval was pulling
# near-identical cross-year copies of the same section (e.g. three years
# of the same share-repurchase header) into the top-8, crowding out
# distinct content. MMR penalizes redundancy so the 8 slots carry more
# unique material (risk factors + MD&A + market risk) rather than
# triplicates. Lambda 0.7 = mostly relevance, modest diversity nudge.
RAG_MMR: bool = True
RAG_MMR_LAMBDA: float = 0.7

# LLM-driven query expansion. Runs 1-2 expansion calls per retrieve().
# On llama3.1:8b (phi3 removed) this is slower per call but more reliable;
# disable for ultra-low-latency paths.
RAG_QUERY_EXPAND: bool = True
RAG_QUERY_EXPAND_MODEL: str = "llama3.1:8b"

# ---- Indexing (Phase 2) -----------------------------------------------------
# Contextualization prepends a short LLM-written summary to each chunk
# before embedding (Anthropic's "contextual retrieval" idea). Costs one
# LLM call per chunk per indexing run — meaningful on a 10-K with 300+
# chunks. Leave OFF until you have a feel for retrieval quality without.
RAG_CONTEXTUALIZE: bool = False
RAG_CONTEXTUALIZE_MODEL: str = "llama3.1:8b"

# ---- Learning loop (Phase 4) ------------------------------------------------
# Self-indexing of closed-position theses. Thresholds are intentionally
# strict — the indexed corpus poisons every future retrieval if quality
# drifts, so under-indexing beats over-indexing.
RAG_LEARNING_ENABLED: bool = True
RAG_LEARNING_INDEX_THRESHOLD: float = 0.7
RAG_LEARNING_DEMOTE_THRESHOLD: float = 0.5
RAG_LEARNING_MAX_AUTHOR_PCT: float = 0.30   # no single author dominates
RAG_LEARNING_MAX_TICKER_PCT: float = 0.25   # no single ticker dominates
RAG_LEARNING_AUDIT_SAMPLES: int = 6
RAG_LEARNING_AUDIT_LOG_DIR: str = str(DATA_DIR / "learning_audits")


# =============================================================================
# SANITY CHECKS (explicit-only - callers invoke after logging is configured)
# =============================================================================

def _warn_if_missing() -> None:
    """Emit warnings for missing-but-not-fatal config values.

    Called by ``main.py`` (and any other entry point) AFTER
    ``logging.basicConfig`` has run, so the messages appear in the
    application log rather than on stderr every time *any* script imports
    config. This function is intentionally NOT auto-invoked at import
    time - callers must explicitly call it once they own the log
    configuration. That avoids double-warnings (the previous version
    auto-fired here AND in ``main._cli_entry``).
    """
    logger = logging.getLogger(__name__)
    if not FRED_API_KEY:
        logger.warning(
            "FRED_API_KEY is empty; macro_data.py calls to FRED will fail. "
            "Set the FRED_API_KEY environment variable."
        )
    if DEFAULT_AGENT_MODEL not in AGENT_MODELS:
        logger.warning(
            "DEFAULT_AGENT_MODEL=%r is not in AGENT_MODELS; base_agent will "
            "fall back to mock mode.",
            DEFAULT_AGENT_MODEL,
        )
    if RAG_ENABLED:
        # Soft RAG environment check — RAG silently degrades on missing
        # backends, so a startup warning helps the user notice before
        # spending an hour debugging empty retrievals.
        chroma_dir = Path(RAG_VECTOR_STORE_PATH)
        if not chroma_dir.exists():
            logger.info(
                "RAG_VECTOR_STORE_PATH=%s does not exist yet; will be "
                "created on first indexer run.",
                RAG_VECTOR_STORE_PATH,
            )