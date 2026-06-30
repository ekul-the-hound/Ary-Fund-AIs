"""
diag_rag_learning.py
===================

Prove the RAG learning loop actually LEARNS — i.e. that a high-quality, winning
(profitable) closed thesis gets scored, curated, and indexed into the vector
store. The earlier in-app run showed 0 indexed only because there were no closed
P&L theses to feed it; this script supplies one and watches the machinery work.

WHAT IT DOES
------------
1. Builds a Curator (writes to data/rag_tracking.db, the real tracking DB).
2. Builds an Indexer (real embedder + vector store).
3. Builds a LearningLoop with a FAKE pnl_lookup_fn returning a strongly
   profitable outcome (so the thesis clears the P&L / quality bar).
4. Feeds ONE fabricated closed thesis with a high review score + real text.
5. Runs process_closed_theses and prints, step by step:
     * corpus stats BEFORE
     * the curator decision (should_index, composite score, any block reasons)
     * the index result
     * corpus stats AFTER  (active count should go 0 -> 1)

If you see `indexed: 1` and active climb, the learning path is proven. If it
blocks, the printed reasons tell you why (e.g. composite below threshold, or an
embedder problem).

HONEST CAVEATS
--------------
* This writes a test thesis into the tracking DB / vector store. It uses a
  clearly-marked id ("DIAG-RAG-*") so you can tell it apart. It does NOT touch
  your portfolio or agent_opinions.
* If ARY_EMBED_BACKEND=ollama isn't set (with Ollama running), the embedder may
  fall back to MiniLM-384; indexing can still succeed (proving the path) but on
  a different dimension than your production store. The script prints which
  embedder/dimension it used.
* This proves the MACHINERY learns. Whether it improves real retrieval depends
  on feeding it real closed positions over time — that's the separate
  portfolio-lifecycle wiring.

Usage (from project root, venv active, ideally with `ollama serve` running):
    python diag_rag_learning.py
"""
from __future__ import annotations

import os
# Match the app's offline flags so embedder/model loads behave the same.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project-root bootstrap so `from rag...` / `from data...` imports resolve.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _imp(*candidates):
    """Import the first module path that works; return the module."""
    import importlib
    last = None
    for c in candidates:
        try:
            return importlib.import_module(c)
        except Exception as e:  # noqa: BLE001
            last = e
    raise ImportError(f"none of {candidates} importable; last error: {last}")


def main() -> None:
    print("=" * 64)
    print("RAG LEARNING DIAGNOSTIC — feeding one winning closed thesis")
    print("=" * 64)

    # --- Import the learning components (try a few package layouts) ---------
    loop_mod = _imp("rag.learning.loop", "loop")
    curator_mod = _imp("rag.learning.curator", "curator")
    indexer_mod = _imp("rag.indexer", "indexer")

    LearningLoop = loop_mod.LearningLoop
    Curator = curator_mod.Curator
    Indexer = indexer_mod.Indexer

    # --- Build the pieces --------------------------------------------------
    print("\n[1] Building Curator + Indexer + LearningLoop…")
    curator = Curator()  # default data/rag_tracking.db
    try:
        indexer = Indexer()
        emb = getattr(indexer, "embedder", None)
        dim = getattr(emb, "dimension", "?")
        name = type(emb).__name__ if emb else "?"
        print(f"    embedder = {name}  (dim={dim})")
    except Exception as e:  # noqa: BLE001
        print(f"    ERROR building Indexer (embedder/store): {e}")
        print("    If this is an Ollama/embedder issue, start `ollama serve` "
              "and/or set ARY_EMBED_BACKEND=ollama, then retry.")
        sys.exit(1)

    # Fake P&L: a strong winner (held ~60 days, +35% vs +5% benchmark).
    def _fake_pnl(thesis: dict):
        return {
            "return_pct": 0.35,
            "days_held": 60,
            "benchmark_return_pct": 0.05,
        }

    loop = LearningLoop(
        curator=curator,
        auditor=None,
        indexer=indexer,
        pnl_lookup_fn=_fake_pnl,
    )

    # --- A fabricated, high-quality closed thesis --------------------------
    now = datetime.now(timezone.utc)
    thesis = {
        "id": f"DIAG-RAG-{now.strftime('%Y%m%d%H%M%S')}",
        "ticker": "NVDA",
        "created_at": (now - timedelta(days=60)).isoformat(),
        "score": 0.9,  # high review score (in [0,1])
        "thesis_text": (
            "NVIDIA (NVDA) — closed long. Thesis: the Blackwell data-center "
            "ramp plus CUDA lock-in would drive durable revenue growth and "
            "margin expansion. Realized: position held ~60 days for a +35% "
            "return versus +5% for the benchmark, validating the AI-demand "
            "thesis. Key driver played out as expected; risk from hyperscaler "
            "capex digestion did not materialize in the window."
        ),
        "outlook": "bullish",
    }

    # --- Corpus stats BEFORE ----------------------------------------------
    def _stats():
        try:
            return curator.stats()
        except Exception:
            return {}
    before = _stats()
    print(f"\n[2] Corpus stats BEFORE: active="
          f"{before.get('active', '?')}  demoted={before.get('demoted', '?')}")

    # --- Run the learning cycle on our one thesis --------------------------
    print("\n[3] Running process_closed_theses on the winning thesis…")
    result = loop.process_closed_theses([thesis])
    print(f"    indexed={result.get('indexed')}  blocked={result.get('blocked')}"
          f"  error={result.get('error')}")
    for d in result.get("decisions", []):
        print(f"    decision: {d.get('decision')}  thesis={d.get('thesis_id')}"
              f"  reasons={d.get('reasons', d.get('block_reasons', ''))}")

    # --- Curator decision detail (re-score to show the composite) ----------
    try:
        decision = curator.decide_indexable(thesis, realized_pnl=_fake_pnl(thesis))
        q = getattr(decision, "quality", None)
        print(f"\n[4] Curator: should_index={getattr(decision, 'should_index', '?')}"
              f"  composite={getattr(q, 'composite', '?')}"
              f"  threshold={curator.index_threshold}")
        if getattr(q, "warnings", None):
            print(f"    warnings: {q.warnings}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[4] (couldn't re-score for detail: {e})")

    # --- Corpus stats AFTER ------------------------------------------------
    after = _stats()
    print(f"\n[5] Corpus stats AFTER:  active="
          f"{after.get('active', '?')}  demoted={after.get('demoted', '?')}")

    # --- Verdict -----------------------------------------------------------
    print("\n" + "=" * 64)
    if result.get("indexed", 0) > 0:
        print("RESULT: ✅ The loop INDEXED the winning thesis — learning path "
              "is proven. The curator scored it, the indexer pushed it into the "
              "vector store, and the corpus grew.")
    else:
        print("RESULT: ⚠️ The thesis was NOT indexed. See the block reasons / "
              "composite above. Common causes: composite below threshold "
              f"({curator.index_threshold}), or an embedder/store issue. This "
              "still shows the pipeline RAN end-to-end; it just declined to "
              "index this input.")
    print("=" * 64)
    print("\nNote: this wrote a test thesis (id starts with DIAG-RAG-) into the "
          "tracking DB / vector store. It does not affect your portfolio.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\diag_rag_learning.py
