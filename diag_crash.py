"""
diag_crash.py — isolate which layer segfaults during RAG indexing.

Runs each stage SEPARATELY with flush-after-every-print so the last
line you see before the crash names the culprit. Native crashes
("Python has stopped working") kill the process without a traceback,
so the ordering of printed lines is the diagnostic.

    python diag_crash.py MSFT
"""
import sys

def out(msg):
    print(msg, flush=True)   # flush so nothing is lost on a native crash

def main(ticker="MSFT"):
    ticker = ticker.upper()

    # ---- Stage 0: imports (torch / chroma / onnx load native libs) ----
    out("STAGE 0: importing config + sec_fetcher ...")
    import config
    from data.sec_fetcher import SECFetcher
    out("  ok")

    out("STAGE 0b: importing rag.embedder ...")
    from rag.embedder import Embedder
    out("  ok")

    out("STAGE 0c: importing rag.vector_store ...")
    from rag.vector_store import VectorStore
    out("  ok")

    out("STAGE 0d: importing chunker ...")
    from rag.chunker import chunk_document
    out("  ok")

    # ---- Stage 1: read ONE filing text from DB (no network/ML) ----
    out("\nSTAGE 1: reading one 10-K text from DB ...")
    sec = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)
    cached = sec._get_cached_filings(ticker, "10-K", 1, None, None)
    out(f"  cached rows: {len(cached)}")
    acc = cached[0]["accession_number"]
    out(f"  accession: {acc}")
    text = sec.get_filing_text(acc)
    out(f"  text length: {len(text):,} chars")

    # ---- Stage 2: chunk it (pure python) ----
    out("\nSTAGE 2: chunking ...")
    chunks = chunk_document(
        raw_text=text, doc_id=f"{ticker}_TEST", doc_type="filing",
        base_metadata={"ticker": ticker, "doc_type": "filing", "doc_id": f"{ticker}_TEST"},
        chunk_tokens=getattr(config, "RAG_CHUNK_TOKENS", 500),
        overlap_tokens=getattr(config, "RAG_OVERLAP_TOKENS", 50),
    )
    out(f"  produced {len(chunks)} chunks")

    # ---- Stage 3: build embedder (probes Ollama) ----
    out("\nSTAGE 3: building Embedder (forces ollama) ...")
    import os
    os.environ.setdefault("ARY_EMBED_BACKEND", "ollama")
    emb = Embedder(cache_db_path=getattr(config, "RAG_EMBEDDING_CACHE_DB", None))
    out(f"  backend={emb.backend_name} dim={emb.dimension}")

    # ---- Stage 4: embed JUST 3 chunks (small Ollama call) ----
    out("\nSTAGE 4: embedding first 3 chunks via Ollama ...")
    sample = [c.text for c in chunks[:3]]
    vecs = emb.embed(sample, role="document")
    out(f"  got embeddings shape: {vecs.shape}")

    # ---- Stage 5: embed ALL chunks (full Ollama load — VRAM test) ----
    out(f"\nSTAGE 5: embedding ALL {len(chunks)} chunks ...")
    all_inputs = [c.text for c in chunks]
    all_vecs = emb.embed(all_inputs, role="document")
    out(f"  got embeddings shape: {all_vecs.shape}")

    # ---- Stage 6: open Chroma store ----
    out("\nSTAGE 6: opening VectorStore (Chroma) ...")
    store = VectorStore(
        persist_path=config.RAG_VECTOR_STORE_PATH,
        embedding_dim=768,
    )
    out("  ok")

    # ---- Stage 7: write to Chroma (the suspected native crash) ----
    out(f"\nSTAGE 7: upserting {len(chunks)} chunks to Chroma ...")
    store.upsert_chunks(chunks, all_vecs)
    out("  ok — WROTE SUCCESSFULLY")

    out("\nALL STAGES PASSED. No crash.")

if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    main(tk)
