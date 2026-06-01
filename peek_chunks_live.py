"""
peek_chunks_live.py — re-run the pipeline's RAG retrieval and dump the
chunks it returns, so we can judge retrieval quality.

The opinion payload doesn't persist retrieved_context, so we call the
exact same function the pipeline uses (_rag_retrieve) with the same
query and filters. Whatever this prints is what the essay prompt saw.

    python peek_chunks_live.py MSFT
"""
import sys

import config
from data import pipeline

ticker = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()

print(f"Running live RAG retrieval for {ticker} ...\n")
chunks = pipeline._rag_retrieve(ticker, config)

if not chunks:
    print("No chunks returned. Either retrieval failed (check logs) or "
          "the store has nothing matching the filters for this ticker.")
    raise SystemExit

print(f"{len(chunks)} chunks retrieved:\n")
for i, ch in enumerate(chunks):
    score = ch.get("score")
    section = ch.get("section") or "-"
    as_of = ch.get("as_of") or "-"
    src = ch.get("source") or "-"
    text = (ch.get("text") or "").strip().replace("\n", " ")
    print(f"--- chunk {i} | {src} | score={score} | as_of={as_of} | section={section} ---")
    print(text[:500])
    print()
