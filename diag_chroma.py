"""
diag_chroma.py — minimal ChromaDB write test.

STAGE 7 of the full pipeline crashes natively on col.upsert. This
tests Chroma in isolation with progressively: version info, a 1-row
write, then a 10-row write. The last printed line tells us whether
ANY write crashes (Chroma broken) or only larger ones (batch issue).

    python diag_chroma.py
"""
import sys

def out(m):
    print(m, flush=True)

def main():
    out("Chroma versions:")
    try:
        import chromadb
        out(f"  chromadb = {getattr(chromadb, '__version__', '?')}")
    except Exception as e:
        out(f"  chromadb import FAILED: {e}")
        return
    try:
        import numpy as np
        out(f"  numpy = {np.__version__}")
    except Exception as e:
        out(f"  numpy import FAILED: {e}")

    import config
    out(f"\nOpening PersistentClient at {config.RAG_VECTOR_STORE_PATH} ...")
    client = chromadb.PersistentClient(path=config.RAG_VECTOR_STORE_PATH)
    out("  ok")

    out("Getting/creating a TEST collection ...")
    col = client.get_or_create_collection(name="diag_test")
    out("  ok")

    # 1-row write
    out("\nWrite 1 row (dim 768) ...")
    col.upsert(
        ids=["diag_1"],
        embeddings=[[0.01] * 768],
        documents=["hello world"],
        metadatas=[{"doc_id": "diag", "ticker": "TEST"}],
    )
    out("  ok — 1 row written")

    # 10-row write
    out("\nWrite 10 rows ...")
    col.upsert(
        ids=[f"diag_{i}" for i in range(10)],
        embeddings=[[0.01 * (i + 1)] * 768 for i in range(10)],
        documents=[f"doc {i}" for i in range(10)],
        metadatas=[{"doc_id": "diag", "ticker": "TEST", "n": i} for i in range(10)],
    )
    out("  ok — 10 rows written")

    out("\nCount in test collection:")
    out(f"  {col.count()}")

    out("\nCleaning up test collection ...")
    client.delete_collection("diag_test")
    out("  ok")

    out("\nALL CHROMA WRITES PASSED.")

if __name__ == "__main__":
    main()
