"""
peek_chunks.py — show the RAG chunks retrieved for the latest opinion.

Tells us whether the retriever is pulling substantive 10-K text
(MD&A, risk factors, segment results) or boilerplate (cover pages,
exhibit lists). That decides whether the next fix is a prompt change
or a retrieval-quality change.

    python peek_chunks.py MSFT
"""
import sqlite3
import json
import sys

import config

ticker = (sys.argv[1] if len(sys.argv) > 1 else "MSFT").upper()

con = sqlite3.connect(config.PORTFOLIO_DB_PATH)
row = con.execute(
    "SELECT payload_json FROM agent_opinions WHERE ticker=? ORDER BY id DESC LIMIT 1",
    (ticker,),
).fetchone()

if not row:
    print(f"No agent_opinions row for {ticker}")
    raise SystemExit

payload = json.loads(row[0])

# retrieved_context may sit at top level or under 'thesis'
rc = (
    payload.get("retrieved_context")
    or payload.get("thesis", {}).get("retrieved_context")
    or []
)

if not rc:
    print("retrieved_context not stored in the payload.")
    print("Top-level keys:", list(payload.keys()))
    th = payload.get("thesis")
    if isinstance(th, dict):
        print("thesis keys:", list(th.keys()))
    raise SystemExit

print(f"{len(rc)} chunks retrieved for {ticker}:\n")
for i, ch in enumerate(rc):
    score = ch.get("score")
    section = ch.get("section") or "-"
    as_of = ch.get("as_of") or "-"
    text = (ch.get("text") or "").strip().replace("\n", " ")
    print(f"--- chunk {i} | score={score} | as_of={as_of} | section={section} ---")
    print(text[:400])
    print()
