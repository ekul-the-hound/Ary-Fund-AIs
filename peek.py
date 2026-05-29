"""Throwaway: print the latest agent opinion for a ticker, checking
both candidate DB files. Run from the project root:  python peek.py AAPL
"""
import sqlite3, json, os, sys, textwrap

ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
candidates = ["data/hedgefund.db", "data/portfolio.db"]

found = False
for db in candidates:
    if not os.path.exists(db):
        continue
    try:
        with sqlite3.connect(db) as conn:
            rows = conn.execute(
                "SELECT id, created_at, payload_json FROM agent_opinions "
                "WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            ).fetchall()
    except sqlite3.OperationalError:
        # No agent_opinions table in this DB — skip.
        continue
    if not rows:
        continue
    found = True
    rid, created_at, payload_json = rows[0]
    payload = json.loads(payload_json)
    print(f"\n{'='*70}\nDB: {db}  |  row id {rid}  |  {created_at}\n{'='*70}")
    # Print the top-level keys so we can see the opinion's shape...
    print("KEYS:", list(payload.keys()), "\n")
    # ...then dump each field, truncating long prose so the console
    # stays readable.
    for k, v in payload.items():
        text = v if isinstance(v, str) else json.dumps(v, indent=2, default=str)
        if isinstance(text, str) and len(text) > 2000:
            text = text[:2000] + f"\n... [truncated, {len(text)} chars total]"
        print(f"\n----- {k} -----\n{text}")

if not found:
    print(f"No opinion rows for {ticker} in any of: {candidates}")