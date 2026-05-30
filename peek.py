"""
Opinion Inspector Script — peek.py

WHAT: Given a ticker, prints the most recent persisted agent opinion (full
      thesis payload) in human-readable form to the console.

WHY:  The pipeline writes verdicts via portfolio_db.save_agent_opinion, but
      there is no built-in reader method on PortfolioDB for pulling a single
      latest opinion. During validation, this script provides a quick window
      into stored output without launching the whole Streamlit UI. Used for
      model comparison and prompt-hardening verification.

HOW:
  - Checks both data/hedgefund.db and data/portfolio.db (tolerates a missing
    agent_opinions table)
  - Selects the most recent row for the requested ticker (ORDER BY id DESC
    LIMIT 1)
  - Parses payload_json, prints row id, timestamp, top-level keys, then each
    field
  - Truncates very long prose (essay, review) so the console stays readable

RUN: python peek.py AAPL   (ticker argument; defaults to AAPL if omitted)

STATUS: Operational script — safe to delete once the dashboard or a dedicated
        reader method supersedes it. Not part of runtime contracts.
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