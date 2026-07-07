"""
peek.py — inspect stored opinions and RAG chunks from the console.

Consolidates the old peek.py / peek_chunks.py / peek_chunks_live.py trio into
one CLI. Read-only quick window into pipeline output without launching Streamlit.

    python peek.py opinion AAPL          # latest persisted agent opinion (full payload)
    python peek.py chunks MSFT           # RAG chunks stored on the latest opinion
    python peek.py chunks MSFT --live     # re-run the pipeline's retrieval instead

Subcommands
-----------
opinion  : dump the most recent agent_opinions row for a ticker (all fields,
           long prose truncated), checking both hedgefund.db and portfolio.db.
chunks   : show the chunks behind the latest opinion. By default reads the
           stored retrieved_context; --live re-runs pipeline._rag_retrieve with
           the same query/filters the essay prompt saw (the payload doesn't
           always persist retrieved_context).

Not part of runtime contracts — safe to delete once the dashboard supersedes it.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys


# ----------------------------------------------------------------------
# opinion — dump the latest persisted opinion for a ticker
# ----------------------------------------------------------------------
def cmd_opinion(args: argparse.Namespace) -> int:
    ticker = args.ticker.upper()
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
        print("KEYS:", list(payload.keys()), "\n")
        for k, v in payload.items():
            text = v if isinstance(v, str) else json.dumps(v, indent=2, default=str)
            if isinstance(text, str) and len(text) > 2000:
                text = text[:2000] + f"\n... [truncated, {len(text)} chars total]"
            print(f"\n----- {k} -----\n{text}")

    if not found:
        print(f"No opinion rows for {ticker} in any of: {candidates}")
        return 1
    return 0


# ----------------------------------------------------------------------
# chunks — show RAG chunks (stored, or re-run live retrieval)
# ----------------------------------------------------------------------
def _print_chunks(chunks: list, snippet: int) -> None:
    print(f"{len(chunks)} chunks:\n")
    for i, ch in enumerate(chunks):
        score = ch.get("score")
        section = ch.get("section") or "-"
        as_of = ch.get("as_of") or "-"
        src = ch.get("source")
        text = (ch.get("text") or "").strip().replace("\n", " ")
        head = f"--- chunk {i} | "
        if src:
            head += f"{src} | "
        head += f"score={score} | as_of={as_of} | section={section} ---"
        print(head)
        print(text[:snippet])
        print()


def cmd_chunks(args: argparse.Namespace) -> int:
    import config

    ticker = args.ticker.upper()

    if args.live:
        # Re-run the exact retrieval the pipeline uses. Whatever this prints is
        # what the essay prompt saw.
        try:
            from data import pipeline
        except Exception:
            import pipeline  # type: ignore
        print(f"Running live RAG retrieval for {ticker} ...\n")
        chunks = pipeline._rag_retrieve(ticker, config)
        if not chunks:
            print("No chunks returned. Either retrieval failed (check logs) or "
                  "the store has nothing matching the filters for this ticker.")
            return 1
        _print_chunks(chunks, snippet=500)
        return 0

    # Stored path: read retrieved_context off the latest opinion payload.
    con = sqlite3.connect(config.PORTFOLIO_DB_PATH)
    row = con.execute(
        "SELECT payload_json FROM agent_opinions WHERE ticker=? "
        "ORDER BY id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if not row:
        print(f"No agent_opinions row for {ticker}")
        return 1

    payload = json.loads(row[0])
    rc = (
        payload.get("retrieved_context")
        or payload.get("thesis", {}).get("retrieved_context")
        or []
    )
    if not rc:
        print("retrieved_context not stored in the payload. "
              "Try --live to re-run retrieval.")
        print("Top-level keys:", list(payload.keys()))
        th = payload.get("thesis")
        if isinstance(th, dict):
            print("thesis keys:", list(th.keys()))
        return 1

    print(f"{len(rc)} chunks retrieved for {ticker}:\n")
    _print_chunks(rc, snippet=400)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect stored opinions and RAG chunks.")
    sub = p.add_subparsers(dest="command", required=True)

    po = sub.add_parser("opinion", help="dump the latest agent opinion for a ticker")
    po.add_argument("ticker", nargs="?", default="AAPL")
    po.set_defaults(func=cmd_opinion)

    pc = sub.add_parser("chunks", help="show RAG chunks behind the latest opinion")
    pc.add_argument("ticker", nargs="?", default="MSFT")
    pc.add_argument("--live", action="store_true",
                    help="re-run pipeline retrieval instead of reading the payload")
    pc.set_defaults(func=cmd_chunks)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
