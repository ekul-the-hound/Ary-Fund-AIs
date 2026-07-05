"""
backfill_news_edges.py
======================
Turn the per-ticker headlines in ``news_articles`` (populated by
``sentiment_news``) into company-pair *suspected* edges for the Money Flow
Graph, written to a ``news_edges`` table that ``data/money_graph.py`` reads.

Signal: ``sentiment_news`` tags each article with the ticker it was fetched for.
When the SAME article URL is tagged under two different tickers, those two
companies were co-mentioned in one story — a strong, name-matching-free
co-mention signal. We keep a pair only when the shared headline carries
partnership / M&A / supply / investment language, so the edges mean something.

Writes:
    news_edges(src_ticker, dst_ticker, headline, url, published_at,
               confidence, event_type)

Run from project root (venv active):
    python backfill_news_edges.py                 # build from existing news_articles
    python backfill_news_edges.py --fetch         # fetch fresh news first (universe)
    python backfill_news_edges.py --fetch --limit 100
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from itertools import combinations
from pathlib import Path

# --- DB path -------------------------------------------------------------
try:
    from data import config as cfg  # type: ignore
except Exception:
    try:
        import config as cfg        # type: ignore
    except Exception:
        cfg = None
MARKET_DB = (getattr(cfg, "MARKET_DB_PATH", None)
             or getattr(cfg, "HEDGEFUND_DB_PATH", None)
             or "data/hedgefund.db")

try:
    from data.money_graph import universe_tickers
except Exception:
    from money_graph import universe_tickers  # type: ignore

# relationship language -> event_type (first match wins), with a confidence.
_EVENT_RULES = [
    (re.compile(r"\b(acquir|acquisition|takeover|buyout|to buy|merger|merge)\b", re.I),
     "m&a", 0.55),
    (re.compile(r"\b(partner|partnership|alliance|joint venture|team up|collaborat)\b", re.I),
     "partnership", 0.5),
    (re.compile(r"\b(supply|supplier|sources? from|chips? for|orders? from)\b", re.I),
     "supply", 0.45),
    (re.compile(r"\b(invest|stake|funding round|backs|backed by)\b", re.I),
     "investment", 0.45),
    (re.compile(r"\b(contract|deal|agreement|licens)\b", re.I),
     "deal", 0.42),
]


def _classify(headline: str):
    for rx, etype, conf in _EVENT_RULES:
        if rx.search(headline or ""):
            return etype, conf
    return None, None


def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_ticker TEXT NOT NULL,
            dst_ticker TEXT NOT NULL,
            headline TEXT,
            url TEXT,
            published_at TEXT,
            confidence REAL,
            event_type TEXT,
            fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(src_ticker, dst_ticker, url)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_edges_pair "
                 "ON news_edges(src_ticker, dst_ticker)")


def _fetch_news(universe, limit: int) -> None:
    """Optionally populate news_articles via sentiment_news (Google-News RSS)."""
    try:
        from data.sentiment_news import SentimentNews
    except Exception:
        from sentiment_news import SentimentNews  # type: ignore
    sn = SentimentNews()
    tickers = universe[:limit] if limit else universe
    print(f"Fetching news for {len(tickers)} tickers…")
    for i, tk in enumerate(tickers, 1):
        try:
            sn.refresh_ticker(tk)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: {e}")
        if i % 25 == 0:
            print(f"  …{i}/{len(tickers)}")
        time.sleep(0.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="fetch fresh news for the universe before extracting")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap tickers when --fetch (0 = all)")
    args = ap.parse_args()

    universe = set(universe_tickers())
    if args.fetch:
        _fetch_news(sorted(universe), args.limit)

    conn = sqlite3.connect(MARKET_DB)
    _ensure_table(conn)

    # articles tagged to in-universe tickers, grouped by URL
    try:
        rows = conn.execute(
            "SELECT url, ticker, title, published_at FROM news_articles "
            "WHERE url IS NOT NULL AND url <> ''").fetchall()
    except sqlite3.OperationalError:
        print("No news_articles table found. Run sentiment_news ingestion first, "
              "or use --fetch.")
        conn.close()
        return

    by_url: dict[str, dict] = {}
    for url, ticker, title, pub in rows:
        tk = (ticker or "").upper()
        if tk not in universe:
            continue
        slot = by_url.setdefault(url, {"tickers": set(), "title": title, "pub": pub})
        slot["tickers"].add(tk)
        if title and not slot["title"]:
            slot["title"] = title

    written = pairs_seen = 0
    for url, slot in by_url.items():
        tks = sorted(slot["tickers"])
        if len(tks) < 2:
            continue
        etype, conf = _classify(slot["title"])
        if not etype:                       # require relationship language
            continue
        for a, b in combinations(tks, 2):   # every co-mentioned pair in the story
            pairs_seen += 1
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO news_edges "
                    "(src_ticker, dst_ticker, headline, url, published_at, "
                    " confidence, event_type) VALUES (?,?,?,?,?,?,?)",
                    (a, b, (slot["title"] or "")[:300], url, slot["pub"],
                     conf, etype))
                written += conn.total_changes and 1 or 0
            except Exception:  # noqa: BLE001
                pass
    conn.commit()
    n_edges = conn.execute("SELECT COUNT(*) FROM news_edges").fetchone()[0]
    conn.close()

    print(f"Articles scanned: {len(rows)} | shared-URL co-mention pairs: "
          f"{pairs_seen} | news_edges rows now: {n_edges}")
    if n_edges == 0:
        print("No pairs found. Co-mentions need the same article tagged under "
              "two universe tickers with partnership/M&A language — run --fetch "
              "to broaden coverage, or ingest more news first.")
    else:
        print("Rebuild the graph -> dashed 'suspected' edges will appear.")


if __name__ == "__main__":
    main()
