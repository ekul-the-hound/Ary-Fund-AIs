"""
sentiment_news.py
=================
Social, news, and analyst signal ingestion for Ary Quant.

Sources (all free, in priority order):
    * yfinance news endpoint  — recent headlines per ticker
    * Stocktwits public API   — message volume & sentiment per ticker
    * Google News RSS         — broad coverage when nothing else has it
    * GDELT 2.0 events        — financial-themed event counts
    * Reddit r/wallstreetbets — mention counts (best-effort scrape)
    * yfinance analyst info   — consensus target, recommendation key

Every fetch writes both:
    1. raw rows into ``social_mentions`` / ``news_articles`` / ``analyst_events``
    2. canonical aggregates into ``data_points`` via the ``DataRegistry``

Design notes
------------
* This file owns the consumer-facing class ``SentimentNews``.
* All fetches are wrapped in best-effort try/except: a failure in one source
  must not block the others.
* No API keys required for the default path. PRAW + a Reddit client_id
  are an optional add-on; Stocktwits and yfinance work without auth.
* VADER sentiment is used as a lightweight default (no GPU); FinBERT is
  loaded only if the user pre-installs ``transformers``.

Usage
-----
    >>> sn = SentimentNews()
    >>> sn.refresh_ticker("AAPL")
    {'news': 18, 'social': 2, 'analyst': 1}
"""

from __future__ import annotations

import logging
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)


# Sources we register. Confidence per CONFIDENCE_GUIDE in data_registry.
SRC_YF_NEWS = "yfinance"
SRC_STOCKTWITS = "stocktwits"
SRC_GOOGLE_NEWS = "google_news_rss"
SRC_GDELT = "gdelt"
SRC_REDDIT_WSB = "reddit_wsb"
SRC_FINNHUB = "finnhub"


# ----------------------------------------------------------------------
# Tiny VADER-style polarity dictionary. Not as good as the full NLTK
# VADER, but good enough as a default signal and avoids a 30 MB install.
# Users who want better can pip install vaderSentiment and we'll prefer
# that automatically.
# ----------------------------------------------------------------------
_POSITIVE = {
    "beat", "beats", "beating", "outperform", "outperforming", "strong",
    "growth", "surged", "rally", "rallying", "bullish", "buy", "upgrade",
    "upgraded", "raised", "expand", "expanding", "record", "soared",
    "jumped", "gain", "gains", "profit", "profits", "win", "winning",
}
_NEGATIVE = {
    "miss", "missed", "missing", "underperform", "weak", "decline",
    "declined", "fell", "dropped", "bearish", "sell", "downgrade",
    "downgraded", "lowered", "cut", "cuts", "warning", "warned", "loss",
    "losses", "lawsuit", "probe", "investigation", "fraud", "bankruptcy",
    "bankrupt", "delisted", "default", "recession", "crash",
}


def _try_load_vader():
    """If the user installed ``vaderSentiment``, use it; else return None."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:  # noqa: BLE001
        return None


def _polarity(text: str, vader=None) -> float:
    """Return sentiment in -1..+1. Uses VADER if available, else word-list fallback."""
    if not text:
        return 0.0
    if vader is not None:
        try:
            return float(vader.polarity_scores(text)["compound"])
        except Exception:  # noqa: BLE001
            pass
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POSITIVE)
    neg = sum(1 for t in tokens if t in _NEGATIVE)
    n = pos + neg
    if n == 0:
        return 0.0
    return (pos - neg) / max(n, 1)


# ----------------------------------------------------------------------
# Class
# ----------------------------------------------------------------------


class SentimentNews:
    """News, social, and analyst-event ingestion. Writes into the registry."""

    def __init__(
        self,
        db_path: str = "data/hedgefund.db",
        registry=None,
        timeout: int = 15,
        user_agent: str = "ary-quant/1.0 (+research)",
    ):
        self.db_path = db_path
        self._registry = registry
        self.timeout = timeout
        self.user_agent = user_agent
        self._vader = _try_load_vader()
        self._init_db()
        self._register_sources()

    # ------------------------------------------------------------------
    # Registry plumbing
    # ------------------------------------------------------------------
    @property
    def registry(self):
        if self._registry is None:
            try:
                from data.data_registry import get_default_registry
                self._registry = get_default_registry(self.db_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("sentiment_news | registry load failed: %s", e)
        return self._registry

    def _register_sources(self) -> None:
        reg = self.registry
        if reg is None:
            return
        try:
            reg.register_source(SRC_YF_NEWS,    "news",    "hourly", base_priority=2)
            reg.register_source(SRC_STOCKTWITS, "social",  "hourly", base_priority=4)
            reg.register_source(SRC_GOOGLE_NEWS,"news",    "hourly", base_priority=3)
            reg.register_source(SRC_GDELT,     "news",    "hourly", base_priority=3)
            reg.register_source(SRC_REDDIT_WSB,"social",  "hourly", base_priority=4)
            reg.register_source(SRC_FINNHUB,   "analyst", "daily",  base_priority=2)
        except Exception as e:  # noqa: BLE001
            logger.debug("sentiment_news | source registration skipped: %s", e)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    publisher TEXT,
                    url TEXT,
                    published_at TEXT,
                    source TEXT NOT NULL,
                    sentiment REAL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(ticker, url)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_tk ON news_articles(ticker, published_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS social_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    posted_at TEXT,
                    body TEXT,
                    sentiment REAL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(ticker, platform, posted_at, body)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_soc_tk ON social_mentions(ticker, posted_at)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyst_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    firm TEXT,
                    detail_json TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(ticker, event_type, occurred_at, firm)
                )
            """)

    # ------------------------------------------------------------------
    # yfinance news (default path - no API key)
    # ------------------------------------------------------------------
    def fetch_yfinance_news(self, ticker: str) -> int:
        """Pull recent news headlines from yfinance.Ticker(...).news."""
        ticker = ticker.upper()
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("sentiment_news | yfinance not installed")
            return 0
        try:
            items = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            logger.warning("yfinance news | %s failed: %s", ticker, e)
            return 0

        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for it in items:
                # yfinance changed shape multiple times. Handle both.
                content = it.get("content") if isinstance(it, dict) else None
                if isinstance(content, dict):
                    title = content.get("title") or ""
                    url = (content.get("canonicalUrl", {}) or {}).get("url") or ""
                    publisher = (content.get("provider", {}) or {}).get("displayName") or ""
                    pub_date = content.get("pubDate") or ""
                else:
                    title = it.get("title", "") if isinstance(it, dict) else ""
                    url = it.get("link", "") if isinstance(it, dict) else ""
                    publisher = it.get("publisher", "") if isinstance(it, dict) else ""
                    ts = it.get("providerPublishTime") if isinstance(it, dict) else None
                    pub_date = (
                        datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                        if ts else ""
                    )
                if not title or not url:
                    continue
                sentiment = _polarity(title, self._vader)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO news_articles
                            (ticker, title, publisher, url, published_at, source, sentiment)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker, title, publisher, url, pub_date, SRC_YF_NEWS, sentiment),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        return n

    # ------------------------------------------------------------------
    # Stocktwits (free, no key)
    # ------------------------------------------------------------------
    def fetch_stocktwits(self, ticker: str) -> int:
        ticker = ticker.upper()
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("stocktwits | %s failed: %s", ticker, e)
            return 0
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for msg in data.get("messages", []):
                body = msg.get("body", "")
                created = msg.get("created_at", "")
                # Stocktwits provides explicit Bull/Bear sentiment
                st_sent = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")
                if st_sent == "Bullish":
                    sent = 1.0
                elif st_sent == "Bearish":
                    sent = -1.0
                else:
                    sent = _polarity(body, self._vader)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO social_mentions
                            (ticker, platform, posted_at, body, sentiment)
                           VALUES (?, 'stocktwits', ?, ?, ?)""",
                        (ticker, created, body, sent),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        return n

    # ------------------------------------------------------------------
    # Google News RSS
    # ------------------------------------------------------------------
    def fetch_google_news_rss(self, ticker: str) -> int:
        """Use Google News RSS as a broad-coverage fallback."""
        ticker = ticker.upper()
        url = (
            f"https://news.google.com/rss/search?q=%22{ticker}%22+stock"
            "&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            root = ET.fromstring(resp.text)
        except Exception as e:  # noqa: BLE001
            logger.debug("google news rss | %s failed: %s", ticker, e)
            return 0
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                pubname = (item.find("source").text if item.find("source") is not None else "")
                if not title or not link:
                    continue
                sentiment = _polarity(title, self._vader)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO news_articles
                            (ticker, title, publisher, url, published_at, source, sentiment)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker, title, pubname, link, pub, SRC_GOOGLE_NEWS, sentiment),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        return n

    # ------------------------------------------------------------------
    # Reddit r/wallstreetbets (best-effort, public JSON)
    # ------------------------------------------------------------------
    def fetch_wsb_mentions(self, ticker: str, limit: int = 100) -> int:
        """Scan recent /r/wallstreetbets posts for ticker mentions.

        Uses Reddit's public ``.json`` endpoint which doesn't require auth
        but is heavily rate-limited and may return 429. We treat 429 as a
        soft failure.
        """
        ticker = ticker.upper()
        url = f"https://www.reddit.com/r/wallstreetbets/new.json?limit={limit}"
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("wsb | %s failed: %s", ticker, e)
            return 0

        pattern = re.compile(rf"\b\$?{re.escape(ticker)}\b")
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for child in (data.get("data") or {}).get("children", []):
                p = child.get("data", {})
                title = p.get("title", "") or ""
                body = p.get("selftext", "") or ""
                combined = f"{title}\n{body}"
                if not pattern.search(combined.upper()):
                    continue
                created = p.get("created_utc")
                created_iso = (
                    datetime.fromtimestamp(int(created), tz=timezone.utc).isoformat()
                    if created else ""
                )
                sentiment = _polarity(combined, self._vader)
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO social_mentions
                            (ticker, platform, posted_at, body, sentiment)
                           VALUES (?, 'wsb', ?, ?, ?)""",
                        (ticker, created_iso, title[:500], sentiment),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        return n

    # ------------------------------------------------------------------
    # GDELT 2.0 — financial event counts
    # ------------------------------------------------------------------
    def fetch_gdelt_events(self, ticker: str, hours_back: int = 24) -> int:
        """Pull GDELT DOC API counts for a ticker over the last N hours.
        Returns the row written (we store one per ticker per call as an
        aggregate)."""
        ticker = ticker.upper()
        # Use the DOC API which returns JSON
        ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_back))
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc"
            f"?query=%22{ticker}%22+stock&mode=ArtList&maxrecords=50"
            f"&startdatetime={ts_str}&format=json"
        )
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            try:
                data = resp.json()
            except ValueError:
                return 0
        except Exception as e:  # noqa: BLE001
            logger.debug("gdelt | %s failed: %s", ticker, e)
            return 0

        articles = data.get("articles") or []
        tones: list[float] = []
        for a in articles:
            t = a.get("tone")
            if t is None:
                continue
            try:
                tones.append(float(t))
            except (TypeError, ValueError):
                continue
        as_of = datetime.now().strftime("%Y-%m-%d")
        if self.registry:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.sentiment.news_count_7d",
                as_of=as_of, source_id=SRC_GDELT,
                value_num=float(len(articles)), confidence=0.8,
            )
            if tones:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.sentiment.news_tone_7d",
                    as_of=as_of, source_id=SRC_GDELT,
                    value_num=float(statistics.mean(tones)), confidence=0.8,
                )
        return 1 if articles else 0

    # ------------------------------------------------------------------
    # Analyst events via yfinance (no key)
    # ------------------------------------------------------------------
    def fetch_analyst_events(self, ticker: str) -> int:
        ticker = ticker.upper()
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}
        except Exception as e:  # noqa: BLE001
            logger.debug("analyst | %s failed: %s", ticker, e)
            return 0

        n = 0
        as_of = datetime.now().strftime("%Y-%m-%d")
        target = info.get("targetMeanPrice")
        n_analysts = info.get("numberOfAnalystOpinions")
        rec = info.get("recommendationKey")
        eps_est = info.get("earningsEstimateAvg") or (info.get("earningsEstimate") or {}).get("avg")
        next_earn = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if isinstance(next_earn, (int, float)) and next_earn:
            try:
                next_earn = datetime.fromtimestamp(int(next_earn), tz=timezone.utc).date().isoformat()
            except (OSError, ValueError):
                next_earn = None

        if self.registry:
            if target is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.analyst.consensus_target",
                    as_of=as_of, source_id=SRC_YF_NEWS,
                    value_num=float(target), confidence=0.9,
                )
                n += 1
            if next_earn:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.analyst.next_earnings_date",
                    as_of=as_of, source_id=SRC_YF_NEWS,
                    value_text=str(next_earn), confidence=0.9,
                )
                n += 1
        # Also push a row into analyst_events as a structured snapshot
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO analyst_events
                        (ticker, event_type, occurred_at, firm, detail_json)
                       VALUES (?, 'consensus_snapshot', ?, 'aggregate', ?)""",
                    (ticker, as_of, str({
                        "target": target, "n_analysts": n_analysts,
                        "rec": rec, "eps_est": eps_est, "next_earn": next_earn,
                    })),
                )
            except sqlite3.IntegrityError:
                pass
        return n

    # ------------------------------------------------------------------
    # Aggregation -> registry
    # ------------------------------------------------------------------
    def refresh_aggregates(self, ticker: str) -> int:
        """Recompute 7-day news count + tone, and 24h WSB mention count."""
        if self.registry is None:
            return 0
        ticker = ticker.upper()
        as_of = datetime.now().strftime("%Y-%m-%d")
        cutoff_7d = (datetime.now() - timedelta(days=7)).isoformat()
        cutoff_24h = (datetime.now() - timedelta(hours=24)).isoformat()
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            news = conn.execute(
                """SELECT COUNT(*), AVG(sentiment) FROM news_articles
                    WHERE ticker = ? AND published_at >= ?""",
                (ticker, cutoff_7d),
            ).fetchone()
            wsb = conn.execute(
                """SELECT COUNT(*), AVG(sentiment) FROM social_mentions
                    WHERE ticker = ? AND platform='wsb' AND posted_at >= ?""",
                (ticker, cutoff_24h),
            ).fetchone()

        if news and news[0] is not None:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.sentiment.news_count_7d",
                as_of=as_of, source_id="derived",
                value_num=float(news[0] or 0), confidence=0.7,
            )
            n += 1
            if news[1] is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.sentiment.news_tone_7d",
                    as_of=as_of, source_id="derived",
                    value_num=float(news[1] or 0), confidence=0.7,
                )
                n += 1
        if wsb and wsb[0] is not None:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.sentiment.wsb_mentions_24h",
                as_of=as_of, source_id="derived",
                value_num=float(wsb[0] or 0), confidence=0.4,
            )
            n += 1
            if wsb[1] is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.sentiment.wsb_score",
                    as_of=as_of, source_id="derived",
                    value_num=float(wsb[1] or 0), confidence=0.4,
                )
                n += 1
        return n

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def refresh_ticker(self, ticker: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, fn in (
            ("yf_news", lambda: self.fetch_yfinance_news(ticker)),
            ("stocktwits", lambda: self.fetch_stocktwits(ticker)),
            ("google_news", lambda: self.fetch_google_news_rss(ticker)),
            ("wsb", lambda: self.fetch_wsb_mentions(ticker)),
            ("gdelt", lambda: self.fetch_gdelt_events(ticker)),
            ("analyst", lambda: self.fetch_analyst_events(ticker)),
        ):
            try:
                out[name] = fn()
            except Exception as e:  # noqa: BLE001
                logger.warning("sentiment_news | %s | %s failed: %s", ticker, name, e)
                out[name] = 0
        try:
            out["aggregates"] = self.refresh_aggregates(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("sentiment_news | aggregates failed: %s", e)
            out["aggregates"] = 0
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sn = SentimentNews()
    print("polarity('beats expectations, strong growth'):",
          _polarity("beats expectations, strong growth"))
    print("polarity('lawsuit, missed guidance, downgrade'):",
          _polarity("lawsuit, missed guidance, downgrade"))
