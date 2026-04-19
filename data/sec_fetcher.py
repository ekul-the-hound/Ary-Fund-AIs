"""
SEC EDGAR Filing Fetcher
========================
Pulls 10-K, 10-Q, 8-K filings from SEC EDGAR using their free REST API.
Handles rate limiting (10 req/sec), caching, CIK lookup, and full-text extraction.

SEC requires a User-Agent header with your name and email.
Set environment variables:
    SEC_AGENT_NAME=YourName
    SEC_AGENT_EMAIL=you@email.com

Usage:
    fetcher = SECFetcher()
    filings = fetcher.get_filings("AAPL", filing_type="10-K", count=5)
    text = fetcher.get_filing_text(filings[0]["accession_number"])
"""

import os
import re
import json
import time
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
RATE_LIMIT_DELAY = 0.12  # SEC allows 10 req/sec — stay under at ~8/sec

FILING_TYPES = {"10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR", "SC 13D"}


class SECFetcher:
    """Fetch and cache SEC EDGAR filings."""

    def __init__(
        self,
        db_path: str = "data/hedgefund.db",
        cache_dir: str = "data/sec_cache",
        agent_name: Optional[str] = None,
        agent_email: Optional[str] = None,
    ):
        self.agent_name = agent_name or os.getenv("SEC_AGENT_NAME", "HedgeFundAI")
        self.agent_email = agent_email or os.getenv("SEC_AGENT_EMAIL", "user@example.com")
        self.headers = {
            "User-Agent": f"{self.agent_name} {self.agent_email}",
            "Accept-Encoding": "gzip, deflate",
        }

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self._last_request_time = 0.0
        self._cik_cache: dict[str, str] = {}

        self._init_db()

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------
    def _init_db(self):
        """Create filings table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sec_filings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    filing_type TEXT NOT NULL,
                    filed_date TEXT NOT NULL,
                    accession_number TEXT NOT NULL UNIQUE,
                    primary_doc_url TEXT,
                    description TEXT,
                    full_text TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(ticker, accession_number)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filings_ticker
                ON sec_filings(ticker, filing_type, filed_date)
            """)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    def _throttle(self):
        """Enforce SEC's 10 requests/second rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, **kwargs) -> requests.Response:
        """Rate-limited GET request."""
        self._throttle()
        resp = requests.get(url, headers=self.headers, timeout=30, **kwargs)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------
    # CIK lookup (ticker -> CIK number)
    # ------------------------------------------------------------------
    def get_cik(self, ticker: str) -> str:
        """
        Resolve a stock ticker to its SEC Central Index Key (CIK).

        The CIK is SEC's unique identifier for every entity that files.
        Think of it like a Social Security number for companies at the SEC.
        """
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        # SEC provides a full ticker->CIK mapping file
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(url)
        data = resp.json()

        # Build reverse lookup: ticker -> zero-padded CIK
        for entry in data.values():
            t = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            self._cik_cache[t] = cik

        if ticker not in self._cik_cache:
            raise ValueError(
                f"Ticker '{ticker}' not found in SEC EDGAR. "
                f"Check spelling or try the company name search."
            )
        return self._cik_cache[ticker]

    # ------------------------------------------------------------------
    # Filing index retrieval
    # ------------------------------------------------------------------
    def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch filing metadata from SEC EDGAR.

        Args:
            ticker:      Stock ticker (e.g., "AAPL")
            filing_type: One of 10-K, 10-Q, 8-K, DEF 14A, S-1, 13F-HR, SC 13D
            count:       Max filings to return
            start_date:  Filter filings after this date (YYYY-MM-DD)
            end_date:    Filter filings before this date (YYYY-MM-DD)

        Returns:
            List of dicts with keys: ticker, cik, filing_type, filed_date,
            accession_number, primary_doc_url, description
        """
        ticker = ticker.upper()
        cik = self.get_cik(ticker)

        # First check the DB cache
        cached = self._get_cached_filings(ticker, filing_type, count, start_date, end_date)
        if len(cached) >= count:
            logger.info(f"Returning {len(cached)} cached filings for {ticker} {filing_type}")
            return cached[:count]

        # Fetch from SEC submissions endpoint
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
        resp = self._get(url)
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning(f"No recent filings found for {ticker}")
            return []

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i in range(len(forms)):
            if forms[i] != filing_type:
                continue

            filed = dates[i]
            if start_date and filed < start_date:
                continue
            if end_date and filed > end_date:
                continue

            acc_clean = accessions[i].replace("-", "")
            doc_url = f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_clean}/{primary_docs[i]}"

            filing = {
                "ticker": ticker,
                "cik": cik,
                "filing_type": filing_type,
                "filed_date": filed,
                "accession_number": accessions[i],
                "primary_doc_url": doc_url,
                "description": descriptions[i] if i < len(descriptions) else "",
            }
            filings.append(filing)

            if len(filings) >= count:
                break

        # Cache to DB
        self._cache_filings(filings)
        return filings

    # ------------------------------------------------------------------
    # Full text extraction
    # ------------------------------------------------------------------
    def get_filing_text(
        self, accession_number: str, max_chars: int = 500_000
    ) -> str:
        """
        Download and extract plain text from a filing.

        SEC filings are HTML/SGML documents. This strips tags and returns
        clean text, truncated to max_chars (default 500K ~ enough for
        your 30B Qwen3 context window with room for the prompt).

        Args:
            accession_number: The SEC accession number (e.g., "0000320193-23-000106")
            max_chars: Truncate text to this length

        Returns:
            Plain text of the filing
        """
        # Check disk cache first
        cache_key = hashlib.md5(accession_number.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
            return text[:max_chars]

        # Check DB for URL
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT primary_doc_url, full_text FROM sec_filings WHERE accession_number = ?",
                (accession_number,),
            ).fetchone()

        if row and row[1]:
            return row[1][:max_chars]

        if not row or not row[0]:
            raise ValueError(
                f"Accession number '{accession_number}' not found in cache. "
                f"Call get_filings() first to populate the index."
            )

        doc_url = row[0]
        logger.info(f"Downloading filing: {doc_url}")

        resp = self._get(doc_url)
        content_type = resp.headers.get("Content-Type", "")

        if "html" in content_type or doc_url.endswith(".htm") or doc_url.endswith(".html"):
            text = self._extract_html(resp.text)
        else:
            text = resp.text

        # Clean up common SEC artifacts
        text = self._clean_sec_text(text)
        text = text[:max_chars]

        # Cache to disk and DB
        cache_file.write_text(text, encoding="utf-8")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sec_filings SET full_text = ? WHERE accession_number = ?",
                (text, accession_number),
            )

        return text

    # ------------------------------------------------------------------
    # Full-text search across cached filings
    # ------------------------------------------------------------------
    def search_filings(
        self, ticker: str, query: str, filing_type: Optional[str] = None
    ) -> list[dict]:
        """
        Search cached filing texts for a keyword/phrase.

        Returns matching filings with a snippet around the match.
        Useful for finding specific disclosures (e.g., "goodwill impairment").
        """
        ticker = ticker.upper()
        sql = """
            SELECT ticker, filing_type, filed_date, accession_number, full_text
            FROM sec_filings
            WHERE ticker = ? AND full_text IS NOT NULL
        """
        params: list = [ticker]

        if filing_type:
            sql += " AND filing_type = ?"
            params.append(filing_type)

        results = []
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(sql, params):
                text = row[4]
                idx = text.lower().find(query.lower())
                if idx == -1:
                    continue

                # Extract snippet: 200 chars around match
                start = max(0, idx - 200)
                end = min(len(text), idx + len(query) + 200)
                snippet = text[start:end]

                results.append({
                    "ticker": row[0],
                    "filing_type": row[1],
                    "filed_date": row[2],
                    "accession_number": row[3],
                    "snippet": f"...{snippet}...",
                })

        return results

    # ------------------------------------------------------------------
    # 8-K event detection (material events)
    # ------------------------------------------------------------------
    def get_recent_8k_events(
        self, ticker: str, days_back: int = 30
    ) -> list[dict]:
        """
        Fetch recent 8-K filings (material events).

        8-Ks are filed when something significant happens: earnings,
        acquisitions, leadership changes, material agreements, etc.
        This is your early-warning system for portfolio holdings.

        Fun fact: Companies must file an 8-K within 4 business days
        of a triggering event. Some hedge funds monitor these in
        real-time for alpha signals.
        """
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        return self.get_filings(ticker, filing_type="8-K", count=20, start_date=start)

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------
    def fetch_portfolio_filings(
        self,
        tickers: list[str],
        filing_type: str = "10-K",
        count: int = 3,
        download_text: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Bulk-fetch filings for an entire portfolio watchlist.

        Args:
            tickers:       List of ticker symbols
            filing_type:   Type of filing to fetch
            count:         Filings per ticker
            download_text: If True, also downloads full text (slow but thorough)

        Returns:
            Dict mapping ticker -> list of filing dicts
        """
        results = {}
        for ticker in tickers:
            try:
                filings = self.get_filings(ticker, filing_type=filing_type, count=count)
                if download_text:
                    for f in filings:
                        try:
                            self.get_filing_text(f["accession_number"])
                        except Exception as e:
                            logger.warning(f"Could not download text for {ticker} "
                                           f"{f['accession_number']}: {e}")
                results[ticker] = filings
                logger.info(f"Fetched {len(filings)} {filing_type} filings for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch filings for {ticker}: {e}")
                results[ticker] = []
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _extract_html(self, html: str) -> str:
        """Strip HTML tags, scripts, styles from SEC filing HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_sec_text(self, text: str) -> str:
        """Remove common SEC filing boilerplate noise."""
        # Remove XBRL inline tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove excessive whitespace
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove page break markers
        text = re.sub(r"-{5,}", "", text)
        # Remove Unicode junk
        text = text.encode("ascii", errors="ignore").decode("ascii")
        return text.strip()

    def _get_cached_filings(
        self, ticker, filing_type, count, start_date, end_date
    ) -> list[dict]:
        """Retrieve filings from SQLite cache."""
        sql = """
            SELECT ticker, cik, filing_type, filed_date,
                   accession_number, primary_doc_url, description
            FROM sec_filings
            WHERE ticker = ? AND filing_type = ?
        """
        params: list = [ticker, filing_type]

        if start_date:
            sql += " AND filed_date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND filed_date <= ?"
            params.append(end_date)

        sql += " ORDER BY filed_date DESC LIMIT ?"
        params.append(count)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def _cache_filings(self, filings: list[dict]):
        """Upsert filings into SQLite."""
        if not filings:
            return
        with sqlite3.connect(self.db_path) as conn:
            for f in filings:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sec_filings
                        (ticker, cik, filing_type, filed_date,
                         accession_number, primary_doc_url, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f["ticker"], f["cik"], f["filing_type"], f["filed_date"],
                        f["accession_number"], f["primary_doc_url"], f["description"],
                    ),
                )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = SECFetcher()

    print("=== SEC Fetcher Test ===\n")

    # Test CIK lookup
    cik = fetcher.get_cik("AAPL")
    print(f"AAPL CIK: {cik}")

    # Fetch recent 10-K filings
    filings = fetcher.get_filings("AAPL", filing_type="10-K", count=3)
    for f in filings:
        print(f"  {f['filed_date']} | {f['filing_type']} | {f['accession_number']}")

    # Fetch text of most recent
    if filings:
        text = fetcher.get_filing_text(filings[0]["accession_number"])
        print(f"\nFirst 500 chars of latest 10-K:\n{text[:500]}")

    # Test 8-K events
    events = fetcher.get_recent_8k_events("AAPL", days_back=90)
    print(f"\n8-K events in last 90 days: {len(events)}")
    for e in events[:3]:
        print(f"  {e['filed_date']} | {e['description']}")
