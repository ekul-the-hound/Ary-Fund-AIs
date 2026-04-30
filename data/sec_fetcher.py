"""
SEC EDGAR Filing Fetcher
========================
Pulls 10-K, 10-Q, 8-K filings AND extends to:

    * XBRL ``companyfacts`` — the single highest-yield endpoint (revenue,
      EPS, capex, R&D, tax rate, goodwill, pension, dilution, SBC, ...)
    * Form 4 insider transactions (XML parse)
    * 13D / 13G ownership filings
    * 13F holdings (institutional snapshots)
    * 10b5-1 plan references (text-search across filings)
    * Share buybacks, secondary offerings, ATM offerings (form-type filters)
    * CEO/CFO turnover and board changes (8-K Item 5.02 detection)
    * Bulk fundamental ingestion that writes into ``data_registry``

Original API is preserved verbatim:

    >>> fetcher = SECFetcher()
    >>> filings = fetcher.get_filings("AAPL", filing_type="10-K", count=5)
    >>> text    = fetcher.get_filing_text(filings[0]["accession_number"])
    >>> events  = fetcher.get_recent_8k_events("AAPL", days_back=30)

New API added below the ``# === EXTENDED API ===`` divider.

Set:
    SEC_AGENT_NAME=YourName
    SEC_AGENT_EMAIL=you@email.com
"""

from __future__ import annotations

import os
import re
import json
import time
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Iterable
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_XBRL_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
EDGAR_XBRL_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
EDGAR_FULLTEXT = "https://efts.sec.gov/LATEST/search-index"  # ?q=...&forms=...
RATE_LIMIT_DELAY = 0.12  # SEC allows 10 req/sec — stay under at ~8/sec

FILING_TYPES = {
    "10-K", "10-Q", "8-K", "DEF 14A", "S-1", "13F-HR", "SC 13D", "SC 13G",
    "4", "424B5", "424B2", "424B3", "S-3", "S-3ASR", "SR",
}


# Source ID strings used for the data_registry layer.
SRC_FILING = "sec_edgar"
SRC_XBRL = "sec_xbrl"
SRC_FORM4 = "sec_form4"
SRC_13D = "sec_13d"
SRC_13G = "sec_13g"
SRC_13F = "sec_13f"


# Mapping of canonical fields -> US-GAAP XBRL concept names. Some fields
# live under multiple plausible concepts (companies tag inconsistently),
# so we keep an ordered fallback list.
XBRL_CONCEPT_MAP: dict[str, list[str]] = {
    "ticker.fundamental.revenue_ttm": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "ticker.fundamental.net_income_ttm": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "ticker.fundamental.eps_diluted_ttm": [
        "EarningsPerShareDiluted",
    ],
    "ticker.fundamental.capex_ttm": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "ticker.fundamental.rd_ttm": [
        "ResearchAndDevelopmentExpense",
    ],
    "ticker.fundamental.tax_rate": [
        "EffectiveIncomeTaxRateContinuingOperations",
    ],
    "ticker.fundamental.goodwill": [
        "Goodwill",
    ],
    "ticker.fundamental.pension_liability": [
        "DefinedBenefitPensionPlanLiabilitiesNoncurrent",
        "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
        "LiabilityForUncertainTaxPositionsNoncurrent",  # last-resort fallback
    ],
    "ticker.fundamental.shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
    "ticker.fundamental.sbc_ttm": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "ticker.fundamental.long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "ticker.fundamental.total_assets": [
        "Assets",
    ],
    "ticker.fundamental.total_liabilities": [
        "Liabilities",
    ],
    "ticker.fundamental.fcf_ttm": [
        # Often not directly tagged; we derive from CFO - capex below.
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}


# Form 4 transaction codes (SEC official codes)
FORM4_BUY_CODES = {"P", "A"}            # Open-market purchase, grant
FORM4_SELL_CODES = {"S", "F", "D"}      # Open-market sale, payment of tax, sale to issuer
# 'M' = exercise, 'G' = gift, etc. -- treated as neutral.

# 8-K item codes that map to governance / corporate-action events
EIGHTK_GOVERNANCE_ITEMS = {
    "5.02": "officer_director_change",
    "5.03": "bylaws_change",
    "8.01": "other_event",       # often used for ratings, buyback announcements
    "1.01": "material_agreement",
    "2.02": "results_of_operations",
    "3.02": "unregistered_sale",  # ATM-related sometimes
    "3.03": "modification_of_rights",
}


class SECFetcher:
    """Fetch and cache SEC EDGAR filings + XBRL facts + insider/ownership data."""

    def __init__(
        self,
        db_path: str = "data/hedgefund.db",
        cache_dir: str = "data/sec_cache",
        agent_name: Optional[str] = None,
        agent_email: Optional[str] = None,
        registry=None,  # data_registry.DataRegistry, optional
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

        # Lazily resolve registry. Imported here, not at module top, so
        # sec_fetcher remains importable even if data_registry has a bug.
        self._registry = registry
        self._init_db()
        self._init_extended_db()
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
                logger.warning("sec_fetcher | could not load data_registry: %s", e)
        return self._registry

    def _register_sources(self) -> None:
        reg = self.registry
        if reg is None:
            return
        try:
            reg.register_source(SRC_FILING, "filing", "hourly", base_priority=1,
                                notes="SEC EDGAR submissions index")
            reg.register_source(SRC_XBRL, "filing", "quarterly", base_priority=1,
                                notes="SEC XBRL companyfacts (gold standard)")
            reg.register_source(SRC_FORM4, "filing", "hourly", base_priority=1,
                                notes="SEC Form 4 insider transactions")
            reg.register_source(SRC_13D, "filing", "daily", base_priority=1,
                                notes="SEC SC 13D ownership filings")
            reg.register_source(SRC_13G, "filing", "daily", base_priority=1,
                                notes="SEC SC 13G ownership filings")
            reg.register_source(SRC_13F, "filing", "weekly", base_priority=1,
                                notes="SEC 13F-HR institutional holdings")
        except Exception as e:  # noqa: BLE001
            logger.debug("sec_fetcher | source registration skipped: %s", e)

    # ------------------------------------------------------------------
    # Database setup
    # ------------------------------------------------------------------
    def _init_db(self):
        """Create filings table if it doesn't exist (existing behavior)."""
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

    def _init_extended_db(self):
        """Create raw tables for extended SEC features."""
        with sqlite3.connect(self.db_path) as conn:
            # Form 4 insider transactions, parsed
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insider_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    insider_name TEXT,
                    insider_title TEXT,
                    transaction_date TEXT,
                    transaction_code TEXT,
                    direction TEXT CHECK(direction IN ('BUY','SELL','OTHER')),
                    shares REAL,
                    price REAL,
                    value_usd REAL,
                    is_10b5_1 INTEGER DEFAULT 0,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(accession_number, insider_name, transaction_date, transaction_code, shares)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ins_ticker ON insider_transactions(ticker, transaction_date)")

            # 13D / 13G ownership filings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ownership_filings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    filing_type TEXT NOT NULL,
                    filed_date TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    filer_name TEXT,
                    pct_owned REAL,
                    shares_owned REAL,
                    is_amendment INTEGER DEFAULT 0,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(accession_number, filer_name)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_own_ticker ON ownership_filings(ticker, filed_date)")

            # 13F holdings (one row per holding within a 13F)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS f13f_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filer_cik TEXT NOT NULL,
                    accession_number TEXT NOT NULL,
                    period_of_report TEXT,
                    cusip TEXT,
                    issuer_name TEXT,
                    class_title TEXT,
                    value_usd REAL,
                    shares REAL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(accession_number, cusip)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_13f_cusip ON f13f_holdings(cusip)")

            # XBRL facts cache (one row per (ticker, concept, period))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS xbrl_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    fy INTEGER,
                    fp TEXT,
                    form TEXT,
                    value REAL,
                    unit TEXT,
                    accession_number TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(ticker, concept, period_end, form)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_xbrl_tk ON xbrl_facts(ticker, concept, period_end)")

            # Corporate actions (buybacks, secondaries, ATMs, exec changes)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    accession_number TEXT,
                    amount_usd REAL,
                    detail_json TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(accession_number, action_type)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corp_ticker ON corporate_actions(ticker, occurred_at)")

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
    # CIK lookup (ticker -> CIK number)  [unchanged]
    # ------------------------------------------------------------------
    def get_cik(self, ticker: str) -> str:
        ticker = ticker.upper()
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        url = "https://www.sec.gov/files/company_tickers.json"
        resp = self._get(url)
        data = resp.json()

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
    # Filing index retrieval  [unchanged]
    # ------------------------------------------------------------------
    def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        count: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> list[dict]:
        ticker = ticker.upper()
        cik = self.get_cik(ticker)

        cached = self._get_cached_filings(ticker, filing_type, count, start_date, end_date)
        if len(cached) >= count:
            logger.info(f"Returning {len(cached)} cached filings for {ticker} {filing_type}")
            return cached[:count]

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

        self._cache_filings(filings)
        return filings

    # ------------------------------------------------------------------
    # Full text extraction  [unchanged]
    # ------------------------------------------------------------------
    def get_filing_text(
        self, accession_number: str, max_chars: int = 500_000
    ) -> str:
        cache_key = hashlib.md5(accession_number.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
            return text[:max_chars]

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

        text = self._clean_sec_text(text)
        text = text[:max_chars]

        cache_file.write_text(text, encoding="utf-8")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sec_filings SET full_text = ? WHERE accession_number = ?",
                (text, accession_number),
            )

        return text

    # ------------------------------------------------------------------
    # Full-text search across cached filings  [unchanged]
    # ------------------------------------------------------------------
    def search_filings(
        self, ticker: str, query: str, filing_type: Optional[str] = None
    ) -> list[dict]:
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
    # 8-K event detection  [unchanged]
    # ------------------------------------------------------------------
    def get_recent_8k_events(
        self, ticker: str, days_back: int = 30
    ) -> list[dict]:
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        return self.get_filings(ticker, filing_type="8-K", count=20, start_date=start)

    # ------------------------------------------------------------------
    # Batch operations  [unchanged]
    # ------------------------------------------------------------------
    def fetch_portfolio_filings(
        self,
        tickers: list[str],
        filing_type: str = "10-K",
        count: int = 3,
        download_text: bool = False,
    ) -> dict[str, list[dict]]:
        results = {}
        for ticker in tickers:
            try:
                filings = self.get_filings(ticker, filing_type=filing_type, count=count)
                if download_text:
                    for f in filings:
                        try:
                            self.get_filing_text(f["accession_number"])
                        except Exception as e:
                            logger.warning(
                                f"Could not download text for {ticker} {f['accession_number']}: {e}"
                            )
                results[ticker] = filings
                logger.info(f"Fetched {len(filings)} {filing_type} filings for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch filings for {ticker}: {e}")
                results[ticker] = []
        return results

    # ==================================================================
    # === EXTENDED API ===
    # ==================================================================

    # ------------------------------------------------------------------
    # XBRL company facts
    # ------------------------------------------------------------------
    def get_company_facts(self, ticker: str) -> dict:
        """Fetch the full XBRL companyfacts JSON for a ticker.

        This single endpoint is the highest-yield call in the entire
        data layer — every numeric concept the company has ever filed,
        keyed by us-gaap concept name.
        """
        cik = self.get_cik(ticker.upper())
        url = EDGAR_XBRL_FACTS.format(cik=cik)
        resp = self._get(url)
        return resp.json()

    def get_xbrl_concept(
        self, ticker: str, concept: str
    ) -> Optional[dict]:
        """Fetch a single XBRL concept (e.g. 'Revenues') for a ticker."""
        cik = self.get_cik(ticker.upper())
        url = EDGAR_XBRL_CONCEPT.format(cik=cik, concept=concept)
        try:
            resp = self._get(url)
            return resp.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise

    def ingest_xbrl_facts(self, ticker: str) -> int:
        """Pull companyfacts and write every relevant concept into both
        the raw ``xbrl_facts`` table and the unified ``data_points`` table.

        Returns the number of canonical (field, period) rows written.
        """
        ticker = ticker.upper()
        cik = self.get_cik(ticker)
        try:
            facts = self.get_company_facts(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("xbrl | %s | facts fetch failed: %s", ticker, e)
            return 0

        us_gaap = (facts.get("facts") or {}).get("us-gaap", {})
        if not us_gaap:
            logger.info("xbrl | %s | no us-gaap facts present", ticker)
            return 0

        # Walk the canonical map, choose the first concept the company has,
        # write each annual + TTM observation.
        n_rows = 0
        registry_rows: list[dict] = []
        with sqlite3.connect(self.db_path) as conn:
            for canonical_field, candidates in XBRL_CONCEPT_MAP.items():
                concept = next((c for c in candidates if c in us_gaap), None)
                if concept is None:
                    continue
                concept_data = us_gaap[concept]
                units = concept_data.get("units", {})
                # Pick the most appropriate unit: USD, USD/shares, pure
                # ratio (typically used for tax rate).
                for unit, observations in units.items():
                    for obs in observations:
                        end = obs.get("end")
                        val = obs.get("val")
                        form = obs.get("form")
                        fy = obs.get("fy")
                        fp = obs.get("fp")
                        accession = obs.get("accn")
                        if end is None or val is None:
                            continue
                        try:
                            conn.execute(
                                """
                                INSERT OR REPLACE INTO xbrl_facts
                                    (ticker, cik, concept, period_end, fy, fp,
                                     form, value, unit, accession_number)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (ticker, cik, concept, end, fy, fp, form,
                                 float(val), unit, accession),
                            )
                        except (sqlite3.IntegrityError, ValueError):
                            continue

                        # Only push annual (10-K) figures into the canonical
                        # data_points table — clearer for the agent.
                        if form == "10-K":
                            registry_rows.append({
                                "entity_id": ticker,
                                "entity_type": "ticker",
                                "field": canonical_field,
                                "as_of": end,
                                "source_id": SRC_XBRL,
                                "value_num": float(val),
                                "confidence": 1.0,
                            })
                            n_rows += 1

        # Push to registry in one bulk transaction
        if self.registry and registry_rows:
            try:
                self.registry.upsert_points_bulk(registry_rows)
            except Exception as e:  # noqa: BLE001
                logger.warning("xbrl | %s | registry bulk write failed: %s", ticker, e)

        # Compute and write derived ratios (rd_intensity, sbc_to_revenue,
        # goodwill_to_assets, dilution_yoy)
        self._write_xbrl_derived(ticker)
        logger.info("xbrl | %s | wrote %d canonical facts", ticker, n_rows)
        return n_rows

    def _write_xbrl_derived(self, ticker: str) -> None:
        """Compute simple derived ratios from the XBRL facts and push to registry."""
        if self.registry is None:
            return

        def latest_value(field: str) -> Optional[tuple[float, str]]:
            row = self.registry.latest(ticker, field)
            if row and row.get("value_num") is not None:
                return float(row["value_num"]), row["as_of"]
            return None

        # rd_intensity
        rev = latest_value("ticker.fundamental.revenue_ttm")
        rd = latest_value("ticker.fundamental.rd_ttm")
        if rev and rd and rev[0] > 0:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.fundamental.rd_intensity",
                as_of=rd[1], source_id=SRC_XBRL,
                value_num=round(rd[0] / rev[0], 6), confidence=1.0,
            )

        # sbc_to_revenue
        sbc = latest_value("ticker.fundamental.sbc_ttm")
        if rev and sbc and rev[0] > 0:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.fundamental.sbc_to_revenue",
                as_of=sbc[1], source_id=SRC_XBRL,
                value_num=round(sbc[0] / rev[0], 6), confidence=1.0,
            )

        # goodwill_to_assets
        gw = latest_value("ticker.fundamental.goodwill")
        ta = latest_value("ticker.fundamental.total_assets")
        if gw and ta and ta[0] > 0:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.fundamental.goodwill_to_assets",
                as_of=gw[1], source_id=SRC_XBRL,
                value_num=round(gw[0] / ta[0], 6), confidence=1.0,
            )

        # dilution_yoy: compare two latest annual shares_diluted
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT period_end, value FROM xbrl_facts
                 WHERE ticker = ? AND concept = ? AND form = '10-K'
                 ORDER BY period_end DESC LIMIT 2
                """,
                (ticker, "WeightedAverageNumberOfDilutedSharesOutstanding"),
            ).fetchall()
        if len(rows) == 2 and rows[1][1] and rows[0][1]:
            yoy = (rows[0][1] - rows[1][1]) / rows[1][1]
            self.registry.upsert_point(
                ticker, "ticker", "ticker.fundamental.dilution_yoy",
                as_of=rows[0][0], source_id=SRC_XBRL,
                value_num=round(float(yoy), 6), confidence=1.0,
            )

        # FCF derivation: CFO - capex
        cfo = latest_value("ticker.fundamental.fcf_ttm")  # currently storing CFO
        capex = latest_value("ticker.fundamental.capex_ttm")
        if cfo and capex:
            fcf = cfo[0] - capex[0]
            self.registry.upsert_point(
                ticker, "ticker", "ticker.fundamental.fcf_ttm",
                as_of=cfo[1], source_id="derived",
                value_num=round(float(fcf), 2), confidence=0.9,
            )

    # ------------------------------------------------------------------
    # Form 4 — insider transactions
    # ------------------------------------------------------------------
    def get_form4_filings(self, ticker: str, count: int = 50) -> list[dict]:
        """Fetch recent Form 4 filings for a ticker (filing index only)."""
        return self.get_filings(ticker, filing_type="4", count=count)

    def ingest_insider_transactions(
        self, ticker: str, count: int = 30
    ) -> int:
        """Download recent Form 4 filings, parse the XML primary doc, and
        write parsed transactions into raw + unified tables. Returns the
        number of transaction rows written.
        """
        ticker = ticker.upper()
        cik = self.get_cik(ticker)
        filings = self.get_form4_filings(ticker, count=count)
        n = 0
        for filing in filings:
            try:
                txns = self._parse_form4(filing)
            except Exception as e:  # noqa: BLE001
                logger.debug("form4 | %s | parse error %s: %s",
                             ticker, filing["accession_number"], e)
                continue
            with sqlite3.connect(self.db_path) as conn:
                for t in txns:
                    try:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO insider_transactions
                                (ticker, cik, accession_number, insider_name,
                                 insider_title, transaction_date, transaction_code,
                                 direction, shares, price, value_usd, is_10b5_1)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (ticker, cik, filing["accession_number"],
                             t.get("insider_name"), t.get("insider_title"),
                             t.get("transaction_date"), t.get("transaction_code"),
                             t.get("direction"), t.get("shares"),
                             t.get("price"), t.get("value_usd"),
                             1 if t.get("is_10b5_1") else 0),
                        )
                        n += 1
                    except sqlite3.IntegrityError:
                        continue
            # Also fire an event row for each
            if self.registry:
                for t in txns:
                    self.registry.upsert_event(
                        event_type=("form4_buy" if t.get("direction") == "BUY"
                                    else "form4_sell" if t.get("direction") == "SELL"
                                    else "form4_other"),
                        occurred_at=t.get("transaction_date") or filing["filed_date"],
                        source_id=SRC_FORM4,
                        entity_id=ticker, entity_type="ticker",
                        severity=0.5,
                        payload={**t, "accession": filing["accession_number"]},
                    )
        # Recompute aggregate fields from the raw table
        self._refresh_insider_aggregates(ticker)
        return n

    def _parse_form4(self, filing: dict) -> list[dict]:
        """Download and parse a single Form 4 filing's XML to a list of dicts."""
        # The primary_doc_url usually points to an .xml or .htm wrapper.
        # The actual parsable file is form4.xml in the index.
        index_url = filing["primary_doc_url"]
        # Convert e.g. .../primary.htm -> .../index.json to find the XML
        accession_clean = filing["accession_number"].replace("-", "")
        cik = filing["cik"].lstrip("0")
        index_json = (
            f"{EDGAR_ARCHIVES}/{cik}/{accession_clean}/index.json"
        )
        try:
            idx = self._get(index_json).json()
        except Exception:
            return []

        xml_url = None
        for item in idx.get("directory", {}).get("item", []):
            name = item.get("name", "")
            if name.endswith(".xml") and "form4" in name.lower():
                xml_url = f"{EDGAR_ARCHIVES}/{cik}/{accession_clean}/{name}"
                break
            if name.endswith(".xml") and not name.startswith("Financial_Report"):
                xml_url = f"{EDGAR_ARCHIVES}/{cik}/{accession_clean}/{name}"
        if not xml_url:
            return []

        resp = self._get(xml_url)
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError:
            return []

        # Extract reporter name + title
        reporter = root.find(".//reportingOwner")
        insider_name = ""
        insider_title = ""
        is_10b5_1 = "10b5-1" in resp.text.lower() or "10b5(c)" in resp.text.lower()
        if reporter is not None:
            owner_id = reporter.find(".//reportingOwnerId/rptOwnerName")
            if owner_id is not None and owner_id.text:
                insider_name = owner_id.text.strip()
            relationship = reporter.find(".//reportingOwnerRelationship")
            if relationship is not None:
                titles = []
                for tag in ("officerTitle", "isOfficer", "isDirector",
                            "isTenPercentOwner", "isOther"):
                    el = relationship.find(tag)
                    if el is not None and el.text:
                        if tag == "officerTitle":
                            titles.append(el.text.strip())
                        elif tag == "isDirector" and el.text in ("1", "true"):
                            titles.append("Director")
                        elif tag == "isTenPercentOwner" and el.text in ("1", "true"):
                            titles.append("10%+ Owner")
                insider_title = ", ".join(titles)

        results = []
        # Non-derivative transactions (most common: open-market trades)
        for txn in root.findall(".//nonDerivativeTransaction"):
            code_el = txn.find(".//transactionCoding/transactionCode")
            code = code_el.text.strip() if (code_el is not None and code_el.text) else ""
            if code in FORM4_BUY_CODES:
                direction = "BUY"
            elif code in FORM4_SELL_CODES:
                direction = "SELL"
            else:
                direction = "OTHER"

            date_el = txn.find(".//transactionDate/value")
            date = date_el.text if date_el is not None and date_el.text else None
            shares_el = txn.find(".//transactionShares/value")
            shares = float(shares_el.text) if shares_el is not None and shares_el.text else None
            price_el = txn.find(".//transactionPricePerShare/value")
            price = float(price_el.text) if price_el is not None and price_el.text else None
            value_usd = (shares * price) if (shares is not None and price is not None) else None

            results.append({
                "insider_name": insider_name,
                "insider_title": insider_title,
                "transaction_date": date,
                "transaction_code": code,
                "direction": direction,
                "shares": shares,
                "price": price,
                "value_usd": value_usd,
                "is_10b5_1": is_10b5_1,
            })
        return results

    def _refresh_insider_aggregates(self, ticker: str) -> None:
        """Recompute 30-day rolling counts and net $ from raw insider table."""
        if self.registry is None:
            return
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            buys = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(value_usd),0)
                     FROM insider_transactions
                    WHERE ticker = ? AND transaction_date >= ? AND direction = 'BUY'""",
                (ticker, cutoff),
            ).fetchone()
            sells = conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(value_usd),0)
                     FROM insider_transactions
                    WHERE ticker = ? AND transaction_date >= ? AND direction = 'SELL'""",
                (ticker, cutoff),
            ).fetchone()
        net_usd = (buys[1] or 0) - (sells[1] or 0)
        today = datetime.now().strftime("%Y-%m-%d")
        self.registry.upsert_point(
            ticker, "ticker", "ticker.ownership.insider_buys_30d",
            as_of=today, source_id=SRC_FORM4, value_num=float(buys[0] or 0),
        )
        self.registry.upsert_point(
            ticker, "ticker", "ticker.ownership.insider_sells_30d",
            as_of=today, source_id=SRC_FORM4, value_num=float(sells[0] or 0),
        )
        self.registry.upsert_point(
            ticker, "ticker", "ticker.ownership.insider_net_usd_30d",
            as_of=today, source_id=SRC_FORM4, value_num=float(net_usd),
        )

    # ------------------------------------------------------------------
    # 13D / 13G ownership filings
    # ------------------------------------------------------------------
    def ingest_ownership_filings(
        self, ticker: str, filing_type: str = "SC 13D", count: int = 20
    ) -> int:
        """Pull recent 13D or 13G filings against a ticker. We can't
        reliably parse the precise pct/shares without an HTML scraper for
        each unique filer template, so we record metadata + filer name and
        leave precision-extraction for a later pass.
        """
        if filing_type not in ("SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"):
            raise ValueError(f"unsupported filing_type: {filing_type}")
        filings = self.get_filings(ticker, filing_type=filing_type, count=count)
        n = 0
        ticker = ticker.upper()
        cik = self.get_cik(ticker)
        with sqlite3.connect(self.db_path) as conn:
            for f in filings:
                # Best-effort filer-name extraction from description
                filer = (f.get("description") or "").split(",")[0]
                is_amend = filing_type.endswith("/A")
                try:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO ownership_filings
                            (ticker, cik, filing_type, filed_date,
                             accession_number, filer_name, is_amendment)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (ticker, cik, filing_type, f["filed_date"],
                         f["accession_number"], filer, int(is_amend)),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        # Aggregate counts
        if self.registry:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(self.db_path) as conn:
                count_13d = conn.execute(
                    "SELECT COUNT(*) FROM ownership_filings WHERE ticker = ? AND filing_type LIKE 'SC 13D%'",
                    (ticker,),
                ).fetchone()[0]
                count_13g = conn.execute(
                    "SELECT COUNT(*) FROM ownership_filings WHERE ticker = ? AND filing_type LIKE 'SC 13G%'",
                    (ticker,),
                ).fetchone()[0]
            self.registry.upsert_point(
                ticker, "ticker", "ticker.ownership.13d_holders",
                as_of=today, source_id=SRC_13D, value_num=float(count_13d),
            )
            self.registry.upsert_point(
                ticker, "ticker", "ticker.ownership.13g_holders",
                as_of=today, source_id=SRC_13G, value_num=float(count_13g),
            )
        return n

    # ------------------------------------------------------------------
    # 13F holdings
    # ------------------------------------------------------------------
    def ingest_13f_filings_by_filer(self, filer_cik: str, count: int = 4) -> int:
        """Pull recent 13F-HR filings for an institutional filer (by CIK)
        and parse the holdings information table XML.
        """
        cik = str(filer_cik).zfill(10)
        url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
        resp = self._get(url)
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        n = 0
        seen = 0
        for i, form in enumerate(forms):
            if form != "13F-HR":
                continue
            seen += 1
            if seen > count:
                break
            acc_clean = accessions[i].replace("-", "")
            base = f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{acc_clean}"
            # Find the *.xml table file from index.json
            try:
                idx = self._get(f"{base}/index.json").json()
            except Exception:
                continue
            xml_name = None
            for item in idx.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and "infotable" in name.lower():
                    xml_name = name
                    break
            if not xml_name:
                # fall back to first xml
                for item in idx.get("directory", {}).get("item", []):
                    name = item.get("name", "")
                    if name.endswith(".xml") and "primary" not in name.lower():
                        xml_name = name
                        break
            if not xml_name:
                continue
            try:
                xml = self._get(f"{base}/{xml_name}").text
                root = ET.fromstring(xml)
            except Exception:
                continue
            ns = "{http://www.sec.gov/edgar/document/thirteenf/informationtable}"
            with sqlite3.connect(self.db_path) as conn:
                for inf in root.findall(f".//{ns}infoTable"):
                    cusip = (inf.findtext(f"{ns}cusip") or "").strip()
                    issuer = (inf.findtext(f"{ns}nameOfIssuer") or "").strip()
                    cls = (inf.findtext(f"{ns}titleOfClass") or "").strip()
                    val = inf.findtext(f"{ns}value")
                    sshprn = inf.find(f"{ns}shrsOrPrnAmt")
                    shares_text = sshprn.findtext(f"{ns}sshPrnamt") if sshprn is not None else None
                    try:
                        value_usd = float(val) * 1000 if val else None  # 13F values in $1000s
                    except ValueError:
                        value_usd = None
                    try:
                        shares = float(shares_text) if shares_text else None
                    except ValueError:
                        shares = None
                    try:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO f13f_holdings
                                (filer_cik, accession_number, period_of_report,
                                 cusip, issuer_name, class_title, value_usd, shares)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (cik, accessions[i],
                             report_dates[i] if i < len(report_dates) else None,
                             cusip, issuer, cls, value_usd, shares),
                        )
                        n += 1
                    except sqlite3.IntegrityError:
                        continue
        return n

    # ------------------------------------------------------------------
    # 10b5-1 plan references
    # ------------------------------------------------------------------
    def find_10b5_1_references(self, ticker: str) -> list[dict]:
        """Search cached filings for '10b5-1' mentions. Useful flag against
        Form 4 sales — sales under a 10b5-1 plan carry a different signal.
        """
        return self.search_filings(ticker, "10b5-1")

    # ------------------------------------------------------------------
    # Buybacks, secondaries, ATMs, governance — 8-K + form-type pulls
    # ------------------------------------------------------------------
    def ingest_corporate_actions(self, ticker: str) -> int:
        """Scan recent 8-K filings + secondary form types to populate
        ``corporate_actions`` and emit canonical events.
        Returns the number of action rows written.
        """
        ticker = ticker.upper()
        cik = self.get_cik(ticker)
        n = 0

        # 8-K events: governance, buybacks, ratings comments
        events_8k = self.get_recent_8k_events(ticker, days_back=180)
        for ev in events_8k:
            text_match = (ev.get("description") or "").lower()
            action_type = None
            if "5.02" in text_match or "officer" in text_match or "director" in text_match:
                action_type = "officer_director_change"
            elif "buyback" in text_match or "repurchase" in text_match:
                action_type = "share_repurchase_announce"
            elif "8.01" in text_match and "rating" in text_match:
                action_type = "rating_change_mention"
            else:
                continue
            n += self._record_corp_action(
                ticker, cik, action_type, ev["filed_date"],
                ev["accession_number"], detail={"desc": ev["description"]},
            )

        # 424B5 (secondaries / ATMs)
        for f in self.get_filings(ticker, filing_type="424B5", count=10):
            n += self._record_corp_action(
                ticker, cik, "secondary_or_atm", f["filed_date"],
                f["accession_number"], detail={"desc": f["description"]},
            )

        # Form SR (issuer purchases of equity) — newer SEC mandate
        for f in self.get_filings(ticker, filing_type="SR", count=10):
            n += self._record_corp_action(
                ticker, cik, "share_repurchase_form_sr", f["filed_date"],
                f["accession_number"], detail={"desc": f["description"]},
            )
        return n

    def _record_corp_action(
        self, ticker: str, cik: str, action_type: str,
        occurred_at: str, accession: Optional[str],
        amount_usd: Optional[float] = None,
        detail: Optional[dict] = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO corporate_actions
                        (ticker, cik, action_type, occurred_at, accession_number,
                         amount_usd, detail_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ticker, cik, action_type, occurred_at, accession,
                     amount_usd, json.dumps(detail or {}, default=str)),
                )
            except sqlite3.IntegrityError:
                return 0
        if self.registry:
            self.registry.upsert_event(
                event_type=action_type,
                occurred_at=occurred_at,
                source_id=SRC_FILING,
                entity_id=ticker, entity_type="ticker",
                severity=0.6 if "repurchase" in action_type or "officer" in action_type else 0.4,
                payload={"accession": accession, "amount_usd": amount_usd, **(detail or {})},
            )
        return 1

    # ------------------------------------------------------------------
    # High-level: refresh everything for one ticker
    # ------------------------------------------------------------------
    def refresh_ticker_filings(self, ticker: str) -> dict[str, int]:
        """Run all extended SEC fetches for one ticker. Returns row counts."""
        out: dict[str, int] = {}
        try:
            out["xbrl_facts"] = self.ingest_xbrl_facts(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_ticker | %s | xbrl failed: %s", ticker, e)
            out["xbrl_facts"] = 0
        try:
            out["form4"] = self.ingest_insider_transactions(ticker, count=30)
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_ticker | %s | form4 failed: %s", ticker, e)
            out["form4"] = 0
        for ft in ("SC 13D", "SC 13G"):
            try:
                out[ft] = self.ingest_ownership_filings(ticker, filing_type=ft, count=10)
            except Exception as e:  # noqa: BLE001
                logger.warning("refresh_ticker | %s | %s failed: %s", ticker, ft, e)
                out[ft] = 0
        try:
            out["corp_actions"] = self.ingest_corporate_actions(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_ticker | %s | corp_actions failed: %s", ticker, e)
            out["corp_actions"] = 0
        return out

    # ------------------------------------------------------------------
    # Private helpers (existing)
    # ------------------------------------------------------------------
    def _extract_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _clean_sec_text(self, text: str) -> str:
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"-{5,}", "", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
        return text.strip()

    def _get_cached_filings(
        self, ticker, filing_type, count, start_date, end_date
    ) -> list[dict]:
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
# CLI test (preserves existing behavior, adds extended summary)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = SECFetcher()

    print("=== SEC Fetcher Test ===\n")

    cik = fetcher.get_cik("AAPL")
    print(f"AAPL CIK: {cik}")

    filings = fetcher.get_filings("AAPL", filing_type="10-K", count=3)
    for f in filings:
        print(f"  {f['filed_date']} | {f['filing_type']} | {f['accession_number']}")

    if filings:
        text = fetcher.get_filing_text(filings[0]["accession_number"])
        print(f"\nFirst 500 chars of latest 10-K:\n{text[:500]}")

    events = fetcher.get_recent_8k_events("AAPL", days_back=90)
    print(f"\n8-K events in last 90 days: {len(events)}")
    for e in events[:3]:
        print(f"  {e['filed_date']} | {e['description']}")

    # Extended demo
    print("\n=== Extended SEC features ===")
    summary = fetcher.refresh_ticker_filings("AAPL")
    for k, v in summary.items():
        print(f"  {k}: {v}")
