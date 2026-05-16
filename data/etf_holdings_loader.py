"""
data/etf_holdings_loader.py
===========================
ETF holdings ingestion orchestrator.

Discovers issuer-hosted holdings files, parses them via the right
provider adapter, normalizes the rows, resolves CUSIPs to tickers where
possible, and writes the result through
``MarketData.set_etf_holdings()``. Idempotent, batch-capable, preserves
historical snapshots via the ``as_of`` dimension of the target table's
primary key.

Operational modes
-----------------
* **Backfill**     — every configured ETF, one ingestion run each.
* **Incremental**  — every configured ETF, latest available snapshot.
  (Same workflow as backfill; issuers publish one file at a time so
  "backfill" really means "current snapshot for everyone".)
* **Single-fund**  — one ETF.
* **Local file**   — ingest a file the user already downloaded.

API contract
------------
The loader only writes through ``MarketData.set_etf_holdings()``, which
accepts ``(etf_ticker, holdings, as_of, source)`` where each ``holding``
is ``{ticker, weight, shares?, market_value?}``. Richer per-holding
metadata (security_name, identifier, issuer, source_url, etc.) is
captured in the sidecar ``etf_holdings_ingestion_log`` table so it's
queryable later without changing the existing schema.

CLI
---
    python -m data.etf_holdings_loader --backfill
    python -m data.etf_holdings_loader --etf XLK
    python -m data.etf_holdings_loader --file holdings.csv --etf XLK --issuer spdr
    python -m data.etf_holdings_loader --report
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests

from data.etf_providers import (
    GenericIssuerConfig,
    GenericIssuerProvider,
    HoldingsProvider,
    IsharesProvider,
    LocalFileProvider,
    ParseResult,
    RawHolding,
    SpdrProvider,
)
from data.market_data import MarketData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Each ETF maps to a small dict telling the loader which provider class
# to use plus any constructor args. New issuers can be added without
# touching code — just append an entry here. iShares product_ids and
# slugs can be looked up by searching for the ETF on ishares.com.
#
# This map is intentionally small at MVP; the loader works fine with one
# entry. Add more by following the pattern.

ETF_PROVIDER_CONFIG: dict[str, dict] = {
    # ---- SPDR sector ETFs (URL is deterministic from ticker) -----------
    "SPY": {"provider": "spdr"},
    "XLK": {"provider": "spdr"},
    "XLF": {"provider": "spdr"},
    "XLV": {"provider": "spdr"},
    "XLI": {"provider": "spdr"},
    "XLE": {"provider": "spdr"},
    "XLP": {"provider": "spdr"},
    "XLY": {"provider": "spdr"},
    "XLU": {"provider": "spdr"},
    "XLB": {"provider": "spdr"},
    "XLRE": {"provider": "spdr"},
    "XLC": {"provider": "spdr"},

    # ---- iShares (product_id + slug come from the ETF's iShares page) --
    # If iShares rotates a slug, update it here only.
    "IVV": {
        "provider": "ishares",
        "product_id": "239726",
        "slug": "ishares-core-sp-500-etf",
    },
    "IWM": {
        "provider": "ishares",
        "product_id": "239710",
        "slug": "ishares-russell-2000-etf",
    },
    "EFA": {
        "provider": "ishares",
        "product_id": "239623",
        "slug": "ishares-msci-eafe-etf",
    },
}


HTTP_TIMEOUT_SEC = 30
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF_BASE = 2.0
USER_AGENT = "ary-fund-research/1.0 (etf-holdings-loader)"

DEFAULT_DB_PATH = "data/hedgefund.db"
DEFAULT_DOWNLOAD_CACHE = Path("data/cache/etf_holdings")


# ---------------------------------------------------------------------------
# Canonical row + run summary
# ---------------------------------------------------------------------------


@dataclass
class CanonicalHolding:
    """The fully-normalized holding shape the loader produces internally.

    Only a subset of these fields fits the ``set_etf_holdings()`` API;
    the rest are kept for the coverage log and any future schema
    expansion. Producing this dataclass keeps the boundary between
    parsing and persistence clean.
    """
    etf_ticker: str
    underlying_ticker: str       # Canonical ticker, "CUSIP:..." for unmapped, or "CASH"
    security_id: Optional[str]   # Original identifier from the issuer file
    identifier_type: Optional[str]
    security_name: Optional[str]
    weight: Optional[float]
    shares: Optional[float]
    market_value: Optional[float]
    as_of_date: str
    issuer: str
    source_url: str
    source_file: str
    source_type: str
    ingested_at: str
    metadata: dict = field(default_factory=dict)

    def to_api_payload(self) -> dict:
        """Subset accepted by :meth:`MarketData.set_etf_holdings`."""
        return {
            "ticker": self.underlying_ticker,
            "weight": self.weight,
            "shares": self.shares,
            "market_value": self.market_value,
        }


@dataclass
class EtfRunResult:
    """Result of ingesting one ETF."""
    etf_ticker: str
    issuer: str
    status: str                  # "success" | "failed" | "skipped"
    rows_loaded: int
    as_of_date: Optional[str]
    source_url: str
    error: Optional[str] = None


@dataclass
class BatchRunSummary:
    """Result of ingesting many ETFs."""
    run_id: str
    started_at: str
    finished_at: str
    results: list[EtfRunResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == "success")

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "results": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# Ticker resolver
# ---------------------------------------------------------------------------


class TickerResolver:
    """Resolve issuer-side security identifiers to canonical tickers.

    For most US-equity ETFs from iShares and SPDR the file already
    contains a ticker column, so the resolver is mostly a pass-through.
    For files that publish only CUSIP/ISIN (some Vanguard funds, all
    fixed-income ETFs), the resolver looks up a ``cusip_ticker_map``
    table populated manually or from SEC's free CUSIP/CIK feed.

    Never silently drops a holding — unmapped identifiers come back
    prefixed (``CUSIP:037833100``, ``ISIN:US0378331005``) so the
    overlap engine can still count them, just won't treat them as a
    real ticker.
    """

    # In-memory hint map (seeded by the user later). Production code
    # should populate the DB table; this is here so tests don't need a
    # disk lookup.
    _STATIC_HINTS: dict[str, str] = {}

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cusip_ticker_map (
                    identifier      TEXT NOT NULL,
                    identifier_type TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (identifier, identifier_type)
                )
                """
            )
            conn.commit()

    def add_mapping(self, identifier: str, identifier_type: str, ticker: str) -> None:
        """Manually seed a CUSIP/ISIN → ticker mapping."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cusip_ticker_map
                   (identifier, identifier_type, ticker)
                   VALUES (?, ?, ?)""",
                (identifier.upper(), identifier_type.lower(), ticker.upper()),
            )
            conn.commit()

    def resolve(self, raw: RawHolding) -> str:
        """Return the canonical ticker for this holding, or a prefixed
        identifier if nothing maps."""
        # 1. If the issuer gave us a ticker that looks real, use it.
        if raw.raw_ticker:
            normalized = self._normalize_ticker(raw.raw_ticker)
            if normalized:
                return normalized

        # 2. Look up by identifier in the static + DB map
        if raw.raw_identifier and raw.identifier_type in ("cusip", "isin", "sedol"):
            key = raw.raw_identifier.upper()
            if key in self._STATIC_HINTS:
                return self._STATIC_HINTS[key]
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT ticker FROM cusip_ticker_map
                       WHERE identifier = ? AND identifier_type = ?""",
                    (key, raw.identifier_type),
                ).fetchone()
                if row and row[0]:
                    return str(row[0]).upper()
            # Unmapped — preserve as prefixed identifier so we never drop it
            return f"{raw.identifier_type.upper()}:{key}"

        # 3. Truly nothing — preserve original identifier or mark cash
        if raw.raw_identifier:
            return f"ID:{raw.raw_identifier.upper()}"
        return "UNKNOWN"

    @staticmethod
    def _normalize_ticker(s: str) -> Optional[str]:
        """Turn a raw 'Ticker' cell into a canonical ticker or a
        deterministic non-equity tag.

        - "AAPL" → "AAPL"
        - "-", "—", "" → "CASH"
        - "USD" → "CASH:USD"
        - "ESM4 Curncy" (Bloomberg-style futures) → "FUT:ESM4"
        - Anything weird → return original uppercased; let the caller decide
        """
        s = s.strip()
        if not s:
            return "CASH"
        if s in {"-", "—", "--", "N/A", "NA"}:
            return "CASH"
        upper = s.upper()
        if upper in {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD"}:
            return f"CASH:{upper}"
        if " " in s and "Curncy" in s.split():
            return f"FUT:{s.split()[0].upper()}"
        # Looks like a normal ticker: alnum + optional dot/hyphen, ≤ 8 chars
        if len(upper) <= 8 and all(c.isalnum() or c in ".-/" for c in upper):
            return upper
        return upper  # caller still gets something; just not pristine


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ETFHoldingsLoader:
    """Orchestrator for ETF holdings ingestion."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        market_data: Optional[MarketData] = None,
        download_cache: Optional[Path] = None,
        provider_config: Optional[dict[str, dict]] = None,
    ) -> None:
        self.db_path = db_path
        self.market = market_data or MarketData(db_path=db_path)
        self.cache_dir = Path(download_cache) if download_cache else DEFAULT_DOWNLOAD_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resolver = TickerResolver(db_path=db_path)
        self.provider_config = provider_config or ETF_PROVIDER_CONFIG
        self._ensure_log_table()

    # ---- Schema ------------------------------------------------------

    def _ensure_log_table(self) -> None:
        """Coverage log table — one row per (run, etf) ingestion attempt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS etf_holdings_ingestion_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id       TEXT NOT NULL,
                    etf_ticker   TEXT NOT NULL,
                    issuer       TEXT,
                    as_of_date   TEXT,
                    status       TEXT NOT NULL,
                    rows_loaded  INTEGER DEFAULT 0,
                    source_url   TEXT,
                    error        TEXT,
                    ingested_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_etf_log_etf "
                "ON etf_holdings_ingestion_log(etf_ticker, ingested_at)"
            )
            conn.commit()

    # ---- Public modes ------------------------------------------------

    def run_single(self, etf_ticker: str, run_id: Optional[str] = None) -> EtfRunResult:
        """Ingest one ETF using its configured provider."""
        run_id = run_id or _new_run_id()
        cfg = self.provider_config.get(etf_ticker.upper())
        if not cfg:
            return self._record_skip(
                run_id, etf_ticker.upper(), "unknown",
                f"no provider configured for {etf_ticker.upper()}",
            )
        provider = _build_provider(cfg)
        return self._ingest_via_http(run_id, etf_ticker.upper(), provider)

    def run_batch(
        self,
        etf_tickers: Optional[Iterable[str]] = None,
    ) -> BatchRunSummary:
        """Ingest multiple ETFs in one job. Partial failures don't stop
        the batch — one bad issuer file doesn't kill everything else."""
        run_id = _new_run_id()
        started = datetime.now().isoformat()
        tickers = list(etf_tickers) if etf_tickers else list(self.provider_config.keys())
        results: list[EtfRunResult] = []
        for t in tickers:
            try:
                r = self.run_single(t, run_id=run_id)
            except Exception as e:  # noqa: BLE001 — last-resort guard
                logger.exception("etf_loader | unexpected failure for %s", t)
                r = self._record_failure(
                    run_id, t.upper(), "unknown", "",
                    f"unhandled exception: {e}",
                )
            results.append(r)
        return BatchRunSummary(
            run_id=run_id,
            started_at=started,
            finished_at=datetime.now().isoformat(),
            results=results,
        )

    def run_backfill(self) -> BatchRunSummary:
        """Alias for batch over every configured ETF."""
        return self.run_batch(self.provider_config.keys())

    def run_incremental(self) -> BatchRunSummary:
        """Same workflow as backfill — issuers publish point-in-time
        snapshots so 'incremental' just means 'fetch the latest one'."""
        return self.run_batch(self.provider_config.keys())

    def run_local_file(
        self,
        etf_ticker: str,
        file_path: str | Path,
        issuer: str,
    ) -> EtfRunResult:
        """Ingest a manually-downloaded file. The most reliable production
        path — issuer URLs rotate constantly."""
        run_id = _new_run_id()
        etf_ticker = etf_ticker.upper()
        path = Path(file_path)
        if not path.exists():
            return self._record_failure(
                run_id, etf_ticker, issuer, str(path),
                f"file not found: {path}",
            )

        # Build the underlying provider, then wrap in LocalFileProvider
        # so the source_url is tagged file:// and no network fires.
        try:
            inner = _build_provider_by_issuer(issuer)
        except KeyError:
            return self._record_failure(
                run_id, etf_ticker, issuer, str(path),
                f"unknown issuer: {issuer}",
            )
        provider = LocalFileProvider(inner)

        raw = path.read_bytes()
        source_url = path.resolve().as_uri()
        parse = provider.parse_bytes(raw, etf_ticker, source_url)
        return self._finalize(run_id, etf_ticker, provider, parse)

    # ---- HTTP ingest path -------------------------------------------

    def _ingest_via_http(
        self,
        run_id: str,
        etf_ticker: str,
        provider: HoldingsProvider,
    ) -> EtfRunResult:
        url = provider.build_url(etf_ticker)
        try:
            raw = self._download_with_retry(url, label=etf_ticker)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "etf_loader | %s | download failed: %s", etf_ticker, e,
            )
            cached = self._load_cached(etf_ticker, provider.issuer)
            if cached is None:
                return self._record_failure(
                    run_id, etf_ticker, provider.issuer, url,
                    f"download failed and no cache: {e}",
                )
            logger.info(
                "etf_loader | %s | using cached file (download failed)",
                etf_ticker,
            )
            raw = cached
        parse = provider.parse_bytes(raw, etf_ticker, url)
        return self._finalize(run_id, etf_ticker, provider, parse)

    def _download_with_retry(self, url: str, label: str) -> bytes:
        last_exc: Optional[Exception] = None
        headers = {"User-Agent": USER_AGENT}
        for attempt in range(1, HTTP_MAX_RETRIES + 1):
            try:
                logger.info(
                    "etf_loader | %s | download attempt %d/%d",
                    label, attempt, HTTP_MAX_RETRIES,
                )
                resp = requests.get(
                    url, headers=headers, timeout=HTTP_TIMEOUT_SEC,
                )
                resp.raise_for_status()
                self._cache_download(label, resp.content)
                return resp.content
            except Exception as e:  # noqa: BLE001
                last_exc = e
                logger.warning(
                    "etf_loader | %s | attempt %d failed: %s",
                    label, attempt, e,
                )
                if attempt < HTTP_MAX_RETRIES:
                    time.sleep(HTTP_RETRY_BACKOFF_BASE ** attempt)
        assert last_exc is not None
        raise last_exc

    def _cache_download(self, label: str, data: bytes) -> None:
        path = self.cache_dir / f"{label}_{datetime.now():%Y%m%d}.bin"
        try:
            path.write_bytes(data)
        except OSError as e:
            logger.debug("etf_loader | cache write failed (%s): %s", path, e)

    def _load_cached(self, label: str, issuer: str) -> Optional[bytes]:
        candidates = sorted(self.cache_dir.glob(f"{label}_*.bin"))
        if not candidates:
            return None
        try:
            return candidates[-1].read_bytes()
        except OSError:
            return None

    # ---- Finalize: validate, write, log -----------------------------

    def _finalize(
        self,
        run_id: str,
        etf_ticker: str,
        provider: HoldingsProvider,
        parse: ParseResult,
    ) -> EtfRunResult:
        # Hard fail if parser errored
        if not parse.ok:
            err = "; ".join(parse.errors) or "empty result"
            return self._record_failure(
                run_id, etf_ticker, provider.issuer, parse.source_url, err,
            )

        # Default as_of to today if the file didn't carry one. Better to
        # take today's date than refuse to write — but log loudly.
        as_of = parse.as_of_date
        if not as_of:
            as_of = datetime.now().strftime("%Y-%m-%d")
            logger.warning(
                "etf_loader | %s | no as_of date in file; defaulting to %s",
                etf_ticker, as_of,
            )

        # Normalize → canonical → API payload
        canonical = self._to_canonical(
            raws=parse.holdings,
            etf_ticker=etf_ticker,
            issuer=provider.issuer,
            as_of=as_of,
            source_url=parse.source_url,
            source_type=parse.source_type,
        )
        canonical = self._dedupe(canonical)

        # Validate
        validation = self._validate_canonical(canonical)
        if validation:
            return self._record_failure(
                run_id, etf_ticker, provider.issuer, parse.source_url,
                f"validation failed: {validation}",
            )

        # Write through the official API
        payload = [c.to_api_payload() for c in canonical]
        try:
            n = self.market.set_etf_holdings(
                etf_ticker=etf_ticker,
                holdings=payload,
                as_of=as_of,
                source=provider.source_tag,
            )
        except Exception as e:  # noqa: BLE001
            return self._record_failure(
                run_id, etf_ticker, provider.issuer, parse.source_url,
                f"set_etf_holdings failed: {e}",
            )

        # Log success
        result = EtfRunResult(
            etf_ticker=etf_ticker,
            issuer=provider.issuer,
            status="success",
            rows_loaded=n,
            as_of_date=as_of,
            source_url=parse.source_url,
        )
        self._log_result(run_id, result)
        return result

    def _to_canonical(
        self,
        raws: list[RawHolding],
        etf_ticker: str,
        issuer: str,
        as_of: str,
        source_url: str,
        source_type: str,
    ) -> list[CanonicalHolding]:
        """Build canonical records from RawHoldings, resolving tickers."""
        now = datetime.now().isoformat()
        # Derive a source_file from the URL tail for readability
        source_file = source_url.rsplit("/", 1)[-1] or source_url

        out: list[CanonicalHolding] = []
        for r in raws:
            ticker = self.resolver.resolve(r)
            out.append(CanonicalHolding(
                etf_ticker=etf_ticker,
                underlying_ticker=ticker,
                security_id=r.raw_identifier,
                identifier_type=r.identifier_type,
                security_name=r.name,
                weight=r.weight,
                shares=r.shares,
                market_value=r.market_value,
                as_of_date=as_of,
                issuer=issuer,
                source_url=source_url,
                source_file=source_file,
                source_type=source_type,
                ingested_at=now,
                metadata=dict(r.extra),
            ))
        return out

    @staticmethod
    def _dedupe(rows: list[CanonicalHolding]) -> list[CanonicalHolding]:
        """Keep the first occurrence of each underlying_ticker.

        Issuer files occasionally double-print a row (e.g., when an ETF
        holds two share classes of the same company). The downstream
        table's PRIMARY KEY would have INSERT OR REPLACE collapsed them
        anyway, but deduping here keeps the rows_loaded count honest.
        """
        seen: set[str] = set()
        out: list[CanonicalHolding] = []
        for r in rows:
            if r.underlying_ticker in seen:
                continue
            seen.add(r.underlying_ticker)
            out.append(r)
        return out

    @staticmethod
    def _validate_canonical(rows: list[CanonicalHolding]) -> Optional[str]:
        """Return None if rows look acceptable, else a human-readable
        error string describing the first violation."""
        if not rows:
            return "no holdings to write"
        # Weights, where present, should be in [0, 1.2] (allow some slack
        # for issuer rounding + cash > 100% edge cases in leveraged ETFs).
        for r in rows:
            if r.weight is not None and not (-0.01 <= r.weight <= 1.5):
                return (
                    f"weight {r.weight!r} for {r.underlying_ticker} outside "
                    "plausible [-0.01, 1.5] range — issuer may have changed "
                    "percent convention"
                )
        # Date format check
        try:
            datetime.strptime(rows[0].as_of_date, "%Y-%m-%d")
        except ValueError:
            return f"as_of_date {rows[0].as_of_date!r} is not ISO YYYY-MM-DD"
        return None

    # ---- Logging helpers --------------------------------------------

    def _record_failure(
        self, run_id: str, etf_ticker: str, issuer: str,
        source_url: str, error: str,
    ) -> EtfRunResult:
        result = EtfRunResult(
            etf_ticker=etf_ticker,
            issuer=issuer,
            status="failed",
            rows_loaded=0,
            as_of_date=None,
            source_url=source_url,
            error=error,
        )
        logger.error(
            "etf_loader | %s | FAILED | %s | %s",
            etf_ticker, issuer, error,
        )
        self._log_result(run_id, result)
        return result

    def _record_skip(
        self, run_id: str, etf_ticker: str, issuer: str, error: str,
    ) -> EtfRunResult:
        result = EtfRunResult(
            etf_ticker=etf_ticker,
            issuer=issuer,
            status="skipped",
            rows_loaded=0,
            as_of_date=None,
            source_url="",
            error=error,
        )
        logger.info(
            "etf_loader | %s | SKIPPED | %s | %s",
            etf_ticker, issuer, error,
        )
        self._log_result(run_id, result)
        return result

    def _log_result(self, run_id: str, r: EtfRunResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO etf_holdings_ingestion_log
                   (run_id, etf_ticker, issuer, as_of_date, status,
                    rows_loaded, source_url, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, r.etf_ticker, r.issuer, r.as_of_date, r.status,
                 r.rows_loaded, r.source_url, r.error),
            )
            conn.commit()

    # ---- Reporting --------------------------------------------------

    def coverage_report(self, last_n_runs: int = 5) -> list[dict]:
        """Return recent ingestion attempts for inspection."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM etf_holdings_ingestion_log
                   ORDER BY id DESC LIMIT ?""",
                (last_n_runs * 50,),  # rough upper bound
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def _build_provider(cfg: dict) -> HoldingsProvider:
    """Construct a provider from a config dict in ETF_PROVIDER_CONFIG."""
    issuer = cfg["provider"]
    if issuer == "ishares":
        return IsharesProvider(product_id=cfg["product_id"], slug=cfg["slug"])
    if issuer == "spdr":
        return SpdrProvider()
    if issuer == "generic":
        return GenericIssuerProvider(GenericIssuerConfig(**cfg["config"]))
    raise KeyError(f"unknown provider: {issuer!r}")


def _build_provider_by_issuer(issuer: str) -> HoldingsProvider:
    """Same as `_build_provider` but takes an issuer string only.

    Used by the local-file path where there's no per-ETF config to look
    up. For iShares this can't fully reconstruct the URL — but the
    LocalFileProvider doesn't need one anyway."""
    iss = issuer.lower()
    if iss == "spdr":
        return SpdrProvider()
    if iss == "ishares":
        # product_id / slug are URL-only; safe placeholders here
        return IsharesProvider(product_id="0", slug="local")
    raise KeyError(issuer)


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data.etf_holdings_loader",
        description="Ingest ETF holdings from issuer-hosted files",
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--verbose", "-v", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backfill", action="store_true",
                       help="Ingest every configured ETF")
    group.add_argument("--incremental", action="store_true",
                       help="Same as backfill — issuers publish snapshots")
    group.add_argument("--etf", metavar="TICKER",
                       help="Ingest one ETF only")
    group.add_argument("--file", metavar="PATH",
                       help="Ingest a local file (requires --etf + --issuer)")
    group.add_argument("--report", action="store_true",
                       help="Print the most recent ingestion log")
    parser.add_argument("--issuer",
                        help="Required with --file: 'ishares' or 'spdr'")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    loader = ETFHoldingsLoader(db_path=args.db)

    if args.report:
        rows = loader.coverage_report()
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if args.file:
        if not args.etf or not args.issuer:
            parser.error("--file requires --etf and --issuer")
        r = loader.run_local_file(args.etf, args.file, args.issuer)
        print(json.dumps(asdict(r), indent=2))
        return 0 if r.status == "success" else 1

    if args.etf:
        r = loader.run_single(args.etf)
        print(json.dumps(asdict(r), indent=2))
        return 0 if r.status == "success" else 1

    # Backfill or incremental — same code path
    summary = loader.run_backfill() if args.backfill else loader.run_incremental()
    print(json.dumps(summary.as_dict(), indent=2))
    return 0 if summary.failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
