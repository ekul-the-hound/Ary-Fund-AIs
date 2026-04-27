"""
geo_supply.py
=============
Geopolitical, supply-chain, energy and commodity ingestion.

Sources (all free)
------------------
* OFAC SDN list (US Treasury) — sanctions snapshot, daily diff -> events
* EU Consolidated Sanctions   — JSON feed
* UK HM Treasury Sanctions    — XML feed
* UN Security Council Sanctions — JSON
* GDELT 2.0 events            — financial-themed event counts
* EIA Open Data (no key)      — oil, natgas, electricity, inventories
* Stooq free index proxies    — Baltic Dry (BDI) etc.
* USAspending.gov             — government contract awards

Behavior
--------
* Each fetch is best-effort. A failure does not block other sources.
* Daily sanctions diff: compare today's set of sanctioned entities against
  yesterday's; new additions are emitted as ``sanctioned_entity_added``
  events on the registry.
* All numeric outputs land in canonical ``data_points`` so the LLM-facing
  snapshot is consistent.

Usage
-----
    >>> gs = GeoSupply()
    >>> gs.refresh_global()       # fetches all geopolitical/global signals
    >>> gs.refresh_ticker("XOM")  # ticker-specific energy/commodity links
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# URLs (all are stable public endpoints; rotate here when they move)
# ----------------------------------------------------------------------
OFAC_SDN_CSV = "https://www.treasury.gov/ofac/downloads/sdn.csv"
OFAC_CONSOLIDATED_CSV = "https://www.treasury.gov/ofac/downloads/consolidated/cons_prim.csv"
EU_SANCTIONS_JSON = (
    "https://webgate.ec.europa.eu/fsd/fsf/public/files/jsonFullSanctionsList_1_1/content"
)
UK_SANCTIONS_CSV = (
    "https://ofsistorage.blob.core.windows.net/publishlive/2022format/ConList.csv"
)
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
EIA_API = "https://api.eia.gov/v2"  # API key optional — pulls work without key for many endpoints

# Source IDs
SRC_OFAC = "ofac"
SRC_EU = "eu_sanctions"
SRC_UK = "uk_sanctions"
SRC_UN = "un_sanctions"
SRC_GDELT = "gdelt"
SRC_EIA = "eia"
SRC_FREIGHT = "stooq"
SRC_USASPENDING = "usaspending"


# ----------------------------------------------------------------------
# Class
# ----------------------------------------------------------------------


class GeoSupply:
    """Geopolitical / supply chain / commodity ingestion. Writes to registry."""

    def __init__(
        self,
        db_path: str = "data/hedgefund.db",
        registry=None,
        eia_api_key: Optional[str] = None,
        timeout: int = 30,
        user_agent: str = "ary-quant/1.0 (+research)",
    ):
        self.db_path = db_path
        self._registry = registry
        self.eia_api_key = eia_api_key
        self.timeout = timeout
        self.user_agent = user_agent
        self._init_db()
        self._register_sources()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------
    @property
    def registry(self):
        if self._registry is None:
            try:
                from data.data_registry import get_default_registry
                self._registry = get_default_registry(self.db_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("geo_supply | registry load failed: %s", e)
        return self._registry

    def _register_sources(self) -> None:
        reg = self.registry
        if reg is None:
            return
        try:
            reg.register_source(SRC_OFAC,        "sanctions",   "daily",  base_priority=1)
            reg.register_source(SRC_EU,          "sanctions",   "daily",  base_priority=1)
            reg.register_source(SRC_UK,          "sanctions",   "daily",  base_priority=1)
            reg.register_source(SRC_UN,          "sanctions",   "daily",  base_priority=1)
            reg.register_source(SRC_GDELT,       "geopolitical","hourly", base_priority=2)
            reg.register_source(SRC_EIA,         "energy",      "daily",  base_priority=1)
            reg.register_source(SRC_FREIGHT,     "freight",     "daily",  base_priority=2)
            reg.register_source(SRC_USASPENDING, "government",  "weekly", base_priority=2)
        except Exception as e:  # noqa: BLE001
            logger.debug("geo_supply | source registration skipped: %s", e)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sanctions_list (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT,
                    listing_id TEXT,
                    country TEXT,
                    program TEXT,
                    listed_at TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(source, entity_name, listing_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sanc_src ON sanctions_list(source, fetched_at)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS geopolitical_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    region TEXT,
                    theme TEXT,
                    title TEXT,
                    url TEXT,
                    tone REAL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(source, url, occurred_at)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_geo_when ON geopolitical_events(occurred_at)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS energy_data (
                    series_id TEXT NOT NULL,
                    period TEXT NOT NULL,
                    value REAL,
                    unit TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (series_id, period)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS freight_indices (
                    index_name TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    value REAL,
                    source TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (index_name, as_of)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS commodity_prices (
                    symbol TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    price_usd REAL,
                    source TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (symbol, as_of)
                )
            """)

    # ------------------------------------------------------------------
    # Sanctions: OFAC SDN
    # ------------------------------------------------------------------
    def fetch_ofac_sdn(self) -> int:
        """Pull the OFAC SDN consolidated CSV. Each row is a sanctioned entity."""
        try:
            resp = requests.get(OFAC_SDN_CSV, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
        except Exception as e:  # noqa: BLE001
            logger.warning("ofac sdn | fetch failed: %s", e)
            return 0

        # The OFAC SDN CSV does NOT have a header row. Columns:
        #   ent_num, SDN_Name, SDN_Type, Program, Title, Call_Sign,
        #   Vess_type, Tonnage, GRT, Vess_flag, Vess_owner, Remarks
        n = self._record_sanctions_diff(SRC_OFAC, self._parse_ofac_csv(resp.text))
        return n

    @staticmethod
    def _parse_ofac_csv(text: str) -> list[dict]:
        rows: list[dict] = []
        reader = csv.reader(io.StringIO(text))
        for row in reader:
            if len(row) < 4:
                continue
            ent_num, name, etype, program = row[0], row[1].strip(), row[2].strip(), row[3].strip()
            if not name or name.lower() == "name":
                continue
            rows.append({
                "entity_name": name,
                "entity_type": etype,
                "listing_id": str(ent_num),
                "program": program,
                "country": None,
            })
        return rows

    # ------------------------------------------------------------------
    # Sanctions: EU consolidated
    # ------------------------------------------------------------------
    def fetch_eu_sanctions(self) -> int:
        try:
            resp = requests.get(EU_SANCTIONS_JSON, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("eu sanctions | fetch failed: %s", e)
            return 0
        # The EU file is large and the schema can vary. Be defensive.
        rows: list[dict] = []
        entries = data if isinstance(data, list) else data.get("sanctionsEntities") or []
        for e in entries:
            try:
                name_alias = e.get("nameAlias") or []
                primary_name = ""
                if name_alias:
                    primary_name = (
                        name_alias[0].get("wholeName")
                        or name_alias[0].get("firstName", "") + " " + name_alias[0].get("lastName", "")
                    ).strip()
                if not primary_name:
                    continue
                listing_id = str(e.get("logicalId") or e.get("id") or "")
                rows.append({
                    "entity_name": primary_name,
                    "entity_type": e.get("subjectType", {}).get("classificationCode"),
                    "listing_id": listing_id,
                    "program": (e.get("regulation") or {}).get("publicationDate", ""),
                    "country": ((e.get("citizenship") or [{}])[0].get("countryDescription")
                                if e.get("citizenship") else None),
                })
            except Exception:  # noqa: BLE001
                continue
        return self._record_sanctions_diff(SRC_EU, rows)

    # ------------------------------------------------------------------
    # Sanctions: UK HMT (CSV)
    # ------------------------------------------------------------------
    def fetch_uk_sanctions(self) -> int:
        try:
            resp = requests.get(UK_SANCTIONS_CSV, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
        except Exception as e:  # noqa: BLE001
            logger.warning("uk sanctions | fetch failed: %s", e)
            return 0
        rows: list[dict] = []
        reader = csv.DictReader(io.StringIO(resp.text))
        for r in reader:
            name = (r.get("Name 6") or r.get("Name") or "").strip()
            if not name:
                continue
            rows.append({
                "entity_name": name,
                "entity_type": (r.get("Group Type") or "").strip(),
                "listing_id": (r.get("Group ID") or "").strip(),
                "program": (r.get("Regime") or "").strip(),
                "country": (r.get("Country of Birth") or r.get("Address Country") or "").strip(),
            })
        return self._record_sanctions_diff(SRC_UK, rows)

    # ------------------------------------------------------------------
    # Sanctions diff -> events
    # ------------------------------------------------------------------
    def _record_sanctions_diff(self, source: str, rows: list[dict]) -> int:
        """Insert any rows we haven't seen before. Each new addition gets
        a ``sanctioned_entity_added`` event in the registry."""
        if not rows:
            return 0
        n = 0
        new_rows = []
        with sqlite3.connect(self.db_path) as conn:
            for r in rows:
                try:
                    cur = conn.execute(
                        """INSERT INTO sanctions_list
                            (source, entity_name, entity_type, listing_id,
                             country, program, listed_at)
                           VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
                        (source, r["entity_name"], r.get("entity_type"),
                         r.get("listing_id"), r.get("country"), r.get("program")),
                    )
                    if cur.rowcount > 0:
                        n += 1
                        new_rows.append(r)
                except sqlite3.IntegrityError:
                    continue
        # Emit events
        if self.registry and new_rows:
            now = datetime.now().isoformat(timespec="seconds")
            for r in new_rows:
                self.registry.upsert_event(
                    event_type="sanctioned_entity_added",
                    occurred_at=now, source_id=source,
                    severity=0.7,
                    payload={"entity": r["entity_name"], "program": r.get("program"),
                             "country": r.get("country")},
                )
            # Aggregate count over last 7 days
            since = (datetime.now() - timedelta(days=7)).isoformat()
            recent = self.registry.recent_events(
                event_type="sanctioned_entity_added", since=since, limit=10_000,
            )
            self.registry.upsert_point(
                "global", "global", "global.sanctions_added_7d",
                as_of=datetime.now().strftime("%Y-%m-%d"),
                source_id="derived", value_num=float(len(recent)),
                confidence=0.9,
            )
        return n

    # ------------------------------------------------------------------
    # GDELT geopolitical events (financial themes)
    # ------------------------------------------------------------------
    def fetch_gdelt_geopolitical(self, hours_back: int = 24) -> int:
        """Pull GDELT articles tagged with financial/geopolitical themes."""
        ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_back))
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        url = (
            f"{GDELT_DOC_API}?query=theme:ECON_BANKRUPTCY OR theme:ARMEDCONFLICT "
            f"OR theme:SANCTIONS&mode=ArtList&maxrecords=100"
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
            logger.debug("gdelt geo | failed: %s", e)
            return 0

        n = 0
        articles = data.get("articles") or []
        with sqlite3.connect(self.db_path) as conn:
            for a in articles:
                url_a = a.get("url", "")
                title = a.get("title", "")
                seendate = a.get("seendate", "")
                tone = a.get("tone")
                try:
                    tone_f = float(tone) if tone is not None else None
                except (TypeError, ValueError):
                    tone_f = None
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO geopolitical_events
                            (occurred_at, region, theme, title, url, tone, source)
                           VALUES (?, ?, 'GDELT_FINANCIAL', ?, ?, ?, 'gdelt')""",
                        (seendate, a.get("sourcecountry"), title, url_a, tone_f),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        if self.registry:
            self.registry.upsert_point(
                "global", "global", "global.geopolitical_event_24h",
                as_of=datetime.now().strftime("%Y-%m-%d"),
                source_id=SRC_GDELT, value_num=float(len(articles)),
                confidence=0.8,
            )
        return n

    # ------------------------------------------------------------------
    # EIA energy data
    # ------------------------------------------------------------------
    def fetch_eia_series(
        self, route: str, params: Optional[dict] = None,
        commodity: Optional[str] = None,
    ) -> int:
        """Fetch a single EIA v2 endpoint and persist values."""
        params = dict(params or {})
        if self.eia_api_key:
            params["api_key"] = self.eia_api_key
        url = f"{EIA_API}/{route}"
        try:
            resp = requests.get(url, params=params,
                                headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("eia | %s failed: %s", route, e)
            return 0
        rows = ((data or {}).get("response") or {}).get("data") or []
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for r in rows:
                period = str(r.get("period") or "")
                series_id = str(r.get("series") or r.get("seriesId") or route)
                val = r.get("value")
                try:
                    val_f = float(val) if val is not None else None
                except (TypeError, ValueError):
                    val_f = None
                if not period or val_f is None:
                    continue
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO energy_data
                            (series_id, period, value, unit)
                           VALUES (?, ?, ?, ?)""",
                        (series_id, period, val_f, r.get("units")),
                    )
                    n += 1
                except sqlite3.IntegrityError:
                    continue
        # If a commodity tag was given, push the most-recent value into the
        # registry under commodity.spot_usd
        if commodity and rows and self.registry:
            latest = max(rows, key=lambda x: str(x.get("period") or ""))
            try:
                self.registry.upsert_point(
                    commodity, "commodity", "commodity.spot_usd",
                    as_of=str(latest.get("period")),
                    source_id=SRC_EIA,
                    value_num=float(latest.get("value")),
                    confidence=1.0,
                )
            except Exception:  # noqa: BLE001
                pass
        return n

    # ------------------------------------------------------------------
    # Freight indices via Stooq
    # ------------------------------------------------------------------
    def fetch_stooq_freight(self, symbol: str = "bdiy") -> int:
        """Stooq publishes CSVs of indices including BDIY (Baltic Dry).
        URL: https://stooq.com/q/d/?s=bdiy&i=d&o=1111111&d1=...&d2=...
        We grab the latest line.
        """
        url = f"https://stooq.com/q/?s={symbol}&f=csv"
        try:
            resp = requests.get(url, headers={"User-Agent": self.user_agent},
                                timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            text = resp.text
        except Exception as e:  # noqa: BLE001
            logger.debug("stooq | %s failed: %s", symbol, e)
            return 0
        # CSV format: "Symbol,Date,Time,Open,High,Low,Close,Volume,..."
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return 0
        header = lines[0].lower().split(",")
        try:
            date_idx = header.index("date")
            close_idx = header.index("close")
        except ValueError:
            return 0
        parts = lines[1].split(",")
        try:
            as_of = parts[date_idx].strip()
            value = float(parts[close_idx])
        except (ValueError, IndexError):
            return 0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO freight_indices
                    (index_name, as_of, value, source) VALUES (?, ?, ?, 'stooq')""",
                (symbol.upper(), as_of, value),
            )
        if self.registry:
            field_map = {"bdiy": "freight.bdiy"}
            f = field_map.get(symbol.lower())
            if f:
                self.registry.upsert_point(
                    "global", "global", f,
                    as_of=as_of, source_id=SRC_FREIGHT,
                    value_num=float(value), confidence=0.7,
                )
        return 1

    # ------------------------------------------------------------------
    # Government contracts (USAspending.gov)
    # ------------------------------------------------------------------
    def fetch_usaspending_for_recipient(self, recipient_name: str, days_back: int = 90) -> int:
        """Query USAspending.gov for recent awards to a recipient by name."""
        url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
        body = {
            "filters": {
                "recipient_search_text": [recipient_name],
                "time_period": [{
                    "start_date": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                    "end_date":   datetime.now().strftime("%Y-%m-%d"),
                }],
                "award_type_codes": ["A", "B", "C", "D"],
            },
            "fields": ["Award ID", "Recipient Name", "Award Amount",
                       "Action Date", "Awarding Agency"],
            "sort": "Action Date", "order": "desc",
            "limit": 50,
        }
        try:
            resp = requests.post(url, json=body,
                                 headers={"User-Agent": self.user_agent,
                                          "Content-Type": "application/json"},
                                 timeout=self.timeout)
            if resp.status_code != 200:
                return 0
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.debug("usaspending | %s failed: %s", recipient_name, e)
            return 0
        results = data.get("results") or []
        if not results:
            return 0
        # Aggregate total $ awarded last N days
        total = 0.0
        for r in results:
            try:
                total += float(r.get("Award Amount") or 0)
            except (TypeError, ValueError):
                continue
        if self.registry and total > 0:
            self.registry.upsert_event(
                event_type="govt_contracts_awarded",
                occurred_at=datetime.now().isoformat(timespec="seconds"),
                source_id=SRC_USASPENDING,
                entity_id=recipient_name, entity_type="company",
                severity=0.4,
                payload={"total_usd": total, "n_awards": len(results),
                         "window_days": days_back},
            )
        return len(results)

    # ------------------------------------------------------------------
    # Orchestrators
    # ------------------------------------------------------------------
    def refresh_global(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, fn in (
            ("ofac",          self.fetch_ofac_sdn),
            ("eu_sanctions",  self.fetch_eu_sanctions),
            ("uk_sanctions",  self.fetch_uk_sanctions),
            ("gdelt",         self.fetch_gdelt_geopolitical),
            ("freight_bdiy",  lambda: self.fetch_stooq_freight("bdiy")),
        ):
            try:
                out[name] = fn()
            except Exception as e:  # noqa: BLE001
                logger.warning("geo_supply.refresh_global | %s failed: %s", name, e)
                out[name] = 0
        return out

    def refresh_ticker(self, ticker: str, recipient_name: Optional[str] = None) -> dict[str, int]:
        """Ticker-specific lookups. ``recipient_name`` is the company's
        registered name for USAspending.gov (defaults to ticker)."""
        out: dict[str, int] = {}
        try:
            out["govt_contracts"] = self.fetch_usaspending_for_recipient(
                recipient_name or ticker
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("geo_supply.refresh_ticker | %s | usaspending failed: %s",
                           ticker, e)
            out["govt_contracts"] = 0
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gs = GeoSupply()
    print("Schema OK; tables present.")
