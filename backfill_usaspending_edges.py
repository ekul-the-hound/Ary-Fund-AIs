"""
backfill_usaspending_edges.py
=============================
Populate a ``contract_awards`` table with federal contract dollars per company,
so the Money Flow Graph draws confirmed **US Government -> company** edges
(``data/money_graph.py``'s ``_edges_from_usaspending`` already reads this table).

Mirrors the working request in ``geo_supply.fetch_usaspending_for_recipient``
(USASpending.gov ``spending_by_award``), but writes a clean pair table instead
of registry events. Free API, no key.

Writes:
    contract_awards(ticker, recipient_name, amount_usd, award_count,
                    last_award_date, window_days)

Most companies have no federal contracts, so this is naturally sparse — expect
edges mainly for defense / industrial / IT / health names (BA, LMT, RTX, NOC,
GD, LHX, MSFT, ORCL, HCA, …).

Run from project root (venv active):
    python backfill_usaspending_edges.py                  # whole universe
    python backfill_usaspending_edges.py --days 365
    python backfill_usaspending_edges.py --tickers BA LMT RTX
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

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

_DONE = Path("data") / "usaspending_done.json"
_UA = "ARY QUANT research (contact: set SEC_USER_AGENT)"
_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Curated ticker -> USASpending recipient search text for the major federal
# contractors. Recipients are named by legal entity, not ticker, and yfinance
# names fuzzy-match poorly, so an explicit map is far more reliable here. Tickers
# not listed fall back to the resolved company name. Extend freely.
_RECIPIENT_MAP = {
    "BA": "Boeing", "LMT": "Lockheed Martin", "RTX": "Raytheon",
    "GD": "General Dynamics", "NOC": "Northrop Grumman", "LHX": "L3Harris",
    "HII": "Huntington Ingalls", "LDOS": "Leidos", "TXT": "Textron",
    "TDG": "TransDigm", "HON": "Honeywell", "GE": "General Electric",
    "CAT": "Caterpillar", "DE": "Deere", "PLTR": "Palantir",
    "ORCL": "Oracle", "IBM": "International Business Machines",
    "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon Web Services",
    "UNH": "UnitedHealth", "MCK": "McKesson", "CAH": "Cardinal Health",
    "COR": "Cencora", "HCA": "HCA Healthcare", "CACI": "CACI",
    "SAIC": "Science Applications", "ACN": "Accenture Federal",
    "DELL": "Dell", "CSCO": "Cisco", "GEHC": "GE HealthCare",
    "AXON": "Axon", "PWR": "Quanta Services",
}


def _load_done() -> set[str]:
    try:
        return set(json.loads(_DONE.read_text()))
    except Exception:
        return set()


def _save_done(done: set[str]) -> None:
    _DONE.parent.mkdir(parents=True, exist_ok=True)
    _DONE.write_text(json.dumps(sorted(done)))


def _make_market_data():
    for path in ("data.market_data", "market_data"):
        try:
            mod = __import__(path, fromlist=["MarketData"])
            return mod.MarketData(db_path=MARKET_DB)
        except Exception:  # noqa: BLE001
            continue
    return None


def _company_name(md, ticker: str) -> str:
    if md is not None:
        try:
            f = md.get_fundamentals(ticker) or {}
            if f.get("name"):
                return f["name"]
        except Exception:  # noqa: BLE001
            pass
    return ticker


def _ensure_table(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contract_awards (
            ticker TEXT PRIMARY KEY,
            recipient_name TEXT,
            amount_usd REAL,
            award_count INTEGER,
            last_award_date TEXT,
            window_days INTEGER,
            fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def _query_awards(recipient_name: str, days: int) -> dict:
    body = {
        "filters": {
            "recipient_search_text": [recipient_name],
            "time_period": [{
                "start_date": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d"),
            }],
            "award_type_codes": ["A", "B", "C", "D"],
        },
        "fields": ["Award ID", "Recipient Name", "Award Amount", "Start Date"],
        "sort": "Award Amount", "order": "desc", "limit": 100,
    }
    r = requests.post(_URL, json=body,
                      headers={"User-Agent": _UA, "Content-Type": "application/json"},
                      timeout=30)
    if r.status_code != 200:
        return {"total": 0.0, "n": 0, "last": None,
                "err": f"HTTP {r.status_code}: {r.text[:120]}"}
    results = (r.json() or {}).get("results") or []
    total, last = 0.0, None
    for row in results:
        try:
            total += float(row.get("Award Amount") or 0)
        except (TypeError, ValueError):
            pass
        d = row.get("Start Date")
        if d and (last is None or d > last):
            last = d
    return {"total": total, "n": len(results), "last": last}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    tickers = [t.upper() for t in (args.tickers or universe_tickers())]
    done = _load_done()
    todo = [t for t in tickers if t not in done]
    print(f"Tickers: {len(tickers)} | done: {len(done)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do (delete data/usaspending_done.json to re-run).")
        return

    md = _make_market_data()
    conn = sqlite3.connect(MARKET_DB)
    _ensure_table(conn)

    hits = 0
    t0 = time.time()
    for i, tk in enumerate(todo, 1):
        name = _RECIPIENT_MAP.get(tk) or _company_name(md, tk)
        try:
            res = _query_awards(name, args.days)
            if res["total"] > 0:
                conn.execute(
                    "INSERT OR REPLACE INTO contract_awards "
                    "(ticker, recipient_name, amount_usd, award_count, "
                    " last_award_date, window_days) VALUES (?,?,?,?,?,?)",
                    (tk, name, res["total"], res["n"], res["last"], args.days))
                conn.commit()
                hits += 1
                print(f"[{i:>3}/{len(todo)}] {tk:<6} {name[:28]:<28} "
                      f"${res['total']/1e9:.2f}B ({res['n']} awards)")
            else:
                why = res.get("err") or "no contracts matched"
                print(f"[{i:>3}/{len(todo)}] {tk:<6} {name[:28]:<28} — {why}")
            done.add(tk)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>3}/{len(todo)}] {tk:<6} ERROR: {e}")
        if i % 25 == 0:
            _save_done(done)
            rate = i / max(1e-6, time.time() - t0)
            print(f"    …{i}/{len(todo)} | {hits} with contracts | "
                  f"ETA {(len(todo)-i)/max(rate,1e-6)/60:.1f}m")
        time.sleep(args.sleep)

    _save_done(done)
    conn.close()
    print(f"\nDone. {hits} companies with federal contracts in "
          f"{(time.time()-t0)/60:.1f} min. Rebuild the graph for gov edges.")


if __name__ == "__main__":
    main()