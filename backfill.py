r"""
backfill.py — one CLI for every data-ingestion backfill.

Consolidates the eight backfill_*.py scripts into a single argparse tool with
subcommands. Each subcommand preserves the exact flags, SQL, resume markers,
and behavior of the original script it replaces, so existing progress files
(data/*_done.json, data/f13f_corrected.json) carry over unchanged.

Run from the project root, venv active (.\hedgefund_ai\Scripts\Activate.ps1):

    python backfill.py filing-text --tickers MSFT AAPL
    python backfill.py filing-text --all
    python backfill.py fundamentals --limit 10
    python backfill.py universe --limit 50
    python backfill.py money-graph NVDA AAPL MSFT
    python backfill.py 13f --count 8
    python backfill.py ownership-filers --ticker NVDA
    python backfill.py news-edges --fetch --limit 100
    python backfill.py usaspending-edges --tickers BA LMT RTX

Subcommand -> replaced script
    filing-text        backfill_filing_text.py
    fundamentals       backfill_fundamentals.py
    universe           backfill_universe.py
    money-graph        backfill_money_graph.py
    13f                backfill_13f.py
    ownership-filers   backfill_ownership_filers.py
    news-edges         backfill_news_edges.py
    usaspending-edges  backfill_usaspending_edges.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Set


# ======================================================================
# Shared helpers (these were duplicated across ~5 of the originals)
# ======================================================================
def _cfg():
    """Return the config module, tolerating data.config or top-level config."""
    try:
        from data import config as cfg  # type: ignore
        return cfg
    except Exception:
        try:
            import config as cfg  # type: ignore
            return cfg
        except Exception:
            return None


def _market_db() -> str:
    """Resolve the market DB path the same way the app/graph code does."""
    cfg = _cfg()
    return (getattr(cfg, "MARKET_DB_PATH", None)
            or getattr(cfg, "HEDGEFUND_DB_PATH", None)
            or "data/hedgefund.db")


def _portfolio_db() -> str:
    import config
    return config.PORTFOLIO_DB_PATH


def _load_done(path: Path) -> Set[str]:
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def _save_done(path: Path, done: Set[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(done)))
    except Exception as e:  # noqa: BLE001
        print(f"  (could not save progress: {e})")


def _universe() -> list:
    try:
        from data.universe import US_UNIVERSE
    except Exception:
        from universe import US_UNIVERSE  # type: ignore
    return list(US_UNIVERSE)


def _sec_fetcher(db_path: str | None = None):
    try:
        from data.sec_fetcher import SECFetcher
    except Exception:
        from sec_fetcher import SECFetcher  # type: ignore
    return SECFetcher(db_path=db_path) if db_path else SECFetcher()


def _money_graph():
    try:
        from data import money_graph as mg
    except Exception:
        import money_graph as mg  # type: ignore
    return mg


# ======================================================================
# filing-text  (was backfill_filing_text.py)
# ======================================================================
def cmd_filing_text(args: argparse.Namespace) -> int:
    db = _portfolio_db()
    sec = _sec_fetcher(db)

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    if args.all:
        rows = conn.execute(
            "SELECT ticker, filing_type, filed_date, accession_number "
            "FROM sec_filings "
            "WHERE full_text IS NULL OR full_text = '' "
            "ORDER BY ticker, filed_date DESC"
        ).fetchall()
    else:
        tickers = [t.upper() for t in args.tickers]
        placeholders = ",".join("?" for _ in tickers)
        rows = conn.execute(
            f"SELECT ticker, filing_type, filed_date, accession_number "
            f"FROM sec_filings "
            f"WHERE ticker IN ({placeholders}) "
            f"AND (full_text IS NULL OR full_text = '') "
            f"ORDER BY ticker, filed_date DESC",
            tickers,
        ).fetchall()
    conn.close()

    if not rows:
        print("Nothing to backfill — all targeted filings already have text.")
        return 0

    print(f"Backfilling {len(rows)} filing(s) (downloading from EDGAR)...\n")
    ok, failed = 0, 0
    for r in rows:
        acc = r["accession_number"]
        label = f"{r['ticker']:6} {r['filing_type']:8} {r['filed_date']}"
        try:
            text = sec.get_filing_text(acc, max_chars=args.max_chars)
            n = len(text or "")
            if n > 0:
                print(f"  OK   {label} | {n:,} chars")
                ok += 1
            else:
                print(f"  WARN {label} | downloaded but empty")
                failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {label} | {e}")
            failed += 1

    print(f"\nDone. {ok} succeeded, {failed} failed.")
    if ok:
        print("Now run:  python fill_rag.py --tickers " +
              " ".join(sorted({r["ticker"] for r in rows})))
    return 0 if ok else 1


# ======================================================================
# fundamentals  (was backfill_fundamentals.py)
# ======================================================================
def _tickers_with_sector(db_path: str) -> Set[str]:
    conn = sqlite3.connect(db_path)
    done: Set[str] = set()
    try:
        rows = conn.execute(
            "SELECT ticker, data_json FROM fundamentals_cache"
        ).fetchall()
    except sqlite3.OperationalError:
        return done
    finally:
        conn.close()
    for tk, blob in rows:
        try:
            d = json.loads(blob)
            if (d.get("sector") or "").strip():
                done.add(tk)
        except Exception:  # noqa: BLE001
            continue
    return done


def cmd_fundamentals(args: argparse.Namespace) -> int:
    try:
        from data.market_data import MarketData
    except Exception:  # noqa: BLE001
        from market_data import MarketData  # type: ignore

    db_path = _portfolio_db()
    universe = list(args.tickers) if args.tickers else _universe()

    done = set() if args.force else _tickers_with_sector(db_path)
    pending = [t for t in universe if t not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    print(f"universe={len(universe)}  already_have_sector={len(done)}  pending={total}")
    if not total:
        print("Nothing to do — all requested tickers already have a sector.")
        return 0

    md = MarketData(db_path=db_path)
    t0 = time.time()
    ok = no_sector = failed = 0

    for i, ticker in enumerate(pending, 1):
        try:
            f = md.get_fundamentals(ticker, use_cache=not args.force)
            sector = (f.get("sector") or "").strip() if isinstance(f, dict) else ""
            if sector:
                ok += 1
                tag = sector
            else:
                no_sector += 1
                tag = "(no sector)"
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"[{i}/{total}] {ticker:6} -> {tag:24} "
                  f"(ok={ok} nosec={no_sector} fail={failed}, ETA ~{eta:.0f}m)")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}. Progress saved — re-run to resume.")
            return 130
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{total}] {ticker:6} -> ERROR ({e})")

    print(f"\nDone. {ok} with sector, {no_sector} no-sector, {failed} errors, "
          f"in {(time.time()-t0)/60:.1f} min.")
    print("Next: re-run the pipeline (peer stats will recompute with sectors).")
    return 0


# ======================================================================
# universe  (was backfill_universe.py)
# ======================================================================
def _tickers_with_text(db_path: str) -> Set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM sec_filings "
            "WHERE full_text IS NOT NULL AND full_text != ''"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _universe_backfill_one(sec, ticker, kinds, max_10k, max_10q, max_chars):
    hydrated = 0
    for kind in kinds:
        count = max_10k if kind == "10-K" else max_10q
        if count <= 0:
            continue
        try:
            filings = sec.get_filings(ticker, filing_type=kind, count=count)
        except Exception as e:
            print(f"    ! {ticker} {kind}: list fetch failed ({e})")
            continue
        for f in filings:
            acc = f.get("accession_number")
            if not acc:
                continue
            try:
                text = sec.get_filing_text(acc, max_chars=max_chars)
                if text:
                    hydrated += 1
            except Exception as e:
                print(f"    ! {ticker} {kind} {acc}: text fetch failed ({e})")
    return hydrated


def cmd_universe(args: argparse.Namespace) -> int:
    db_path = _portfolio_db()
    universe = list(args.tickers) if args.tickers else _universe()
    done = set() if args.force else _tickers_with_text(db_path)
    pending = [t for t in universe if t not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    print(f"universe={len(universe)}  already_done={len(done)}  pending={total}")
    if not total:
        print("Nothing to do.")
        return 0

    sec = _sec_fetcher(db_path)
    t0 = time.time()
    ok = failed = 0
    for i, ticker in enumerate(pending, 1):
        try:
            n = _universe_backfill_one(
                sec, ticker, args.kinds, args.max_10k, args.max_10q, args.max_chars)
            if n:
                ok += 1
            else:
                failed += 1
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(f"[{i}/{total}] {ticker:6} -> {n} filings   "
                  f"(ok={ok} fail={failed}, ETA ~{eta:.0f}m)")
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}. Progress saved — re-run to resume.")
            return 130
        except Exception as e:
            failed += 1
            print(f"[{i}/{total}] {ticker:6} -> ERROR ({e})")
    print(f"\nDone. {ok} hydrated, {failed} no-text, in {(time.time()-t0)/60:.1f} min.")
    return 0


# ======================================================================
# money-graph  (was backfill_money_graph.py)
# ======================================================================
def cmd_money_graph(args: argparse.Namespace) -> int:
    done_file = Path("data") / "money_graph_backfill_done.json"
    tickers = [t.upper() for t in args.tickers] or _universe()
    done = _load_done(done_file)
    todo = [t for t in tickers if t not in done]

    print(f"Universe: {len(tickers)} | already done: {len(done)} | "
          f"to process: {len(todo)}")
    if not todo:
        print("Nothing to do — delete data/money_graph_backfill_done.json to "
              "force a full re-run.")
        return 0

    f = _sec_fetcher()
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        try:
            counts = f.refresh_ticker_filings(t)  # XBRL + Form 4 + SC 13D/13G
            own = counts.get("SC 13D", 0) + counts.get("SC 13G", 0)
            print(f"[{i:>3}/{len(todo)}] {t:<6} ownership={own} "
                  f"form4={counts.get('form4', 0)} xbrl={counts.get('xbrl_facts', 0)}")
            done.add(t)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:>3}/{len(todo)}] {t:<6} ERROR: {e}")
        if i % 20 == 0:
            _save_done(done_file, done)
            rate = i / max(1e-6, time.time() - t0)
            eta = (len(todo) - i) / max(1e-6, rate)
            print(f"    ...{i} done, ~{rate:.1f}/s, ETA ~{eta/60:.1f} min")

    _save_done(done_file, done)
    print(f"\nDone. Processed {len(todo)} tickers in {(time.time()-t0)/60:.1f} min.")
    print("Now open the app, choose Scope = 'Full universe', and hit Rebuild.")
    return 0


# ======================================================================
# 13f  (was backfill_13f.py)
# ======================================================================
def _correct_post2023_values(market_db: str, corrected_path: Path) -> int:
    """SEC switched 13F <value> from $thousands to whole dollars in 2023-Q1,
    but the fetcher multiplies by 1000 unconditionally — so 2023+ filings come
    out 1000x too large. Divide those back down, at most once per accession."""
    corrected = set()
    try:
        corrected = set(json.loads(corrected_path.read_text()))
    except Exception:
        pass
    fixed = 0
    try:
        with sqlite3.connect(market_db) as conn:
            accs = [r[0] for r in conn.execute(
                "SELECT DISTINCT accession_number FROM f13f_holdings "
                "WHERE period_of_report >= '2023-01-01'").fetchall()]
            for acc in accs:
                if acc in corrected:
                    continue
                conn.execute(
                    "UPDATE f13f_holdings SET value_usd = value_usd/1000.0 "
                    "WHERE accession_number=? AND period_of_report >= '2023-01-01'",
                    (acc,))
                corrected.add(acc)
                fixed += 1
        corrected_path.parent.mkdir(parents=True, exist_ok=True)
        corrected_path.write_text(json.dumps(sorted(corrected)))
    except Exception as e:  # noqa: BLE001
        print(f"  (value correction skipped: {e})")
    return fixed


def cmd_13f(args: argparse.Namespace) -> int:
    try:
        from data.filer_canonical import CIK_TO_NAME, name_for_cik
    except Exception:
        from filer_canonical import CIK_TO_NAME, name_for_cik  # type: ignore

    market_db = _market_db()
    done_file = Path("data") / "f13f_backfill_done.json"
    corrected_file = Path("data") / "f13f_corrected.json"

    ciks = [str(c) for c in (args.cik or list(CIK_TO_NAME.keys()))]
    done = _load_done(done_file)
    todo = [c for c in ciks if c not in done]

    print(f"Institutions: {len(ciks)} | already done: {len(done)} | "
          f"to process: {len(todo)}")

    total_rows = 0
    t0 = time.time()
    if todo:
        f = _sec_fetcher()
        for i, cik in enumerate(todo, 1):
            name = name_for_cik(cik) or f"CIK {cik}"
            try:
                n = f.ingest_13f_filings_by_filer(cik, count=args.count)
                total_rows += n
                flag = "" if n else "  (0 rows — check CIK / no recent 13F)"
                print(f"[{i:>2}/{len(todo)}] {name:<26} CIK {cik:<9} holdings={n}{flag}")
                done.add(cik)
            except Exception as e:  # noqa: BLE001
                print(f"[{i:>2}/{len(todo)}] {name:<26} CIK {cik:<9} ERROR: {e}")
            _save_done(done_file, done)
            time.sleep(args.sleep)
        print(f"\nDone. {total_rows} total holdings rows across {len(todo)} "
              f"institutions in {(time.time()-t0)/60:.1f} min.")
    else:
        print("All filers already ingested — running correction + index only.")

    fixed = _correct_post2023_values(market_db, corrected_file)
    print(f"Corrected 2023+ value scaling on {fixed} new filing(s).")

    try:
        with sqlite3.connect(market_db) as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_13f_value "
                         "ON f13f_holdings(value_usd DESC)")
        print("Ensured idx_13f_value index.")
    except Exception as e:  # noqa: BLE001
        print(f"  (index skipped: {e})")
    print("Rebuild the graph:  Scope = Full universe -> Rebuild "
          "(institutions now merge across 13D/G + 13F).")
    return 0


# ======================================================================
# ownership-filers  (was backfill_ownership_filers.py)
# ======================================================================
import re  # noqa: E402  (used only by ownership-filers + news-edges)

_HDR_RE = re.compile(
    r'FILED BY[:\s].*?COMPANY CONFORMED NAME[:\s]*([^\r\n]+)', re.I | re.S)
_ANY_RE = re.compile(r'COMPANY CONFORMED NAME[:\s]*([^\r\n]+)', re.I)
_JUNK_FILER = re.compile(
    r"^(none|n/?a|null|form\s+sc\s+13[dg](/a)?|sc\s+13[dg](/a)?|"
    r"sec\s+schedule\s+13[dg](/a)?|schedule\s+13[dg](/a)?|13[dg](/a)?|"
    r"ms\s+initial|initial|amendment|amended)$", re.I)


def _needs_resolution(name: str) -> bool:
    s = (name or "").strip()
    return (not s) or (_JUNK_FILER.match(s) is not None)


def _header_url(cik, accession) -> str:
    acc_nodash = str(accession).replace("-", "")
    return (f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_nodash}/{accession}.txt")


def _get_header(fetcher, url) -> str:
    sess = getattr(fetcher, "session", None) or getattr(fetcher, "_session", None)
    if sess is not None:
        try:
            r = sess.get(url, headers={"Range": "bytes=0-32767"}, timeout=30)
            if r.status_code in (200, 206) and r.text:
                return r.text
        except Exception:
            pass
    try:
        return fetcher._get(url).text[:32768]
    except Exception:
        return ""


def _parse_filer(header: str):
    m = _HDR_RE.search(header)
    if m:
        return m.group(1).strip() or None
    names = _ANY_RE.findall(header)          # [0]=subject, [1]=filer
    if len(names) >= 2 and names[1].strip():
        return names[1].strip()
    return None


def cmd_ownership_filers(args: argparse.Namespace) -> int:
    market_db = _market_db()
    done_file = Path("data") / "ownership_filers_done.json"

    where = "1=1"
    params: tuple = ()
    if args.ticker:
        where = "ticker=?"
        params = (args.ticker.upper(),)
    sql = (f"SELECT DISTINCT accession_number, cik, filer_name "
           f"FROM ownership_filings WHERE {where} ORDER BY accession_number DESC")

    with sqlite3.connect(market_db) as conn:
        all_rows = conn.execute(sql, params).fetchall()
    rows = [(a, c) for (a, c, name) in all_rows if _needs_resolution(name)]

    done = _load_done(done_file)
    todo = [(a, c) for (a, c) in rows if a not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"DB: {market_db}")
    print(f"Accessions to resolve (empty or form-label filer): {len(rows)} | "
          f"to process now: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    fetcher = _sec_fetcher()
    wconn = sqlite3.connect(market_db)
    wconn.create_function(
        "isjunk", 1, lambda s: 1 if _needs_resolution(s) else 0)

    resolved = failed = 0
    t0 = time.time()
    try:
        for i, (acc, cik) in enumerate(todo, 1):
            try:
                header = _get_header(fetcher, _header_url(cik, acc))
                filer = _parse_filer(header) if header else None
                if filer:
                    wconn.execute(
                        "UPDATE OR IGNORE ownership_filings SET filer_name=? "
                        "WHERE accession_number=? AND isjunk(filer_name)=1",
                        (filer[:120], acc))
                    wconn.commit()
                    resolved += 1
                else:
                    failed += 1
                done.add(acc)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  {acc}: {e}")
            if i % 25 == 0:
                _save_done(done_file, done)
                rate = i / max(1e-6, time.time() - t0)
                print(f"  [{i}/{len(todo)}] resolved={resolved} failed={failed} "
                      f"~{rate:.1f}/s ETA {(len(todo)-i)/max(rate,1e-6)/60:.1f}m")
            time.sleep(args.sleep)
    finally:
        wconn.close()

    _save_done(done_file, done)
    print(f"\nDone. resolved={resolved} failed={failed} "
          f"in {(time.time()-t0)/60:.1f} min.")
    print("Now rebuild the graph:  Scope = Full universe -> Rebuild.")
    return 0


# ======================================================================
# news-edges  (was backfill_news_edges.py)
# ======================================================================
from itertools import combinations  # noqa: E402

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


def _ensure_news_table(conn) -> None:
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
    try:
        from data.sentiment_news import SentimentNews
    except Exception:
        from sentiment_news import SentimentNews  # type: ignore
    sn = SentimentNews()
    tickers = universe[:limit] if limit else universe
    print(f"Fetching news for {len(tickers)} tickers...")
    for i, tk in enumerate(tickers, 1):
        try:
            sn.refresh_ticker(tk)
        except Exception as e:  # noqa: BLE001
            print(f"  {tk}: {e}")
        if i % 25 == 0:
            print(f"  ...{i}/{len(tickers)}")
        time.sleep(0.2)


def cmd_news_edges(args: argparse.Namespace) -> int:
    mg = _money_graph()
    market_db = _market_db()

    universe = set(mg.universe_tickers())
    if args.fetch:
        _fetch_news(sorted(universe), args.limit)

    conn = sqlite3.connect(market_db)
    _ensure_news_table(conn)

    try:
        rows = conn.execute(
            "SELECT url, ticker, title, published_at FROM news_articles "
            "WHERE url IS NOT NULL AND url <> ''").fetchall()
    except sqlite3.OperationalError:
        print("No news_articles table found. Run sentiment_news ingestion first, "
              "or use --fetch.")
        conn.close()
        return 1

    by_url: dict = {}
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
        for a, b in combinations(tks, 2):
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
    return 0


# ======================================================================
# usaspending-edges  (was backfill_usaspending_edges.py)
# ======================================================================
from datetime import datetime, timedelta  # noqa: E402

_UA = "ARY QUANT research (contact: set SEC_USER_AGENT)"
_USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
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


def _make_market_data(market_db: str):
    for path in ("data.market_data", "market_data"):
        try:
            mod = __import__(path, fromlist=["MarketData"])
            return mod.MarketData(db_path=market_db)
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


def _ensure_contracts_table(conn) -> None:
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
    import requests
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
    r = requests.post(_USASPENDING_URL, json=body,
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


def cmd_usaspending_edges(args: argparse.Namespace) -> int:
    mg = _money_graph()
    market_db = _market_db()
    done_file = Path("data") / "usaspending_done.json"

    tickers = [t.upper() for t in (args.tickers or mg.universe_tickers())]
    done = _load_done(done_file)
    todo = [t for t in tickers if t not in done]
    print(f"Tickers: {len(tickers)} | done: {len(done)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do (delete data/usaspending_done.json to re-run).")
        return 0

    md = _make_market_data(market_db)
    conn = sqlite3.connect(market_db)
    _ensure_contracts_table(conn)

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
            _save_done(done_file, done)
            rate = i / max(1e-6, time.time() - t0)
            print(f"    ...{i}/{len(todo)} | {hits} with contracts | "
                  f"ETA {(len(todo)-i)/max(rate,1e-6)/60:.1f}m")
        time.sleep(args.sleep)

    _save_done(done_file, done)
    conn.close()
    print(f"\nDone. {hits} companies with federal contracts in "
          f"{(time.time()-t0)/60:.1f} min. Rebuild the graph for gov edges.")
    return 0


# ======================================================================
# Parser
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified data-ingestion backfills.")
    sub = p.add_subparsers(dest="command", required=True)

    # filing-text
    ft = sub.add_parser("filing-text",
                        help="download full_text for cached filings")
    g = ft.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+")
    g.add_argument("--all", action="store_true")
    ft.add_argument("--max-chars", type=int, default=500_000)
    ft.set_defaults(func=cmd_filing_text)

    # fundamentals
    fu = sub.add_parser("fundamentals",
                        help="fetch fundamentals (incl. sector) for the universe")
    fu.add_argument("--tickers", nargs="+")
    fu.add_argument("--limit", type=int, default=0)
    fu.add_argument("--force", action="store_true")
    fu.set_defaults(func=cmd_fundamentals)

    # universe
    un = sub.add_parser("universe",
                        help="fetch + hydrate 10-K/10-Q text across the universe")
    un.add_argument("--tickers", nargs="+")
    un.add_argument("--limit", type=int, default=0)
    un.add_argument("--kinds", nargs="+", default=["10-K", "10-Q"])
    un.add_argument("--max-10k", type=int, default=1)
    un.add_argument("--max-10q", type=int, default=2)
    un.add_argument("--max-chars", type=int, default=500000)
    un.add_argument("--force", action="store_true")
    un.set_defaults(func=cmd_universe)

    # money-graph
    mgp = sub.add_parser("money-graph",
                         help="ingest ownership/insider/XBRL for graph edges")
    mgp.add_argument("tickers", nargs="*")
    mgp.set_defaults(func=cmd_money_graph)

    # 13f
    f13 = sub.add_parser("13f", help="ingest 13F institutional holdings")
    f13.add_argument("--count", type=int, default=4,
                     help="quarters of 13F history per institution (default 4)")
    f13.add_argument("--cik", nargs="*", default=None,
                     help="specific CIKs (default: all in filer_canonical)")
    f13.add_argument("--sleep", type=float, default=0.3)
    f13.set_defaults(func=cmd_13f)

    # ownership-filers
    of = sub.add_parser("ownership-filers",
                        help="resolve missing filer_name on ownership rows")
    of.add_argument("--limit", type=int, default=0, help="max accessions (0=all)")
    of.add_argument("--ticker", default=None, help="restrict to one ticker")
    of.add_argument("--sleep", type=float, default=0.13)
    of.set_defaults(func=cmd_ownership_filers)

    # news-edges
    ne = sub.add_parser("news-edges",
                        help="build co-mention edges from news_articles")
    ne.add_argument("--fetch", action="store_true",
                    help="fetch fresh news for the universe first")
    ne.add_argument("--limit", type=int, default=0,
                    help="cap tickers when --fetch (0 = all)")
    ne.set_defaults(func=cmd_news_edges)

    # usaspending-edges
    us = sub.add_parser("usaspending-edges",
                        help="populate federal contract dollars per company")
    us.add_argument("--days", type=int, default=365)
    us.add_argument("--tickers", nargs="*", default=None)
    us.add_argument("--sleep", type=float, default=0.3)
    us.set_defaults(func=cmd_usaspending_edges)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
