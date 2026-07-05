"""
backfill_ownership_filers.py
============================
Fill in the missing `filer_name` on ownership_filings rows.

Why: sec_fetcher records 13D/13G filings from the SUBJECT company's submissions
feed, where the institutional filer name isn't present, so `filer_name` is
blank on ~all rows and the Money Flow Graph has no ownership edges. The real
filer ("FILED BY") lives in each filing's SEC header. This script fetches just
the header (Range request, ~32KB), parses the filer's conformed name, and
writes it back.

Rate-limited, resumable (skips accessions already resolved), safe to re-run.

Run from project root (venv active):
    python backfill_ownership_filers.py               # all empty-filer rows
    python backfill_ownership_filers.py --limit 50    # small test first
    python backfill_ownership_filers.py --ticker NVDA # one company
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

# --- DB path (same resolution as the app) --------------------------------
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

# --- reuse SECFetcher's session (correct User-Agent + rate limiting) ------
try:
    from data.sec_fetcher import SECFetcher
except Exception:
    from sec_fetcher import SECFetcher  # type: ignore

_DONE = Path("data") / "ownership_filers_done.json"
_HDR_RE = re.compile(
    r'FILED BY[:\s].*?COMPANY CONFORMED NAME[:\s]*([^\r\n]+)', re.I | re.S)
_ANY_RE = re.compile(r'COMPANY CONFORMED NAME[:\s]*([^\r\n]+)', re.I)

# filer_name values that are really form/document labels, not institutions.
# These leaked in from primaryDocDescription and must be re-resolved too.
_JUNK_FILER = re.compile(
    r"^(none|n/?a|null|form\s+sc\s+13[dg](/a)?|sc\s+13[dg](/a)?|"
    r"sec\s+schedule\s+13[dg](/a)?|schedule\s+13[dg](/a)?|13[dg](/a)?|"
    r"ms\s+initial|initial|amendment|amended)$", re.I)


def _needs_resolution(name: str) -> bool:
    s = (name or "").strip()
    return (not s) or (_JUNK_FILER.match(s) is not None)


def _load_done() -> set[str]:
    try:
        return set(json.loads(_DONE.read_text()))
    except Exception:
        return set()


def _save_done(done: set[str]) -> None:
    _DONE.parent.mkdir(parents=True, exist_ok=True)
    _DONE.write_text(json.dumps(sorted(done)))


def _header_url(cik, accession) -> str:
    acc_nodash = str(accession).replace("-", "")
    return (f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{acc_nodash}/{accession}.txt")


def _get_header(fetcher, url) -> str:
    """Fetch just the top of the submission (header). Prefer a Range request
    through the fetcher's session so we reuse its UA + don't download exhibits."""
    sess = getattr(fetcher, "session", None) or getattr(fetcher, "_session", None)
    if sess is not None:
        try:
            r = sess.get(url, headers={"Range": "bytes=0-32767"}, timeout=30)
            if r.status_code in (200, 206) and r.text:
                return r.text
        except Exception:
            pass
    # fallback: fetcher's own rate-limited GET, sliced
    try:
        return fetcher._get(url).text[:32768]
    except Exception:
        return ""


def _parse_filer(header: str) -> str | None:
    m = _HDR_RE.search(header)
    if m:
        return m.group(1).strip() or None
    names = _ANY_RE.findall(header)          # [0]=subject, [1]=filer
    if len(names) >= 2 and names[1].strip():
        return names[1].strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max accessions (0=all)")
    ap.add_argument("--ticker", default=None, help="restrict to one ticker")
    ap.add_argument("--sleep", type=float, default=0.13, help="seconds between requests")
    args = ap.parse_args()

    # distinct accessions whose filer_name is empty OR a form/junk label
    where = "1=1"
    params: tuple = ()
    if args.ticker:
        where = "ticker=?"
        params = (args.ticker.upper(),)
    sql = (f"SELECT DISTINCT accession_number, cik, filer_name "
           f"FROM ownership_filings WHERE {where} ORDER BY accession_number DESC")

    with sqlite3.connect(MARKET_DB) as conn:
        all_rows = conn.execute(sql, params).fetchall()
    rows = [(a, c) for (a, c, name) in all_rows if _needs_resolution(name)]

    done = _load_done()
    todo = [(a, c) for (a, c) in rows if a not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"DB: {MARKET_DB}")
    print(f"Accessions to resolve (empty or form-label filer): {len(rows)} | "
          f"to process now: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    fetcher = SECFetcher()
    # one persistent writable connection; expose the junk test to SQL so we
    # overwrite empty AND form-label rows (but never a real institution name).
    wconn = sqlite3.connect(MARKET_DB)
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
                _save_done(done)
                rate = i / max(1e-6, time.time() - t0)
                print(f"  [{i}/{len(todo)}] resolved={resolved} failed={failed} "
                      f"~{rate:.1f}/s ETA {(len(todo)-i)/max(rate,1e-6)/60:.1f}m")
            time.sleep(args.sleep)
    finally:
        wconn.close()

    _save_done(done)
    print(f"\nDone. resolved={resolved} failed={failed} "
          f"in {(time.time()-t0)/60:.1f} min.")
    print("Now rebuild the graph:  Scope = Full universe -> Rebuild.")


if __name__ == "__main__":
    main()