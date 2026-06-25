"""
fix_screener_data_xbrl.py
========================

Add an ``xbrl`` subcommand to data/screener_data.py that ingests SEC XBRL
companyfacts for the whole universe into the ``xbrl_facts`` table, so the
screener can later read real Operating Income / Total Assets / CapEx from
filings.

WHAT IT ADDS
------------
* ``_import_sec_fetcher()`` — tolerant import (data.sec_fetcher | sec_fetcher).
* ``cmd_xbrl(symbols, *, sleep, limit)`` — loops the universe and calls the
  existing ``SECFetcher.ingest_xbrl_facts(ticker)`` for each, with progress and
  SEC-friendly throttling. (ingest_xbrl_facts already fetches companyfacts,
  writes xbrl_facts + the registry, and derives ratios — we just drive it over
  the universe.)
* Wires ``xbrl`` into the argparse ``choices`` and the dispatch in ``main``.

USAGE AFTER PATCH
-----------------
    python data/screener_data.py xbrl

HONEST COST NOTE
----------------
This hits SEC ~once per symbol, and each companyfacts JSON is several MB, so a
full-universe run takes a while (≈10-20 min) and downloads a few GB. XBRL only
changes on new filings, so this is a once-per-quarter job, not daily. SEC asks
for a descriptive User-Agent and ≤10 req/s — SECFetcher already throttles; the
default --sleep here adds a little extra headroom. Be polite to SEC's servers.

SAFETY
------
* Targets data/screener_data.py.
* Backs up to data/screener_data.py.bak before writing.
* Idempotent: detects cmd_xbrl and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_screener_data_xbrl.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("data") / "screener_data.py"

# --- 1. Add _import_sec_fetcher after _import_market_data ----------------
IMPORT_ANCHOR = '''def _import_market_data():
    for path in ("data.market_data", "market_data"):
        try:
            mod = __import__(path, fromlist=["MarketData"])
            return mod.MarketData
        except Exception:
            continue
    print("ERROR: could not import MarketData (tried data.market_data and "
          "market_data).")
    sys.exit(1)'''

IMPORT_INSERT = IMPORT_ANCHOR + '''


def _import_sec_fetcher():
    for path in ("data.sec_fetcher", "sec_fetcher"):
        try:
            mod = __import__(path, fromlist=["SECFetcher"])
            return mod.SECFetcher
        except Exception:
            continue
    print("ERROR: could not import SECFetcher (tried data.sec_fetcher and "
          "sec_fetcher).")
    sys.exit(1)


def cmd_xbrl(symbols: list[str], *, sleep: float,
             limit: "Optional[int]" = None) -> None:
    """Ingest SEC XBRL companyfacts for each symbol into the xbrl_facts table.

    Drives the existing SECFetcher.ingest_xbrl_facts over the universe. Each
    call fetches the company's companyfacts JSON (several MB), writes every
    mapped concept (revenue, net income, total assets, capex, operating
    income, ...) to xbrl_facts + the registry, and derives ratios.
    """
    SECFetcher = _import_sec_fetcher()
    sec = SECFetcher()

    if limit is not None:
        symbols = symbols[:limit]
    total = len(symbols)

    ingested = 0
    no_facts: list[str] = []
    failed: list[str] = []
    t0 = time.time()
    print(f"[xbrl] Ingesting SEC XBRL companyfacts for {total} symbols into "
          f"the xbrl_facts table…")
    print("[xbrl] Each companyfacts JSON is several MB; this is a once-per-"
          "quarter job. Be patient and kind to SEC's servers.")
    print()

    for i, sym in enumerate(symbols, 1):
        try:
            n = sec.ingest_xbrl_facts(sym)
            if n > 0:
                ingested += 1
            else:
                no_facts.append(sym)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{sym} ({type(e).__name__})")

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / rate if rate > 0 else 0.0
            print(f"[xbrl] {i}/{total}  ingested={ingested}  "
                  f"no_facts={len(no_facts)}  failed={len(failed)}  "
                  f"~{eta:5.0f}s left  (last: {sym})")

        if sleep:
            time.sleep(sleep)

    dt = time.time() - t0
    print()
    print(f"[xbrl] Done. Ingested facts for {ingested}/{total} in {dt:.0f}s.")
    if no_facts:
        print(f"[xbrl] {len(no_facts)} symbol(s) had no us-gaap facts "
              f"(e.g. foreign filers, ETFs): {', '.join(no_facts[:20])}"
              + (" ..." if len(no_facts) > 20 else ""))
    if failed:
        print(f"[xbrl] {len(failed)} symbol(s) errored: {', '.join(failed[:20])}"
              + (" ..." if len(failed) > 20 else ""))
    print()
    print("Now restart the screener — Op Income / Total Assets / CapEx should")
    print("fill from XBRL for names that report those concepts. (Some "
          "financials don't file a GAAP operating-income line, so those stay "
          "blank — that's a filing reality, not a bug.)")'''

# --- 2. Add 'xbrl' to the argparse choices -------------------------------
CHOICES_ANCHOR = '''    parser.add_argument("command", choices=["check", "warm", "all"],
                        help="check = report delisted; warm = fill cache; "
                             "all = check then warm the survivors")'''

CHOICES_INSERT = '''    parser.add_argument("command", choices=["check", "warm", "all", "xbrl"],
                        help="check = report delisted; warm = fill cache; "
                             "all = check then warm the survivors; "
                             "xbrl = ingest SEC XBRL facts (Op Income/Total "
                             "Assets/CapEx) for the universe")'''

# --- 3. Add the dispatch branch ------------------------------------------
DISPATCH_ANCHOR = '''    elif args.command == "all":
        good = cmd_check(symbols, sleep=args.sleep)
        print(f"[all] Warming the {len(good)} symbols that passed check…\\n")
        cmd_warm(good, sleep=args.sleep, limit=args.limit)'''

DISPATCH_INSERT = '''    elif args.command == "all":
        good = cmd_check(symbols, sleep=args.sleep)
        print(f"[all] Warming the {len(good)} symbols that passed check…\\n")
        cmd_warm(good, sleep=args.sleep, limit=args.limit)
    elif args.command == "xbrl":
        cmd_xbrl(symbols, sleep=args.sleep, limit=args.limit)'''


def _fail(msg: str) -> None:
    print(f"[fix_screener_data_xbrl] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "def cmd_xbrl(" in src:
        print("[fix_screener_data_xbrl] Already applied — cmd_xbrl present. "
              "Nothing to do.")
        return

    for anchor, name in (
        (IMPORT_ANCHOR, "_import_market_data (for cmd_xbrl insert)"),
        (CHOICES_ANCHOR, "argparse choices line"),
        (DISPATCH_ANCHOR, "dispatch 'all' branch"),
    ):
        if anchor not in src:
            _fail(f"could not find the {name} anchor. The file may have "
                  "changed. Not editing blindly.")

    src = src.replace(IMPORT_ANCHOR, IMPORT_INSERT, 1)
    src = src.replace(CHOICES_ANCHOR, CHOICES_INSERT, 1)
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_data_xbrl] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added _import_sec_fetcher() + cmd_xbrl()")
    print("  • Wired 'xbrl' into the command choices and dispatch")
    print()
    print("Run it (full universe — takes a while, downloads a few GB):")
    print("    python data/screener_data.py xbrl")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_data_xbrl.py
