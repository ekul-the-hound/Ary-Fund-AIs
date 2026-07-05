"""
diag_money_graph.py — figure out why the Money Flow Graph is nearly empty.

Run from project root (venv active):
    python diag_money_graph.py
"""
from __future__ import annotations
import sqlite3

# resolve the market DB path the same way the app does
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
PORTFOLIO_DB = getattr(cfg, "PORTFOLIO_DB_PATH", None) or "data/portfolio.db"
print(f"market DB   = {MARKET_DB}")
print(f"portfolio DB= {PORTFOLIO_DB}\n")


def q(db, sql, params=()):
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            return c.execute(sql, params).fetchall()
    except Exception as e:  # noqa: BLE001
        print(f"  query error: {e}")
        return []


# 1) does the universe load?
try:
    from data.money_graph import universe_tickers
except Exception:
    from money_graph import universe_tickers  # type: ignore
uni = universe_tickers()
print(f"[1] universe_tickers() -> {len(uni)} names "
      f"(expect ~560).  first few: {uni[:8]}\n")

# 2) ownership_filings health — THE key check
total = q(MARKET_DB, "SELECT COUNT(*) FROM ownership_filings")
total = total[0][0] if total else 0
nonempty = q(MARKET_DB,
             "SELECT COUNT(*) FROM ownership_filings "
             "WHERE filer_name IS NOT NULL AND TRIM(filer_name) <> ''")
nonempty = nonempty[0][0] if nonempty else 0
distinct = q(MARKET_DB,
             "SELECT COUNT(DISTINCT filer_name) FROM ownership_filings "
             "WHERE filer_name IS NOT NULL AND TRIM(filer_name) <> ''")
distinct = distinct[0][0] if distinct else 0
tickers = q(MARKET_DB, "SELECT COUNT(DISTINCT ticker) FROM ownership_filings")
tickers = tickers[0][0] if tickers else 0
print(f"[2] ownership_filings:")
print(f"      total rows            = {total}")
print(f"      rows WITH filer_name  = {nonempty}   <-- edges can only come from these")
print(f"      distinct filer_names  = {distinct}")
print(f"      distinct tickers      = {tickers}")
if total:
    print(f"      -> {100*nonempty/total:.1f}% of ownership rows are usable")
print("      top filer_name values (by count):")
for name, n in q(MARKET_DB,
                 "SELECT filer_name, COUNT(*) c FROM ownership_filings "
                 "GROUP BY filer_name ORDER BY c DESC LIMIT 12"):
    shown = repr(name)[:50]
    print(f"        {n:>5}  {shown}")

# 3) 13F holdings (the DENSE source — likely empty)
f13f = q(MARKET_DB, "SELECT COUNT(*) FROM f13f_holdings")
f13f = f13f[0][0] if f13f else 0
print(f"\n[3] f13f_holdings rows = {f13f}   "
      f"(0 = you haven't ingested 13F yet; this is the real density source)")

# 4) what the builder actually produces at full-universe scope
try:
    from data.money_graph import build_money_graph
except Exception:
    from money_graph import build_money_graph  # type: ignore
g = build_money_graph(tickers=uni or None, portfolio_db_path=PORTFOLIO_DB,
                      market_db_path=MARKET_DB, market_data=None,
                      drop_isolated=True, max_nodes=600)
m = g["meta"]
print(f"\n[4] build_money_graph(full universe, offline):")
print(f"      source        = {m.get('source')}")
print(f"      scope_size    = {m.get('scope_size')}")
print(f"      rendered nodes= {m.get('rendered_nodes', len(g['nodes']))}")
print(f"      rendered edges= {m.get('rendered_edges', len(g['edges']))}")
print(f"      provenance    = {m.get('provenance')}")

print("\nDIAGNOSIS:")
if nonempty < total * 0.2:
    print("  -> Most ownership rows have EMPTY filer_name (filer lives in the")
    print("     filing header, not the subject feed). Run backfill_ownership_filers.py.")
if f13f == 0:
    print("  -> No 13F data. For a dense who-owns-whom graph, ingest 13F for the")
    print("     big institutions (Vanguard/BlackRock/State Street/Berkshire).")
