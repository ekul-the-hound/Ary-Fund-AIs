"""
diag_universe_coverage.py — how much of US_UNIVERSE already has filing data.

Stage 2 (peer-relative risk-count) needs a risk count for enough names per
sector to form a distribution. That requires hydrated filing text. This
reports how many universe tickers already have filings cached (cheap) vs
need a fresh EDGAR fetch (the slow part), so we know the true ingest cost
before committing to the backfill.

    python diag_universe_coverage.py
"""
import sqlite3
import config
try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE

c = sqlite3.connect(config.PORTFOLIO_DB_PATH)
uni = set(US_UNIVERSE)

# Tables vary; guard each query.
def _distinct(where=""):
    try:
        q = "SELECT DISTINCT ticker FROM sec_filings"
        if where:
            q += " WHERE " + where
        return set(r[0] for r in c.execute(q).fetchall())
    except sqlite3.OperationalError as e:
        print("  (query failed:", e, ")")
        return set()

have = _distinct()
withtext = _distinct("full_text IS NOT NULL AND full_text != ''")
have_uni = have & uni

print("universe size              :", len(uni))
print("universe w/ ANY filing row :", len(have_uni))
print("universe w/ full_text      :", len(have_uni & withtext))
print("universe needing FETCH     :", len(uni - have))
print()
# Rough fetch-time estimate: SEC EDGAR fair-access ~10 req/s, but be polite
# (~3-5 req/s typical). Each new ticker ~ 1 list call + ~3-5 text calls.
need = len(uni - have)
calls = need * 5
print(f"~{need} tickers to fetch, ~{calls} EDGAR calls")
print(f"at ~5 req/s polite rate: ~{calls/5/60:.0f} min of fetching (rough)")
