"""
age_fundamentals.py — force the fundamentals_cache stale condition for testing.

Backdates every fundamentals_cache row's fetched_at to 2020, so the OLD
cache-first code would treat them all as stale and fall through to yfinance.
With the fixed _sector_for (direct table read, no TTL), the recompute should
stay fully offline regardless. This is the test harness for that fix.

    python age_fundamentals.py

Harmless: a normal `python main.py ...` run afterward will re-fetch and
repopulate fundamentals naturally. Sector values themselves are untouched.
"""
import sqlite3
import config

conn = sqlite3.connect(config.PORTFOLIO_DB_PATH)
conn.execute("UPDATE fundamentals_cache SET fetched_at = '2020-01-01 00:00:00'")
conn.commit()
print(f"aged {conn.total_changes} fundamentals_cache rows to 2020 (forced stale)")
conn.close()
