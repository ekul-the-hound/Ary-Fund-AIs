import sqlite3
# derived signals live in the registry (data_points table in hedgefund.db)
c = sqlite3.connect("data/hedgefund.db")
# what signal rows exist for NVDA?
try:
    rows = c.execute("""
        SELECT key, value FROM data_points
        WHERE key LIKE 'ticker.signal.%' AND entity='NVDA'
        ORDER BY key LIMIT 30
    """).fetchall()
    print("NVDA signal rows in data_points:", len(rows))
    for k, v in rows:
        print("  ", k, "=", v)
except Exception as e:
    print("data_points query failed:", e)
    # try alternate schema
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(data_points)").fetchall()]
        print("data_points columns:", cols)
    except Exception as e2:
        print("no data_points table:", e2)
