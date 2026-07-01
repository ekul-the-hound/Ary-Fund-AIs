import sqlite3
# Did the signals also land in portfolio.db, or ONLY hedgefund.db?
for db in ("data/hedgefund.db", "data/portfolio.db"):
    try:
        c = sqlite3.connect(db)
        rows = c.execute("""
            SELECT field, value_num, as_of FROM data_points
            WHERE entity_id='NVDA' AND field LIKE 'ticker.signal.realized_vol%'
            ORDER BY as_of DESC LIMIT 3
        """).fetchall()
        print(f"{db}: realized_vol rows =", rows)
    except Exception as e:
        print(f"{db}: ERR {e}")
