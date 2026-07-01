import sqlite3
for db in ("data/hedgefund.db", "data/portfolio.db"):
    c = sqlite3.connect(db)
    n = c.execute("SELECT COUNT(*) FROM data_points WHERE entity_id='NVDA'").fetchone()[0]
    fields = c.execute("SELECT COUNT(DISTINCT field) FROM data_points WHERE entity_id='NVDA'").fetchone()[0]
    signals = c.execute("SELECT COUNT(*) FROM data_points WHERE entity_id='NVDA' AND field LIKE 'ticker.signal.%'").fetchone()[0]
    print(f"{db}: {n} NVDA rows, {fields} distinct fields, {signals} signal rows")
