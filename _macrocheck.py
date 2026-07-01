import sqlite3
c = sqlite3.connect("data/hedgefund.db")
# macro lives under entity 'US' or 'MACRO' or in a macro_data table
print("=== macro_data table rows ===")
try:
    rows = c.execute("SELECT series_id, value, date FROM macro_data ORDER BY date DESC LIMIT 15").fetchall()
    for r in rows: print("  ", r)
except Exception as e:
    print("  macro_data query failed:", e)
    cols = [r[1] for r in c.execute("PRAGMA table_info(macro_data)").fetchall()]
    print("  macro_data columns:", cols)

print()
print("=== macro-ish entities in data_points ===")
rows2 = c.execute("""
    SELECT DISTINCT entity_id, field FROM data_points
    WHERE field LIKE '%macro%' OR field LIKE '%vix%' OR field LIKE '%recession%'
       OR field LIKE '%yield%' OR entity_id IN ('US','MACRO','ECONOMY')
    LIMIT 20
""").fetchall()
for r in rows2: print("  ", r)
