import sqlite3
c = sqlite3.connect("data/hedgefund.db")

# 1. What signal fields exist for NVDA?
rows = c.execute("""
    SELECT field, value_num, value_text, as_of
    FROM data_points
    WHERE entity_id='NVDA' AND field LIKE '%vol%'
       OR entity_id='NVDA' AND field LIKE '%drawdown%'
       OR entity_id='NVDA' AND field LIKE '%rsi%'
    ORDER BY field LIMIT 40
""").fetchall()
print("NVDA vol/drawdown/rsi rows:", len(rows))
for f, vn, vt, ao in rows:
    print(f"  {f} = {vn if vn is not None else vt}   (as_of {ao})")

print()
# 2. Broader: ANY signal-like fields for NVDA at all?
rows2 = c.execute("""
    SELECT DISTINCT field FROM data_points
    WHERE entity_id='NVDA' AND field LIKE 'signal%'
    ORDER BY field LIMIT 40
""").fetchall()
print("NVDA fields starting 'signal':", [r[0] for r in rows2])

print()
# 3. What DO we have for NVDA? (sample of all field names)
rows3 = c.execute("""
    SELECT DISTINCT field FROM data_points WHERE entity_id='NVDA'
    ORDER BY field LIMIT 60
""").fetchall()
print("All NVDA field names (up to 60):")
for r in rows3:
    print("  ", r[0])
