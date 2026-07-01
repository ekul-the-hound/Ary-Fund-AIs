import sys
sys.path.insert(0, ".")
from data.derived_signals import DerivedSignals

ds = DerivedSignals()
print("Recomputing derived signals for NVDA...")
result = ds.recompute_for("NVDA")
print("Result:", result)

# Verify they landed in data_points
import sqlite3
c = sqlite3.connect("data/hedgefund.db")
rows = c.execute("""
    SELECT field, value_num, as_of FROM data_points
    WHERE entity_id='NVDA' AND field LIKE 'ticker.signal.%'
    ORDER BY field
""").fetchall()
print("\nNVDA signals now in registry:")
for f, v, ao in rows:
    print(f"  {f} = {v}  (as_of {ao})")
