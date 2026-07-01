# Check the modification times of the patched files vs a marker.
# If the running Streamlit started BEFORE these were patched, it has stale code.
import os, datetime
for f in ("data/pipeline.py", "agent/risk_scanner.py", "main.py"):
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    print(f"{f}: last modified {mt}")

# Also: is the all-clear code actually IN the risk_scanner on disk?
src = open("agent/risk_scanner.py").read()
print()
print("risk_scanner has _reasons_for (all-clear patch):", "_reasons_for" in src)
print("pipeline has _reg_db resolver:", "_reg_db = None" in open("data/pipeline.py").read())
