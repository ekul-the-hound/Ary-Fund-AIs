import sqlite3, json
c = sqlite3.connect("data/portfolio.db")
row = c.execute("SELECT payload_json FROM agent_opinions WHERE ticker='NVDA' ORDER BY id DESC LIMIT 1").fetchone()
d = json.loads(row[0])
# Does the opinion carry any macro context echo?
print("payload keys:", sorted(d.keys()))
# Some opinions store the macro dict or a macro echo
for k in ("macro","macro_context","context_macro"):
    if k in d:
        print(f"{k}:", d[k])
