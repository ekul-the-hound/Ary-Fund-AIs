import sqlite3, json
c = sqlite3.connect("data/portfolio.db")
row = c.execute("SELECT payload_json FROM agent_opinions WHERE ticker='NVDA' ORDER BY id DESC LIMIT 1").fetchone()
d = json.loads(row[0])
print("TOP-LEVEL KEYS:", sorted(d.keys()))
for k in ("outlook","price_direction","direction","confidence","rationale","summary","key_risks","risk_flags"):
    v = d.get(k)
    print(k, "=>", (str(v)[:90]) if v else v)
