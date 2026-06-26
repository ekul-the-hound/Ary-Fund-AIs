import sqlite3, glob
for f in glob.glob("*.db") + glob.glob("data/*.db"):
    try:
        tables = [r[0] for r in sqlite3.connect(f).execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f, "->", tables)
    except Exception as e:
        print(f, "ERR", e)
