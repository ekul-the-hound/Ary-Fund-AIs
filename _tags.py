import urllib.request, json
r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10)
d = json.loads(r.read())
print("Models the 11434 server sees:")
for m in d.get("models", []):
    print("  ", m.get("name"))
