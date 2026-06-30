import urllib.request, json
for ep in ("/api/version", "/api/tags"):
    try:
        r = urllib.request.urlopen("http://localhost:11434"+ep, timeout=10)
        print(ep, "->", json.loads(r.read()))
    except Exception as e:
        print(ep, "FAIL", e)
