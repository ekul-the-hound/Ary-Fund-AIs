import urllib.request, json
req = urllib.request.Request(
    "http://localhost:11434/api/embed",
    data=json.dumps({"model": "nomic-embed-text", "input": "test"}).encode(),
    headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=60)
    d = json.loads(r.read())
    embs = d.get("embeddings") or []
    print("EMBED OK — dim:", len(embs[0]) if embs else "no embeddings key")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, "BODY:", e.read().decode()[:300])
except Exception as e:
    print("ERR:", e)
