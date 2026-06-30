import urllib.request, json

# 1. Is Ollama even up? Check the version endpoint.
try:
    r = urllib.request.urlopen("http://localhost:11434/api/version", timeout=5)
    print("Ollama up:", json.loads(r.read()))
except Exception as e:
    print("Ollama /api/version FAILED:", e)

# 2. Try the embed endpoint (newer Ollama: /api/embed with "input")
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": "nomic-embed-text", "input": "test"}).encode(),
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    embs = d.get("embeddings") or []
    print("/api/embed OK, dim:", len(embs[0]) if embs else "no embeddings key")
except Exception as e:
    print("/api/embed FAILED:", e)

# 3. Try the legacy endpoint (/api/embeddings with "prompt")
try:
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=json.dumps({"model": "nomic-embed-text", "prompt": "test"}).encode(),
        headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=30)
    d = json.loads(r.read())
    print("/api/embeddings OK, dim:", len(d.get("embedding", [])))
except Exception as e:
    print("/api/embeddings FAILED:", e)
