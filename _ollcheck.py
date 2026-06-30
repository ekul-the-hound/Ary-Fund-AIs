import urllib.request, json
for path in ("/api/embed", "/api/embeddings"):
    try:
        body = {"model":"nomic-embed-text","input":"test"} if "embed" == path.split("/")[-1] else {"model":"nomic-embed-text","prompt":"test"}
        req = urllib.request.Request("http://localhost:11434"+path,
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        r = urllib.request.urlopen(req, timeout=10)
        d = json.loads(r.read())
        emb = d.get("embedding") or (d.get("embeddings") or [[]])[0]
        print(path, "-> OK, dim", len(emb))
    except Exception as e:
        print(path, "-> FAIL", e)
