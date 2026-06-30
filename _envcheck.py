import os
# Did .env get loaded? Check raw, then with dotenv
print("Before load_dotenv, ARY_EMBED_BACKEND =", repr(os.environ.get("ARY_EMBED_BACKEND")))
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("After load_dotenv,  ARY_EMBED_BACKEND =", repr(os.environ.get("ARY_EMBED_BACKEND")))
except Exception as e:
    print("dotenv err:", e)
# Is Ollama serving nomic?
import urllib.request, json
try:
    req = urllib.request.Request("http://localhost:11434/api/embeddings",
        data=json.dumps({"model":"nomic-embed-text","prompt":"test"}).encode(),
        headers={"Content-Type":"application/json"})
    r = urllib.request.urlopen(req, timeout=10)
    d = json.loads(r.read())
    print("Ollama nomic embedding dim:", len(d.get("embedding", [])))
except Exception as e:
    print("Ollama probe failed:", e)
