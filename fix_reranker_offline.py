"""
fix_reranker_offline.py — skip the reranker's ~15s HuggingFace startup check.

Run from D:\\Ary Fund:   python fix_reranker_offline.py

The reranker re-validates the cross-encoder against huggingface.co on every
startup (a string of HTTP HEAD/GET requests, ~15s). The model is already
cached locally, so this network round-trip is pure startup latency.

FIX: set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE before constructing the
CrossEncoder so it loads from the local cache without hitting the network.
Includes a fallback: if the offline load fails (e.g. a fresh machine without
the cached model), it retries once with offline mode disabled so the model
can still download. So startup is fast on a warm cache and still works on a
cold one.

Surgical: replaces only the model-load block in the `model` property.
Refuses to write unless the block matches exactly once, re-parses with ast
before and after, writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\rag\reranker.py"

OLD = (
    '        if self._model is None:\n'
    '            try:\n'
    '                from sentence_transformers import CrossEncoder\n'
    '                self._model = CrossEncoder(self.model_name)\n'
    '                logger.info("Reranker loaded | %s", self.model_name)\n'
    '            except ImportError as e:\n'
    '                raise RuntimeError(\n'
    '                    "Reranker requires sentence-transformers. "\n'
    '                    "Install with: pip install sentence-transformers"\n'
    '                ) from e\n'
)

NEW = (
    '        if self._model is None:\n'
    '            try:\n'
    '                import os\n'
    '                from sentence_transformers import CrossEncoder\n'
    '                # Load from local cache without re-validating against the\n'
    '                # HuggingFace Hub on every startup (~15s of HTTP round-\n'
    '                # trips). The model is already cached locally.\n'
    '                _prev_hub = os.environ.get("HF_HUB_OFFLINE")\n'
    '                _prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")\n'
    '                os.environ["HF_HUB_OFFLINE"] = "1"\n'
    '                os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
    '                try:\n'
    '                    self._model = CrossEncoder(self.model_name)\n'
    '                except Exception:  # noqa: BLE001\n'
    '                    # Cold cache (e.g. fresh machine): allow a networked\n'
    '                    # download by restoring the prior offline setting and\n'
    '                    # retrying once.\n'
    '                    if _prev_hub is None:\n'
    '                        os.environ.pop("HF_HUB_OFFLINE", None)\n'
    '                    else:\n'
    '                        os.environ["HF_HUB_OFFLINE"] = _prev_hub\n'
    '                    if _prev_tf is None:\n'
    '                        os.environ.pop("TRANSFORMERS_OFFLINE", None)\n'
    '                    else:\n'
    '                        os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf\n'
    '                    self._model = CrossEncoder(self.model_name)\n'
    '                logger.info("Reranker loaded | %s", self.model_name)\n'
    '            except ImportError as e:\n'
    '                raise RuntimeError(\n'
    '                    "Reranker requires sentence-transformers. "\n'
    '                    "Install with: pip install sentence-transformers"\n'
    '                ) from e\n'
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    count = src.count(OLD)
    if count == 0:
        print("ERROR: expected reranker load block not found. No changes made.")
        if 'os.environ["HF_HUB_OFFLINE"] = "1"' in src:
            print("NOTE: looks like it's already patched.")
        return 1
    if count > 1:
        print(f"ERROR: block found {count} times (expected 1). Aborting.")
        return 1

    patched = src.replace(OLD, NEW)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched source does not parse ({e}). No changes written.")
        return 1

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(patched)

    with io.open(PATH, "r", encoding="utf-8") as f:
        check = f.read()
    try:
        ast.parse(check)
    except SyntaxError as e:
        print(f"ERROR: file on disk does not parse after write ({e}).")
        return 1

    print("PATCHED OK — reranker now loads offline (warm cache) with a "
          "network fallback for a cold cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
