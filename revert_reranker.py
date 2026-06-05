"""
revert_reranker.py — undo the non-working reranker offline patch.

Run from D:\\Ary Fund:   python revert_reranker.py

The in-property HF_HUB_OFFLINE patch did not suppress the HuggingFace startup
calls (the env var is read at huggingface library import time, which happens
before this property runs — likely via the embedder loading
sentence_transformers first). Rather than leave dead, misleading complexity in
reranker.py, this reverts that block to the original simple load. The real fix
(setting the env vars at the top of main.py, before any import) is applied
separately.

Surgical: replaces the patched block with the original. Refuses to write
unless the patched block matches exactly once; re-parses before and after;
writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\rag\reranker.py"

# The patched (non-working) block currently on disk.
PATCHED = (
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

ORIGINAL = (
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


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    if ORIGINAL in src and PATCHED not in src:
        print("Already at original — nothing to revert.")
        return 0

    count = src.count(PATCHED)
    if count == 0:
        print("ERROR: patched block not found; cannot revert automatically. "
              "No changes made.")
        return 1
    if count > 1:
        print(f"ERROR: patched block found {count} times (expected 1). Aborting.")
        return 1

    reverted = src.replace(PATCHED, ORIGINAL)

    try:
        ast.parse(reverted)
    except SyntaxError as e:
        print(f"ERROR: reverted source does not parse ({e}). No changes written.")
        return 1

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(reverted)

    with io.open(PATH, "r", encoding="utf-8") as f:
        check = f.read()
    try:
        ast.parse(check)
    except SyntaxError as e:
        print(f"ERROR: file on disk does not parse after write ({e}).")
        return 1

    print("REVERTED OK — reranker.py back to original simple load.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
