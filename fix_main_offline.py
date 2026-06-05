"""
fix_main_offline.py — set HuggingFace offline mode at the entry point.

Run from D:\\Ary Fund:   python fix_main_offline.py

The reranker (and embedder) re-validate against huggingface.co on every
startup. Setting HF_HUB_OFFLINE inside the reranker's lazy property was too
late: the huggingface libraries read the flag at import time, and the embedder
imports sentence_transformers before the reranker property runs. The robust
fix is to set the env vars at the very top of main.py, before ANY import can
pull in a huggingface library.

This inserts an os.environ block immediately after
`from __future__ import annotations` (which must remain the first statement).
A network fallback isn't needed here: if the model isn't cached, the
individual loaders still raise clearly, and you'd unset these for a one-time
download. The model is cached on this machine.

Surgical: inserts one block at a single stable anchor. Refuses to write if the
anchor isn't found exactly once, or if the block is already present. Re-parses
before and after; writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\main.py"

ANCHOR = "from __future__ import annotations\n"

BLOCK = (
    "from __future__ import annotations\n"
    "\n"
    "# Load cached HuggingFace models (reranker cross-encoder, embedder)\n"
    "# without re-validating against huggingface.co on every startup. Must be\n"
    "# set before any import pulls in a huggingface library, so it lives here\n"
    "# at the top of the entry point rather than in a lazy loader.\n"
    "import os as _os\n"
    "_os.environ.setdefault(\"HF_HUB_OFFLINE\", \"1\")\n"
    "_os.environ.setdefault(\"TRANSFORMERS_OFFLINE\", \"1\")\n"
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    if 'setdefault("HF_HUB_OFFLINE"' in src:
        print("Already patched — HF_HUB_OFFLINE block present in main.py.")
        return 0

    count = src.count(ANCHOR)
    if count == 0:
        print("ERROR: anchor 'from __future__ import annotations' not found. "
              "No changes made. Paste the top of main.py and I'll adjust.")
        return 1
    if count > 1:
        print(f"ERROR: anchor found {count} times (expected 1). Aborting.")
        return 1

    patched = src.replace(ANCHOR, BLOCK, 1)

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

    print("PATCHED OK — HF offline env vars set at top of main.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
