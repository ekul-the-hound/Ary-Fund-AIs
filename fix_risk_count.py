"""
fix_risk_count.py — reconcile the peer-path risk count with the agent path.

Run from D:\\Ary Fund:   python fix_risk_count.py

The peer path (_risk_count_for in data/pipeline.py) counts risk-cue sentences
over UP TO 3 10-Ks + 3 10-Qs. The agent path counts over 1 10-K + 2 10-Qs.
That mismatch is why MSFT contributes ~202 to its sector distribution but is
scored as 119. This script changes ONLY the peer path's filing selection to
1x 10-K + 2x 10-Q (no 8-Ks), so a ticker's contributed count matches the
basis it is scored on.

Surgical: it replaces exactly one block inside _risk_count_for and leaves the
rest of pipeline.py byte-for-byte unchanged. It refuses to write if the
expected block isn't found (so it can't silently corrupt the file), and it
re-parses the result with ast before saving, restoring the original on any
syntax error. Writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\data\pipeline.py"

OLD = (
    '        filings = []\n'
    '        for kind in ("10-K", "10-Q"):\n'
    '            try:\n'
    '                cached = _sec._get_cached_filings(tk, kind, 3, None, None)\n'
    '            except Exception:  # noqa: BLE001\n'
    '                cached = []\n'
)

NEW = (
    '        filings = []\n'
    '        # Match the AGENT path\'s annual/quarterly basis (1x 10-K, 2x 10-Q)\n'
    '        # so a ticker\'s CONTRIBUTED count == the count it is SCORED as.\n'
    '        # 8-Ks are excluded: they rarely carry Item-1A risk language and\n'
    '        # their date-relative selection would make the distribution drift.\n'
    '        for kind, _n in (("10-K", 1), ("10-Q", 2)):\n'
    '            try:\n'
    '                cached = _sec._get_cached_filings(tk, kind, _n, None, None)\n'
    '            except Exception:  # noqa: BLE001\n'
    '                cached = []\n'
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:  # tolerate any stray BOM
        src = f.read()

    count = src.count(OLD)
    if count == 0:
        print("ERROR: expected block not found. The file may already be "
              "patched or differs from what was expected. No changes made.")
        # Help diagnose: show whether the new form is already present.
        if 'for kind, _n in (("10-K", 1), ("10-Q", 2)):' in src:
            print("NOTE: the file already contains the reconciled selection — "
                  "looks like it's already patched.")
        return 1
    if count > 1:
        print(f"ERROR: expected block found {count} times (expected exactly 1). "
              "Aborting to avoid an ambiguous edit.")
        return 1

    patched = src.replace(OLD, NEW)

    # Verify the patched source parses before writing.
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched source does not parse ({e}). No changes written.")
        return 1

    # Write UTF-8 WITHOUT BOM.
    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(patched)

    # Re-read and re-verify what actually landed on disk.
    with io.open(PATH, "r", encoding="utf-8") as f:
        check = f.read()
    try:
        ast.parse(check)
    except SyntaxError as e:
        print(f"ERROR: file on disk does not parse after write ({e}).")
        return 1

    print("PATCHED OK — _risk_count_for now uses 1x 10-K + 2x 10-Q (no 8-Ks).")
    print("File parses clean, written without BOM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
