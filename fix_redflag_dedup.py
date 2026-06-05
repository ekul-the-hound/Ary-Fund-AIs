"""
fix_redflag_dedup.py — stop one event being counted as multiple red flags.

Run from D:\\Ary Fund:   python fix_redflag_dedup.py

BUG (reproduced via diag_distressed.py): _find_red_flags dedups on the regex
PATTERN, so the same pattern can't fire twice — but two DIFFERENT patterns can
both match the same sentence (e.g. a "material weakness" pattern and a
"controls not effective" pattern hitting one material-weakness sentence),
producing two flags for one event. That inflates the red-flag count (and the
-0.3-per-flag penalty) on distressed filings.

FIX: positional dedup. Track the text spans already flagged and skip a match
whose span overlaps one already recorded. Two patterns on the same sentence
count once; genuinely separate events elsewhere in the text still count
separately. Detection coverage is unchanged — only double-counting of a single
location is removed.

Surgical: replaces the scan loop in _find_red_flags. Refuses to write unless
the block matches exactly once; re-parses before and after; writes UTF-8
without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\agent\filing_analyzer.py"

OLD = (
    '    hits: List[str] = []\n'
    '    seen: set = set()\n'
    '    for pat in (*_AFFIRMATIVE_SEVERE_PATTERNS, *_RED_FLAG_PATTERNS):\n'
    '        m = pat.search(text)\n'
    '        if not m:\n'
    '            continue\n'
    '        key = pat.pattern\n'
    '        if key in seen:\n'
    '            continue\n'
    '        seen.add(key)\n'
    '        # ~80-char context window for human review.\n'
    '        start = max(0, m.start() - 40)\n'
    '        end = min(len(text), m.end() + 40)\n'
    '        snippet = re.sub(r"\\s+", " ", text[start:end]).strip()\n'
    '        hits.append(snippet)\n'
    '    return hits\n'
)

NEW = (
    '    hits: List[str] = []\n'
    '    seen: set = set()\n'
    '    flagged_spans: List[tuple] = []  # (start, end) of matched cores\n'
    '    for pat in (*_AFFIRMATIVE_SEVERE_PATTERNS, *_RED_FLAG_PATTERNS):\n'
    '        m = pat.search(text)\n'
    '        if not m:\n'
    '            continue\n'
    '        key = pat.pattern\n'
    '        if key in seen:\n'
    '            continue\n'
    '        # Positional dedup: if this match overlaps a span already\n'
    '        # flagged, it is the SAME underlying event picked up by a second\n'
    '        # pattern (e.g. "material weakness" and "controls not effective"\n'
    '        # in one sentence). Count the event once.\n'
    '        ms, me = m.start(), m.end()\n'
    '        if any(ms < fe and me > fs for fs, fe in flagged_spans):\n'
    '            continue\n'
    '        seen.add(key)\n'
    '        flagged_spans.append((ms, me))\n'
    '        # ~80-char context window for human review.\n'
    '        start = max(0, ms - 40)\n'
    '        end = min(len(text), me + 40)\n'
    '        snippet = re.sub(r"\\s+", " ", text[start:end]).strip()\n'
    '        hits.append(snippet)\n'
    '    return hits\n'
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    if "flagged_spans: List[tuple]" in src:
        print("Already patched — positional dedup present.")
        return 0

    count = src.count(OLD)
    if count == 0:
        print("ERROR: expected _find_red_flags loop not found. No changes made.")
        print("Paste the function and I'll match your on-disk version.")
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

    print("PATCHED OK — _find_red_flags now dedups overlapping matches "
          "(one event = one flag).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
