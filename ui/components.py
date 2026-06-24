"""
ui/components.py
================

Shared rendering primitives for the ARY QUANT v2 dashboard.

This module is the single source of visual vocabulary for every other UI
file (desk, board, lab, pipeline_rail, palette, app_v2). It encodes the
four orthogonal signal channels described in the design spec so they never
collide visually:

    - Conviction   (agent confidence 0..1)      -> pip meter
    - Bias         (bias_score -1..+1)           -> diverging bar
    - Risk         (LOW/MEDIUM/HIGH per axis)    -> three-axis triplet
    - Provenance   (source / as_of / confidence) -> freshness chip

Plus a handful of structural helpers (section anchors, fallback border,
evidence cards, sector-relative z-score bars, review scorecards).

Design constraints honored here
-------------------------------
* PURE: nothing in this module touches Ollama, SQLite, the network, or any
  ``data.*`` / ``agent.*`` module. It only renders. That means importing it
  can never break a running dashboard, and it can be unit-tested headless.
* DEFENSIVE: every helper tolerates ``None`` / missing keys and renders a
  neutral placeholder ("—") rather than raising. The backend context
  contract uses explicit absence (None / {} / []), and this module mirrors
  that: absent data is shown as absent, never as a fabricated zero.
* SHAPE-FAITHFUL: the helpers consume the exact shapes the backend emits.
    - risk_flags:        {"levels": {fundamental, macro, market, combined},
                          "reasons": {fundamental: [...], macro: [...], ...}}
    - provenance[field]: {"source_id", "as_of", "confidence"}
    - freshness[section]: latest_as_of_iso (str)
    - review:            {"scores": {section: 1..10, "overall": float},
                          "text": str}
    - altman/distress:   {"z", "variant", "distress" (0..1), "zone"}
    - zscore tier:       LOW/MEDIUM/HIGH via the same ramp as levels.

All public helpers are prefixed by their kind (``badge_*``, ``meter_*``,
``chip_*``, ``card_*``, ``panel_*``, ``section_*``) so call sites read
clearly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st


# ======================================================================
# Color system
# ======================================================================
# One ramp for "risk-like" magnitude (used by risk levels AND z-score
# tiers, so a HIGH risk and a HIGH z-tier read identically). One ramp for
# direction/outlook. Greys for unknown/neutral. These are the only colors
# the dashboard should use for semantic state — everything else is
# structural (borders, text, backgrounds).

RISK_COLORS: dict[str, str] = {
    "low": "#16a34a",        # green-600
    "medium": "#ca8a04",     # amber-600
    "moderate": "#ca8a04",   # alias used by portfolio concentration bucket
    "high": "#dc2626",       # red-600
    "severe": "#991b1b",     # red-800
    "unknown": "#6b7280",    # gray-500
}

OUTLOOK_COLORS: dict[str, str] = {
    "bullish": "#16a34a",
    "neutral": "#6b7280",
    "bearish": "#dc2626",
    "unknown": "#6b7280",
}

# Distress zones from risk_scanner.altman_z (zone in {distress, grey, safe}).
ZONE_COLORS: dict[str, str] = {
    "safe": "#16a34a",
    "grey": "#ca8a04",
    "distress": "#dc2626",
    "unknown": "#6b7280",
}

# Freshness staleness ramp, applied to a section's latest as_of.
FRESH_FRESH = "#16a34a"     # < 1 day
FRESH_RECENT = "#ca8a04"    # < 7 days
FRESH_STALE = "#6b7280"     # older / unknown

_NEUTRAL_TEXT = "#9ca3af"
_HAIRLINE = "#1f2630"
_CARD_BG = "rgba(148,163,184,0.06)"


# ======================================================================
# Small internal utilities
# ======================================================================
def _norm(s: Any) -> str:
    """Lowercase, stripped string key; '' for None."""
    return (str(s).strip().lower()) if s is not None else ""


def _risk_color(level: Any) -> str:
    return RISK_COLORS.get(_norm(level), RISK_COLORS["unknown"])


def _outlook_color(direction: Any) -> str:
    return OUTLOOK_COLORS.get(_norm(direction), OUTLOOK_COLORS["unknown"])


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_as_of(as_of: Any) -> datetime | None:
    """Parse an ISO date/datetime string to an aware datetime, or None.

    Accepts 'YYYY-MM-DD' and full ISO timestamps (with or without 'Z').
    Returns None on anything unparseable so callers can fall back to a
    neutral 'unknown' freshness rather than raising.
    """
    if not as_of or not isinstance(as_of, str):
        return None
    txt = as_of.strip()
    if not txt:
        return None
    # Normalize a trailing Z to +00:00 for fromisoformat.
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        # Date-only fallback.
        try:
            dt = datetime.strptime(txt[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _staleness_color(as_of: Any) -> str:
    dt = _parse_as_of(as_of)
    if dt is None:
        return FRESH_STALE
    age = datetime.now(timezone.utc) - dt
    if age.total_seconds() < 86_400:        # < 1 day
        return FRESH_FRESH
    if age.total_seconds() < 7 * 86_400:    # < 7 days
        return FRESH_RECENT
    return FRESH_STALE


def _age_label(as_of: Any) -> str:
    """Human 'as of' label, e.g. 'today', '3d ago', '2025-01-04'."""
    dt = _parse_as_of(as_of)
    if dt is None:
        return "as of —"
    age = datetime.now(timezone.utc) - dt
    days = age.days
    if days <= 0:
        return "as of today"
    if days == 1:
        return "as of 1d ago"
    if days < 7:
        return f"as of {days}d ago"
    # Older: show the date itself.
    return f"as of {dt.date().isoformat()}"


# ======================================================================
# Badges — risk level and outlook pills
# ======================================================================
def badge_risk(level: Any) -> str:
    """Return an HTML span styled as a colored risk pill (LOW/MEDIUM/HIGH).

    Render with ``st.markdown(badge_risk(x), unsafe_allow_html=True)``.
    Accepts any case; unknown values render as a grey 'UNKNOWN' pill.
    """
    key = _norm(level) or "unknown"
    color = _risk_color(key)
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.78em;"
        f"font-weight:700;letter-spacing:0.03em;"
        f"text-transform:uppercase;white-space:nowrap;'>{key}</span>"
    )


def badge_outlook(direction: Any) -> str:
    """Return an HTML span styled as a colored outlook pill
    (BULLISH/NEUTRAL/BEARISH)."""
    key = _norm(direction) or "unknown"
    color = _outlook_color(key)
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.78em;"
        f"font-weight:700;letter-spacing:0.03em;"
        f"text-transform:uppercase;white-space:nowrap;'>{key}</span>"
    )


def badge_zone(zone: Any) -> str:
    """Altman distress-zone pill (SAFE/GREY/DISTRESS)."""
    key = _norm(zone) or "unknown"
    color = ZONE_COLORS.get(key, ZONE_COLORS["unknown"])
    label = {"grey": "GREY ZONE"}.get(key, key.upper())
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.78em;"
        f"font-weight:700;letter-spacing:0.03em;white-space:nowrap;'>{label}</span>"
    )


# ======================================================================
# Conviction pip meter — agent confidence 0..1 as 5 segments
# ======================================================================
def meter_conviction(confidence: Any, *, segments: int = 5) -> str:
    """Render confidence (0..1) as a discrete pip meter, never a bare %.

    A lone "62%" reads as spurious precision for an LLM-derived figure; a
    5-pip meter communicates "moderate conviction" at a glance and keeps
    the exact value as a trailing label for those who want it. Returns an
    HTML string; render with ``unsafe_allow_html=True``.

    Filled pips are tinted by conviction band (low/med/high) so the meter
    carries magnitude in both fill-count and color.
    """
    if not _is_num(confidence):
        # Unknown conviction: all-empty meter + em dash.
        pips = "".join(_pip(False, _NEUTRAL_TEXT) for _ in range(segments))
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;'>"
            f"{pips}<span style='color:{_NEUTRAL_TEXT};font-size:0.8em;'>—</span></span>"
        )

    c = max(0.0, min(1.0, float(confidence)))
    filled = int(round(c * segments))
    # Color band by conviction tier.
    if c >= 0.66:
        tint = RISK_COLORS["low"]      # high conviction -> green
    elif c >= 0.33:
        tint = RISK_COLORS["medium"]   # moderate -> amber
    else:
        tint = RISK_COLORS["high"]     # low conviction -> red
    pips = "".join(
        _pip(i < filled, tint) for i in range(segments)
    )
    return (
        f"<span style='display:inline-flex;align-items:center;gap:6px;'>"
        f"{pips}"
        f"<span style='color:{_NEUTRAL_TEXT};font-size:0.8em;"
        f"font-variant-numeric:tabular-nums;'>{c:.0%}</span></span>"
    )


def _pip(filled: bool, color: str) -> str:
    if filled:
        return (
            f"<span style='display:inline-block;width:14px;height:8px;"
            f"border-radius:2px;background:{color};'></span>"
        )
    return (
        f"<span style='display:inline-block;width:14px;height:8px;"
        f"border-radius:2px;background:transparent;"
        f"border:1px solid {_HAIRLINE};'></span>"
    )


# ======================================================================
# Diverging bias bar — bias_score -1..+1 centered at zero
# ======================================================================
def meter_bias(bias_score: Any, *, width_px: int = 160) -> str:
    """Render bias_score (-1..+1) as a centered diverging bar.

    Negative (bearish) fills left in red; positive (bullish) fills right in
    green; a center tick marks zero. This is distinct from the conviction
    meter on purpose: conviction is "how sure", bias is "which way and how
    hard". Returns HTML; render with ``unsafe_allow_html=True``.
    """
    half = width_px // 2
    if not _is_num(bias_score):
        return (
            f"<span style='display:inline-flex;align-items:center;gap:8px;'>"
            f"<span style='position:relative;display:inline-block;"
            f"width:{width_px}px;height:10px;background:{_CARD_BG};"
            f"border-radius:5px;'>"
            f"<span style='position:absolute;left:{half}px;top:-2px;"
            f"width:1px;height:14px;background:{_HAIRLINE};'></span></span>"
            f"<span style='color:{_NEUTRAL_TEXT};font-size:0.8em;'>—</span></span>"
        )

    b = max(-1.0, min(1.0, float(bias_score)))
    fill = int(abs(b) * half)
    if b >= 0:
        # Fill rightward from center.
        bar = (
            f"<span style='position:absolute;left:{half}px;top:0;"
            f"width:{fill}px;height:10px;background:{OUTLOOK_COLORS['bullish']};"
            f"border-radius:0 5px 5px 0;'></span>"
        )
    else:
        # Fill leftward from center.
        bar = (
            f"<span style='position:absolute;left:{half - fill}px;top:0;"
            f"width:{fill}px;height:10px;background:{OUTLOOK_COLORS['bearish']};"
            f"border-radius:5px 0 0 5px;'></span>"
        )
    return (
        f"<span style='display:inline-flex;align-items:center;gap:8px;'>"
        f"<span style='position:relative;display:inline-block;"
        f"width:{width_px}px;height:10px;background:{_CARD_BG};"
        f"border-radius:5px;'>"
        f"{bar}"
        f"<span style='position:absolute;left:{half}px;top:-2px;"
        f"width:1px;height:14px;background:{_NEUTRAL_TEXT};'></span></span>"
        f"<span style='color:{_NEUTRAL_TEXT};font-size:0.8em;"
        f"font-variant-numeric:tabular-nums;'>{b:+.2f}</span></span>"
    )


# ======================================================================
# Freshness / provenance chip
# ======================================================================
def chip_freshness(as_of: Any, *, source_id: Any = None,
                   confidence: Any = None, label: str | None = None) -> str:
    """A small 'as of' chip colored by staleness, with optional source.

    Pulls directly from the context's ``provenance[field]`` /
    ``freshness[section]`` blocks. A green dot = fresh (<1d), amber =
    recent (<7d), grey = stale/unknown. Hovering shows source + confidence
    via the title attribute. Returns HTML; render with
    ``unsafe_allow_html=True``.
    """
    dot = _staleness_color(as_of)
    text = label or _age_label(as_of)
    title_bits: list[str] = []
    if source_id:
        title_bits.append(f"source: {source_id}")
    if _is_num(confidence):
        title_bits.append(f"confidence: {float(confidence):.0%}")
    if as_of:
        title_bits.append(f"as_of: {as_of}")
    title = " · ".join(title_bits) if title_bits else "no provenance recorded"

    src_suffix = ""
    if source_id:
        src_suffix = (
            f"<span style='color:{_NEUTRAL_TEXT};opacity:0.7;'> · {source_id}</span>"
        )
    return (
        f"<span title='{title}' style='display:inline-flex;align-items:center;"
        f"gap:5px;font-size:0.74em;color:{_NEUTRAL_TEXT};"
        f"border:1px solid {_HAIRLINE};border-radius:8px;padding:1px 7px;"
        f"white-space:nowrap;'>"
        f"<span style='width:7px;height:7px;border-radius:50%;background:{dot};"
        f"display:inline-block;'></span>{text}{src_suffix}</span>"
    )


def chip_from_provenance(context: Mapping[str, Any], field: str,
                         *, label: str | None = None) -> str:
    """Convenience: build a freshness chip from ``context['provenance'][field]``.

    Falls back to ``context['freshness'][section]`` semantics is left to
    the caller; this one is keyed on the per-field provenance map. If the
    field is absent, renders a grey 'no provenance' chip rather than
    raising.
    """
    prov = (context or {}).get("provenance") or {}
    row = prov.get(field) or {}
    return chip_freshness(
        row.get("as_of"),
        source_id=row.get("source_id"),
        confidence=row.get("confidence"),
        label=label,
    )


# ======================================================================
# Three-axis risk triplet
# ======================================================================
def panel_risk_triplet(risk_flags: Mapping[str, Any] | None,
                       *, show_reasons: bool = False,
                       top_reasons: int = 1) -> None:
    """Render the fundamental/macro/market/combined risk decomposition.

    The backend's risk object is::

        {"levels":  {"fundamental","macro","market","combined"},
         "reasons": {"fundamental":[...], "macro":[...], "market":[...]}}

    The current dashboard collapses this to a single pill plus a flat list;
    this helper keeps the three-axis structure visible, which is the whole
    point of the risk_scanner design. ``show_reasons`` adds the top N reason
    per axis inline (used in the Desk risk section); leave it off for compact
    contexts like table rows or the ticker header (use ``inline_risk_triplet``
    there instead).

    Renders directly via Streamlit (writes to the page). Returns None.
    """
    levels = (risk_flags or {}).get("levels") or {}
    reasons = (risk_flags or {}).get("reasons") or {}

    if not levels:
        st.markdown(
            f"<span style='color:{_NEUTRAL_TEXT};'>No risk decomposition yet — "
            f"run the agent chain to compute fundamental / macro / market "
            f"levels.</span>",
            unsafe_allow_html=True,
        )
        return

    combined = levels.get("combined", "unknown")
    st.markdown(
        f"**Combined risk** &nbsp; {badge_risk(combined)}",
        unsafe_allow_html=True,
    )

    axes = [
        ("Fundamental", "fundamental"),
        ("Macro", "macro"),
        ("Market", "market"),
        ("Agent (LLM)", "agent"),
    ]
    for label, key in axes:
        level = levels.get(key, "unknown")
        line = f"<div style='margin:4px 0;'>{badge_risk(level)} " \
               f"<span style='font-size:0.9em;'>&nbsp;{label}</span></div>"
        st.markdown(line, unsafe_allow_html=True)
        if show_reasons:
            axis_reasons = reasons.get(key) or []
            # Filter the 'no data' placeholder the scanner emits.
            axis_reasons = [r for r in axis_reasons if _norm(r) != "no data"]
            for r in axis_reasons[:top_reasons]:
                st.markdown(
                    f"<div style='margin:0 0 6px 2px;font-size:0.85em;"
                    f"color:{_NEUTRAL_TEXT};'>↳ {r}</div>",
                    unsafe_allow_html=True,
                )


def inline_risk_triplet(risk_flags: Mapping[str, Any] | None) -> str:
    """Compact one-line F/M/M risk triplet for headers and table rows.

    Returns an HTML string showing three small dots (fundamental, macro,
    market) colored by level, with a combined pill. Render with
    ``unsafe_allow_html=True``. Designed to fit in a single grid cell.
    """
    levels = (risk_flags or {}).get("levels") or {}
    if not levels:
        return f"<span style='color:{_NEUTRAL_TEXT};font-size:0.8em;'>risk —</span>"

    def _dot(key: str, letter: str) -> str:
        color = _risk_color(levels.get(key, "unknown"))
        return (
            f"<span title='{letter}: {levels.get(key, 'unknown')}' "
            f"style='display:inline-flex;align-items:center;gap:2px;'>"
            f"<span style='width:8px;height:8px;border-radius:50%;"
            f"background:{color};display:inline-block;'></span>"
            f"<span style='font-size:0.72em;color:{_NEUTRAL_TEXT};'>{letter}</span>"
            f"</span>"
        )

    dots = "&nbsp;".join([
        _dot("fundamental", "F"),
        _dot("macro", "M"),
        _dot("market", "Mk"),
        _dot("agent", "A"),
    ])
    combined = levels.get("combined", "unknown")
    return (
        f"<span style='display:inline-flex;align-items:center;gap:8px;'>"
        f"{dots}&nbsp;{badge_risk(combined)}</span>"
    )


# ======================================================================
# Sector-relative z-score bar
# ======================================================================
def panel_zscore_bar(label: str, z: Any, *, width_px: int = 180) -> str:
    """Render a single sector-relative z-score as a diverging bar.

    Convention (from risk_scanner.zscore_risk_signed): **negative z = worse
    than peers**, regardless of which side of the mean is "bad". So the bar
    fills LEFT/red for negative (riskier than peers) and RIGHT/green for
    positive (safer than peers), with the LOW/MEDIUM/HIGH tier coloring the
    label. Clamped to [-3, +3] for display. Returns HTML; render with
    ``unsafe_allow_html=True``.
    """
    half = width_px // 2
    if not _is_num(z):
        bar_inner = ""
        tier_txt = "—"
        tier_color = _NEUTRAL_TEXT
    else:
        zc = max(-3.0, min(3.0, float(z)))
        frac = abs(zc) / 3.0
        fill = int(frac * half)
        if zc >= 0:
            color = RISK_COLORS["low"]
            bar_inner = (
                f"<span style='position:absolute;left:{half}px;top:0;"
                f"width:{fill}px;height:9px;background:{color};"
                f"border-radius:0 5px 5px 0;'></span>"
            )
        else:
            color = RISK_COLORS["high"]
            bar_inner = (
                f"<span style='position:absolute;left:{half - fill}px;top:0;"
                f"width:{fill}px;height:9px;background:{color};"
                f"border-radius:5px 0 0 5px;'></span>"
            )
        # Tier label using the scanner's ztier thresholds (approx; the exact
        # cutoffs live in risk_scanner, this is purely for the color hint).
        if zc < -1.0:
            tier_txt, tier_color = "HIGH", RISK_COLORS["high"]
        elif zc < 0.0:
            tier_txt, tier_color = "MED", RISK_COLORS["medium"]
        else:
            tier_txt, tier_color = "LOW", RISK_COLORS["low"]
        tier_txt = f"{zc:+.1f}σ · {tier_txt}"

    return (
        f"<div style='display:flex;align-items:center;gap:10px;margin:3px 0;'>"
        f"<span style='width:130px;font-size:0.85em;'>{label}</span>"
        f"<span style='position:relative;display:inline-block;width:{width_px}px;"
        f"height:9px;background:{_CARD_BG};border-radius:5px;'>"
        f"<span style='position:absolute;left:{half}px;top:-2px;width:1px;"
        f"height:13px;background:{_NEUTRAL_TEXT};'></span>{bar_inner}</span>"
        f"<span style='font-size:0.78em;color:{tier_color};"
        f"font-variant-numeric:tabular-nums;min-width:70px;'>{tier_txt}</span>"
        f"</div>"
    )


# ======================================================================
# Fallback / deterministic-output border
# ======================================================================
def panel_fallback_notice(is_fallback: bool, *, what: str = "output") -> None:
    """Render a striped amber notice when content is deterministic-fallback.

    Anytime ``essay_meta.fallback`` or an analyzer's ``fallback`` flag is
    true, the content was produced WITHOUT the LLM (Ollama down or returned
    junk). It must never be mistaken for a real model verdict. Call this
    immediately above the affected block. No-op when not in fallback.
    """
    if not is_fallback:
        return
    st.markdown(
        f"<div style='border:1px solid {RISK_COLORS['medium']};"
        f"background:repeating-linear-gradient(45deg,"
        f"rgba(202,138,4,0.08),rgba(202,138,4,0.08) 8px,"
        f"transparent 8px,transparent 16px);"
        f"border-radius:8px;padding:8px 12px;margin:6px 0;"
        f"font-size:0.85em;color:{RISK_COLORS['medium']};'>"
        f"⚠ Deterministic fallback — this {what} was produced without the "
        f"LLM (Ollama unavailable or returned an unusable response). "
        f"Connect Ollama and regenerate for a real model verdict."
        f"</div>",
        unsafe_allow_html=True,
    )


def card_open(border: str = _HAIRLINE) -> str:
    """Open a lightweight bordered card div. Pair with ``card_close()``.

    Returned string must be rendered with ``unsafe_allow_html=True``. Use
    for grouping a summary tile when ``st.container(border=True)`` is too
    heavy or you need custom border color (e.g. risk-tinted).
    """
    return (
        f"<div style='border:1px solid {border};border-radius:10px;"
        f"padding:12px 14px;background:{_CARD_BG};margin:4px 0;'>"
    )


def card_close() -> str:
    return "</div>"


# ======================================================================
# Evidence card — a single RAG retrieved_context chunk
# ======================================================================
def card_evidence(chunk: Mapping[str, Any]) -> None:
    """Render one RAG chunk from ``context['retrieved_context']``.

    Each chunk has: source, doc_id, ticker, as_of, section, score, text.
    The score is shown as a small relevance bar; the source/section as a
    header; the text truncated to a snippet with the full text in a
    tooltip. If a ``primary_doc_url`` is present (filings), a link is shown.
    Renders directly via Streamlit. Returns None.
    """
    source = chunk.get("source") or "?"
    section = chunk.get("section")
    score = chunk.get("score")
    text = (chunk.get("text") or "").strip()
    as_of = chunk.get("as_of")
    url = chunk.get("primary_doc_url") or chunk.get("url")

    score_bar = ""
    if _is_num(score):
        pct = max(0.0, min(1.0, float(score)))
        score_bar = (
            f"<span style='display:inline-block;width:48px;height:6px;"
            f"background:{_CARD_BG};border-radius:3px;position:relative;'>"
            f"<span style='position:absolute;left:0;top:0;height:6px;"
            f"width:{int(pct*48)}px;background:{OUTLOOK_COLORS['bullish']};"
            f"border-radius:3px;'></span></span>"
            f"<span style='font-size:0.7em;color:{_NEUTRAL_TEXT};"
            f"margin-left:4px;font-variant-numeric:tabular-nums;'>"
            f"{pct:.2f}</span>"
        )

    header = (
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"gap:8px;margin-bottom:4px;'>"
        f"<span style='font-size:0.8em;font-weight:600;'>{source}"
        + (f" · {section}" if section else "")
        + f"</span>{score_bar}</div>"
    )

    # Filing excerpts contain newlines and leading-space-indented lines like
    # "  (973)" / "Capital spending". st.markdown runs the MARKDOWN PARSER
    # before honoring our HTML wrapper, so any 4+ space indent or blank-line
    # break becomes a green <code>/<pre> block regardless of escaping or CSS.
    # Fix: flatten the excerpt to a single continuous prose line (collapse all
    # whitespace runs to one space), THEN escape. A compact evidence card reads
    # better as one blob anyway, and markdown has no indentation left to
    # misinterpret.
    import html as _html
    import re as _re
    flat = _re.sub(r"\s+", " ", text).strip()
    snippet = flat if len(flat) <= 320 else flat[:317].rstrip() + "…"
    safe_snippet = _html.escape(snippet)
    safe_title = _html.escape(flat[:1000])
    body = (
        f"<div style='font-size:0.84em;line-height:1.5;color:#cbd5e1;"
        f"white-space:normal;' "
        f"title='{safe_title}'>{safe_snippet}</div>"
    )

    footer_bits = []
    if as_of:
        footer_bits.append(_age_label(as_of))
    footer = ""
    if footer_bits or url:
        link = f" · <a href='{url}' target='_blank'>open source</a>" if url else ""
        footer = (
            f"<div style='font-size:0.72em;color:{_NEUTRAL_TEXT};margin-top:6px;'>"
            f"{' · '.join(footer_bits)}{link}</div>"
        )

    st.markdown(
        card_open() + header + body + footer + card_close(),
        unsafe_allow_html=True,
    )


def panel_evidence_list(chunks: Sequence[Mapping[str, Any]] | None,
                        *, limit: int = 8, empty_hint: str | None = None) -> None:
    """Render a list of RAG evidence cards, or a neutral empty hint.

    The current dashboard never shows ``retrieved_context``, so the analyst
    can't see what evidence the thesis rests on. This makes it visible.
    """
    chunks = list(chunks or [])
    if not chunks:
        msg = empty_hint or (
            "No retrieved evidence for this ticker. RAG may be disabled, the "
            "vector store empty, or nothing indexed yet for this name."
        )
        st.markdown(
            f"<span style='color:{_NEUTRAL_TEXT};font-size:0.9em;'>{msg}</span>",
            unsafe_allow_html=True,
        )
        return
    for chunk in chunks[:limit]:
        card_evidence(chunk)


# ======================================================================
# Review scorecard — thesis_review output
# ======================================================================
# Canonical section order + display labels. The scanner/reviewer emits keys
# like 'executive_summary', 'valuation', etc.; we render them in memo order
# and skip any that are absent.
_REVIEW_SECTION_LABELS: list[tuple[str, str]] = [
    ("executive_summary", "Executive Summary"),
    ("business_performance", "Business & Financial Performance"),
    ("valuation", "Valuation"),
    ("filings_signal", "Filings & Management Signal"),
    ("macro_context", "Macro Context"),
    ("risks", "Risks"),
    ("catalysts", "Catalysts"),
    ("verdict", "Verdict"),
]


def _score_color(score: float) -> str:
    """Map a 1-10 review score to the risk ramp (low score = red)."""
    if score >= 8.0:
        return RISK_COLORS["low"]
    if score >= 6.0:
        return RISK_COLORS["medium"]
    return RISK_COLORS["high"]


def panel_review_scorecard(review: Mapping[str, Any] | None) -> None:
    """Render the thesis_review scores as a per-section quality strip.

    Shape::

        {"scores": {"executive_summary": 7, "valuation": 4, ...,
                    "overall": 6.2},
         "text": "...full review prose..."}

    A low 'valuation: 4/10' is exactly the kind of self-critique that should
    be visible next to the memo, not buried. Shows the overall score as a
    headline, each section as a small 0-10 bar, and flags the weakest
    section. Renders directly. Returns None.
    """
    scores = (review or {}).get("scores") or {}
    if not scores:
        st.markdown(
            f"<span style='color:{_NEUTRAL_TEXT};'>No review yet — the memo "
            f"has not been self-reviewed.</span>",
            unsafe_allow_html=True,
        )
        return

    overall = scores.get("overall")
    if _is_num(overall):
        ov = float(overall)
        color = _score_color(ov)
        st.markdown(
            f"<div style='font-size:1.05em;margin-bottom:8px;'>"
            f"<b>Self-review score:</b> "
            f"<span style='color:{color};font-weight:700;"
            f"font-variant-numeric:tabular-nums;'>{ov:.1f}/10</span>"
            + (f" &nbsp;<span style='color:{RISK_COLORS['high']};font-size:0.82em;'>"
               f"· auto-revised threshold is 8.0</span>" if ov < 8.0 else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    # Per-section bars.
    section_scores: list[tuple[str, float]] = []
    for key, label in _REVIEW_SECTION_LABELS:
        val = scores.get(key)
        if _is_num(val):
            section_scores.append((label, float(val)))

    if section_scores:
        weakest = min(section_scores, key=lambda kv: kv[1])
        for label, val in section_scores:
            frac = max(0.0, min(1.0, val / 10.0))
            color = _score_color(val)
            is_weak = (label == weakest[0])
            weak_mark = (
                f" <span style='color:{RISK_COLORS['high']};font-size:0.72em;'>"
                f"weakest</span>" if is_weak else ""
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:2px 0;'>"
                f"<span style='width:230px;font-size:0.85em;'>{label}{weak_mark}</span>"
                f"<span style='position:relative;display:inline-block;width:140px;"
                f"height:8px;background:{_CARD_BG};border-radius:4px;'>"
                f"<span style='position:absolute;left:0;top:0;height:8px;"
                f"width:{int(frac*140)}px;background:{color};border-radius:4px;'>"
                f"</span></span>"
                f"<span style='font-size:0.78em;color:{color};"
                f"font-variant-numeric:tabular-nums;'>{val:.0f}</span></div>",
                unsafe_allow_html=True,
            )

    # Full review reasoning (the prose behind the scores). The reviewer emits
    # a 'text' field explaining each section's score and the main weakness;
    # surfacing it lets the analyst see WHY a section scored low, not just THAT
    # it did.
    review_text = (review or {}).get("text")
    if review_text and isinstance(review_text, str) and review_text.strip():
        with st.expander("Why these scores? (full review)", expanded=False):
            st.markdown(review_text)


# ======================================================================
# Section anchor — for the Desk's sticky in-page nav
# ======================================================================
def section_anchor(anchor_id: str, title: str, *, subtitle: str | None = None) -> None:
    """Render a section header with an HTML anchor for in-page navigation.

    The Desk is a scrollable single page with a sticky nav; each major
    section drops one of these so the nav can jump to it. Renders directly.
    """
    sub = (
        f"<div style='font-size:0.85em;color:{_NEUTRAL_TEXT};margin-top:-2px;'>"
        f"{subtitle}</div>" if subtitle else ""
    )
    st.markdown(
        f"<div id='{anchor_id}' style='scroll-margin-top:80px;'></div>"
        f"<div style='margin:6px 0 2px 0;'>"
        f"<span style='font-size:1.25em;font-weight:700;'>{title}</span></div>{sub}",
        unsafe_allow_html=True,
    )


def section_nav(items: Sequence[tuple[str, str]]) -> None:
    """Render a horizontal in-page anchor nav.

    ``items`` is a sequence of (anchor_id, label). Renders as a row of
    same-page links. Streamlit re-renders strip ``#anchor`` scrolling in
    some embeds, so this is best-effort smooth nav, with the anchors from
    ``section_anchor`` as the targets.
    """
    if not items:
        return
    links = " &nbsp;·&nbsp; ".join(
        f"<a href='#{aid}' style='text-decoration:none;font-size:0.85em;"
        f"color:#93c5fd;'>{label}</a>"
        for aid, label in items
    )
    st.markdown(
        f"<div style='position:sticky;top:0;z-index:5;padding:6px 0;"
        f"border-bottom:1px solid {_HAIRLINE};margin-bottom:8px;"
        f"backdrop-filter:blur(4px);'>{links}</div>",
        unsafe_allow_html=True,
    )


# ======================================================================
# Number formatting helpers (display-only)
# ======================================================================
def fmt_money(x: Any, *, decimals: int = 2) -> str:
    """Format a number as $ with thousands separators, or '—' if not numeric."""
    if not _is_num(x):
        return "—"
    return f"${float(x):,.{decimals}f}"


def fmt_big(x: Any) -> str:
    """Format a large number compactly ($1.2B, $890M, $4.5K), or '—'."""
    if not _is_num(x):
        return "—"
    v = float(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return f"{sign}${v/1e12:.2f}T"
    if v >= 1e9:
        return f"{sign}${v/1e9:.2f}B"
    if v >= 1e6:
        return f"{sign}${v/1e6:.2f}M"
    if v >= 1e3:
        return f"{sign}${v/1e3:.1f}K"
    return f"{sign}${v:,.0f}"


def fmt_pct(x: Any, *, decimals: int = 1, signed: bool = False) -> str:
    """Format a fraction (0.123) as a percent (12.3%), or '—'.

    Set ``signed`` to always show +/-. Pass already-percent values divided
    by 100 first; this assumes a 0..1 fraction.
    """
    if not _is_num(x):
        return "—"
    v = float(x) * 100.0
    if signed:
        return f"{v:+.{decimals}f}%"
    return f"{v:.{decimals}f}%"


def fmt_ratio(x: Any, *, decimals: int = 2, suffix: str = "") -> str:
    """Format a plain ratio (P/E, debt/EBITDA), or '—'."""
    if not _is_num(x):
        return "—"
    return f"{float(x):.{decimals}f}{suffix}"


__all__ = [
    # colors
    "RISK_COLORS", "OUTLOOK_COLORS", "ZONE_COLORS",
    # badges
    "badge_risk", "badge_outlook", "badge_zone",
    # meters
    "meter_conviction", "meter_bias",
    # chips
    "chip_freshness", "chip_from_provenance",
    # risk
    "panel_risk_triplet", "inline_risk_triplet", "panel_zscore_bar",
    # cards / containers
    "card_open", "card_close", "panel_fallback_notice",
    # evidence
    "card_evidence", "panel_evidence_list",
    # review
    "panel_review_scorecard",
    # layout
    "section_anchor", "section_nav",
    # formatting
    "fmt_money", "fmt_big", "fmt_pct", "fmt_ratio",
]

# D:\Ary Fund\ui\components.py