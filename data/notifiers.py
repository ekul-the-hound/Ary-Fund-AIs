"""
data/notifiers.py
=================

Outbound notification adapters. Today this is just Slack via incoming
webhooks; the module is structured so additional channels (Discord,
email, PagerDuty) can be slotted in without touching call sites.

Slack incoming webhooks
-----------------------
Configure a webhook URL once per workspace at
``https://api.slack.com/messaging/webhooks`` and expose it to the app as
``SLACK_WEBHOOK_URL`` in the environment (or in ``.env``). With that
single secret, any layer can post:

    >>> from data.notifiers import notify_slack
    >>> notify_slack("AAPL flagged HIGH (macro + market)")

Calls are best-effort. A network failure or missing webhook URL logs a
warning but does NOT raise — the caller's primary job (a risk scan, a
refresh) must never fail just because Slack is down or unconfigured.

For richer alerts use the structured helper:

    >>> from data.notifiers import notify_risk_flags
    >>> notify_risk_flags("AAPL", risk_flags_dict)

which formats the dict produced by ``risk_scanner.compute_risk_flags``
into a readable Slack message with severity emoji and the offending
reasons grouped by domain.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Optional

import requests

logger = logging.getLogger(__name__)


_HTTP_TIMEOUT = 8
_DEFAULT_USERNAME = "ARY QUANT"

# Visual cues per severity level. Slack renders these as inline emoji.
_SEVERITY_EMOJI: dict[str, str] = {
    "HIGH":   ":rotating_light:",
    "MEDIUM": ":warning:",
    "LOW":    ":white_check_mark:",
}


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
def _resolve_webhook_url(explicit: Optional[str] = None) -> Optional[str]:
    """Pick the first non-empty source: explicit arg, then env var.

    Returns ``None`` if neither is set; callers must treat that as
    "Slack disabled" rather than an error condition.
    """
    if explicit:
        return explicit
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    return url or None


def slack_configured(webhook_url: Optional[str] = None) -> bool:
    """Return True iff a Slack webhook URL is resolvable."""
    return _resolve_webhook_url(webhook_url) is not None


# ---------------------------------------------------------------------------
# Low-level send
# ---------------------------------------------------------------------------
def notify_slack(
    text: str,
    *,
    webhook_url: Optional[str] = None,
    username: str = _DEFAULT_USERNAME,
    blocks: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """Post a message to Slack. Returns True on success, False otherwise.

    Never raises. A missing webhook is logged at INFO (expected on
    laptops without a webhook configured) and returns False quietly.
    """
    url = _resolve_webhook_url(webhook_url)
    if not url:
        logger.info("notifiers | slack disabled (SLACK_WEBHOOK_URL not set)")
        return False

    payload: dict[str, Any] = {"text": text, "username": username}
    if blocks:
        payload["blocks"] = blocks

    try:
        resp = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("notifiers | slack send failed: %s", e)
        return False
    if not resp.ok:
        logger.warning(
            "notifiers | slack returned %s: %s",
            resp.status_code, resp.text[:200],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Structured risk-flag formatter
# ---------------------------------------------------------------------------
def format_risk_flags(
    ticker: str,
    flags: dict[str, Any],
    *,
    include_levels: Iterable[str] = ("HIGH", "MEDIUM"),
) -> Optional[str]:
    """Render a ``risk_scanner.compute_risk_flags()`` dict as Slack text.

    Returns ``None`` if the combined severity is below
    ``include_levels`` — caller can use ``None`` as a "don't bother
    sending" sentinel.
    """
    if not isinstance(flags, dict):
        return None
    levels = flags.get("levels") or {}
    combined = str(levels.get("combined", "LOW")).upper()
    if combined not in {l.upper() for l in include_levels}:
        return None

    reasons = flags.get("reasons") or {}
    emoji = _SEVERITY_EMOJI.get(combined, ":grey_question:")
    header = f"{emoji} *{ticker}* — combined risk: *{combined}*"

    # Per-domain breakdown. Only surface domains that actually contribute
    # to the elevated score, so the message stays short.
    parts: list[str] = [header]
    domain_order = ("fundamental", "macro", "market", "agent")
    for domain in domain_order:
        lvl = str(levels.get(domain, "LOW")).upper()
        if lvl == "LOW":
            continue
        rsn = reasons.get(domain) or []
        rsn = [r for r in rsn if r and r != "no data"]
        if not rsn:
            parts.append(f"  • _{domain}_ ({lvl})")
            continue
        # Keep reasons concise — three lines max per domain.
        trimmed = rsn[:3]
        more = f" (+{len(rsn) - 3} more)" if len(rsn) > 3 else ""
        parts.append(f"  • _{domain}_ ({lvl}): " + "; ".join(trimmed) + more)
    return "\n".join(parts)


def notify_risk_flags(
    ticker: str,
    flags: dict[str, Any],
    *,
    webhook_url: Optional[str] = None,
    include_levels: Iterable[str] = ("HIGH", "MEDIUM"),
) -> bool:
    """Send a risk-flag summary to Slack if severity warrants it.

    Returns True if a message was sent, False otherwise (severity too
    low OR webhook unavailable OR network error).
    """
    text = format_risk_flags(ticker, flags, include_levels=include_levels)
    if text is None:
        return False
    return notify_slack(text, webhook_url=webhook_url)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_flags = {
        "levels": {
            "fundamental": "MEDIUM",
            "macro": "HIGH",
            "market": "LOW",
            "agent": "MEDIUM",
            "combined": "HIGH",
        },
        "reasons": {
            "fundamental": ["debt/EBITDA 4.1x > 3.0", "interest coverage 1.6x < 2.0"],
            "macro": ["recession_prob 0.62 > 0.6", "VIX 31.4 > 28"],
            "market": ["no data"],
            "agent": ["regulatory investigation", "customer concentration"],
        },
    }
    print("--- preview ---")
    print(format_risk_flags("AAPL", sample_flags))
    print("--- send attempt ---")
    ok = notify_risk_flags("AAPL", sample_flags)
    print("sent:", ok)
