"""
ui/palette.py
=============

The command layer + the job tray.

Two cooperating pieces, both deliberately decoupled from navigation (they
return intents / render status; the orchestrator in ``app_v2`` decides what
to do):

1. COMMAND PARSER + INPUT
   A verb-noun command box (Bloomberg-terminal style) that turns short typed
   commands into structured ``Command`` intents the orchestrator dispatches:

       nvda                -> go to ticker NVDA (Desk)
       gen msft            -> enqueue full analysis for MSFT (non-blocking)
       memo tsla           -> go to TSLA, Memo section / draft memo
       risk aapl           -> go to AAPL, Risk section
       quant amd           -> go to AMD in the Quant Lab
       decide nvda         -> go to NVDA, Decision stage
       screen              -> open the Screener
       board               -> open Mission Control
       lab                 -> open the Quant Lab
       report nvda         -> enqueue a PDF report for NVDA
       help                -> show the command cheat-sheet

   The parser is forgiving: a bare symbol is treated as "go to ticker", verbs
   are case-insensitive, and unknown input returns an ``unknown`` command the
   caller can surface as a hint rather than an error.

   HONEST LIMITATION
   -----------------
   Streamlit has no native OS-level keyboard shortcut or modal dialog. A true
   "Cmd-K opens a floating command palette over the app" is not something the
   framework exposes. What this module provides instead is:
     * an always-present command input (no modal needed), and
     * an OPTIONAL best-effort JS snippet (``inject_shortcut_hint``) that
       focuses the command box on Cmd/Ctrl-K. It can only focus an element
       inside the component iframe — it cannot open a true overlay — so it is
       offered as a convenience and labeled as such, not dressed up as a real
       palette.

2. JOB TRAY
   The visible front-end for ``ui.state``'s background queue. Shows each job's
   kind/ticker, live state (queued/running/done/error) with elapsed time and
   error text, and controls to clear finished jobs. This is what makes the
   non-blocking ``gen`` / ``report`` commands legible: fire them, keep working,
   watch the tray.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import streamlit as st

from ui import components as C
from ui import state as S

logger = logging.getLogger("ary_quant.ui.palette")


# ======================================================================
# Command model
# ======================================================================
class Action(str, Enum):
    """The set of intents the parser can emit."""
    GOTO_TICKER = "goto_ticker"      # load a name into the Desk
    GOTO_SECTION = "goto_section"    # load a name + focus a Desk section
    GENERATE = "generate"            # enqueue full analysis for a ticker
    REPORT = "report"                # enqueue a PDF report for a ticker
    OPEN_VIEW = "open_view"          # switch destination (screen/board/lab)
    GOTO_DECISION = "goto_decision"  # load a name + Decision stage
    GOTO_QUANT = "goto_quant"        # load a name + Quant Lab
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class Command:
    """A parsed command intent. ``raw`` is the original text for messaging."""
    action: Action
    ticker: Optional[str] = None
    view: Optional[str] = None        # for OPEN_VIEW: 'screen'|'board'|'lab'|'desk'
    section: Optional[str] = None     # for GOTO_SECTION: 'risk'|'memo'|'quant'|...
    raw: str = ""


# Verb -> handler mapping table. Each verb takes an optional ticker argument.
_VIEW_VERBS = {"screen", "screener", "board", "monitor", "lab", "quant_lab",
               "desk", "home"}
_SECTION_VERBS = {
    "risk": "sec-risk",
    "memo": "sec-thesis",
    "thesis": "sec-thesis",
    "filings": "sec-filings",
    "macro": "sec-macro",
    "evidence": "sec-evidence",
}


def parse_command(text: str) -> Command:
    """Parse a raw command string into a ``Command`` intent.

    Grammar (all case-insensitive):
        <symbol>                      -> GOTO_TICKER
        gen|generate|analyze <sym>    -> GENERATE
        report|pdf <sym>              -> REPORT
        decide|decision <sym>         -> GOTO_DECISION
        quant <sym>                   -> GOTO_QUANT
        risk|memo|thesis|filings|macro|evidence <sym?>  -> GOTO_SECTION
        screen|board|lab|desk         -> OPEN_VIEW
        help|?                         -> HELP
        (anything else)               -> UNKNOWN
    """
    raw = (text or "").strip()
    if not raw:
        return Command(Action.UNKNOWN, raw=raw)

    tokens = raw.split()
    head = tokens[0].lower()
    arg = tokens[1].upper() if len(tokens) > 1 else None

    # Help.
    if head in ("help", "?", "h"):
        return Command(Action.HELP, raw=raw)

    # Pure view switches.
    if head in _VIEW_VERBS:
        view = {
            "screen": "screen", "screener": "screen",
            "board": "board", "monitor": "board",
            "lab": "lab", "quant_lab": "lab",
            "desk": "desk", "home": "desk",
        }.get(head, "desk")
        # A view verb may still carry a ticker ("board nvda" -> set name then board).
        return Command(Action.OPEN_VIEW, view=view, ticker=arg, raw=raw)

    # Generate / analyze.
    if head in ("gen", "generate", "analyze", "analyse", "run"):
        return Command(Action.GENERATE, ticker=arg, raw=raw)

    # Report.
    if head in ("report", "pdf", "memo_pdf"):
        return Command(Action.REPORT, ticker=arg, raw=raw)

    # Decision.
    if head in ("decide", "decision", "size"):
        return Command(Action.GOTO_DECISION, ticker=arg, raw=raw)

    # Quant lab for a name.
    if head in ("quant", "model", "models"):
        return Command(Action.GOTO_QUANT, ticker=arg, raw=raw)

    # Section jumps.
    if head in _SECTION_VERBS:
        return Command(Action.GOTO_SECTION, ticker=arg,
                       section=_SECTION_VERBS[head], raw=raw)

    # Bare symbol -> go to ticker. Heuristic: 1-5 letters, optionally with a
    # dot (BRK.B). Anything else is unknown.
    if len(tokens) == 1 and _looks_like_ticker(head):
        return Command(Action.GOTO_TICKER, ticker=head.upper(), raw=raw)

    return Command(Action.UNKNOWN, raw=raw)


def _looks_like_ticker(tok: str) -> bool:
    t = tok.replace(".", "").replace("-", "")
    return 1 <= len(t) <= 5 and t.isalpha()


# ======================================================================
# Command input widget
# ======================================================================
_CMD_INPUT_KEY = "ary_command_input"


def render_command_bar(*, key_suffix: str = "") -> Optional[Command]:
    """Render the command input. Returns a parsed Command on submit, else None.

    Placed by the orchestrator in the header or sidebar. On Enter, the text is
    parsed and the intent returned for dispatch; the input is then cleared so
    the next command starts fresh. Unknown commands are returned too (action
    UNKNOWN) so the caller can show a one-line hint.
    """
    placeholder = "Command…  e.g.  nvda · gen msft · risk aapl · board · help"
    text = st.text_input(
        "⌘ Command",
        key=f"{_CMD_INPUT_KEY}{key_suffix}",
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    # Streamlit text_input returns the current value every run; we only want
    # to act when it's non-empty AND changed since last dispatch. Track the
    # last-dispatched value in session.
    last_key = f"_ary_last_cmd{key_suffix}"
    last = st.session_state.get(last_key, "")
    if text and text != last:
        st.session_state[last_key] = text
        return parse_command(text)
    if not text:
        # Reset the dispatch guard when the box is cleared.
        st.session_state[last_key] = ""
    return None


def render_help() -> None:
    """Render the command cheat-sheet (shown on the 'help' command)."""
    st.markdown("#### Command reference")
    rows = [
        ("`nvda`", "Open a ticker in the Desk"),
        ("`gen msft`", "Run full analysis for a name (non-blocking)"),
        ("`report tsla`", "Generate a PDF report (non-blocking)"),
        ("`risk aapl`", "Open a name at its Risk section"),
        ("`memo amd`", "Open a name at its Memo section"),
        ("`decide nvda`", "Open a name at the Decision stage"),
        ("`quant nvda`", "Open a name in the Quant Lab"),
        ("`screen`", "Open the Screener"),
        ("`board`", "Open Mission Control"),
        ("`lab`", "Open the Quant Lab"),
        ("`help`", "Show this reference"),
    ]
    for cmd, desc in rows:
        st.markdown(
            f"<div style='display:flex;gap:12px;font-size:0.88em;margin:2px 0;'>"
            f"<span style='min-width:120px;'>{cmd}</span>"
            f"<span style='color:#9ca3af;'>{desc}</span></div>",
            unsafe_allow_html=True)
    st.caption("Tip: a bare symbol (just `nvda`) jumps straight to that name. "
               "Verbs are case-insensitive.")


def inject_shortcut_hint() -> None:
    """OPTIONAL best-effort Cmd/Ctrl-K focus of the command box.

    HONEST SCOPE: Streamlit cannot register a true global shortcut or open a
    modal. This injects a tiny script that, when the user presses Cmd/Ctrl-K,
    attempts to focus the first text input on the page (the command bar). It
    works within the app's DOM but cannot create a floating overlay, so it is
    a convenience, not a real command-palette modal. Safe to omit entirely.
    """
    st.components.v1.html(
        """
        <script>
        const doc = window.parent.document;
        doc.addEventListener('keydown', function(e) {
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                const inputs = doc.querySelectorAll('input[type="text"]');
                if (inputs.length) { inputs[0].focus(); }
            }
        });
        </script>
        """,
        height=0,
    )


# ======================================================================
# Job tray
# ======================================================================
def render_job_tray(*, ticker: Optional[str] = None,
                    compact: bool = False) -> None:
    """Render the background-job tray over ``ui.state``'s queue.

    Parameters
    ----------
    ticker:
        If given, scope the tray to that ticker's jobs (used on the Desk).
        If None, show all jobs (used in the global header / sidebar).
    compact:
        If True, render a single-line summary (for a tight header). If False,
        render the full per-job list with controls.

    Reads the queue via ``poll_jobs`` (safe every rerun). Applies completed
    opinion jobs' cache invalidation as a side effect so finished analyses
    show up without a manual refresh.
    """
    jobs = S.poll_jobs(ticker=ticker)
    active = [j for j in jobs if j.state in (S.JobState.QUEUED, S.JobState.RUNNING)]
    finished = [j for j in jobs if j.state in (S.JobState.DONE, S.JobState.ERROR)]

    # Side effect: when an opinion/report job finishes, clear cached reads so
    # the new state is visible. We do this once per finished job by consuming
    # a session-tracked set of "already applied" ids.
    _apply_finished_side_effects(finished)

    if compact:
        if active:
            kinds = ", ".join(f"{j.kind}:{j.ticker}" for j in active[:3])
            st.markdown(
                f"<span style='font-size:0.82em;color:#93c5fd;'>"
                f"⏳ {len(active)} running · {kinds}</span>",
                unsafe_allow_html=True)
        elif finished:
            errs = sum(1 for j in finished if j.state == S.JobState.ERROR)
            tail = f" · {errs} error(s)" if errs else ""
            st.markdown(
                f"<span style='font-size:0.82em;color:#9ca3af;'>"
                f"✓ {len(finished)} finished{tail}</span>",
                unsafe_allow_html=True)
        return

    # Full tray.
    if not jobs:
        st.caption("No background jobs. Use `gen <ticker>` or `report "
                   "<ticker>` to run analysis without blocking the UI.")
        return

    head_l, head_r = st.columns([3, 1])
    head_l.markdown("**Background jobs**")
    if head_r.button("Clear finished", key="tray_clear",
                     use_container_width=True):
        S.clear_finished_jobs()
        st.rerun()

    # --- Saved reports: Read inline / Open / Download per row ---------------
    try:
        import os as _os
        from pathlib import Path as _P
        import datetime as _dt
        import streamlit.components.v1 as _components
        _root = _P(__file__).resolve().parent.parent
        _rdir = _root / "reports"
        _pdfs = sorted(_rdir.glob("*.pdf"),
                       key=lambda p: p.stat().st_mtime, reverse=True) \
            if _rdir.exists() else []
        if _pdfs:
            st.markdown("**Saved reports**")
            for _i, _pdf in enumerate(_pdfs[:12]):
                _ts = _dt.datetime.fromtimestamp(_pdf.stat().st_mtime)
                _nc, _rc, _oc, _dc = st.columns([3, 1, 1, 1])
                _nc.caption(f"{_pdf.name}  ·  {_ts.strftime('%Y-%m-%d %H:%M')}")
                _rk = f"rep_read_{_i}"
                if _rc.button("Read", key=f"rep_readbtn_{_i}",
                              use_container_width=True):
                    st.session_state[_rk] = not st.session_state.get(_rk, False)
                if _oc.button("Open", key=f"rep_open_{_i}",
                              use_container_width=True):
                    try:
                        _os.startfile(str(_pdf))  # noqa: B606 (Windows-only)
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"Open failed ({_oe}); use Download.")
                try:
                    with open(_pdf, "rb") as _fh:
                        _dc.download_button(
                            "⬇", data=_fh.read(), file_name=_pdf.name,
                            mime="application/pdf", key=f"rep_dl_{_i}",
                            use_container_width=True)
                except Exception as _de:  # noqa: BLE001
                    _dc.caption("dl?")
                # Inline reader: show the HTML sidecar content.
                if st.session_state.get(_rk, False):
                    _html = _pdf.with_suffix(".html")
                    if _html.exists():
                        try:
                            _components.html(_html.read_text(encoding="utf-8"),
                                             height=520, scrolling=True)
                        except Exception as _he:  # noqa: BLE001
                            st.caption(f"couldn't render ({_he}).")
                    else:
                        st.caption("No inline text for this report (it predates "
                                   "the in-app reader). Regenerate it with "
                                   "`report <ticker>` to enable Read.")
        else:
            st.caption("No reports saved yet — run `report <ticker>` to make one.")
    except Exception as _pe:  # noqa: BLE001
        st.caption(f"saved-reports list unavailable: {_pe}")

    for j in jobs:
        _render_job_row(j)


def _render_job_row(job: S.Job) -> None:
    if job.state == S.JobState.RUNNING:
        glyph, color, status = "⏳", "#3b82f6", f"running · {job.elapsed():.0f}s"
    elif job.state == S.JobState.QUEUED:
        glyph, color, status = "•", C._NEUTRAL_TEXT, "queued"
    elif job.state == S.JobState.DONE:
        glyph, color, status = "✓", C.RISK_COLORS["low"], f"done · {job.elapsed():.0f}s"
    else:  # ERROR
        glyph, color, status = "✕", C.RISK_COLORS["high"], "error"

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;padding:4px 0;"
        f"border-bottom:1px solid {C._HAIRLINE};'>"
        f"<span style='color:{color};font-size:1.05em;'>{glyph}</span>"
        f"<span style='font-size:0.88em;font-weight:600;min-width:140px;'>"
        f"{job.kind} · {job.ticker}</span>"
        f"<span style='font-size:0.82em;color:{color};'>{status}</span></div>",
        unsafe_allow_html=True)
    if job.state == S.JobState.ERROR and job.error:
        st.caption(f"↳ {job.error[:200]}")

    # Finished report jobs expose the PDF path as job.result — offer it.
    if job.state == S.JobState.DONE and job.kind == "report":
        _pdf = getattr(job, "result", None)
        try:
            from pathlib import Path as _P
            if _pdf and str(_pdf).lower().endswith(".pdf") and _P(_pdf).exists():
                _p = _P(_pdf)
                with open(_p, "rb") as _fh:
                    st.download_button(
                        label=f"⬇ Download {_p.name}",
                        data=_fh.read(),
                        file_name=_p.name,
                        mime="application/pdf",
                        key=f"dl_{job.kind}_{job.ticker}_{_p.name}",
                        use_container_width=False,
                    )
                # Open buttons — work because Streamlit runs on this machine.
                import os as _os
                _oc1, _oc2, _oc3 = st.columns([1, 1, 3])
                if _oc1.button("Open PDF",
                               key=f"open_{job.kind}_{job.ticker}_{_p.name}"):
                    try:
                        _os.startfile(str(_p))  # noqa: B606 (Windows-only, local)
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"could not open ({_oe}); use Download.")
                if _oc2.button("Open folder",
                               key=f"openf_{job.kind}_{job.ticker}_{_p.name}"):
                    try:
                        _os.startfile(str(_p.parent))  # noqa: B606
                    except Exception as _oe:  # noqa: BLE001
                        st.caption(f"could not open folder ({_oe}).")
                st.caption(f"↳ saved to {_p}")
            elif _pdf:
                st.caption(f"↳ report result: {_pdf}")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"↳ report ready but link failed: {_e}")


# ======================================================================
# Finished-job side effects (cache invalidation)
# ======================================================================
_APPLIED_KEY = "_ary_applied_job_ids"


def _apply_finished_side_effects(finished: list[S.Job]) -> None:
    """Clear cached board/opinion reads once per newly-finished job.

    Without this, a finished ``gen`` job wouldn't show up until the user hit
    a manual refresh, because the board/desk reads are cached. We track which
    job ids we've already reacted to in session state so this fires exactly
    once per job.
    """
    applied: set[str] = st.session_state.setdefault(_APPLIED_KEY, set())
    newly = [j for j in finished if j.id not in applied]
    if not newly:
        return
    # Invalidate the cheap cached readers used by the board (and indirectly
    # the desk via the app's own cached loaders, which the orchestrator
    # clears in its own completion hook).
    try:
        from ui import board as _board
        _board._load_latest_opinions.clear()
        _board._load_positions.clear()
    except Exception:
        pass
    for j in newly:
        applied.add(j.id)


__all__ = [
    "Action", "Command", "parse_command",
    "render_command_bar", "render_help", "inject_shortcut_hint",
    "render_job_tray",
]

# D:\Ary Fund\ui\palette.py
