"""
main.py
=======

Master agent loop for the hedge-fund AI research system.

Usage
-----
Run with the default watchlist from :mod:`config`::

    python3 main.py

Or override the ticker list from the CLI::

    python3 main.py --tickers AAPL MSFT NVDA

Flow per ticker
---------------
For every ticker, ``main`` does the following in order:

    1. ``pipeline.run_daily_refresh(tickers, db_path, config)`` — once, up
       front. Ingests SEC filings, market data, and macro series.
    2. ``pipeline.build_agent_context(ticker, db_path, config)``.
    3. ``filing_analyzer.summarize_filings_by_year(...)``.
    4. ``filing_analyzer.extract_key_metrics_for_agent(...)``.
    5. ``base_agent.ask_agent(AgentRequest(...))`` — currently mock.
    6. ``risk_scanner.compute_risk_flags(...)`` using the agent's risk list.
    7. ``thesis_generator.generate_thesis(...)``.
    8. ``portfolio_db.save_agent_opinion(ticker, final_opinion, db_path=db_path)``.

Errors on any single ticker are logged and skipped; the batch continues.
A compact summary table is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from data import pipeline, portfolio_db
from agent import base_agent, filing_analyzer, risk_scanner, thesis_generator
from agent.base_agent import AgentRequest


logger = logging.getLogger("hedgefund_ai.main")


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_agent_prompt(
    ticker: str,
    context: Dict[str, Any],
    filings_summary: Dict[str, Any],
    key_metrics: Dict[str, Any],
) -> str:
    """Render the agent prompt from structured context.

    The mock backend ignores prompt content (returns a fixed JSON), but a
    real Phi-3 / Qwen model will read this. Keep it compact, structured, and
    JSON-instructional so the backend's ``format=json`` path works cleanly.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    context:
        Full dict from ``pipeline.build_agent_context``. We include only the
        macro slice here since fundamentals come via ``key_metrics``.
    filings_summary:
        Output of ``filing_analyzer.summarize_filings_by_year``.
    key_metrics:
        Output of ``filing_analyzer.extract_key_metrics_for_agent``.
    """
    macro = context.get("macro") or {}

    payload = {
        "ticker": ticker,
        "filings": {
            "management_tone": filings_summary.get("management_tone"),
            "red_flags": filings_summary.get("red_flags", []),
            "risk_factors_sample": (filings_summary.get("risk_factors") or [])[:8],
            "summary": filings_summary.get("summary"),
        },
        "metrics": key_metrics,
        "macro": {
            "recession_probability": macro.get("recession_probability"),
            "vix": macro.get("vix"),
            "yield_curve_spread": macro.get("yield_curve_spread"),
            "yield_curve_inverted": macro.get("yield_curve_inverted"),
        },
    }

    return (
        "You are a hedge-fund sell-side analyst. Analyse the ticker below and "
        "return ONLY a JSON object with the following keys:\n"
        '  "risks": list of strings, each prefixed with "HIGH:", "MEDIUM:", '
        'or "LOW:" and a short phrase.\n'
        '  "thesis": short string like "BULLISH 1Y", "NEUTRAL 1Y", "BEARISH 1Y".\n'
        '  "price_direction": one of "strong_up", "moderate_up", "neutral", '
        '"moderate_down", "strong_down".\n'
        '  "confidence": float in [0.0, 1.0].\n\n'
        "Do not include any prose outside the JSON object.\n\n"
        "CONTEXT:\n"
        f"{json.dumps(payload, default=str, indent=2)}"
    )


# =============================================================================
# RESULT ASSEMBLY
# =============================================================================

def _assemble_final_opinion(
    ticker: str,
    agent_response: Any,
    filings_summary: Dict[str, Any],
    key_metrics: Dict[str, Any],
    risk_flags: Dict[str, Any],
    thesis: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge all layer outputs into a single persistable opinion dict.

    This is the shape passed to ``portfolio_db.save_agent_opinion``. Keep
    it flat-ish and JSON-serialisable so the DB layer can store it as a
    blob without further massaging.
    """
    return {
        "ticker": ticker,
        "as_of": datetime.now(timezone.utc).isoformat(),
        # Thesis is the headline; splat its keys so downstream UIs can read
        # them without drilling into a nested dict.
        "outlook": thesis.get("outlook"),
        "time_horizon": thesis.get("time_horizon", "1Y"),
        "price_direction": thesis.get("price_direction"),
        "confidence": thesis.get("confidence"),
        "key_risks": thesis.get("key_risks", []),
        "key_opportunities": thesis.get("key_opportunities", []),
        "rationale": thesis.get("rationale"),
        "bias_score": thesis.get("bias_score"),

        # Essay and review (new).
        "essay": thesis.get("essay"),
        "essay_meta": thesis.get("essay_meta"),
        "review": thesis.get("review"),
        "essay_revised": thesis.get("essay_revised", False),

        # Nested supporting detail.
        "risk_flags": risk_flags,
        "key_metrics": key_metrics,
        "filings_summary": {
            # Drop the potentially huge by_year blob here; keep what's useful.
            "management_tone": filings_summary.get("management_tone"),
            "red_flags": filings_summary.get("red_flags", []),
            "filings_considered": filings_summary.get("filings_considered", 0),
            "summary": filings_summary.get("summary"),
        },
        "agent_raw": {
            "model_used": getattr(agent_response, "model_used", None),
            "generated_json": getattr(agent_response, "generated_json", {}),
            "tokens_in": getattr(agent_response, "tokens_in", 0),
            "tokens_out": getattr(agent_response, "tokens_out", 0),
            "elapsed_ms": getattr(agent_response, "elapsed_ms", 0.0),
        },
    }


# =============================================================================
# PER-TICKER WORKER
# =============================================================================

def _process_ticker(
    ticker: str,
    db_path: str,
    cfg: Any,
) -> Optional[Dict[str, Any]]:
    """Run the full agent chain for one ticker. Returns the final opinion
    dict on success, ``None`` on failure.

    Exceptions are caught and logged; one bad ticker must never kill a batch.
    """
    try:
        # 1. Build context from the freshly-refreshed data layer.
        context = pipeline.build_agent_context(ticker, db_path, cfg)
        if not context:
            logger.warning("main | %s | empty context; skipping", ticker)
            return None

        filings = context.get("filings") or []
        metrics_raw = context.get("metrics") or {}
        macro = context.get("macro") or {}
        prices = context.get("prices") or {}
        # Latest price may be supplied as a scalar or under 'close' / 'last'.
        price = (
            prices.get("last")
            or prices.get("close")
            or prices.get("price")
            or 0.0
        )

        # 2. Shape filings & metrics for the agent.
        max_filings = getattr(cfg, "MAX_FILINGS_PER_TICKER", 10)
        filings_summary = filing_analyzer.summarize_filings_by_year(
            ticker, filings, max_filings=max_filings
        )
        key_metrics = filing_analyzer.extract_key_metrics_for_agent(
            ticker, metrics_raw, float(price or 0.0)
        )

        # 3. Ask the agent (mock today, Phi-3 later).
        prompt = build_agent_prompt(ticker, context, filings_summary, key_metrics)
        request = AgentRequest(
            prompt=prompt,
            context=context,
            tools=["filings", "prices", "macro"],
        )
        response = base_agent.ask_agent(request, cfg)
        agent_risks: List[str] = list(response.generated_json.get("risks", []))

        # 4. Rule-based risk flags (reads the agent's risks).
        risk_flags = risk_scanner.compute_risk_flags(
            ticker=ticker,
            metrics=key_metrics,
            macro=macro,
            agent_risks=agent_risks,
            config=cfg,
        )

        # 5. Heuristic thesis (shape-compatible with future LLM version).
        thesis = thesis_generator.generate_thesis(
            ticker=ticker,
            filings_summary=filings_summary,
            metrics=key_metrics,
            macro=macro,
            risk_flags=risk_flags,
        )

        # 5b. Generate the institutional-memo essay and review it.
        from agent.thesis_essay import generate_thesis_essay
        from agent.thesis_review import review_essay, revise_essay

        essay = generate_thesis_essay(
            ticker=ticker,
            thesis=thesis,
            filings_summary=filings_summary,
            metrics=key_metrics,
            macro=macro,
            risk_flags=risk_flags,
            config=cfg,
        )
        thesis["essay"] = essay["text"]
        thesis["essay_meta"] = {
            "model": essay["model_used"],
            "elapsed_ms": essay["elapsed_ms"],
            "fallback": essay["fallback"],
            "word_count": essay["word_count"],
        }

        # Optional: review the essay and auto-revise if score is low
        review = review_essay(
            ticker=ticker,
            essay_text=essay["text"],
            metrics=key_metrics,
            macro=macro,
            risk_flags=risk_flags,
            config=cfg,
        )
        thesis["review"] = review

        # Auto-revise if the review score is below 8.0 AND we're not in fallback mode
        # (fallback mode means LLM is unavailable, so revision would also fail)
        if review["overall_score"] < 8.0 and not review["fallback"]:
            revised = revise_essay(
                ticker=ticker,
                original_essay=essay["text"],
                review_text=review["text"],
                metrics=key_metrics,
                macro=macro,
                risk_flags=risk_flags,
                config=cfg,
            )
            if revised["revision_applied"]:
                thesis["essay"] = revised["text"]
                thesis["essay_revised"] = True
                logger.info(
                    "main | %s | essay revised | score %.1f -> revision applied",
                    ticker,
                    review["overall_score"],
                )

        # 6. Merge into one persistable opinion.
        final_opinion = _assemble_final_opinion(
            ticker=ticker,
            agent_response=response,
            filings_summary=filings_summary,
            key_metrics=key_metrics,
            risk_flags=risk_flags,
            thesis=thesis,
        )

        # 7. Persist.
        try:
            portfolio_db.save_agent_opinion(ticker, final_opinion, db_path=db_path)
        except Exception as db_exc:  # noqa: BLE001
            logger.error(
                "main | %s | save_agent_opinion failed: %s",
                ticker,
                db_exc,
            )
            # Return the opinion anyway so the summary prints. The DB failure
            # is logged for follow-up but should not hide the analysis.

        return final_opinion

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "main | %s | ticker failed: %s\n%s",
            ticker,
            exc,
            traceback.format_exc(),
        )
        return None


# =============================================================================
# SUMMARY PRINTER
# =============================================================================

def _print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a compact table of per-ticker results.

    Columns: ticker | combined_risk | outlook | price_direction | confidence.
    """
    print()
    print("=" * 78)
    print(f"{'TICKER':<8} {'RISK':<8} {'OUTLOOK':<10} {'DIRECTION':<16} {'CONF':<6}")
    print("-" * 78)

    if not results:
        print("(no results)")
        print("=" * 78)
        return

    for r in results:
        levels = (r.get("risk_flags") or {}).get("levels", {})
        combined = levels.get("combined", "?")
        outlook = r.get("outlook") or "?"
        direction = r.get("price_direction") or "?"
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        print(
            f"{r.get('ticker', '?'):<8} "
            f"{combined:<8} "
            f"{outlook:<10} "
            f"{direction:<16} "
            f"{conf_str:<6}"
        )
    print("=" * 78)
    print(f"Processed {len(results)} ticker(s).")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(
    tickers: List[str],
    db_path: str,
    cfg: Any,
) -> List[Dict[str, Any]]:
    """Run the full agent workflow for the given tickers.

    Parameters
    ----------
    tickers:
        Universe to process.
    db_path:
        Portfolio SQLite path (usually ``cfg.PORTFOLIO_DB_PATH``).
    cfg:
        The project's ``config`` module.

    Returns
    -------
    list of dict
        Final opinion dicts for tickers that processed successfully.
    """
    if not tickers:
        logger.warning("main | no tickers provided; nothing to do")
        return []

    logger.info(
        "main | starting run | tickers=%s | model=%s",
        tickers,
        getattr(cfg, "DEFAULT_AGENT_MODEL", "?"),
    )

    # Data refresh is a single batched call; it's the expensive step.
    try:
        pipeline.run_daily_refresh(tickers, db_path, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "main | run_daily_refresh failed: %s. "
            "Continuing with cached data.\n%s",
            exc,
            traceback.format_exc(),
        )

    results: List[Dict[str, Any]] = []
    for ticker in tickers:
        opinion = _process_ticker(ticker, db_path, cfg)
        if opinion is not None:
            results.append(opinion)

    _print_summary(results)
    logger.info(
        "main | run complete | processed=%d / %d",
        len(results),
        len(tickers),
    )
    return results


# =============================================================================
# CLI WIRING
# =============================================================================

def _configure_logging(cfg: Any) -> None:
    """Set up root logging from config. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. running under pytest / notebook).

    level = getattr(cfg, "LOG_LEVEL", logging.INFO)
    fmt = getattr(cfg, "LOG_FORMAT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log_file = getattr(cfg, "LOG_FILE", None)

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        try:
            handlers.append(logging.FileHandler(log_file))
        except OSError as exc:
            # Don't kill the run if the log file can't be opened (perms,
            # disk full, missing dir). Stdout handler is enough.
            print(f"warning: could not open log file {log_file}: {exc}", file=sys.stderr)

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hedge-fund AI agent workflow over a ticker universe.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers to process (default: config.WATCHLIST).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to portfolio SQLite DB (default: config.PORTFOLIO_DB_PATH).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override config.DEFAULT_AGENT_MODEL for this run "
             "(e.g. 'mock', 'dev', 'prod').",
    )
    return parser.parse_args(argv)


def _cli_entry() -> int:
    """CLI wrapper. Returns an exit code."""
    args = _parse_args()
    _configure_logging(config)

    # Optional one-run override of the default model.
    if args.model is not None:
        logger.info("main | overriding DEFAULT_AGENT_MODEL -> %r", args.model)
        config.DEFAULT_AGENT_MODEL = args.model

    tickers = args.tickers or getattr(config, "WATCHLIST", [])
    db_path = args.db_path or getattr(config, "PORTFOLIO_DB_PATH", "portfolio.db")

    try:
        main(tickers, db_path, config)
        return 0
    except KeyboardInterrupt:
        logger.warning("main | interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("main | fatal error: %s\n%s", exc, traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(_cli_entry())