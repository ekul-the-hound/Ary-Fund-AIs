"""
ui/data_point_analyzer_section.py
==================================

Self-contained Streamlit section for the per-data-point analyzer.

This module exports a single function, ``render_data_point_analyzer_section``,
that the existing ``ui/app.py`` (or ``app.py``) can call to render the entire
interactive analyzer panel. It does not modify global Streamlit state outside
its own widgets.

Integration
-----------
In your existing app.py, after the agent context has been built, add::

    from ui.data_point_analyzer_section import render_data_point_analyzer_section

    render_data_point_analyzer_section(
        ticker=ticker,
        context=ctx,         # output of build_agent_context()
        config=config,       # the imported config module
    )

That's it. The section renders its own header, checkboxes, button, and output.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


# ----------------------------------------------------------------------
# Defensive import — if the analyzer module is missing, render an error
# banner instead of crashing the whole dashboard.
# ----------------------------------------------------------------------
try:
    from agent.data_point_analyzer import (
        AVAILABLE_DATA_POINTS,
        analyze_data_points,
        get_categories,
        get_display_name,
        get_formatted_value,
    )
    _ANALYZER_AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - defensive
    _ANALYZER_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def render_data_point_analyzer_section(
    ticker: str,
    context: Dict[str, Any],
    config: Any,
) -> None:
    """Render the full analyzer section for the given ticker.

    The section includes:
      - A category-grouped checkbox grid for selecting data points.
      - A "Generate Analysis" button.
      - The rendered overview + per-paragraph output (with collapsible expanders).
      - Metadata (model used, elapsed time, fallback status, word count).

    All UI state is keyed by the ticker so switching tickers does not
    leak selections between them.
    """
    st.markdown("---")
    st.subheader("📝 Data-Point Investment Analysis")
    st.caption(
        "Pick the data points you want analyzed. The model writes one overview "
        "paragraph plus one paragraph per selected point (150-200 words each). "
        "No extra API calls are made — values come from the loaded context."
    )

    if not _ANALYZER_AVAILABLE:
        st.error(
            f"Data-point analyzer module is not available: {_IMPORT_ERROR}. "
            "Place `agent/data_point_analyzer.py` in your project."
        )
        return

    if not context:
        st.warning("No agent context loaded — generate one for this ticker first.")
        return

    # --- Selection UI ----------------------------------------------------
    state_key = f"dpa_selected_{ticker}"
    if state_key not in st.session_state:
        # Sensible default: pre-select 4 high-signal points if they exist
        defaults = [
            "prices.last",
            "metrics.trailing_pe",
            "metrics.freeCashflow",
            "macro.financial_conditions.vix",
        ]
        st.session_state[state_key] = [
            k for k in defaults if k in AVAILABLE_DATA_POINTS
        ]

    selected: List[str] = list(st.session_state[state_key])
    categories = get_categories()

    with st.expander("Select data points to analyze", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select all", key=f"dpa_all_{ticker}"):
                selected = list(AVAILABLE_DATA_POINTS.keys())
                st.session_state[state_key] = selected
                st.rerun()
        with col_b:
            if st.button("Clear all", key=f"dpa_clear_{ticker}"):
                selected = []
                st.session_state[state_key] = selected
                st.rerun()

        # Render category-grouped checkboxes in a 3-column grid for compactness
        category_names = list(categories.keys())
        n_cols = 3
        cols = st.columns(n_cols)
        for idx, cat_name in enumerate(category_names):
            target_col = cols[idx % n_cols]
            with target_col:
                st.markdown(f"**{cat_name}**")
                for key in categories[cat_name]:
                    display = get_display_name(key)
                    formatted_value = get_formatted_value(context, key)
                    label = f"{display} ({formatted_value})"
                    checkbox_key = f"dpa_cb_{ticker}_{key}"
                    is_checked = st.checkbox(
                        label,
                        value=(key in selected),
                        key=checkbox_key,
                    )
                    if is_checked and key not in selected:
                        selected.append(key)
                    elif not is_checked and key in selected:
                        selected.remove(key)
                st.markdown("")  # spacing

    st.session_state[state_key] = selected

    # --- Status row ------------------------------------------------------
    n_selected = len(selected)
    total_paragraphs = 1 + n_selected  # overview + per-point
    estimated_words = total_paragraphs * 175  # midpoint of 150-200
    st.info(
        f"**{n_selected}** data point{'s' if n_selected != 1 else ''} selected. "
        f"Output will be **{total_paragraphs} paragraphs** "
        f"(~{estimated_words} words)."
    )

    # --- Generate button -------------------------------------------------
    generate = st.button(
        "🔍 Generate Analysis",
        type="primary",
        disabled=(n_selected == 0),
        key=f"dpa_go_{ticker}",
    )

    result_key = f"dpa_result_{ticker}"

    if generate:
        with st.spinner(
            f"Generating {total_paragraphs}-paragraph analysis for {ticker}... "
            "(this may take 30-180 seconds depending on model and selection size)"
        ):
            try:
                result = analyze_data_points(
                    ticker=ticker,
                    selected_keys=selected,
                    context=context,
                    config=config,
                )
                st.session_state[result_key] = result
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

    # --- Render result ---------------------------------------------------
    result = st.session_state.get(result_key)
    if not result:
        return

    # Metadata row
    meta_cols = st.columns(4)
    meta_cols[0].metric("Model", result.get("model_used", "n/a"))
    meta_cols[1].metric("Word count", f"{result.get('word_count', 0):,}")
    meta_cols[2].metric("Elapsed", f"{result.get('elapsed_ms', 0) / 1000:.1f}s")
    fallback_label = "⚠ Fallback" if result.get("fallback") else "✓ LLM"
    meta_cols[3].metric("Source", fallback_label)

    if result.get("fallback"):
        st.warning(
            "This output was produced by the deterministic fallback because "
            "the LLM was unavailable or returned an unusable response. "
            "Connect to Ollama and retry to get the full analysis."
        )

    # Overview
    overview = result.get("overview", "")
    if overview:
        st.markdown("### 🎯 Overview")
        st.write(overview)
    elif result.get("text"):
        # Parser couldn't split — render the raw text
        st.markdown("### 📄 Analysis")
        st.write(result["text"])
        return

    # Per-point paragraphs in expanders
    paragraphs = result.get("paragraphs", {}) or {}
    selected_keys = result.get("selected_keys", selected)

    if paragraphs:
        st.markdown("### 📊 Data-Point Analysis")
        for key in selected_keys:
            display = get_display_name(key)
            formatted_value = get_formatted_value(context, key)
            paragraph = paragraphs.get(key)
            with st.expander(f"**{display}** — {formatted_value}", expanded=False):
                if paragraph:
                    st.write(paragraph)
                else:
                    st.caption(
                        "(The model did not produce a clearly-delimited paragraph "
                        "for this data point. See the raw output below.)"
                    )

    # Raw output (collapsed by default for debugging)
    with st.expander("Show raw model output", expanded=False):
        st.code(result.get("text", ""), language="markdown")
