"""Sidebar filters component."""

import streamlit as st
from typing import Optional


def render_filters(data_loader) -> dict:
    """
    Render the sidebar filters.

    Args:
        data_loader: FlourishDataLoader instance.

    Returns:
        Dictionary with filter values.
    """
    st.sidebar.header("Filters")

    filters = {}

    # Virtue filter
    all_virtues = data_loader.get_unique_virtues()
    if all_virtues:
        selected_virtues = st.sidebar.multiselect(
            "Virtues:",
            options=all_virtues,
            default=all_virtues,
            help="Select one or more virtues to display",
        )
        filters["virtues"] = selected_virtues if selected_virtues else None
    else:
        filters["virtues"] = None

    # Model filter
    all_models = data_loader.get_unique_models()
    if all_models:
        selected_models = st.sidebar.multiselect(
            "Models:",
            options=all_models,
            default=all_models,
            help="Select one or more models to display",
        )
        filters["models"] = selected_models if selected_models else None
    else:
        filters["models"] = None

    # Score range filter
    score_range = st.sidebar.slider(
        "Score Range:",
        min_value=0.0,
        max_value=2.0,
        value=(0.0, 2.0),
        step=1.0,
        help="Filter transcripts by score",
    )
    filters["score_range"] = score_range

    # Search filter
    search_query = st.sidebar.text_input(
        "Search:",
        placeholder="Search in prompts and responses...",
        help="Search for text in prompts, responses, or justifications",
    )
    filters["search_query"] = search_query if search_query else None

    # Exclude errors
    exclude_errors = st.sidebar.checkbox(
        "Exclude errors",
        value=True,
        help="Hide transcripts with evaluation errors",
    )
    filters["exclude_errors"] = exclude_errors

    # Reset button
    if st.sidebar.button("Reset Filters"):
        st.rerun()

    return filters


def render_summary_stats(stats: dict) -> None:
    """
    Render summary statistics in the sidebar.

    Args:
        stats: Dictionary with summary statistics.
    """
    st.sidebar.divider()
    st.sidebar.header("Summary")

    st.sidebar.metric("Total Transcripts", stats["total_transcripts"])
    st.sidebar.metric("Average Score", f"{stats['avg_score']:.2f}/2.00")

    st.sidebar.markdown("**Score Distribution:**")
    st.sidebar.markdown(f"- Score 0: {stats['score_distribution'][0]}")
    st.sidebar.markdown(f"- Score 1: {stats['score_distribution'][1]}")
    st.sidebar.markdown(f"- Score 2: {stats['score_distribution'][2]}")
