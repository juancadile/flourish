"""Transcript display component."""

import streamlit as st
import pandas as pd


def render_score_badge(score: float) -> str:
    """
    Render a colored score badge.

    Args:
        score: Score value (0-2).

    Returns:
        Formatted string for the badge.
    """
    if pd.isna(score):
        return "**N/A**"

    score = int(score)
    badges = {
        0: ":red[**0/2** - No Evidence]",
        1: ":orange[**1/2** - Partial Evidence]",
        2: ":green[**2/2** - Full Evidence]",
    }
    return badges.get(score, f"**{score}/2**")


def render_transcript(row: pd.Series, index: int, total: int) -> None:
    """
    Render a single transcript.

    Args:
        row: DataFrame row containing transcript data.
        index: Current transcript index (1-based).
        total: Total number of transcripts.
    """
    st.markdown(f"### Transcript {index}/{total}")

    # Header with scenario ID, virtue, and score
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"**Scenario:** `{row['scenario_id']}`")

    with col2:
        st.markdown(f"**Virtue:** {row['virtue']}")

    with col3:
        st.markdown(render_score_badge(row["score"]))

    # Model info
    st.markdown(f"**Model:** {row['model']}")
    if "judge" in row.index:
        st.markdown(f"**Judge:** {row['judge']}")

    st.divider()

    # Prompt (collapsible)
    with st.expander("**Prompt**", expanded=False):
        st.markdown(row["prompt"])

    # Response
    st.markdown("#### Model Response")
    st.markdown(row["response"])

    # Judge reasoning
    if "justification" in row.index and row["justification"]:
        st.markdown("#### Judge Reasoning")
        st.markdown(row["justification"])

    # Summary (if available from multi-stage judge)
    if "summary" in row.index and row["summary"]:
        with st.expander("**Judge Summary**", expanded=False):
            st.markdown(row["summary"])

    # Highlights (if available from multi-stage judge)
    if "highlights" in row.index and row["highlights"]:
        with st.expander("**Key Highlights**", expanded=False):
            if isinstance(row["highlights"], list):
                for i, highlight in enumerate(row["highlights"], 1):
                    if isinstance(highlight, dict):
                        desc = highlight.get("description", "")
                        parts = highlight.get("parts", [])
                        st.markdown(f"**{i}.** {desc}")
                        for part in parts:
                            if isinstance(part, dict):
                                quoted_text = part.get("quoted_text", "")
                                st.markdown(f"> {quoted_text}")
                    else:
                        st.markdown(f"- {highlight}")
            else:
                st.markdown(row["highlights"])

    # Additional metadata
    if "timestamp" in row.index:
        st.caption(f"Timestamp: {row['timestamp']}")

    # Error indicator
    if "error" in row.index and row["error"]:
        st.error("Warning: This evaluation encountered an error")


def render_transcripts_view(df: pd.DataFrame) -> None:
    """
    Render the main transcripts view with pagination.

    Args:
        df: Filtered DataFrame to display.
    """
    if df is None or len(df) == 0:
        st.info("No transcripts match the current filters.")
        return

    # Pagination
    transcripts_per_page = st.selectbox(
        "Transcripts per page:",
        options=[10, 25, 50, 100],
        index=0,
    )

    total_pages = max(1, (len(df) - 1) // transcripts_per_page + 1)

    # Initialize page in session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Page navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("First"):
            st.session_state.current_page = 1

    with col2:
        if st.button("Previous"):
            st.session_state.current_page = max(1, st.session_state.current_page - 1)

    with col3:
        page_num = st.number_input(
            f"Page (of {total_pages})",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page,
            key="page_input",
        )
        if page_num != st.session_state.current_page:
            st.session_state.current_page = page_num

    with col4:
        if st.button("Next"):
            st.session_state.current_page = min(
                total_pages, st.session_state.current_page + 1
            )

    with col5:
        if st.button("Last"):
            st.session_state.current_page = total_pages

    st.markdown(f"Showing page {st.session_state.current_page} of {total_pages}")

    # Calculate slice
    start_idx = (st.session_state.current_page - 1) * transcripts_per_page
    end_idx = min(start_idx + transcripts_per_page, len(df))

    # Display transcripts
    for i, (_, row) in enumerate(df.iloc[start_idx:end_idx].iterrows(), start=start_idx + 1):
        render_transcript(row, i, len(df))
        st.divider()
