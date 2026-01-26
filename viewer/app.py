"""Flourish Transcript Viewer - Main Application

Interactive viewer for Flourish evaluation results.
Inspired by Bloom's transcript viewer, adapted for Flourish's curated evaluation approach.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from viewer.data_loader import FlourishDataLoader
from viewer.components.filters import render_filters, render_summary_stats
from viewer.components.transcript_view import render_transcripts_view
from viewer.components.comparison_view import render_comparison_view
from viewer.components.analytics import render_analytics_view


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="Flourish Transcript Viewer",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("Flourish Transcript Viewer")
    st.markdown(
        "Interactive viewer for Flourish behavioral evaluation results. "
        "Filter, compare, and analyze model responses across virtue scenarios."
    )

    # Initialize data loader in session state
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = FlourishDataLoader()
        st.session_state.df = None

    data_loader = st.session_state.data_loader

    # File upload
    st.sidebar.header("Load Data")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Flourish Results CSV",
        type=["csv"],
        help="Upload a CSV file exported from Flourish evaluations",
    )

    # Or load from path
    csv_path = st.sidebar.text_input(
        "Or enter CSV path:",
        placeholder="results/detailed_20260126.csv",
        help="Path to a local CSV file",
    )

    load_button = st.sidebar.button("Load CSV")

    # Load data
    if uploaded_file is not None:
        try:
            data_loader.load_csv(uploaded_file)
            st.session_state.df = data_loader.df
            st.sidebar.success(f"Loaded {len(data_loader.df)} transcripts")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    elif load_button and csv_path:
        try:
            data_loader.load_csv(csv_path)
            st.session_state.df = data_loader.df
            st.sidebar.success(f"Loaded {len(data_loader.df)} transcripts")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # Check if data is loaded
    if data_loader.df is None or len(data_loader.df) == 0:
        st.info(
            "Please upload a Flourish results CSV file or enter a path to get started."
        )
        st.markdown("### Example Usage")
        st.code(
            """
# Run evaluations first
flourish --model claude-sonnet-4 --eval evals/empathy_in_action.yaml

# Then view the results
streamlit run viewer/app.py
            """,
            language="bash",
        )
        return

    # Render filters
    filters = render_filters(data_loader)

    # Apply filters
    filtered_df = data_loader.filter_data(**filters)

    # Show summary stats in sidebar
    stats = data_loader.get_summary_stats(filtered_df)
    render_summary_stats(stats)

    # Main content area
    st.divider()

    # View mode selector
    view_mode = st.radio(
        "View Mode:",
        options=["Transcripts", "Comparison", "Analytics"],
        horizontal=True,
        help="Choose how to view the data",
    )

    st.divider()

    # Render selected view
    if view_mode == "Transcripts":
        render_transcripts_view(filtered_df)

    elif view_mode == "Comparison":
        render_comparison_view(filtered_df, data_loader)

    elif view_mode == "Analytics":
        render_analytics_view(filtered_df, data_loader)

    # Export functionality
    st.sidebar.divider()
    st.sidebar.header("Export")

    if st.sidebar.button("Export Filtered Data"):
        if filtered_df is not None and len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name="flourish_filtered_results.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.warning("No data to export")

    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Flourish Transcript Viewer | "
        "Inspired by [Bloom](https://github.com/safety-research/bloom)"
    )


if __name__ == "__main__":
    main()
