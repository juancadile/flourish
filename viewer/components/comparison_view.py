"""Model comparison component."""

import streamlit as st
import pandas as pd


def render_comparison_view(df: pd.DataFrame, data_loader) -> None:
    """
    Render the comparison view for comparing models on the same scenario.

    Args:
        df: Filtered DataFrame.
        data_loader: FlourishDataLoader instance.
    """
    st.header("Model Comparison")
    st.markdown("Compare how different models performed on the same scenario.")

    if df is None or len(df) == 0:
        st.info("No data available for comparison.")
        return

    # Select scenario
    scenarios = sorted(df["scenario_id"].unique().tolist())

    if not scenarios:
        st.warning("No scenarios available in the filtered data.")
        return

    selected_scenario = st.selectbox(
        "Select a scenario to compare:",
        options=scenarios,
        help="Choose a scenario to see how different models responded",
    )

    # Get all responses for this scenario
    scenario_df = data_loader.get_scenario_comparison(selected_scenario)

    # Filter by current filters (virtues, models, etc.)
    if "virtue" in df.columns:
        virtues = df["virtue"].unique()
        scenario_df = scenario_df[scenario_df["virtue"].isin(virtues)]

    if "model" in df.columns:
        models = df["model"].unique()
        scenario_df = scenario_df[scenario_df["model"].isin(models)]

    if len(scenario_df) == 0:
        st.warning("No models found for this scenario with current filters.")
        return

    # Show scenario details
    st.subheader(f"Scenario: {selected_scenario}")

    first_row = scenario_df.iloc[0]
    st.markdown(f"**Virtue:** {first_row['virtue']}")

    with st.expander("**View Prompt**", expanded=False):
        st.markdown(first_row["prompt"])

    st.divider()

    # Display each model's response
    st.subheader(f"Model Responses ({len(scenario_df)} models)")

    for idx, (_, row) in enumerate(scenario_df.iterrows(), 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {idx}. {row['model']}")

            with col2:
                score = row["score"]
                if pd.notna(score):
                    score_int = int(score)
                    if score_int == 0:
                        st.markdown(":red[**0/2**]")
                    elif score_int == 1:
                        st.markdown(":orange[**1/2**]")
                    else:
                        st.markdown(":green[**2/2**]")
                else:
                    st.markdown("**N/A**")

            # Response
            st.markdown("**Response:**")
            st.markdown(row["response"])

            # Judge reasoning
            if "justification" in row.index and row["justification"]:
                with st.expander("**Judge Reasoning**"):
                    st.markdown(row["justification"])

            st.divider()

    # Summary statistics for this scenario
    st.subheader("Score Summary")

    valid_scores = scenario_df[scenario_df["score"].notna()]["score"]

    if len(valid_scores) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average Score", f"{valid_scores.mean():.2f}/2.00")

        with col2:
            st.metric("Score 0", int((valid_scores == 0).sum()))

        with col3:
            st.metric("Score 1", int((valid_scores == 1).sum()))

        with col4:
            st.metric("Score 2", int((valid_scores == 2).sum()))
