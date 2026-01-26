"""Analytics and visualization component."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_analytics_view(df: pd.DataFrame, data_loader) -> None:
    """
    Render the analytics dashboard with visualizations.

    Args:
        df: Filtered DataFrame.
        data_loader: FlourishDataLoader instance.
    """
    st.header("Analytics Dashboard")

    if df is None or len(df) == 0:
        st.info("No data available for analytics.")
        return

    # Summary statistics
    stats = data_loader.get_summary_stats(df)

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transcripts", stats["total_transcripts"])

    with col2:
        st.metric("Average Score", f"{stats['avg_score']:.2f}/2.00")

    with col3:
        st.metric("Virtues", stats["virtues_count"])

    with col4:
        st.metric("Models", stats["models_count"])

    st.divider()

    # Score distribution
    st.subheader("Score Distribution")

    score_dist = stats["score_distribution"]
    score_df = pd.DataFrame({
        "Score": ["0", "1", "2"],
        "Count": [score_dist[0], score_dist[1], score_dist[2]],
    })

    fig_dist = px.bar(
        score_df,
        x="Score",
        y="Count",
        title="Distribution of Scores",
        labels={"Count": "Number of Transcripts"},
        color="Score",
        color_discrete_map={"0": "#ff4b4b", "1": "#ffa500", "2": "#00cc66"},
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()

    # Model x Virtue heatmap
    st.subheader("Model Performance by Virtue")

    matrix = data_loader.get_model_virtue_matrix(df)

    if not matrix.empty:
        fig_heatmap = px.imshow(
            matrix,
            labels=dict(x="Virtue", y="Model", color="Average Score"),
            title="Average Scores: Model × Virtue",
            color_continuous_scale="RdYlGn",
            aspect="auto",
            zmin=0,
            zmax=2,
        )
        fig_heatmap.update_xaxes(side="bottom")
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Show the data table
        with st.expander("**View Data Table**"):
            st.dataframe(matrix, use_container_width=True)
    else:
        st.info("Not enough data to generate heatmap.")

    st.divider()

    # Per-virtue breakdown
    st.subheader("Performance by Virtue")

    virtue_stats = df.groupby("virtue")["score"].agg(["mean", "count"]).reset_index()
    virtue_stats.columns = ["Virtue", "Average Score", "Count"]
    virtue_stats = virtue_stats.sort_values("Average Score", ascending=False)

    fig_virtue = px.bar(
        virtue_stats,
        x="Virtue",
        y="Average Score",
        title="Average Score by Virtue",
        labels={"Average Score": "Average Score (0-2)"},
        text="Average Score",
        hover_data={"Count": True},
    )
    fig_virtue.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_virtue.update_layout(yaxis_range=[0, 2.2])
    st.plotly_chart(fig_virtue, use_container_width=True)

    st.divider()

    # Per-model breakdown
    st.subheader("Performance by Model")

    model_stats = df.groupby("model")["score"].agg(["mean", "count"]).reset_index()
    model_stats.columns = ["Model", "Average Score", "Count"]
    model_stats = model_stats.sort_values("Average Score", ascending=False)

    fig_model = px.bar(
        model_stats,
        x="Model",
        y="Average Score",
        title="Average Score by Model",
        labels={"Average Score": "Average Score (0-2)"},
        text="Average Score",
        hover_data={"Count": True},
    )
    fig_model.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_model.update_layout(yaxis_range=[0, 2.2])
    st.plotly_chart(fig_model, use_container_width=True)

    st.divider()

    # Score distribution per model
    if df["model"].nunique() > 1:
        st.subheader("Score Distribution per Model")

        model_score_dist = df.groupby(["model", "score"]).size().reset_index(name="count")

        fig_model_dist = px.bar(
            model_score_dist,
            x="model",
            y="count",
            color="score",
            title="Score Distribution by Model",
            labels={"count": "Count", "model": "Model", "score": "Score"},
            barmode="group",
            color_discrete_map={0: "#ff4b4b", 1: "#ffa500", 2: "#00cc66"},
        )
        st.plotly_chart(fig_model_dist, use_container_width=True)
