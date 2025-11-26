import streamlit as st
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from app.db.memory import ChatRepository


@st.cache_data(ttl=60)
def _load_metrics(limit: int) -> list[dict]:
    repo = ChatRepository()
    return repo.get_assistant_metrics(limit=limit)


@st.cache_data(ttl=60)
def _load_breakdown() -> dict[str, int]:
    repo = ChatRepository()
    return repo.get_success_breakdown()


def render() -> None:
    st.title("Metrics Dashboard")

    # Load Data
    metrics = _load_metrics(limit=500)
    breakdown = _load_breakdown()

    if not metrics:
        st.info("No data yet. Use the CLI to chat: `python -m app.main`")
        return

    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- KPI Row ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", len(df))
    with col2:
        st.metric("Avg Latency", f"{df['total_latency'].mean():.2f}s")
    with col3:
        st.metric("Total Cost", f"${df['cost'].sum():.4f}")
    with col4:
        # Handle case where avg_retrieval_distance might be all None
        avg_dist = df["avg_retrieval_distance"].mean()
        val_str = f"{avg_dist:.2f}" if pd.notna(avg_dist) else "N/A"
        st.metric("Avg Retrieval Dist", val_str)

    # --- Chart 1: Latency & Cost Over Time ---
    st.subheader("Latency & Cost Over Time")

    # Dual axis chart
    fig_lc = go.Figure()
    fig_lc.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["total_latency"],
            name="Latency (s)",
            mode="lines+markers",
            line=dict(color="firebrick"),
        )
    )
    fig_lc.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["cost"],
            name="Cost ($)",
            mode="lines+markers",
            line=dict(color="royalblue"),
            yaxis="y2",
        )
    )

    fig_lc.update_layout(
        yaxis=dict(title="Latency (s)"),
        yaxis2=dict(title="Cost ($)", overlaying="y", side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_lc, use_container_width=True)

    # --- Chart 2: Retrieval Accuracy (Distance) ---
    st.subheader("Retrieval Accuracy (Distance)")

    # Filter out NaN for distance chart if needed, or Plotly handles it
    fig_dist = px.scatter(
        df,
        x="timestamp",
        y="avg_retrieval_distance",
        color="rag_success",
        title="Vector Distance (Lower is Better)",
        labels={"avg_retrieval_distance": "Distance Score", "rag_success": "Success?"},
    )
    # Add threshold line
    fig_dist.add_hline(y=1.0, line_dash="dash", annotation_text="Threshold (1.0)")
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Chart 3: Success Breakdown ---
    st.subheader("Success/Failure Breakdown")

    labels = list(breakdown.keys())
    values = list(breakdown.values())

    fig_pie = px.pie(
        names=labels,
        values=values,
        title="Response Status Distribution",
        color=labels,
        color_discrete_map={
            "full_success": "green",
            "partial": "orange",
            "error": "red",
        },
    )
    st.plotly_chart(fig_pie, use_container_width=True)
