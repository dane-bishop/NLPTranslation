import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from circle_visualization.nllb_geo_chord_investigate_lang import (
    load_language_name_map,
    build_distance_df_from_embedding_csv,
    compute_similarity_from_distances,
    build_label_map,
    get_top_k_neighbors,
    build_star_edges,
)


st.title("NLLB-200 Circle Visualization")
st.write("Explore the top-k nearest languages from your exported NLLB embeddings.")


@st.cache_data(show_spinner=False)
def load_nllb_data(script_dir_str: str):
    script_dir = Path(script_dir_str)
    embedding_csv_path = script_dir / "language_embeddings.csv"
    flores_names_path = script_dir / "flores_language_names.csv"

    if not embedding_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {embedding_csv_path}. "
            "Put language_embeddings.csv in ui/circle_visualization/."
        )

    flores_name_map = load_language_name_map(flores_names_path)
    embedding_df, distance_df = build_distance_df_from_embedding_csv(embedding_csv_path)
    similarity_df = compute_similarity_from_distances(distance_df)
    label_map = build_label_map(embedding_df, flores_name_map)

    return embedding_df, distance_df, similarity_df, label_map


def _circle_positions(node_ids: list[str], radius: float = 1.0) -> dict[str, tuple[float, float]]:
    n = len(node_ids)
    if n == 0:
        return {}

    positions = {}
    for i, node in enumerate(node_ids):
        angle = math.pi / 2 - (2 * math.pi * i / n)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions[node] = (x, y)
    return positions


def _similarity_to_color(similarity: float, min_similarity: float, max_similarity: float) -> str:
    """
    Map similarity to a cool->warm color.
    Lower similarity = cooler color, higher similarity = warmer color.
    """
    if max_similarity - min_similarity < 1e-12:
        t = 0.5
    else:
        t = (similarity - min_similarity) / (max_similarity - min_similarity)

    return px.colors.sample_colorscale("Turbo", [t])[0]


def _make_edge_traces(
    src: str,
    dst: str,
    positions: dict[str, tuple[float, float]],
    src_label: str,
    dst_label: str,
    distance: float,
    similarity: float,
    line_width: float,
    edge_color: str,
):
    x0, y0 = positions[src]
    x1, y1 = positions[dst]

    mid_x = (x0 + x1) / 2 * 0.55
    mid_y = (y0 + y1) / 2 * 0.55

    hover_text = (
        f"<b>{src_label}</b> → <b>{dst_label}</b><br><br>"
        f"<b>Distance:</b> {distance:.4f}<br>"
        f"<b>Similarity:</b> {similarity:.4f}"
    )

    line_trace = go.Scatter(
        x=[x0, mid_x, x1],
        y=[y0, mid_y, y1],
        mode="lines",
        line={"width": line_width, "color": edge_color, "shape": "spline"},
        hoverinfo="skip",
        showlegend=False,
    )

    midpoint_hover_trace = go.Scatter(
        x=[mid_x],
        y=[mid_y],
        mode="markers",
        marker={
            "size": 10,
            "color": edge_color,
            "opacity": 0.001,
        },
        hovertemplate=hover_text + "<extra></extra>",
        showlegend=False,
    )

    endpoint_hover_trace = go.Scatter(
        x=[x1],
        y=[y1],
        mode="markers",
        marker={
            "size": 14,
            "color": edge_color,
            "opacity": 0.001,
        },
        hovertemplate=hover_text + "<extra></extra>",
        showlegend=False,
    )

    return line_trace, midpoint_hover_trace, endpoint_hover_trace


def build_interactive_circle_figure(
    selected_lang: str,
    selected_and_neighbors: list[str],
    edge_df: pd.DataFrame,
    label_map: dict[str, str],
) -> go.Figure:
    positions = _circle_positions(selected_and_neighbors, radius=1.0)

    fig = go.Figure()

    if not edge_df.empty:
        min_similarity = float(edge_df["similarity"].min())
        max_similarity = float(edge_df["similarity"].max())
    else:
        min_similarity = 0.0
        max_similarity = 1.0

    for _, row in edge_df.iterrows():
        src = row["src"]
        dst = row["dst"]
        similarity = float(row["similarity"])

        edge_color = _similarity_to_color(similarity, min_similarity, max_similarity)

        line_trace, midpoint_hover_trace, endpoint_hover_trace = _make_edge_traces(
            src=src,
            dst=dst,
            positions=positions,
            src_label=label_map.get(src, src),
            dst_label=label_map.get(dst, dst),
            distance=float(row["distance"]),
            similarity=similarity,
            line_width=float(row["plot_weight"]),
            edge_color=edge_color,
        )

        fig.add_trace(line_trace)
        fig.add_trace(midpoint_hover_trace)
        fig.add_trace(endpoint_hover_trace)

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in selected_and_neighbors:
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(label_map.get(node, node))

        is_selected = node == selected_lang
        node_size.append(20 if is_selected else 12)
        node_color.append("#d62728" if is_selected else "#2ca02c")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont={
                "size": 14,
                "color": "#111111",
                "family": "Arial Black",
            },
            hoverinfo="skip",
            marker={
                "size": node_size,
                "color": node_color,
                "line": {"width": 1, "color": "#222222"},
            },
            showlegend=False,
        )
    )

    # Add a hidden marker trace only to show the similarity color legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "colorscale": "Turbo",
                "showscale": True,
                "cmin": 0.0,
                "cmax": 1.0,
                "color": [0.5],
                "size": 0.1,
                "colorbar": {
                    "title": {"text": "Similarity"},
                    "tickvals": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                    "len": 0.75,
                    "thickness": 18,
                    "x": 1.02,
                },
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 80, "t": 60, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x", "scaleratio": 1},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"color": "#111111"},
        hoverlabel={
            "bgcolor": "white",
            "font_size": 16,
            "font_family": "Arial",
            "font_color": "#111111",
            "bordercolor": "#333333",
        },
        title={
            "text": f"{label_map.get(selected_lang, selected_lang)} and Top Similar Languages",
            "x": 0.5,
            "font": {"color": "#111111", "size": 20},
        },
    )

    return fig


script_dir = Path(__file__).parent / "circle_visualization"

try:
    embedding_df, distance_df, similarity_df, label_map = load_nllb_data(str(script_dir))
except Exception as e:
    st.error(str(e))
    st.stop()

language_options = sorted(label_map.keys(), key=lambda x: label_map.get(x, x))

default_idx = 0
if "eng_Latn" in language_options:
    default_idx = language_options.index("eng_Latn")

left_col, right_col = st.columns([1, 2])

with left_col:
    selected_lang = st.selectbox(
        "Choose a language",
        options=language_options,
        index=default_idx,
        format_func=lambda x: f"{label_map.get(x, x)} ({x})",
    )

    max_neighbors = max(1, len(distance_df.index) - 1)
    k = st.slider(
        "Number of similar languages",
        min_value=1,
        max_value=min(50, max_neighbors),
        value=min(10, max_neighbors),
    )

selected_and_neighbors = get_top_k_neighbors(distance_df, selected_lang, k)
subset_distance_df = distance_df.loc[selected_and_neighbors, selected_and_neighbors]
subset_similarity_df = similarity_df.loc[selected_and_neighbors, selected_and_neighbors]
edge_df = build_star_edges(selected_lang, subset_distance_df, subset_similarity_df)

neighbor_rows = []
for _, row in edge_df.sort_values("distance").iterrows():
    neighbor_rows.append(
        {
            "Language": label_map.get(row["dst"], row["dst"]),
            "Code": row["dst"],
            "Distance": float(row["distance"]),
            "Similarity": float(row["similarity"]),
        }
    )

with left_col:
    st.subheader("Nearest neighbors")
    st.dataframe(pd.DataFrame(neighbor_rows), use_container_width=True)

with right_col:
    fig = build_interactive_circle_figure(
        selected_lang=selected_lang,
        selected_and_neighbors=selected_and_neighbors,
        edge_df=edge_df,
        label_map=label_map,
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Debug info"):
    st.write("Selected language code:", selected_lang)
    st.write("All nodes in this view:", selected_and_neighbors)
    st.write("Distance matrix index sample:", list(distance_df.index[:15]))