from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT_DIR / "configs" / "precomputed_language_embeddings_registry.json"
LANGUAGE_LABELS = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "nld_Latn": "Dutch",
    "swe_Latn": "Swedish",
    "spa_Latn": "Spanish",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "pol_Latn": "Polish",
    "ces_Latn": "Czech",
}


def make_preview(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def display_language(lang_code: str) -> str:
    return LANGUAGE_LABELS.get(lang_code, lang_code)


def display_backbone_name(backbone_name: str) -> str:
    normalized = backbone_name.strip().lower()
    if "mdeberta" in normalized:
        return "mDeBERTa"
    if "nllb" in normalized:
        return "NLLB"
    return backbone_name


def get_file_cache_token(path: Path) -> int:
    return path.stat().st_mtime_ns


@st.cache_data
def load_precomputed_registry():
    with open(REGISTRY_PATH, "r") as stream:
        return json.load(stream)


@st.cache_data
def load_precomputed_config(config_path: str, cache_token: int):
    with open(config_path, "r") as stream:
        return json.load(stream)


@st.cache_data
def load_precomputed_metadata(metadata_path: str, cache_token: int):
    with open(metadata_path, "r") as stream:
        return json.load(stream)


@st.cache_data
def load_precomputed_artifact(artifact_path: str, cache_token: int):
    data = np.load(artifact_path, allow_pickle=True)
    layer_indices = [int(layer_idx) for layer_idx in data["layer_indices"].tolist()]
    return {
        "texts": data["texts"].tolist(),
        "langs": data["langs"].tolist(),
        "layer_indices": layer_indices,
        "counts_by_lang": {
            str(lang): int(count)
            for lang, count in zip(data["counts_langs"].tolist(), data["counts_values"].tolist())
        },
        "coords_by_layer": {
            layer_idx: (
                data[f"tsne_coords_layer_{layer_idx}"].astype(np.float32)
                if f"tsne_coords_layer_{layer_idx}" in data
                else data[f"coords_layer_{layer_idx}"].astype(np.float32)
            )
            for layer_idx in layer_indices
        },
    }


def build_scatter_frame(coords: np.ndarray, langs: list[str], texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "lang_code": langs,
            "language": [display_language(lang) for lang in langs],
            "text": texts,
            "preview": [make_preview(text) for text in texts],
        }
    )


def build_centroid_distance_frame(df: pd.DataFrame) -> pd.DataFrame:
    unique_langs = sorted(df["lang_code"].unique())
    rows = []

    for idx, lang_a in enumerate(unique_langs):
        a_points = df.loc[df["lang_code"] == lang_a, ["x", "y"]].to_numpy()
        centroid_a = a_points.mean(axis=0)

        for lang_b in unique_langs[idx + 1:]:
            b_points = df.loc[df["lang_code"] == lang_b, ["x", "y"]].to_numpy()
            centroid_b = b_points.mean(axis=0)
            rows.append(
                {
                    "language_a": display_language(lang_a),
                    "language_b": display_language(lang_b),
                    "distance": float(np.linalg.norm(centroid_a - centroid_b)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["language_a", "language_b", "distance"])

    return pd.DataFrame(rows).sort_values("distance")


def build_scatter_figure(
    df: pd.DataFrame, backbone_name: str, layer_idx: int, projection_label: str
):
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="language",
        render_mode="webgl",
        custom_data=["language", "preview"],
        opacity=0.78,
    )
    fig.update_traces(
        marker={"size": 8},
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        title=f"{display_backbone_name(backbone_name)} - Layer {layer_idx}",
        xaxis_title=f"{projection_label} 1",
        yaxis_title=f"{projection_label} 2",
        legend_title_text="Language",
        height=860,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


st.title("Precomputed Language Embeddings")
st.caption("Browse precomputed multilingual language embedding projections by layer without recomputing.")

registry = load_precomputed_registry()
artifact_entries = {entry["key"]: entry for entry in registry["artifacts"]}
artifact_labels = {entry["key"]: entry["label"] for entry in registry["artifacts"]}
default_key = registry["default_key"]

selected_key = st.selectbox(
    "Embedding Family",
    options=list(artifact_entries.keys()),
    index=list(artifact_entries.keys()).index(default_key),
    format_func=lambda key: artifact_labels[key],
)

selected_entry = artifact_entries[selected_key]
config_path = ROOT_DIR / selected_entry["config_path"]
config = load_precomputed_config(str(config_path), get_file_cache_token(config_path))
artifact_path = ROOT_DIR / config["output_path"]
metadata_path = artifact_path.with_suffix(".json")

if not artifact_path.exists():
    st.error(f"Precomputed artifact not found: {artifact_path}")
    st.code(
        "python mllm/precompute_language_embeddings.py "
        f"{selected_entry['config_path']}"
    )
    st.stop()

artifact = load_precomputed_artifact(str(artifact_path), get_file_cache_token(artifact_path))
metadata = (
    load_precomputed_metadata(str(metadata_path), get_file_cache_token(metadata_path))
    if metadata_path.exists()
    else {}
)
projection_label = str(
    metadata.get("browse_projection_method", metadata.get("projection_method", "Projection"))
).upper()

available_layers = [int(layer_idx) for layer_idx in artifact["layer_indices"]]
available_langs = list(dict.fromkeys(artifact["langs"]))

control_left, control_right = st.columns([1, 2])

with control_left:
    selected_layer = st.slider(
        "Layer",
        min_value=min(available_layers),
        max_value=max(available_layers),
        value=config["layer_indices"][0],
        step=1,
    )

with control_right:
    selected_langs = st.multiselect(
        "Languages",
        options=available_langs,
        default=available_langs,
        format_func=display_language,
    )

if not selected_langs:
    st.warning("Select at least one language to display points.")
    st.stop()

coords = artifact["coords_by_layer"][selected_layer]
df = build_scatter_frame(coords, artifact["langs"], artifact["texts"])
filtered_df = df[df["lang_code"].isin(selected_langs)].reset_index(drop=True)
counts_df = (
    filtered_df.groupby("language")
    .size()
    .reset_index(name="count")
    .sort_values("language")
)
centroid_df = build_centroid_distance_frame(filtered_df)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Points", len(filtered_df))
metric_b.metric("Languages", len(selected_langs))
metric_c.metric("Layer", selected_layer)

st.plotly_chart(
    build_scatter_figure(
        filtered_df,
        backbone_name=config["backbone_name"],
        layer_idx=selected_layer,
        projection_label=projection_label,
    ),
    use_container_width=True,
)

tab_counts, tab_distances, tab_samples, tab_config = st.tabs(
    ["Counts", "Centroid Distances", "Sample Sentences", "Artifact Config"]
)

with tab_counts:
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

with tab_distances:
    st.dataframe(centroid_df, use_container_width=True, hide_index=True)

with tab_samples:
    st.dataframe(
        filtered_df[["language", "text"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_config:
    st.json(config)
