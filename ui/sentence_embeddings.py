from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
MLLM_DIR = ROOT_DIR / "mllm"
REGISTRY_PATH = ROOT_DIR / "configs" / "precomputed_language_embeddings_registry.json"

if str(MLLM_DIR) not in sys.path:
    sys.path.append(str(MLLM_DIR))

from backbone import MLLMBackbone
from train import masked_mean_pool


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
DEFAULT_QUERY = "The quick brown fox jumps over the lazy dog."
TEXT_KEY = "sentence_embeddings_query_text"


def make_preview(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def display_language(lang_code: str) -> str:
    return LANGUAGE_LABELS.get(lang_code, lang_code)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def get_file_cache_token(path: Path) -> int:
    return path.stat().st_mtime_ns


@st.cache_data
def load_precomputed_registry():
    with open(REGISTRY_PATH, "r") as stream:
        return json.load(stream)


@st.cache_data
def load_json(path_str: str, cache_token: int):
    with open(path_str, "r") as stream:
        return json.load(stream)


@st.cache_data
def load_precomputed_artifact(artifact_path: str, cache_token: int):
    data = np.load(artifact_path, allow_pickle=True)
    layer_indices = [int(layer_idx) for layer_idx in data["layer_indices"].tolist()]
    artifact = {
        "texts": data["texts"].tolist(),
        "langs": data["langs"].tolist(),
        "layer_indices": layer_indices,
        "counts_by_lang": {
            str(lang): int(count)
            for lang, count in zip(data["counts_langs"].tolist(), data["counts_values"].tolist())
        },
        "coords_by_layer": {
            layer_idx: (
                data[f"umap_coords_layer_{layer_idx}"].astype(np.float32)
                if f"umap_coords_layer_{layer_idx}" in data
                else data[f"coords_layer_{layer_idx}"].astype(np.float32)
            )
            for layer_idx in layer_indices
        },
        "embeddings_by_layer": {},
    }

    for layer_idx in layer_indices:
        key = f"embeddings_layer_{layer_idx}"
        if key in data:
            artifact["embeddings_by_layer"][layer_idx] = data[key].astype(np.float32)

    return artifact


@st.cache_resource
def load_backbone(model_name: str) -> MLLMBackbone:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MLLMBackbone(device, model_name)


@st.cache_resource
def load_reducer(reducer_path: str, cache_token: int):
    import joblib

    return joblib.load(reducer_path)


@torch.no_grad()
def embed_query_text(
    backbone: MLLMBackbone,
    text: str,
    layer_idx: int,
    max_length: int,
    encoder_only: bool,
):
    acts = backbone.extract_layer_activations(
        texts=[text],
        layer_idx=layer_idx,
        max_length=max_length,
        encoder_only=encoder_only,
    )
    pooled = masked_mean_pool(acts["layer_tensor"], acts["valid_mask"])
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.detach().cpu().numpy().astype(np.float32), int(acts["valid_mask"].sum().item())


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


def build_color_map(languages: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Safe + px.colors.qualitative.Set2 + px.colors.qualitative.Plotly
    unique_languages = list(dict.fromkeys(languages))
    return {
        language: palette[idx % len(palette)]
        for idx, language in enumerate(unique_languages)
    }


def explain_backbone_load_error(exc: Exception) -> str:
    message = str(exc)

    if "BitGenerator module" in message or "known BitGenerator" in message:
        return (
            "The saved UMAP reducer is incompatible with the current NumPy/joblib stack. "
            "Re-run precompute in the current environment to regenerate the `.umap.joblib` files."
        )

    if "torchvision" in message:
        return (
            "The text model load is failing because the current environment has an incompatible "
            "`torch`/`torchvision` pair. This app does not need torchvision, but a broken "
            "torchvision install can still break `transformers` imports."
        )

    if "upgrade torch to at least v2.6" in message or "CVE-2025-32434" in message:
        return (
            "The installed `transformers` version is refusing to load PyTorch `.bin` weights with "
            "the current `torch` version. Use a repo-supported `transformers<5` build, or upgrade "
            "`torch` to a compatible newer version and keep `torchvision` matched to it."
        )

    if "Temporary failure in name resolution" in message:
        return (
            "The model loader tried to reach Hugging Face but network access was unavailable. "
            "This usually means the model weights are not fully cached locally in the expected format."
        )

    return f"Backbone load failed: {message}"


def build_sentence_figure(
    df: pd.DataFrame,
    highlighted_df: pd.DataFrame,
    query_coord: np.ndarray | None,
    query_text: str,
    projection_label: str,
    color_map: dict[str, str],
    ring_radius: float | None,
):
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="language",
        color_discrete_map=color_map,
        render_mode="webgl",
        custom_data=["language", "preview"],
        opacity=0.12,
    )
    fig.update_traces(
        marker={"size": 6},
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>",
    )

    if not highlighted_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=highlighted_df["x"],
                y=highlighted_df["y"],
                mode="markers",
                name="Highlighted neighbors",
                marker={
                    "size": 11,
                    "color": [color_map[language] for language in highlighted_df["language"]],
                    "line": {"color": "white", "width": 1.2},
                    "opacity": 0.95,
                },
                customdata=np.stack(
                    [
                        highlighted_df["language"],
                        highlighted_df["preview"],
                        highlighted_df["cosine"].map(lambda value: f"{value:.4f}"),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Cosine: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    if query_coord is not None:
        fig.add_trace(
            go.Scattergl(
                x=[float(query_coord[0])],
                y=[float(query_coord[1])],
                mode="markers",
                name="Query sentence",
                marker={
                    "size": 16,
                    "symbol": "star",
                    "color": "#111111",
                    "line": {"color": "#f5b301", "width": 2},
                },
                text=[make_preview(query_text, limit=200)],
                hovertemplate="<b>Query</b><br>%{text}<extra></extra>",
            )
        )

    if query_coord is not None and ring_radius is not None and ring_radius > 0:
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=float(query_coord[0] - ring_radius),
            y0=float(query_coord[1] - ring_radius),
            x1=float(query_coord[0] + ring_radius),
            y1=float(query_coord[1] + ring_radius),
            line={"color": "#f5b301", "width": 1.5, "dash": "dot"},
        )

    fig.update_layout(
        title="Live Query Over Precomputed Language Projection",
        xaxis_title=f"{projection_label} 1",
        yaxis_title=f"{projection_label} 2",
        legend_title_text="Language",
        height=860,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


st.title("Sentence Embeddings")
st.caption("Project a user-provided sentence into a precomputed language embedding map and light up its nearest neighborhood.")

try:
    import joblib  # noqa: F401
    import umap  # noqa: F401
except ImportError:
    st.error("This page requires `joblib` and `umap-learn`. Install them before running the UI.")
    st.code("pip install joblib umap-learn")
    st.stop()

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
config_path = resolve_repo_path(selected_entry["config_path"])
config = load_json(str(config_path), get_file_cache_token(config_path))
artifact_path = resolve_repo_path(config["output_path"])
metadata_path = artifact_path.with_suffix(".json")

if not artifact_path.exists():
    st.error(f"Precomputed artifact not found: {artifact_path}")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

if not metadata_path.exists():
    st.error(f"Artifact metadata not found: {metadata_path}")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

metadata = load_json(str(metadata_path), get_file_cache_token(metadata_path))
artifact = load_precomputed_artifact(str(artifact_path), get_file_cache_token(artifact_path))
projection_label = str(
    metadata.get("query_projection_method", metadata.get("projection_method", "Projection"))
).upper()

if not artifact["embeddings_by_layer"]:
    st.error("This artifact does not include per-layer embedding matrices. Re-run precompute with the updated script.")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

available_layers = [int(layer_idx) for layer_idx in artifact["layer_indices"]]
available_langs = list(dict.fromkeys(artifact["langs"]))

control_left, control_right, control_far_right = st.columns([1, 2, 1])

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

with control_far_right:
    top_k = st.slider("Neighbors", min_value=3, max_value=50, value=15, step=1)
    radius_scale = st.slider("Ring Scale", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

if not selected_langs:
    st.warning("Select at least one language to display points.")
    st.stop()

query_text = st.text_area(
    "Type a sentence",
    value=st.session_state.get(TEXT_KEY, DEFAULT_QUERY),
    key=TEXT_KEY,
    height=120,
    placeholder="Enter a sentence to project into the selected embedding space.",
)

coords = artifact["coords_by_layer"][selected_layer]
base_df = build_scatter_frame(coords, artifact["langs"], artifact["texts"])
filtered_df = base_df[base_df["lang_code"].isin(selected_langs)].copy()
filtered_df["source_idx"] = filtered_df.index
filtered_df = filtered_df.reset_index(drop=True)
color_map = build_color_map(filtered_df["language"].tolist())

if not query_text.strip():
    st.info("Enter a sentence to compute its embedding and project it onto the background map.")
    st.plotly_chart(
        build_sentence_figure(
            df=filtered_df,
            highlighted_df=filtered_df.iloc[0:0].copy(),
            query_coord=None,
            query_text="",
            projection_label=projection_label,
            color_map=color_map,
            ring_radius=None,
        ),
        use_container_width=True,
    )
    st.stop()

umap_paths_by_layer = metadata.get("umap_paths_by_layer", {})
reducer_path_raw = umap_paths_by_layer.get(str(selected_layer))

if not reducer_path_raw:
    st.error("The selected artifact is missing the saved UMAP reducer for this layer. Re-run precompute with the updated script.")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

reducer_path = resolve_repo_path(reducer_path_raw)
if not reducer_path.exists():
    st.error(f"Saved UMAP reducer not found: {reducer_path}")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

embedding_matrix = artifact["embeddings_by_layer"].get(selected_layer)
if embedding_matrix is None:
    st.error("The selected layer is missing saved embedding vectors. Re-run precompute with the updated script.")
    st.code(f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}")
    st.stop()

display_indices = filtered_df["source_idx"].to_numpy()
filtered_embeddings = embedding_matrix[display_indices]

try:
    with st.spinner("Embedding query text and projecting it with the saved reducer..."):
        backbone = load_backbone(config["backbone_name"])
        reducer = load_reducer(str(reducer_path), get_file_cache_token(reducer_path))
        query_embedding, valid_token_count = embed_query_text(
            backbone=backbone,
            text=query_text.strip(),
            layer_idx=selected_layer,
            max_length=int(config["max_length"]),
            encoder_only=bool(config["encoder_only"]),
        )
        query_coord = reducer.transform(query_embedding)[0].astype(np.float32)
except Exception as exc:
    st.error(explain_backbone_load_error(exc))
    st.code(
        f"python mllm/precompute_language_embeddings.py {selected_entry['config_path']}"
    )
    st.stop()

cosines = filtered_embeddings @ query_embedding[0]
effective_top_k = min(top_k, len(filtered_df))
top_positions = np.argsort(cosines)[-effective_top_k:][::-1]
highlighted_df = filtered_df.iloc[top_positions].copy().reset_index(drop=True)
highlighted_df["cosine"] = cosines[top_positions]
highlighted_df["distance_2d"] = np.linalg.norm(
    highlighted_df[["x", "y"]].to_numpy() - query_coord[None, :],
    axis=1,
)
ring_radius = float(highlighted_df["distance_2d"].max() * radius_scale) if not highlighted_df.empty else None

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Visible Points", len(filtered_df))
metric_b.metric("Highlighted Neighbors", effective_top_k)
metric_c.metric("Query Tokens", valid_token_count)
metric_d.metric("Best Cosine", f"{highlighted_df['cosine'].max():.4f}" if not highlighted_df.empty else "n/a")

st.plotly_chart(
    build_sentence_figure(
        df=filtered_df,
        highlighted_df=highlighted_df,
        query_coord=query_coord,
        query_text=query_text.strip(),
        projection_label=projection_label,
        color_map=color_map,
        ring_radius=ring_radius,
    ),
    use_container_width=True,
)

tab_neighbors, tab_query, tab_artifact = st.tabs(["Neighbors", "Query Details", "Artifact Config"])

with tab_neighbors:
    st.dataframe(
        highlighted_df[["language", "cosine", "distance_2d", "text"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_query:
    st.write("Current sentence")
    st.write(query_text.strip())
    st.write(f"Projected coordinate: ({query_coord[0]:.4f}, {query_coord[1]:.4f})")
    if ring_radius is not None:
        st.write(f"Highlight ring radius: {ring_radius:.4f}")

with tab_artifact:
    st.json(metadata)
