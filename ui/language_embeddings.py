from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
MLLM_DIR = ROOT_DIR / "mllm"

if str(MLLM_DIR) not in sys.path:
    sys.path.append(str(MLLM_DIR))

from backbone import MLLMBackbone
from cluster import ClusterConf, collect_embeddings, collate_records, run_tsne
from dataset import BalancedNLLBDataset


CONFIG_PATH = ROOT_DIR / "configs" / "eval" / "cluster.json"
SESSION_KEY = "language_embeddings_cluster_conf"
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


@st.cache_data
def load_default_conf() -> ClusterConf:
    with open(CONFIG_PATH, "r") as stream:
        return ClusterConf(**json.load(stream))


@st.cache_resource
def load_backbone(model_name: str) -> MLLMBackbone:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MLLMBackbone(device, model_name)


@st.cache_data(show_spinner=False)
def run_clustering_job(
    backbone_name: str,
    langs: tuple[str, ...],
    pairs: tuple[str, ...],
    batch_size: int,
    max_length: int,
    num_batches: int,
    layer_idx: int,
    points_per_lang_cap: int | None,
    shuffle_buffer_size: int | None,
    stream_shuffle_seed: int | None,
    random_seed: int,
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    backbone = load_backbone(backbone_name)
    dataset = BalancedNLLBDataset(
        pair_configs=list(pairs),
        langs=list(langs),
        shuffle_buffer_size=shuffle_buffer_size,
        seed=stream_shuffle_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_records,
        num_workers=0,
    )

    embeddings, collected_langs, texts, counts_by_lang = collect_embeddings(
        backbone=backbone,
        loader=loader,
        langs_to_collect=list(langs),
        num_batches=num_batches,
        max_length=max_length,
        layer_idx=layer_idx,
        cap_per_lang=points_per_lang_cap,
    )
    coords = run_tsne(embeddings, random_state=random_seed)

    return {
        "coords": coords,
        "langs": collected_langs,
        "texts": texts,
        "counts_by_lang": dict(counts_by_lang),
    }


def parse_optional_int(raw_value: str) -> int | None:
    value = raw_value.strip()
    if not value:
        return None
    return int(value)


def make_preview(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def display_language(lang_code: str) -> str:
    return LANGUAGE_LABELS.get(lang_code, lang_code)


def build_lang_pair_map(conf: ClusterConf) -> dict[str, str]:
    return {lang: pair for lang, pair in zip(conf.langs, conf.pairs)}


def build_scatter_frame(coords: np.ndarray, langs: list[str], texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tsne_1": coords[:, 0],
            "tsne_2": coords[:, 1],
            "lang_code": langs,
            "language": [display_language(lang) for lang in langs],
            "text": texts,
            "preview": [make_preview(text) for text in texts],
        }
    )


def build_counts_frame(counts_by_lang: dict[str, int], ordered_langs: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "language": [display_language(lang) for lang in ordered_langs],
            "count": [counts_by_lang.get(lang, 0) for lang in ordered_langs],
        }
    )


def build_centroid_distance_frame(coords: np.ndarray, langs: list[str]) -> pd.DataFrame:
    unique_langs = sorted(set(langs))
    centroids = {}

    for lang in unique_langs:
        idxs = [idx for idx, value in enumerate(langs) if value == lang]
        centroids[lang] = coords[idxs].mean(axis=0)

    rows = []
    for idx, lang_a in enumerate(unique_langs):
        for lang_b in unique_langs[idx + 1:]:
            dist = float(np.linalg.norm(centroids[lang_a] - centroids[lang_b]))
            rows.append(
                {
                    "language_a": display_language(lang_a),
                    "language_b": display_language(lang_b),
                    "distance": dist,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["language_a", "language_b", "distance"])

    return pd.DataFrame(rows).sort_values("distance")


def build_scatter_figure(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="tsne_1",
        y="tsne_2",
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
        title="Multilingual Sentence Embeddings",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend_title_text="Language",
        height=860,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


st.title("Language Embeddings")
st.caption("Run multilingual clustering from the UI using an in-memory cluster config built from the form.")

default_conf = load_default_conf()
lang_pair_map = build_lang_pair_map(default_conf)

with st.form("language_embedding_form"):
    left, right = st.columns(2)

    with left:
        backbone_name = st.text_input("Backbone Name", value=default_conf.backbone_name)
        selected_langs = st.multiselect(
            "Languages",
            options=default_conf.langs,
            default=default_conf.langs,
            format_func=display_language,
        )
        layer_idx = st.number_input("Layer Index", min_value=0, value=default_conf.layer_idx, step=1)
        max_length = st.number_input("Max Length", min_value=8, value=default_conf.max_length, step=8)
        batch_size = st.number_input("Batch Size", min_value=1, value=default_conf.batch_size, step=1)

    with right:
        num_batches = st.number_input("Num Batches", min_value=1, value=default_conf.num_batches, step=1)
        points_per_lang_cap = st.number_input(
            "Points Per Language Cap",
            min_value=1,
            value=default_conf.points_per_lang_cap or 1000,
            step=1,
        )
        shuffle_buffer_size = st.number_input(
            "Shuffle Buffer Size",
            min_value=1,
            value=default_conf.shuffle_buffer_size or 10000,
            step=100,
        )
        random_seed = st.number_input("Random Seed", min_value=0, value=default_conf.random_seed, step=1)
        stream_shuffle_seed_raw = st.text_input(
            "Stream Shuffle Seed",
            value="" if default_conf.stream_shuffle_seed is None else str(default_conf.stream_shuffle_seed),
            help="Leave blank to keep the stream shuffle seed unset.",
        )

    submitted = st.form_submit_button("Run Clustering", type="primary")

if submitted:
    if not selected_langs:
        st.error("Select at least one language before running clustering.")
    else:
        try:
            stream_shuffle_seed = parse_optional_int(stream_shuffle_seed_raw)
        except ValueError:
            st.error("Stream Shuffle Seed must be an integer or left blank.")
            st.stop()

        conf = ClusterConf(
            backbone_name=backbone_name,
            langs=selected_langs,
            pairs=[lang_pair_map[lang] for lang in selected_langs],
            batch_size=int(batch_size),
            max_length=int(max_length),
            num_batches=int(num_batches),
            layer_idx=int(layer_idx),
            points_per_lang_cap=int(points_per_lang_cap),
            shuffle_buffer_size=int(shuffle_buffer_size),
            stream_shuffle_seed=stream_shuffle_seed,
            random_seed=int(random_seed),
            output_path=default_conf.output_path,
        )
        st.session_state[SESSION_KEY] = asdict(conf)

saved_conf = st.session_state.get(SESSION_KEY)

if not saved_conf:
    st.info("Configure the form and run clustering to generate a projection.")
    st.stop()

active_conf = ClusterConf(**saved_conf)

with st.expander("Current Cluster Config", expanded=False):
    st.json(asdict(active_conf))

with st.spinner("Collecting sentence embeddings and running t-SNE..."):
    results = run_clustering_job(
        backbone_name=active_conf.backbone_name,
        langs=tuple(active_conf.langs),
        pairs=tuple(active_conf.pairs),
        batch_size=active_conf.batch_size,
        max_length=active_conf.max_length,
        num_batches=active_conf.num_batches,
        layer_idx=active_conf.layer_idx,
        points_per_lang_cap=active_conf.points_per_lang_cap,
        shuffle_buffer_size=active_conf.shuffle_buffer_size,
        stream_shuffle_seed=active_conf.stream_shuffle_seed,
        random_seed=active_conf.random_seed,
    )

coords = results["coords"]
langs = results["langs"]
texts = results["texts"]
counts_by_lang = results["counts_by_lang"]

scatter_df = build_scatter_frame(coords, langs, texts)
counts_df = build_counts_frame(counts_by_lang, active_conf.langs)
centroid_df = build_centroid_distance_frame(coords, langs)
figure = build_scatter_figure(scatter_df)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Points", len(scatter_df))
metric_b.metric("Languages", len(set(langs)))
metric_c.metric("Layer", active_conf.layer_idx)

st.plotly_chart(figure, use_container_width=True)

tab_counts, tab_distances, tab_samples = st.tabs(
    ["Counts", "Centroid Distances", "Sample Sentences"]
)

with tab_counts:
    st.dataframe(counts_df, use_container_width=True, hide_index=True)

with tab_distances:
    st.dataframe(centroid_df, use_container_width=True, hide_index=True)

with tab_samples:
    st.dataframe(
        scatter_df[["language", "text"]],
        use_container_width=True,
        hide_index=True,
    )
