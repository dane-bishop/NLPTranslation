"""
streamlit_tab.py

Interactive Streamlit tab for exploring SAE feature activations over
NLLB / FLORES-200 sentences.

Drop this file into your Streamlit app and call  render_tab()  from the
parent page, or run it standalone:

    streamlit run streamlit_tab.py

Expects the output directory produced by precompute_embeddings.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from openTSNE import TSNE

# Optional – silently degrade to PCA if umap-learn is absent
try:
    import umap  # type: ignore
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark-research aesthetic)
# ──────────────────────────────────────────────────────────────────────────────

LANG_COLOURS = [
    "#7DF9C4", "#FF6B6B", "#FFD166", "#6A9EFF", "#C77DFF",
    "#FF9A3C", "#4DD9D9", "#F7717D", "#A8DADC", "#E9C46A",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');

html, body, [data-testid="stApp"] {
    background: #0d0f14;
    color: #e0e4ef;
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.02em;
}
.feature-card {
    background: #161a24;
    border: 1px solid #2a2f3f;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.83rem;
}
.feature-card .rank { color: #7DF9C4; font-weight: 600; }
.feature-card .val  { color: #FFD166; }
.feature-bar-bg {
    background: #2a2f3f;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
}
.feature-bar-fg {
    background: linear-gradient(90deg, #7DF9C4, #6A9EFF);
    border-radius: 4px;
    height: 6px;
}
.sentence-box {
    background: #161a24;
    border-left: 4px solid #7DF9C4;
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    font-size: 1.05rem;
    line-height: 1.65;
    margin-bottom: 18px;
}
.lang-tag {
    display: inline-block;
    background: #1e2335;
    border: 1px solid #3a4060;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #a0aacc;
    margin-bottom: 10px;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading activations ...")
def load_data(cache_dir: str):
    p = Path(cache_dir)
    activations = np.load(p / "activations.npy")  # [N, D]

    with open(p / "sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    with open(p / "metadata.json") as f:
        meta = json.load(f)

    lang_tags = meta.get("lang_tags", ["unknown"] * len(sentences))
    return activations, sentences, lang_tags, meta


@st.cache_data(show_spinner="Projecting to 2-D ...")
def compute_projection(
    activations: np.ndarray,
    method: str,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    pca_components: int = 50,
) -> np.ndarray:
    """Return 2-D projection of SAE activation vectors."""
    # PCA pre-reduction for speed; activations are often high-dimensional and sparse
    n_comp = min(pca_components, activations.shape[1], activations.shape[0] - 1)
    pca_coords = PCA(n_components=n_comp, random_state=42).fit_transform(activations)

    if method == "UMAP" and _UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
            metric="cosine",
        )
        return reducer.fit_transform(pca_coords)

    elif method == "TSNE":
        tsne = TSNE(perplexity = 50, metric="cosine")
        return tsne.fit(pca_coords)

    # Fall back to PCA 2-D
    return PCA(n_components=2, random_state=42).fit_transform(activations)


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def feature_bar(rank: int, feat_idx: int, value: float, max_val: float) -> str:
    pct = int(100 * value / max_val) if max_val > 0 else 0
    return (
        f'<div class="feature-card">'
        f'  <span class="rank">#{rank}</span> &nbsp; feature <b>{feat_idx}</b>'
        f'  &nbsp; <span class="val">{value:.4f}</span>'
        f'  <div class="feature-bar-bg"><div class="feature-bar-fg" style="width:{pct}%"></div></div>'
        f'</div>'
    )


def render_sentence_panel(
    sentences: list[str],
    activations: np.ndarray,
    lang_tags: list[str],
    selected_idx: int,
    top_k: int,
):
    sentence  = sentences[selected_idx]
    act       = activations[selected_idx]           # [D]
    lang      = lang_tags[selected_idx]

    st.markdown(f'<div class="lang-tag">{lang}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sentence-box">{sentence}</div>', unsafe_allow_html=True)

    st.markdown("#### Top activating features")
    top_indices = np.argsort(act)[::-1][:top_k]
    max_val     = act[top_indices[0]] if len(top_indices) else 1.0

    html_blocks = "".join(
        feature_bar(i + 1, int(feat_idx), float(act[feat_idx]), max_val)
        for i, feat_idx in enumerate(top_indices)
        if act[feat_idx] > 0
    )
    if not html_blocks:
        st.info("All feature activations are zero for this sentence.")
    else:
        st.markdown(html_blocks, unsafe_allow_html=True)


def render_cluster_plot(
    coords_2d: np.ndarray,
    sentences: list[str],
    lang_tags: list[str],
    activations: np.ndarray,
    selected_idx: Optional[int],
    colour_by: str,
    feature_idx: Optional[int],
    unique_langs: list[str],
):
    lang_to_colour = {lang: LANG_COLOURS[i % len(LANG_COLOURS)] for i, lang in enumerate(unique_langs)}

    if colour_by == "Language":
        colours = [lang_to_colour[lt] for lt in lang_tags]
        colour_label = lang_tags
        colour_discrete = lang_to_colour
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "lang": lang_tags,
            "sentence": [s[:80] + "…" if len(s) > 80 else s for s in sentences],
            "idx": list(range(len(sentences))),
        })
        fig = px.scatter(
            df, x="x", y="y",
            color="lang",
            color_discrete_map=colour_discrete,
            hover_data={"sentence": True, "lang": True, "x": False, "y": False, "idx": False},
            custom_data=["idx"],
        )

    elif colour_by == "Feature activation" and feature_idx is not None:
        feat_vals = activations[:, feature_idx]
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "activation": feat_vals,
            "lang": lang_tags,
            "sentence": [s[:80] + "…" if len(s) > 80 else s for s in sentences],
            "idx": list(range(len(sentences))),
        })
        fig = px.scatter(
            df, x="x", y="y",
            color="activation",
            color_continuous_scale=[[0, "#161a24"], [0.3, "#2a2f3f"], [0.7, "#6A9EFF"], [1, "#7DF9C4"]],
            hover_data={"sentence": True, "lang": True, "activation": ":.4f", "x": False, "y": False, "idx": False},
            custom_data=["idx"],
        )
    else:
        # Sparsity (fraction of active features)
        sparsity = (activations > 0).mean(axis=1)
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "sparsity": sparsity,
            "lang": lang_tags,
            "sentence": [s[:80] + "…" if len(s) > 80 else s for s in sentences],
            "idx": list(range(len(sentences))),
        })
        fig = px.scatter(
            df, x="x", y="y",
            color="sparsity",
            color_continuous_scale="Teal",
            hover_data={"sentence": True, "lang": True, "sparsity": ":.3f", "x": False, "y": False, "idx": False},
            custom_data=["idx"],
        )

    # Style
    fig.update_traces(
        marker=dict(size=5, opacity=0.8, line=dict(width=0)),
    )

    # Highlight selected point
    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[coords_2d[selected_idx, 0]],
            y=[coords_2d[selected_idx, 1]],
            mode="markers",
            marker=dict(size=14, color="#FF6B6B", symbol="star",
                        line=dict(color="white", width=1.5)),
            name="selected",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#0d0f14",
        font=dict(family="IBM Plex Mono", color="#a0aacc", size=11),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(bgcolor="#161a24", bordercolor="#2a2f3f", borderwidth=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=460,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def render_tab(cache_dir: str = "./cached_embeddings"):
    """Call this from your parent Streamlit app inside a st.tab block."""

    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar / settings ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")
        cache_dir_input = st.text_input("Cache directory", value=cache_dir)
        top_k = st.slider("Top-K features", min_value=5, max_value=50, value=15)
        proj_method = st.selectbox(
            "Projection method",
            ["UMAP", "PCA", "TSNE"] if _UMAP_AVAILABLE else ["PCA", "TSNE"],
        )
        if proj_method == "UMAP":
            n_neighbors = st.slider("UMAP n_neighbors", 5, 50, 20)
            min_dist    = st.slider("UMAP min_dist", 0.01, 0.5, 0.1, step=0.01)
        else:
            n_neighbors, min_dist = 20, 0.1

        colour_by = st.selectbox(
            "Colour by",
            ["Language", "Feature activation", "Sparsity"],
        )
        feature_idx = None
        if colour_by == "Feature activation":
            feature_idx = st.number_input("Feature index", min_value=0, value=0, step=1)

        st.markdown("---")
        st.markdown("### Search")
        search_query = st.text_input("Filter sentences", placeholder="e.g. bank, river …")
        lang_filter  = st.multiselect("Filter by language", options=[], default=[])  # populated below

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        activations, sentences, lang_tags, meta = load_data(cache_dir_input)
    except FileNotFoundError as e:
        st.error(f"Cache not found: {e}\n\nRun `precompute_embeddings.py` first.")
        return

    unique_langs = sorted(set(lang_tags))
    # Back-fill sidebar multiselect options
    lang_filter = st.sidebar.multiselect(
        "Filter by language", options=unique_langs,
        default=unique_langs, key="lang_filter_2"
    )

    # ── Apply filters ──────────────────────────────────────────────────────────
    indices = [
        i for i, (s, lt) in enumerate(zip(sentences, lang_tags))
        if (not lang_filter or lt in lang_filter)
        and (not search_query or search_query.lower() in s.lower())
    ]

    if not indices:
        st.warning("No sentences match the current filters.")
        return

    f_sentences   = [sentences[i]   for i in indices]
    f_lang_tags   = [lang_tags[i]   for i in indices]
    f_activations = activations[indices]

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("# SAE Feature Explorer")
    st.markdown(
        f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#5a6080'>"
        f"model: {meta['config'].get('backbone_name','?')} &nbsp;|&nbsp; "
        f"layer {meta.get('layer_idx','?')} &nbsp;|&nbsp; "
        f"sae: {meta['config'].get('sae_type','?')} &nbsp;|&nbsp; "
        f"{len(f_sentences):,} sentences shown"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── 2-D projection ─────────────────────────────────────────────────────────
    with st.spinner("Computing 2-D projection ..."):
        coords_2d = compute_projection(
            f_activations, proj_method, n_neighbors, min_dist
        )

    # ── Layout: left = cluster plot, right = sentence panel ────────────────────
    col_plot, col_detail = st.columns([3, 2], gap="large")

    with col_plot:
        st.markdown("#### SAE feature space")
        st.markdown(
            "<span style='font-size:0.75rem;color:#5a6080'>"
            "Click a point to inspect its top-activating features"
            "</span>",
            unsafe_allow_html=True,
        )

        if "selected_local" not in st.session_state:
            st.session_state["selected_local"] = 0

        fig = render_cluster_plot(
            coords_2d, f_sentences, f_lang_tags, f_activations,
            st.session_state["selected_local"], colour_by,
            int(feature_idx) if feature_idx is not None else None,
            unique_langs,
        )

        clicked = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="scatter",
        )

        # Handle click → update selected
        if clicked and clicked.get("selection") and clicked["selection"].get("points"):
            pt = clicked["selection"]["points"][0]
            # custom_data carries the local index
            if "customdata" in pt and pt["customdata"]:
                new_local = int(pt["customdata"][0])
                if new_local != st.session_state["selected_local"]:
                    st.session_state["selected_local"] = new_local
                    st.rerun()

        # Legend / colour bar explainer
        if colour_by == "Language":
            cols = st.columns(min(5, len(unique_langs)))
            for i, lang in enumerate(unique_langs):
                colour = LANG_COLOURS[i % len(LANG_COLOURS)]
                cols[i % 5].markdown(
                    f"<span style='color:{colour};font-family:IBM Plex Mono;font-size:0.72rem'>■ {lang}</span>",
                    unsafe_allow_html=True,
                )

    with col_detail:
        st.markdown("#### Sentence inspector")

        # Sentence selector fallback
        local_idx = st.session_state.get("selected_local", 0)
        local_idx = st.number_input(
            "Sentence index (or click plot)",
            min_value=0, max_value=len(f_sentences) - 1,
            value=local_idx, step=1,
        )
        st.session_state["selected_local"] = local_idx

        render_sentence_panel(
            f_sentences, f_activations, f_lang_tags,
            local_idx, top_k,
        )

    # ── Feature statistics ─────────────────────────────────────────────────────
    with st.expander("Global feature statistics", expanded=False):
        mean_act = f_activations.mean(axis=0)
        freq_act = (f_activations > 0).mean(axis=0)

        top_mean_feats = np.argsort(mean_act)[::-1][:20]
        top_freq_feats = np.argsort(freq_act)[::-1][:20]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top features by mean activation**")
            df_mean = pd.DataFrame({
                "Feature": top_mean_feats,
                "Mean activation": mean_act[top_mean_feats].round(5),
            })
            st.dataframe(df_mean, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Top features by activation frequency**")
            df_freq = pd.DataFrame({
                "Feature": top_freq_feats,
                "Frequency": freq_act[top_freq_feats].round(4),
            })
            st.dataframe(df_freq, use_container_width=True, hide_index=True)

        # Sparsity histogram
        sparsity = (f_activations > 0).mean(axis=1)
        fig_hist = px.histogram(
            x=sparsity, nbins=60,
            labels={"x": "Fraction of active features"},
            color_discrete_sequence=["#7DF9C4"],
        )
        fig_hist.update_layout(
            paper_bgcolor="#0d0f14", plot_bgcolor="#161a24",
            font=dict(family="IBM Plex Mono", color="#a0aacc"),
            margin=dict(l=20, r=20, t=30, b=20),
            height=280,
            title="Activation sparsity distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="../artifacts/sae_features/nllb/")
    args, _ = parser.parse_known_args()

    st.set_page_config(
        page_title="SAE Feature Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_tab(cache_dir=args.cache_dir)
