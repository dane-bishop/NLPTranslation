"""
sparse_autoencoders.py

Interactive Streamlit tab for exploring SAE feature activations over
NLLB / FLORES-200 sentences.

Expects the output directory produced by precompute_embeddings.py:
    sentence_projections.npy   [N, 2]   t-SNE coords per sentence
    feature_projections.npy    [F, 2]   t-SNE coords per feature
    sentences.json             list[str]
    topk_sentence_lookup.json  list[F] of list[ [sent_idx, token_str] ]
    metadata.json

Run standalone:
    streamlit run sparse_autoencoders.py -- --cache_dir ./cached_embeddings
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


LANG_COLOURS = [
    "#7DF9C4", "#FF6B6B", "#FFD166", "#6A9EFF", "#C77DFF",
    "#FF9A3C", "#4DD9D9", "#F7717D", "#A8DADC", "#E9C46A",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500;700&display=swap');
html, body, [data-testid="stApp"] {
    background: #0d0f14; color: #e0e4ef;
    font-family: 'IBM Plex Sans', sans-serif;
}
h1,h2,h3,h4 { font-family:'IBM Plex Mono',monospace; letter-spacing:-0.02em; }
.sentence-row {
    background:#161a24; border:1px solid #2a2f3f; border-radius:8px;
    padding:12px 16px; margin-bottom:8px; font-size:0.92rem; line-height:1.6;
}
.sentence-row .rank-badge {
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    color:#5a6080; margin-bottom:4px;
}
.token-highlight {
    background:#2a1f00; color:#FFD166; border-radius:3px;
    padding:1px 4px; font-family:'IBM Plex Mono',monospace; font-weight:600;
}
.sentence-box {
    background:#161a24; border-left:4px solid #7DF9C4;
    padding:14px 18px; border-radius:0 8px 8px 0;
    font-size:1.05rem; line-height:1.65; margin-bottom:18px;
}
.lang-tag {
    display:inline-block; background:#1e2335; border:1px solid #3a4060;
    border-radius:4px; padding:2px 8px; font-family:'IBM Plex Mono',monospace;
    font-size:0.75rem; color:#a0aacc; margin-bottom:10px;
}
.feature-header {
    font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:#7DF9C4;
    margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #2a2f3f;
}
.dead-feature {
    color:#5a6080; font-family:'IBM Plex Mono',monospace; font-size:0.85rem;
    padding:12px; background:#161a24; border-radius:6px;
}
</style>
"""


def discover_models(root_dir: str) -> dict[str, Path]:
    """
    Scan *root_dir* for sub-directories that look like model caches.
    A valid model directory must contain at least sentence_projections.npy
    and sentences.json.  Returns {model_name: Path} sorted alphabetically.
    """
    root = Path(root_dir)
    required = {"sentence_projections.npy", "sentences.json"}
    models: dict[str, Path] = {}
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and required.issubset({f.name for f in child.iterdir()}):
                models[child.name] = child
    return models


@st.cache_data(show_spinner="Loading data ...")
def load_data(cache_dir: str):
    p = Path(cache_dir)
    sentence_proj = np.load(p / "sentence_projections.npy")   # [N, 2]
    feature_proj  = np.load(p / "feature_projections.npy")    # [F, 2]
    alive_idx = np.load(p / "alive_feature_indices.npy")
    with open(p / "sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)
    with open(p / "topk_sentence_lookup.json", encoding="utf-8") as f:
        topk_lookup = json.load(f)
    with open(p / "metadata.json") as f:
        meta = json.load(f)
    lang_tags = meta.get("lang_tags", ["unknown"] * len(sentences))
    return sentence_proj, feature_proj, sentences, lang_tags, topk_lookup, alive_idx, meta


def _base_layout(height: int = 500) -> dict:
    return dict(
        paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14",
        font=dict(family="IBM Plex Mono", color="#a0aacc", size=11),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        legend=dict(bgcolor="#161a24", bordercolor="#2a2f3f", borderwidth=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
    )


def build_sentence_plot(
    coords: np.ndarray,
    sentences: list[str],
    lang_tags: list[str],
    unique_langs: list[str],
    selected_idx: Optional[int],
) -> go.Figure:
    lang_colour = {lang: LANG_COLOURS[i % len(LANG_COLOURS)] for i, lang in enumerate(unique_langs)}
    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "lang": lang_tags,
        "text": [s[:90] + "..." if len(s) > 90 else s for s in sentences],
        "idx":  list(range(len(sentences))),
    })
    fig = px.scatter(df, x="x", y="y", color="lang",
                     color_discrete_map=lang_colour,
                     hover_data={"text": True, "lang": True, "x": False, "y": False, "idx": False},
                     custom_data=["idx"])
    fig.update_traces(marker=dict(size=5, opacity=0.8, line=dict(width=0)))
    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[coords[selected_idx, 0]], y=[coords[selected_idx, 1]],
            mode="markers",
            marker=dict(size=13, color="#FF6B6B", symbol="star",
                        line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(**_base_layout())
    return fig


def build_feature_plot(
    coords: np.ndarray,
    topk_lookup: list,
    selected_feat: Optional[int],
    alive_idx
) -> go.Figure:
    n = coords.shape[0]

    def token_preview(f: int) -> str:
        if f >= len(topk_lookup) or not topk_lookup[f]:
            return "dead"
        return " / ".join(e[1] for e in topk_lookup[f][:3] if e[1])

    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "feature": [int(alive_idx[f]) for f in range(n)],  # original idx, for hover label
        "local_idx": list(range(n)),                        # alive-local idx, for click handling
        "active":  ["active" if (f < len(topk_lookup) and topk_lookup[f]) else "dead"
                    for f in range(n)],
        "preview": [token_preview(f) for f in range(n)],
    })
    fig = px.scatter(df, x="x", y="y", color="active",
                     color_discrete_map={"active": "#7DF9C4", "dead": "#2a2f3f"},
                     hover_data={"feature": True, "preview": True,
                                 "active": False, "x": False, "y": False,
                                 "local_idx": False},
                     custom_data=["local_idx"])  # click returns local index
    fig.update_traces(marker=dict(size=5, opacity=0.85, line=dict(width=0)))
    if selected_feat is not None and selected_feat < n:
        fig.add_trace(go.Scatter(
            x=[coords[selected_feat, 0]], y=[coords[selected_feat, 1]],
            mode="markers",
            marker=dict(size=13, color="#FF6B6B", symbol="star",
                        line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(**_base_layout())
    return fig


def _highlight_token(sentence: str, token: str) -> str:
    """Wrap first occurrence of token (stripping SentencePiece/WordPiece prefix) in a highlight span."""
    if not token:
        return sentence
    clean = token.lstrip("\u2581").lstrip("##").strip()
    if not clean:
        return sentence
    try:
        highlighted = re.sub(
            re.escape(clean),
            lambda m: f'<span class="token-highlight">{m.group(0)}</span>',
            sentence, count=1, flags=re.IGNORECASE,
        )
        return highlighted
    except re.error:
        return sentence


def render_sentence_detail(sentences: list[str], lang_tags: list[str], idx: int):
    lang = lang_tags[idx]
    st.markdown(f'<div class="lang-tag">{lang}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sentence-box">{sentences[idx]}</div>', unsafe_allow_html=True)
    st.caption(f"sentence index: {idx}")


def render_feature_detail(
    feature_idx: int,
    topk_lookup: list,
    sentences: list[str],
    lang_tags: list[str],
    alive_idx=None,
):
    orig_idx = int(alive_idx[feature_idx]) if alive_idx is not None and feature_idx < len(alive_idx) else feature_idx
    st.markdown(f'<div class="feature-header">Feature {orig_idx} (local idx {feature_idx})</div>', unsafe_allow_html=True)

    if feature_idx >= len(topk_lookup) or not topk_lookup[feature_idx]:
        st.markdown('<div class="dead-feature">No activating sentences for this feature.</div>',
                    unsafe_allow_html=True)
        return

    entries = topk_lookup[feature_idx]
    st.caption(f"{len(entries)} stored examples")

    for rank, entry in enumerate(entries, start=1):
        sent_idx, token_str = int(entry[0]), entry[1]
        sentence = sentences[sent_idx] if sent_idx < len(sentences) else "(out of range)"
        lang     = lang_tags[sent_idx] if sent_idx < len(lang_tags) else "?"
        body     = _highlight_token(sentence, token_str)
        token_label = (f'peak token: <span class="token-highlight">{token_str}</span>'
                       if token_str else "")
        st.markdown(
            f'<div class="sentence-row">'
            f'<div class="rank-badge">#{rank} &nbsp; {lang} &nbsp; sent {sent_idx}'
            f'&nbsp;&nbsp; {token_label}</div>'
            f'{body}</div>',
            unsafe_allow_html=True,
        )


def render_tab(cache_dir: str = "./cached_embeddings"):
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Settings")
        root_dir_input = st.text_input("Root embeddings directory", value=cache_dir)

        # ── Model selector ────────────────────────────────────────────────────
        models = discover_models(root_dir_input)
        if models:
            model_names = list(models.keys())

            # Restore previous selection if it still exists in the new root
            prev = st.session_state.get("selected_model", model_names[0])
            default_idx = model_names.index(prev) if prev in model_names else 0

            selected_model = st.radio(
                "Model",
                options=model_names,
                index=default_idx,
                key="model_radio",
                help="Switch between available model embeddings. "
                     "Each folder under the root directory is listed here.",
            )

            # Reset per-model UI state when the model changes
            if st.session_state.get("selected_model") != selected_model:
                st.session_state["selected_model"] = selected_model
                st.session_state["sel_sentence"] = 0
                st.session_state["sel_feature"] = 0
                st.rerun()

            st.session_state["selected_model"] = selected_model
            active_cache_dir = str(models[selected_model])
        else:
            # No sub-directories found – treat root itself as the cache dir
            st.info(
                "No model sub-directories detected. Place each model's embeddings "
                "in a sub-folder of the root directory (e.g. root/nllb, root/mmbert), "
                "or point directly to a cache directory above."
            )
            active_cache_dir = root_dir_input

        st.markdown("---")
        plot_mode = st.selectbox(
            "View",
            ["Sentences", "Features"],
            help="Sentences: points coloured by language.\n"
                 "Features: click a point to see its top activating sentences.",
        )
        st.markdown("---")
        st.markdown("### Search")
        search_query = st.text_input("Filter sentences", placeholder="e.g. bank, river ...")

    try:
        sentence_proj, feature_proj, sentences, lang_tags, topk_lookup, alive_idx, meta = \
            load_data(active_cache_dir)
    except FileNotFoundError as e:
        st.error(f"Cache not found: {e}\n\nRun `precompute_embeddings.py` first.")
        return

    unique_langs = sorted(set(lang_tags))
    lang_filter = st.sidebar.multiselect(
        "Filter by language", options=unique_langs, default=unique_langs, key="lang_filter"
    )

    # ── Header ────────────────────────────────────────────────────────────────
    active_model_label = st.session_state.get("selected_model", Path(active_cache_dir).name)
    st.markdown(
        f"# SAE Feature Explorer "
        f"<span style='font-family:IBM Plex Mono;font-size:0.7rem;color:#5a6080;"
        f"background:#161a24;border:1px solid #2a2f3f;border-radius:4px;"
        f"padding:3px 8px;vertical-align:middle'>{active_model_label}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#5a6080'>"
        f"model: {meta['config'].get('backbone_name','?')} &nbsp;|&nbsp; "
        f"layer {meta.get('layer_idx','?')} &nbsp;|&nbsp; "
        f"sae: {meta['config'].get('sae_type','?')} &nbsp;|&nbsp; "
        f"{len(sentences):,} sentences &nbsp;|&nbsp; "
        f"{feature_proj.shape[0]:,} features"
        f"</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Filter indices
    sent_indices = [
        i for i, (s, lt) in enumerate(zip(sentences, lang_tags))
        if (not lang_filter or lt in lang_filter)
        and (not search_query or search_query.lower() in s.lower())
    ]
    if not sent_indices and plot_mode == "Sentences":
        st.warning("No sentences match the current filters.")
        return

    f_sentences = [sentences[i] for i in sent_indices]
    f_lang_tags = [lang_tags[i] for i in sent_indices]
    f_sent_proj = sentence_proj[sent_indices]

    # Session state defaults
    if "sel_sentence" not in st.session_state:
        st.session_state["sel_sentence"] = 0
    if "sel_feature" not in st.session_state:
        st.session_state["sel_feature"] = 0
    if "lang_filter" not in st.session_state:
        st.session_state["lang_filter"] = unique_langs

    col_plot, col_detail = st.columns([3, 2], gap="large")

    # ── SENTENCE MODE ─────────────────────────────────────────────────────────
    if plot_mode == "Sentences":
        with col_plot:
            st.markdown("#### Sentence space")
            st.markdown(
                "<span style='font-size:0.75rem;color:#5a6080'>"
                "One point per sentence, coloured by language. Click to inspect."
                "</span>", unsafe_allow_html=True,
            )
            sel_global = st.session_state["sel_sentence"]
            sel_local = sent_indices.index(sel_global) if sel_global in sent_indices else None
            fig = build_sentence_plot(f_sent_proj, f_sentences, f_lang_tags, unique_langs, sel_local)
            clicked = st.plotly_chart(fig, use_container_width=True,
                                      on_select="rerun", selection_mode="points",
                                      key="sent_scatter")
            if clicked and clicked.get("selection", {}).get("points"):
                pt = clicked["selection"]["points"][0]
                if pt.get("customdata"):
                    local_new = int(pt["customdata"][0])
                    global_new = sent_indices[local_new]
                    if global_new != st.session_state["sel_sentence"]:
                        st.session_state["sel_sentence"] = global_new
                        st.rerun()
            legend_cols = st.columns(min(5, len(unique_langs)))
            for i, lang in enumerate(unique_langs):
                legend_cols[i % 5].markdown(
                    f"<span style='color:{LANG_COLOURS[i % len(LANG_COLOURS)]};"
                    f"font-family:IBM Plex Mono;font-size:0.72rem'>■ {lang}</span>",
                    unsafe_allow_html=True,
                )

        with col_detail:
            st.markdown("#### Sentence inspector")
            local_idx = st.number_input(
                "Sentence index (or click plot)",
                min_value=0, max_value=len(f_sentences) - 1,
                value=min(sel_local or 0, len(f_sentences) - 1),
                step=1, key="sent_input",
            )
            st.session_state["sel_sentence"] = sent_indices[local_idx]
            render_sentence_detail(f_sentences, f_lang_tags, local_idx)

    # ── FEATURE MODE ──────────────────────────────────────────────────────────
    else:
        n_features = feature_proj.shape[0]
        with col_plot:
            st.markdown("#### Feature space")
            st.markdown(
                "<span style='font-size:0.75rem;color:#5a6080'>"
                "One point per SAE feature. Active features in green. "
                "Hover to preview top tokens; click to inspect."
                "</span>", unsafe_allow_html=True,
            )
            sel_feat = st.session_state["sel_feature"]
            fig = build_feature_plot(feature_proj, topk_lookup, sel_feat, alive_idx)
            clicked = st.plotly_chart(fig, use_container_width=True,
                                      on_select="rerun", selection_mode="points",
                                      key="feat_scatter")
            if clicked and clicked.get("selection", {}).get("points"):
                pt = clicked["selection"]["points"][0]
                if pt.get("customdata"):
                    new = int(pt["customdata"][0])
                    if new != st.session_state["sel_feature"]:
                        st.session_state["sel_feature"] = new
                        st.rerun()
            active_count = sum(1 for f in topk_lookup if f)
            st.caption(f"{active_count} active / {n_features} total features")

        n_alive = len(topk_lookup)
        clamped = min(st.session_state["sel_feature"], n_alive - 1)
        with col_detail:
            st.markdown("#### Feature inspector")
            feat_idx = st.number_input(
                "Feature index (or click plot)",
                min_value=0, max_value=n_alive - 1,
                value=clamped,
                step=1, key="feat_input",
            )
            if "sel_feature" not in st.session_state:
                st.session_state["sel_feature"] = 0
            else:
                st.session_state["sel_feature"] = min(
                    st.session_state["sel_feature"], len(topk_lookup) - 1
                )
            render_feature_detail(feat_idx, topk_lookup, sentences, lang_tags, alive_idx)

    # ── Stats ─────────────────────────────────────────────────────────────────
    with st.expander("Global feature statistics", expanded=False):
        active_feats = sum(1 for e in topk_lookup if e)
        dead_feats   = len(topk_lookup) - active_feats
        c1, c2, c3 = st.columns(3)
        c1.metric("Total features",  feature_proj.shape[0])
        c2.metric("Active features", active_feats)
        c3.metric("Dead features",   dead_feats)

        all_tokens = [e[1] for entries in topk_lookup for e in entries if e[1]]
        if all_tokens:
            st.markdown("**Most frequent peak tokens**")
            df_tok = pd.DataFrame(Counter(all_tokens).most_common(30),
                                  columns=["Token", "Count"])
            st.dataframe(df_tok, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="artifacts/sae_features/",
        help="Root directory containing one sub-folder per model "
             "(e.g. root/nllb, root/mmbert).  "
             "Falls back to treating this path as a direct cache dir if no "
             "sub-directories with the required files are found.",
    )
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="SAE Feature Explorer", layout="wide",
                       initial_sidebar_state="expanded")
    render_tab(cache_dir=args.cache_dir)
