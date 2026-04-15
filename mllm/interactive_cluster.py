import random
import textwrap

import matplotlib


def _select_interactive_backend() -> None:
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        return

    for candidate in ("TkAgg", "QtAgg", "MacOSX"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


_select_interactive_backend()

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from backbone import MLLMBackbone
from cluster import (
    BATCH_SIZE,
    LANGS,
    LAYER_IDX,
    MAX_LENGTH,
    NUM_BATCHES,
    PAIR_CONFIGS,
    POINTS_PER_LANG_CAP,
    RANDOM_SEED,
    SHUFFLE_BUFFER_SIZE,
    STREAM_SHUFFLE_SEED,
    collect_embeddings,
    collate_records,
    print_centroid_distances,
    run_tsne,
)
from dataset import BalancedNLLBDataset


def _format_hover_text(lang: str, text: str, width: int = 70) -> str:
    wrapped = textwrap.fill(text, width=width)
    return f"{lang}\n\n{wrapped}"


def plot_tsne_interactive(coords: np.ndarray, langs: list[str], texts: list[str]) -> None:
    unique_langs = sorted(set(langs))
    cmap = plt.get_cmap("tab10", len(unique_langs))

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter_entries = []

    for color_idx, lang in enumerate(unique_langs):
        idxs = [i for i, value in enumerate(langs) if value == lang]
        pts = coords[idxs]
        lang_texts = [texts[i] for i in idxs]

        scatter = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=18,
            alpha=0.75,
            label=lang,
            color=cmap(color_idx),
            picker=True,
        )
        scatter_entries.append((scatter, pts, lang, lang_texts))

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.4", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "0.3"},
    )
    annot.set_visible(False)

    def hide_annotation() -> None:
        if annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

    def on_hover(event) -> None:
        if event.inaxes != ax:
            hide_annotation()
            return

        for scatter, pts, lang, lang_texts in scatter_entries:
            contains, details = scatter.contains(event)
            point_indices = details.get("ind", [])
            if not contains or len(point_indices) == 0:
                continue

            point_idx = point_indices[0]
            annot.xy = pts[point_idx]
            annot.set_text(_format_hover_text(lang, lang_texts[point_idx]))
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return

        hide_annotation()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("figure_leave_event", lambda event: hide_annotation())

    ax.set_title("mDeBERTa sentence embeddings — t-SNE (interactive)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.5, fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Use the Matplotlib toolbar to zoom or pan. Hover a point to inspect its language and sentence.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        raise RuntimeError(
            "An interactive Matplotlib backend is required. "
            f"Current backend: {matplotlib.get_backend()}. "
            "Run this in a local GUI session with Tk/Qt support."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    dataset = BalancedNLLBDataset(
        pair_configs=PAIR_CONFIGS,
        langs=LANGS,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        seed=STREAM_SHUFFLE_SEED,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_records,
        num_workers=0,
    )

    backbone = MLLMBackbone(device)

    embeddings, langs, texts, counts_by_lang = collect_embeddings(
        backbone=backbone,
        loader=loader,
        num_batches=NUM_BATCHES,
        max_length=MAX_LENGTH,
        layer_idx=LAYER_IDX,
        cap_per_lang=POINTS_PER_LANG_CAP,
    )

    print("\nCollected sentence counts by language:")
    for lang in LANGS:
        print(f"{lang}: {counts_by_lang[lang]}")

    print(f"\nEmbeddings shape: {embeddings.shape}")

    coords = run_tsne(embeddings, random_state=RANDOM_SEED)
    print_centroid_distances(coords, langs)
    plot_tsne_interactive(coords, langs, texts)


if __name__ == "__main__":
    main()
