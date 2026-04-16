import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset import BalancedNLLBDataset
from backbone import MLLMBackbone


@dataclass
class ClusterConf:
    backbone_name: str
    langs: list[str]
    pairs: list[str]
    batch_size: int
    max_length: int
    num_batches: int
    layer_idx: int
    points_per_lang_cap: int | None
    shuffle_buffer_size: int | None
    stream_shuffle_seed: int | None
    random_seed: int
    output_path: str = "tsne_languages.png"


def collate_records(batch):
    return {
        "texts": [item["text"] for item in batch],
        "langs": [item["lang"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }


@torch.no_grad()
def extract_sentence_embeddings_last_layer(
    backbone: MLLMBackbone,
    texts: list[str],
    max_length: int = 128,
    layer_idx: int = 12,
) -> torch.Tensor:
    """
    Returns mean-pooled sentence embeddings of shape (B, H)
    from the requested hidden-state layer.
    """
    tokenizer = backbone.tokenizer
    model = backbone.model
    device = backbone.device

    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states[layer_idx]   # (B, T, H)

    attention_mask = batch["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)

    masked_hidden = hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)  # (B, H)
    counts = attention_mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)

    sent_embs = summed / counts
    sent_embs = F.normalize(sent_embs, p=2, dim=-1)

    return sent_embs


def collect_embeddings(
    backbone: MLLMBackbone,
    loader: DataLoader,
    langs_to_collect: list[str],
    num_batches: int,
    max_length: int,
    layer_idx: int,
    cap_per_lang: int | None = None,
):
    """
    Collect sentence embeddings and metadata from the data stream.
    """
    embeddings = []
    langs = []
    texts = []

    counts_by_lang = defaultdict(int)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        batch_texts = batch["texts"]
        batch_langs = batch["langs"]

        embs = extract_sentence_embeddings_last_layer(
            backbone=backbone,
            texts=batch_texts,
            max_length=max_length,
            layer_idx=layer_idx,
        )  # (B, H)

        embs = embs.detach().cpu()

        for i in range(len(batch_texts)):
            lang = batch_langs[i]

            if cap_per_lang is not None and counts_by_lang[lang] >= cap_per_lang:
                continue

            embeddings.append(embs[i].numpy())
            langs.append(lang)
            texts.append(batch_texts[i])
            counts_by_lang[lang] += 1

        if batch_idx % 10 == 0:
            print(f"[collect] processed batch {batch_idx}")

        # stop early if all languages hit cap
        if cap_per_lang is not None:
            if all(counts_by_lang[lang] >= cap_per_lang for lang in langs_to_collect):
                break

    embeddings = np.stack(embeddings, axis=0)

    return embeddings, langs, texts, counts_by_lang


def run_tsne(embeddings: np.ndarray, random_state: int = 42):
    """
    Run t-SNE on sentence embeddings.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    coords = tsne.fit_transform(embeddings)
    return coords


def plot_tsne(coords: np.ndarray, langs: list[str], output_path: str = "tsne_languages.png"):
    """
    Scatter plot colored by language.
    """
    unique_langs = sorted(set(langs))
    lang_to_idx = {lang: i for i, lang in enumerate(unique_langs)}

    plt.figure(figsize=(12, 9))

    for lang in unique_langs:
        idxs = [i for i, l in enumerate(langs) if l == lang]
        pts = coords[idxs]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=14,
            alpha=0.7,
            label=lang,
        )

    plt.title("mDeBERTa sentence embeddings — t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def print_centroid_distances(coords: np.ndarray, langs: list[str]):
    """
    Print pairwise distances between language centroids in t-SNE space.
    This is only a rough visualization aid, not a rigorous metric.
    """
    unique_langs = sorted(set(langs))
    centroids = {}

    for lang in unique_langs:
        idxs = [i for i, l in enumerate(langs) if l == lang]
        centroids[lang] = coords[idxs].mean(axis=0)

    print("\nLanguage centroid distances in t-SNE space:")
    for i, lang_a in enumerate(unique_langs):
        for lang_b in unique_langs[i + 1:]:
            dist = np.linalg.norm(centroids[lang_a] - centroids[lang_b])
            print(f"{lang_a:10s} <-> {lang_b:10s}: {dist:.4f}")


def main():
    parser = argparse.ArgumentParser(
        "Cluster",
        description="Run multilingual sentence embedding clustering",
    )
    parser.add_argument("config_path", help="path to clustering config. Required.")

    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        conf = ClusterConf(**json.load(stream))

    random.seed(conf.random_seed)
    np.random.seed(conf.random_seed)
    torch.manual_seed(conf.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BalancedNLLBDataset(
        pair_configs=conf.pairs,
        langs=conf.langs,
        shuffle_buffer_size=conf.shuffle_buffer_size,
        seed=conf.stream_shuffle_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_records,
        num_workers=0,
    )

    backbone = MLLMBackbone(device, conf.backbone_name)

    embeddings, langs, texts, counts_by_lang = collect_embeddings(
        backbone=backbone,
        loader=loader,
        langs_to_collect=conf.langs,
        num_batches=conf.num_batches,
        max_length=conf.max_length,
        layer_idx=conf.layer_idx,
        cap_per_lang=conf.points_per_lang_cap,
    )

    print("\nCollected sentence counts by language:")
    for lang in conf.langs:
        print(f"{lang}: {counts_by_lang[lang]}")

    print(f"\nEmbeddings shape: {embeddings.shape}")

    coords = run_tsne(embeddings, random_state=conf.random_seed)

    plot_tsne(coords, langs, output_path=conf.output_path)
    print_centroid_distances(coords, langs)

    # Print a few example sentences per language for sanity
    print("\nSample collected sentences:")
    shown = defaultdict(int)
    for lang, text in zip(langs, texts):
        if shown[lang] < 2:
            print(f"[{lang}] {text}")
            shown[lang] += 1
        if all(shown[l] >= 2 for l in set(langs)):
            break


if __name__ == "__main__":
    main()
