# cluster.py

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset import BalancedNLLBDataset
from backbone import MLLMBackbone


PAIR_CONFIGS = [
    "eng_Latn-fra_Latn",
    "eng_Latn-fra_Latn",
    "deu_Latn-eng_Latn",
    "eng_Latn-nld_Latn",
    "eng_Latn-swe_Latn",
    "eng_Latn-spa_Latn",
    "eng_Latn-ita_Latn",
    "eng_Latn-por_Latn",
    "eng_Latn-pol_Latn",
    "ces_Latn-eng_Latn",
]

LANGS = [
    "eng_Latn",
    "fra_Latn",
    "deu_Latn",
    "nld_Latn",
    "swe_Latn",
    "spa_Latn",
    "ita_Latn",
    "por_Latn",
    "pol_Latn",
    "ces_Latn",
]

BATCH_SIZE = 32
MAX_LENGTH = 128
NUM_BATCHES = 320
LAYER_IDX = 12   # final transformer layer output for mDeBERTa-v3-base
POINTS_PER_LANG_CAP = 1000
RANDOM_SEED = 42


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
            if all(counts_by_lang[lang] >= cap_per_lang for lang in LANGS):
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
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = BalancedNLLBDataset(
        pair_configs=PAIR_CONFIGS,
        langs=LANGS,
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

    plot_tsne(coords, langs, output_path=f"tsne_languages_layer{LAYER_IDX}.png")
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