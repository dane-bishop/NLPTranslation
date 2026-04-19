import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import MLLMBackbone
from cluster import collate_records, run_tsne
from dataset import BalancedNLLBDataset
from train import masked_mean_pool


@dataclass
class PrecomputedLanguageEmbeddingsConf:
    backbone_name: str
    encoder_only: bool
    langs: list[str]
    pairs: list[str]
    batch_size: int
    max_length: int
    num_batches: int
    points_per_lang_cap: int | None
    shuffle_buffer_size: int | None
    stream_shuffle_seed: int | None
    random_seed: int
    layer_indices: list[int]
    output_path: str


def collect_text_rows(
    loader: DataLoader,
    langs_to_collect: list[str],
    num_batches: int,
    cap_per_lang: int | None,
):
    texts = []
    langs = []
    counts_by_lang = defaultdict(int)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        for text, lang in zip(batch["texts"], batch["langs"]):
            if cap_per_lang is not None and counts_by_lang[lang] >= cap_per_lang:
                continue

            texts.append(text)
            langs.append(lang)
            counts_by_lang[lang] += 1

        if batch_idx % 10 == 0:
            print(f"[collect-text] processed batch {batch_idx}")

        if cap_per_lang is not None and all(
            counts_by_lang[lang] >= cap_per_lang for lang in langs_to_collect
        ):
            break

    return texts, langs, dict(counts_by_lang)


@torch.no_grad()
def compute_layer_coords(
    backbone: MLLMBackbone,
    texts: list[str],
    batch_size: int,
    max_length: int,
    layer_idx: int,
    encoder_only: bool,
    random_seed: int,
):
    embeddings = []

    for start_idx in tqdm(
        range(0, len(texts), batch_size),
        desc=f"layer {layer_idx}",
        leave=False,
    ):
        batch_texts = texts[start_idx : start_idx + batch_size]
        acts = backbone.extract_layer_activations(
            texts=batch_texts,
            layer_idx=layer_idx,
            max_length=max_length,
            encoder_only=encoder_only,
        )
        embs = masked_mean_pool(acts["layer_tensor"], acts["valid_mask"])
        embs = F.normalize(embs, p=2, dim=-1)
        embeddings.append(embs.detach().cpu().numpy())

    embedding_matrix = np.concatenate(embeddings, axis=0)
    coords = run_tsne(embedding_matrix, random_state=random_seed)
    return coords.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        "PrecomputeLanguageEmbeddings",
        description="Precompute multilingual sentence embedding projections across layers",
    )
    parser.add_argument("config_path", help="path to precompute config. Required.")

    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        conf = PrecomputedLanguageEmbeddingsConf(**json.load(stream))

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

    texts, langs, counts_by_lang = collect_text_rows(
        loader=loader,
        langs_to_collect=conf.langs,
        num_batches=conf.num_batches,
        cap_per_lang=conf.points_per_lang_cap,
    )

    print(f"Collected {len(texts)} texts for precompute")
    for lang in conf.langs:
        print(f"{lang}: {counts_by_lang.get(lang, 0)}")

    backbone = MLLMBackbone(device, conf.backbone_name)

    save_arrays = {
        "texts": np.array(texts, dtype=object),
        "langs": np.array(langs, dtype=object),
        "layer_indices": np.array(conf.layer_indices, dtype=np.int16),
        "counts_langs": np.array(list(counts_by_lang.keys()), dtype=object),
        "counts_values": np.array(list(counts_by_lang.values()), dtype=np.int32),
    }

    for layer_idx in conf.layer_indices:
        print(f"Computing coordinates for layer {layer_idx}")
        coords = compute_layer_coords(
            backbone=backbone,
            texts=texts,
            batch_size=conf.batch_size,
            max_length=conf.max_length,
            layer_idx=layer_idx,
            encoder_only=conf.encoder_only,
            random_seed=conf.random_seed,
        )
        save_arrays[f"coords_layer_{layer_idx}"] = coords

    output_path = Path(conf.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_arrays)

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as stream:
        json.dump(asdict(conf), stream, indent=2)

    print(f"Saved precomputed embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
