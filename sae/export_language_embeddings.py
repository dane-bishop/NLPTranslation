import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import MLLMBackbone
from dataset import BalancedNLLBDataset
from datasets import get_dataset_config_names



@dataclass
class ExportConf:
    backbone_name: str
    batch_size: int
    max_steps: int
    layer_idx: int
    encoder_only: bool
    max_length: int
    reduction: str
    output_csv: str
    output_npy: str | None = None
    pairs: list[str] | None = None
    langs: list[str] | None = None


def collate_records(batch):
    return {
        "texts": [item["text"] for item in batch],
        "langs": [item["lang"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }


def masked_pool(layer_tensor: torch.Tensor, valid_mask: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    layer_tensor: (B, T, H)
    valid_mask:   (B, T) boolean
    returns:      (B, H)
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    mask = valid_mask.unsqueeze(-1).to(layer_tensor.dtype)  # (B, T, 1)
    masked = layer_tensor * mask

    if reduction == "sum":
        return masked.sum(dim=1)

    counts = mask.sum(dim=1).clamp_min(1.0)  # (B, 1)
    return masked.sum(dim=1) / counts


def export_language_embeddings(conf: ExportConf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(37)

    backbone = MLLMBackbone(device, conf.backbone_name)
    if conf.pairs and conf.langs:
        pairs = conf.pairs
        langs = conf.langs
    else:
        pairs, langs = build_all_language_streams()

    print(f"Using {len(langs)} language streams")
    dataset = BalancedNLLBDataset(pairs, langs)
    loader = DataLoader(dataset, batch_size=conf.batch_size, collate_fn=collate_records)

    # Store example embeddings by language
    per_language_vectors = defaultdict(list)
    per_language_counts = defaultdict(int)

    pbar = tqdm(total=conf.max_steps, desc="Exporting embeddings")

    for step, batch in enumerate(loader):
        if step >= conf.max_steps:
            break

        texts = batch["texts"]
        langs = batch["langs"]

        with torch.no_grad():
            acts = backbone.extract_layer_activations(
                texts=texts,
                layer_idx=conf.layer_idx,
                max_length=conf.max_length,
                encoder_only=conf.encoder_only,
            )

            # Pool to one vector per example
            pooled = masked_pool(
                acts["layer_tensor"].float(),
                acts["valid_mask"],
                conf.reduction,
            )  # (B, H)

            pooled = pooled.cpu().numpy()

        for lang, vec in zip(langs, pooled):
            per_language_vectors[lang].append(vec)
            per_language_counts[lang] += 1

        pbar.update(1)

    pbar.close()

    rows = []
    embedding_matrix = []

    for lang in sorted(per_language_vectors.keys()):
        vectors = np.stack(per_language_vectors[lang], axis=0)  # (N_lang, H)
        lang_embedding = vectors.mean(axis=0)  # (H,)

        row = {
            "iso3": lang,
            "num_examples": per_language_counts[lang],
        }
        for i, value in enumerate(lang_embedding, start=1):
            row[f"dim{i}"] = float(value)

        rows.append(row)
        embedding_matrix.append(lang_embedding)

    df = pd.DataFrame(rows)
    out_csv = Path(conf.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if conf.output_npy:
        out_npy = Path(conf.output_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, np.stack(embedding_matrix, axis=0))

    print(f"Saved CSV to: {out_csv}")
    if conf.output_npy:
        print(f"Saved NPY to: {out_npy}")


def build_all_language_streams():
    """
    Build one stream per language automatically from the available
    allenai/nllb pair configs.

    Preference:
    1. Use an English-pivot pair if available for that language
    2. Otherwise use the first available pair we see
    """
    configs = get_dataset_config_names("allenai/nllb", trust_remote_code=True)

    lang_to_pair = {}

    # Prefer English-pivot configs first
    sorted_configs = sorted(
        configs,
        key=lambda c: (0 if "eng_Latn" in c else 1, c)
    )

    for cfg in sorted_configs:
        if "-" not in cfg:
            continue

        a, b = cfg.split("-")

        if a not in lang_to_pair:
            lang_to_pair[a] = cfg
        if b not in lang_to_pair:
            lang_to_pair[b] = cfg

    langs = sorted(lang_to_pair.keys())
    pairs = [lang_to_pair[lang] for lang in langs]

    return pairs, langs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export one embedding per language from NLLB backbone activations")
    parser.add_argument("config_path", help="Path to export config JSON")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        conf = ExportConf(**json.load(f))

    export_language_embeddings(conf)


if __name__ == "__main__":
    main()
