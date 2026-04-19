import argparse
import contextlib
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from dataset import BalancedNLLBDataset
from backbone import MLLMBackbone
from sae import SAE, GatedSparseAutoEncoder
from train import TrainingConf, masked_mean_pool

@dataclass
class AnalysisConf:
    weight_path: str
    batch_size: int
    num_batches: int
    top_k_features: int
    top_k_examples: int
    output_dir: str
    langs: list[str] | None
    pairs: list[str] | None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_constructors = {"vanilla": SAE, "gated": GatedSparseAutoEncoder}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# -------------------------
# Collate
# -------------------------
def collate_records(batch):
    return {
        "texts": [item["text"] for item in batch],
        "langs": [item["lang"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }


# -------------------------
# Helper: decode valid tokens aligned with x_valid rows
# -------------------------
def get_valid_tokens_and_langs(
    backbone: MLLMBackbone,
    acts: dict[str, torch.Tensor],
    batch_langs: list[str],
):
    """
    Returns lists aligned with acts["token_activations"] rows:
      - valid_tokens[i]
      - valid_langs[i]
      - valid_positions[i]
      - valid_sentence_indices[i]
    """
    input_ids = acts["input_ids"]              # (B, T)
    valid_mask = acts["valid_mask"]            # (B, T)

    valid_tokens = []
    valid_langs = []
    valid_positions = []
    valid_sentence_indices = []

    input_ids_cpu = input_ids.detach().cpu()
    valid_mask_cpu = valid_mask.detach().cpu()

    batch_size, seq_len = input_ids_cpu.shape

    for sent_idx in range(batch_size):
        for tok_idx in range(seq_len):
            if valid_mask_cpu[sent_idx, tok_idx]:
                tok_id = int(input_ids_cpu[sent_idx, tok_idx].item())
                tok_str = backbone.tokenizer.convert_ids_to_tokens([tok_id])[0]

                valid_tokens.append(tok_str)
                valid_langs.append(batch_langs[sent_idx])
                valid_positions.append(tok_idx)
                valid_sentence_indices.append(sent_idx)

    return valid_tokens, valid_langs, valid_positions, valid_sentence_indices


# -------------------------
# Main analysis collector
# -------------------------
@torch.no_grad()
def collect_token_analysis_rows(
    backbone: MLLMBackbone,
    sae: SAE,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    max_length: int,
    encoder_only: bool,
    num_batches: int,
):
    """
    Collects token-level rows with:
      - token metadata
      - latent activations z
      - reconstruction loss per token
    """
    rows = []

    sae.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        texts = batch["texts"]
        langs = batch["langs"]

        acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=layer_idx,
            max_length=max_length,
            encoder_only=encoder_only,
        )

        x = acts["token_activations"]  # (N, 768)
        if x.numel() == 0:
            continue

        # Match training-time centering behavior if you used it there
        x = x - x.mean(dim=0, keepdim=True)

        out = sae(x)
        x_hat = out["x_hat"]
        z = out["z"]

        per_token_recon = ((x_hat - x) ** 2).mean(dim=1)  # (N,)

        valid_tokens, valid_langs, valid_positions, valid_sentence_indices = \
            get_valid_tokens_and_langs(backbone, acts, langs)

        assert len(valid_tokens) == x.shape[0], (
            f"Mismatch: {len(valid_tokens)} valid tokens vs {x.shape[0]} activations"
        )

        z_cpu = z.detach().cpu()
        per_token_recon_cpu = per_token_recon.detach().cpu()

        for i in range(x.shape[0]):
            sent_idx = valid_sentence_indices[i]

            rows.append({
                "token": valid_tokens[i],
                "lang": valid_langs[i],
                "position": valid_positions[i],
                "sentence": texts[sent_idx],
                "sentence_idx_in_batch": sent_idx,
                "recon_loss": float(per_token_recon_cpu[i].item()),
                "z": z_cpu[i],  # shape: (latent_dim,)
            })

        if batch_idx % 10 == 0:
            print(f"[collect] processed batch {batch_idx}")

    return rows


@torch.no_grad()
def collect_pooled_analysis_rows(
    backbone: MLLMBackbone,
    sae: SAE,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    max_length: int,
    encoder_only: bool,
    num_batches: int,
):
    """
    Collects sentence-level rows for SAEs trained on pooled sentence embeddings.
    """
    rows = []

    sae.eval()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        texts = batch["texts"]
        langs = batch["langs"]

        acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=layer_idx,
            max_length=max_length,
            encoder_only=encoder_only,
        )

        x = masked_mean_pool(acts["layer_tensor"], acts["valid_mask"])
        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)

        out = sae(x)
        x_hat = out["x_hat"]
        z = out["z"]

        per_sentence_recon = ((x_hat - x) ** 2).mean(dim=1)

        z_cpu = z.detach().cpu()
        per_sentence_recon_cpu = per_sentence_recon.detach().cpu()

        for sent_idx in range(x.shape[0]):
            rows.append(
                {
                    "lang": langs[sent_idx],
                    "sentence": texts[sent_idx],
                    "recon_loss": float(per_sentence_recon_cpu[sent_idx].item()),
                    "z": z_cpu[sent_idx],
                }
            )

        if batch_idx % 10 == 0:
            print(f"[collect] processed batch {batch_idx}")

    return rows


# -------------------------
# Global stats
# -------------------------
def print_global_stats(rows, latent_dim: int):
    num_rows = len(rows)
    if num_rows == 0:
        print("No rows collected.")
        return

    z = torch.stack([r["z"] for r in rows], dim=0)  # (N, D)

    recon_losses = torch.tensor([r["recon_loss"] for r in rows])
    max_by_feature = z.max(dim=0).values
    mean_by_feature = z.mean(dim=0)
    mean_abs_by_feature = z.abs().mean(dim=0)

    print("\n=== GLOBAL STATS ===")
    print(f"num rows: {num_rows}")
    print(f"latent dim: {latent_dim}")
    print(f"mean recon loss per token: {recon_losses.mean().item():.6f}")
    print(f"std recon loss per token:  {recon_losses.std().item():.6f}")
    print(f"top 10 feature max z:      {[round(float(x), 6) for x in torch.sort(max_by_feature, descending=True).values[:10]]}")
    print(f"top 10 feature mean z:     {[round(float(x), 6) for x in torch.sort(mean_by_feature, descending=True).values[:10]]}")
    print(f"top 10 feature mean |z|:   {[round(float(x), 6) for x in torch.sort(mean_abs_by_feature, descending=True).values[:10]]}")


# -------------------------
# Top examples for features
# -------------------------
def get_top_examples_for_feature(
    rows,
    feature_idx: int,
    top_k: int = 10,
    use_abs: bool = False,
):
    scored = []
    for r in rows:
        raw_score = float(r["z"][feature_idx].item())
        score = abs(raw_score) if use_abs else raw_score
        scored.append((score, raw_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def print_top_examples_for_features(rows, feature_indices: list[int], top_k: int = 10):
    print("\n=== TOP ACTIVATING EXAMPLES ===")
    for feat_idx in feature_indices:
        print(f"\n--- Feature {feat_idx} ---")
        top_examples_by_z = get_top_examples_for_feature(
            rows,
            feat_idx,
            top_k=top_k,
            use_abs=False,
        )
        top_examples_by_abs = get_top_examples_for_feature(
            rows,
            feat_idx,
            top_k=top_k,
            use_abs=True,
        )

        if not top_examples_by_z:
            print("No examples.")
            continue

        print("Top by z:")
        for rank, (_, raw_score, r) in enumerate(top_examples_by_z, start=1):
            if "token" in r:
                print(
                    f"[{rank:02d}] score={raw_score:.6f} "
                    f"abs_score={abs(raw_score):.6f} "
                    f"lang={r['lang']} token={r['token']} pos={r['position']} "
                    f"sentence={r['sentence']}"
                )
            else:
                print(
                    f"[{rank:02d}] score={raw_score:.6f} "
                    f"abs_score={abs(raw_score):.6f} "
                    f"lang={r['lang']} sentence={r['sentence']}"
                )

        print("Top by |z|:")
        for rank, (_, raw_score, r) in enumerate(top_examples_by_abs, start=1):
            if "token" in r:
                print(
                    f"[{rank:02d}] score={raw_score:.6f} "
                    f"abs_score={abs(raw_score):.6f} "
                    f"lang={r['lang']} token={r['token']} pos={r['position']} "
                    f"sentence={r['sentence']}"
                )
            else:
                print(
                    f"[{rank:02d}] score={raw_score:.6f} "
                    f"abs_score={abs(raw_score):.6f} "
                    f"lang={r['lang']} sentence={r['sentence']}"
                )


# -------------------------
# Feature-level stats
# -------------------------
def compute_feature_stats(rows, latent_dim: int, langs: list[str]):
    """
    Returns per-feature ranking stats without imposing an activity threshold.
    """
    z = torch.stack([r["z"] for r in rows], dim=0)  # (N, D)

    feature_stats = []

    for j in range(latent_dim):
        vals = z[:, j]

        feature_stats.append({
            "feature_idx": j,
            "mean_z": float(vals.mean().item()),
            "max_z": float(vals.max().item()),
            "mean_abs_z": float(vals.abs().mean().item()),
            "max_abs_z": float(vals.abs().max().item()),
        })

    return feature_stats


def print_top_features_by_max(feature_stats, top_k: int = 20):
    print("\n=== TOP FEATURES BY MAX Z ===")
    ranked = sorted(
        feature_stats,
        key=lambda d: (d["max_z"], d["mean_z"]),
        reverse=True,
    )[:top_k]

    for fs in ranked:
        print(
            f"feature={fs['feature_idx']:4d} "
            f"max_z={fs['max_z']:.6f} "
            f"mean_z={fs['mean_z']:.6f} "
            f"mean_abs_z={fs['mean_abs_z']:.6f} "
            f"max_abs_z={fs['max_abs_z']:.6f}"
        )


def print_top_features_by_mean(feature_stats, top_k: int = 20):
    print("\n=== TOP FEATURES BY MEAN Z ===")
    ranked = sorted(
        feature_stats,
        key=lambda d: (d["mean_z"], d["max_z"]),
        reverse=True,
    )[:top_k]

    for fs in ranked:
        print(
            f"feature={fs['feature_idx']:4d} "
            f"mean_z={fs['mean_z']:.6f} "
            f"max_z={fs['max_z']:.6f} "
            f"mean_abs_z={fs['mean_abs_z']:.6f} "
            f"max_abs_z={fs['max_abs_z']:.6f}"
        )


def print_top_features_by_mean_abs(feature_stats, top_k: int = 20):
    print("\n=== TOP FEATURES BY MEAN |Z| ===")
    ranked = sorted(
        feature_stats,
        key=lambda d: (d["mean_abs_z"], d["max_abs_z"]),
        reverse=True,
    )[:top_k]

    for fs in ranked:
        print(
            f"feature={fs['feature_idx']:4d} "
            f"mean_abs_z={fs['mean_abs_z']:.6f} "
            f"max_abs_z={fs['max_abs_z']:.6f} "
            f"mean_z={fs['mean_z']:.6f} "
            f"max_z={fs['max_z']:.6f}"
        )


# -------------------------
# Per-language summaries
# -------------------------
def print_per_language_summary(rows, latent_dim: int, langs: list[str]):
    print("\n=== PER-LANGUAGE SUMMARY ===")

    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    for lang in langs:
        lang_rows = by_lang[lang]
        if not lang_rows:
            print(f"{lang}: no rows")
            continue

        mean_recon = sum(r["recon_loss"] for r in lang_rows) / len(lang_rows)

        print(
            f"{lang}: "
            f"num_rows={len(lang_rows)} "
            f"mean_recon={mean_recon:.6f}"
        )


# -------------------------
# Choose example features automatically
# -------------------------
def choose_example_features(feature_stats, top_k: int = 10):
    """
    Picks a mix of:
      - largest peak activations
      - strongest average magnitude
    """
    by_max = sorted(
        feature_stats,
        key=lambda d: (d["max_z"], d["mean_z"]),
        reverse=True,
    )[:top_k]

    by_mean_abs = sorted(
        feature_stats,
        key=lambda d: (d["mean_abs_z"], d["max_abs_z"]),
        reverse=True,
    )[:top_k]

    chosen = []
    seen = set()

    for group in (by_max, by_mean_abs):
        for fs in group:
            idx = fs["feature_idx"]
            if idx not in seen:
                chosen.append(idx)
                seen.add(idx)

    return chosen[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="path to analysis config. Required.")
    
    args = parser.parse_args()

    # -------------------------
    # Load analysis config / saved training config
    # -------------------------
    with open(args.config_path, "r") as stream:
        conf = AnalysisConf(**json.load(stream))

    output_dir = Path(conf.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "analysis.txt"

    weight_path = Path(conf.weight_path)
    model_dir = weight_path.parent

    with open(model_dir / "config.json", "r") as stream:
        train_conf = TrainingConf(**json.load(stream))

    # -- Default to training langs/pairs if none provided
    langs = conf.langs if conf.langs else train_conf.langs
    pairs = conf.pairs if conf.pairs else train_conf.pairs

    # -------------------------
    # Create NLLB Dataset
    # -------------------------
    dataset = BalancedNLLBDataset(
        pair_configs=pairs,
        langs=langs,
    )

    loader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        collate_fn=collate_records,
        num_workers=0,
    )

    # -------------------------
    # Load backbone / SAE from train config
    # -------------------------
    backbone = MLLMBackbone(device, model_name=train_conf.backbone_name)

    model_constructor: SAE | GatedSparseAutoEncoder = sae_constructors[train_conf.sae_type]

    sae = model_constructor(
        d_act=train_conf.model_hidden_size,
        d_hidden=train_conf.sae_hidden_size,
    ).to(device)

    state_dict = torch.load(weight_path, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval()

    with open(output_path, "w") as output_stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, output_stream)):
            print(f"Loaded SAE checkpoint from {weight_path}")
            print(f"Saving analysis output to {output_path}")

            if train_conf.pool_features:
                print("Using sentence-level pooled analysis to match training-time pooling.")
                if train_conf.reduction != "mean":
                    raise ValueError("Pooled SAE analysis currently requires reduction='mean'")
                rows = collect_pooled_analysis_rows(
                    backbone=backbone,
                    sae=sae,
                    loader=loader,
                    device=device,
                    layer_idx=train_conf.layer_idx,
                    max_length=train_conf.max_length,
                    encoder_only=train_conf.encoder_only,
                    num_batches=conf.num_batches,
                )
            else:
                rows = collect_token_analysis_rows(
                    backbone=backbone,
                    sae=sae,
                    loader=loader,
                    device=device,
                    layer_idx=train_conf.layer_idx,
                    max_length=train_conf.max_length,
                    encoder_only=train_conf.encoder_only,
                    num_batches=conf.num_batches,
                )

            print_global_stats(rows, latent_dim=train_conf.sae_hidden_size)
            print_per_language_summary(rows, latent_dim=train_conf.sae_hidden_size, langs=langs)

            feature_stats = compute_feature_stats(rows, latent_dim=train_conf.sae_hidden_size, langs=langs)

            print_top_features_by_max(feature_stats, top_k=conf.top_k_features)
            print_top_features_by_mean(feature_stats, top_k=conf.top_k_features)
            print_top_features_by_mean_abs(feature_stats, top_k=conf.top_k_features)

            chosen_features = choose_example_features(
                feature_stats,
                top_k=conf.top_k_features,
            )

            print_top_examples_for_features(
                rows,
                feature_indices=chosen_features,
                top_k=conf.top_k_examples,
            )


if __name__ == "__main__":
    main()
