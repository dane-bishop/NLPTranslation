import math
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader

from dataset import BalancedNLLBDataset
from backbone import MLLMBackbone
from sae import SAE, GatedSparseAutoEncoder

from train import TrainingConf, sae_constructors

# -------------------------
# Config
# -------------------------
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

LAYER_IDX = 6
CHECKPOINT_PATH = "sae_layer6.pt"

BATCH_SIZE = 32
MAX_LENGTH = 128
NUM_BATCHES_TO_ANALYZE = 50

IN_DIM = 1024
LATENT_DIM = 4096

TOP_K_FEATURES_TO_PRINT = 20
TOP_K_EXAMPLES_PER_FEATURE = 10
ACTIVE_THRESHOLD = 1e-6


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
# Helper: entropy
# -------------------------
def normalized_entropy(counter: Counter, num_langs: int) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0

    probs = [v / total for v in counter.values() if v > 0]
    h = -sum(p * math.log(p) for p in probs)

    if num_langs <= 1:
        return 0.0

    return h / math.log(num_langs)


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
def collect_analysis_rows(
    backbone: MLLMBackbone,
    sae: GatedSparseAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int,
    max_length: int,
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
            encoder_only=True
        )

        x = acts["token_activations"]  # (N, 768)
        if x.numel() == 0:
            continue

        # Match training-time centering behavior if you used it there
        x = x - x.mean(dim=0, keepdim=True)

        f,_ = sae.encode(x)
        x_hat = sae.decode(f)
        #z = out["f"]

        per_token_recon = ((x_hat - x) ** 2).mean(dim=1)  # (N,)

        valid_tokens, valid_langs, valid_positions, valid_sentence_indices = \
            get_valid_tokens_and_langs(backbone, acts, langs)

        assert len(valid_tokens) == x.shape[0], (
            f"Mismatch: {len(valid_tokens)} valid tokens vs {x.shape[0]} activations"
        )

        z_cpu = f.detach().cpu()
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


# -------------------------
# Global stats
# -------------------------
def print_global_stats(rows, latent_dim: int):
    num_rows = len(rows)
    if num_rows == 0:
        print("No rows collected.")
        return

    z = torch.stack([r["z"] for r in rows], dim=0)  # (N, D)

    active_mask = z > ACTIVE_THRESHOLD
    active_frac = active_mask.float().mean().item()

    feature_firing = active_mask.float().mean(dim=0)
    dead_features = int((feature_firing < 1e-6).sum().item())

    recon_losses = torch.tensor([r["recon_loss"] for r in rows])

    print("\n=== GLOBAL STATS ===")
    print(f"num token rows: {num_rows}")
    print(f"latent dim: {latent_dim}")
    print(f"mean recon loss per token: {recon_losses.mean().item():.6f}")
    print(f"std recon loss per token:  {recon_losses.std().item():.6f}")
    print(f"active fraction:           {active_frac:.6f}")
    print(f"dead features:             {dead_features}")

    firing_sorted, _ = torch.sort(feature_firing, descending=True)
    print(f"top 10 feature firing rates: {[round(float(x), 6) for x in firing_sorted[:10]]}")


# -------------------------
# Top examples for features
# -------------------------
def get_top_examples_for_feature(rows, feature_idx: int, top_k: int = 10):
    scored = []
    for r in rows:
        score = float(r["z"][feature_idx].item())
        if score > ACTIVE_THRESHOLD:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def print_top_examples_for_features(rows, feature_indices: list[int], top_k: int = 10):
    print("\n=== TOP ACTIVATING EXAMPLES ===")
    for feat_idx in feature_indices:
        print(f"\n--- Feature {feat_idx} ---")
        top_examples = get_top_examples_for_feature(rows, feat_idx, top_k=top_k)

        if not top_examples:
            print("No active examples.")
            continue

        for rank, (score, r) in enumerate(top_examples, start=1):
            print(
                f"[{rank:02d}] score={score:.6f} "
                f"lang={r['lang']} token={r['token']} pos={r['position']} "
                f"sentence={r['sentence']}"
            )


# -------------------------
# Feature-level stats
# -------------------------
def compute_feature_stats(rows, latent_dim: int, langs: list[str]):
    """
    Returns per-feature stats:
      - firing rate
      - mean activation when active
      - language counter among active examples
      - normalized language entropy
      - selectivity = 1 - entropy
    """
    z = torch.stack([r["z"] for r in rows], dim=0)  # (N, D)
    row_langs = [r["lang"] for r in rows]

    feature_stats = []

    for j in range(latent_dim):
        vals = z[:, j]
        active = vals > ACTIVE_THRESHOLD

        firing_rate = float(active.float().mean().item())

        if active.any():
            mean_active = float(vals[active].mean().item())
            lang_counter = Counter(
                row_langs[i] for i in range(len(rows)) if bool(active[i].item())
            )
            ent = normalized_entropy(lang_counter, len(langs))
            selectivity = 1.0 - ent
        else:
            mean_active = 0.0
            lang_counter = Counter()
            ent = 0.0
            selectivity = 0.0

        feature_stats.append({
            "feature_idx": j,
            "firing_rate": firing_rate,
            "mean_active": mean_active,
            "lang_counter": lang_counter,
            "lang_entropy": ent,
            "lang_selectivity": selectivity,
        })

    return feature_stats


def print_most_language_selective_features(feature_stats, top_k: int = 20):
    print("\n=== MOST LANGUAGE-SELECTIVE FEATURES ===")
    ranked = sorted(
        feature_stats,
        key=lambda d: (d["lang_selectivity"], d["mean_active"]),
        reverse=True,
    )

    count = 0
    for fs in ranked:
        if fs["firing_rate"] <= 0:
            continue

        print(
            f"feature={fs['feature_idx']:4d} "
            f"firing={fs['firing_rate']:.6f} "
            f"mean_active={fs['mean_active']:.6f} "
            f"lang_selectivity={fs['lang_selectivity']:.6f} "
            f"lang_entropy={fs['lang_entropy']:.6f} "
            f"langs={dict(fs['lang_counter'].most_common(4))}"
        )
        count += 1
        if count >= top_k:
            break


def print_most_common_features(feature_stats, top_k: int = 20):
    print("\n=== MOST COMMONLY FIRING FEATURES ===")
    ranked = sorted(
        feature_stats,
        key=lambda d: (d["firing_rate"], d["mean_active"]),
        reverse=True,
    )[:top_k]

    for fs in ranked:
        print(
            f"feature={fs['feature_idx']:4d} "
            f"firing={fs['firing_rate']:.6f} "
            f"mean_active={fs['mean_active']:.6f} "
            f"lang_selectivity={fs['lang_selectivity']:.6f} "
            f"langs={dict(fs['lang_counter'].most_common(4))}"
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

        z = torch.stack([r["z"] for r in lang_rows], dim=0)
        active_frac = float((z > ACTIVE_THRESHOLD).float().mean().item())
        mean_recon = sum(r["recon_loss"] for r in lang_rows) / len(lang_rows)

        print(
            f"{lang}: "
            f"num_tokens={len(lang_rows)} "
            f"active_frac={active_frac:.6f} "
            f"mean_recon={mean_recon:.6f}"
        )


# -------------------------
# Choose example features automatically
# -------------------------
def choose_example_features(feature_stats, top_k: int = 10):
    """
    Picks a mix of:
      - most common features
      - most selective features
    """
    common = sorted(
        feature_stats,
        key=lambda d: (d["firing_rate"], d["mean_active"]),
        reverse=True,
    )[:top_k]

    selective = sorted(
        feature_stats,
        key=lambda d: (d["lang_selectivity"], d["mean_active"]),
        reverse=True,
    )[:top_k]

    chosen = []
    seen = set()

    for group in (common, selective):
        for fs in group:
            idx = fs["feature_idx"]
            if idx not in seen and fs["firing_rate"] > 0:
                chosen.append(idx)
                seen.add(idx)

    return chosen[:top_k]


# -------------------------
# Main
# -------------------------
def main():

    import json
    import sys
    assert len(sys.argv) > 1
    with open(sys.argv[1],'r') as stream:
        conf = TrainingConf(**json.load(stream))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BalancedNLLBDataset(
        pair_configs=conf.pairs,
        langs=conf.langs,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_records,
        num_workers=0,
    )

    backbone = MLLMBackbone(device,model_name=conf.backbone_name)
    sae_constructor = sae_constructors[conf.sae_type]
    sae = sae_constructor(conf.model_hidden_size, conf.sae_hidden_size).to(device)
    state_dict = torch.load(conf.weight_path, map_location=device)
    sae.load_state_dict(state_dict)
    sae.eval()

    print(f"Loaded SAE checkpoint from {conf.weight_path}")

    rows = collect_analysis_rows(
        backbone=backbone,
        sae=sae,
        loader=loader,
        device=device,
        layer_idx=LAYER_IDX,
        max_length=MAX_LENGTH,
        num_batches=NUM_BATCHES_TO_ANALYZE,
    )

    print_global_stats(rows, latent_dim=conf.sae_hidden_size)
    print_per_language_summary(rows, latent_dim=conf.sae_hidden_size, langs=conf.langs)

    feature_stats = compute_feature_stats(rows, latent_dim=conf.sae_hidden_size, langs=conf.langs)

    print_most_common_features(feature_stats, top_k=20)
    print_most_language_selective_features(feature_stats, top_k=20)

    chosen_features = choose_example_features(
        feature_stats,
        top_k=TOP_K_FEATURES_TO_PRINT,
    )

    print_top_examples_for_features(
        rows,
        feature_indices=chosen_features,
        top_k=TOP_K_EXAMPLES_PER_FEATURE,
    )


if __name__ == "__main__":
    main()
