import argparse
import math
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce
from openTSNE import TSNE
from sklearn.preprocessing import normalize
from datasets import load_dataset
from tqdm import tqdm
import umap

from sae import GatedSparseAutoEncoder, SAE
from train import TrainingConf
from backbone import MLLMBackbone
from dataset import BalancedFLORESDataset

seed = 37
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_constructors = {"vanilla": SAE, "gated": GatedSparseAutoEncoder}

def get_tokens(backbone: MLLMBackbone, acts: dict[str, torch.Tensor], batch_langs: list[str]):
    input_ids = acts["input_ids"].cpu()
    valid_mask = acts["valid_mask"].cpu()

    valid_tokens = []
    for idx in range(input_ids.shape[1]):
        if valid_mask[0, idx]:
            tok_id = int(input_ids[0, idx].item())
            tok_str = backbone.tokenizer.convert_ids_to_tokens([tok_id])[0]
            valid_tokens.append(tok_str)

    return valid_tokens


def get_sentence_activations(backbone, sae, dataloader, num_sentences, conf):
    step = 0
    activations = []        # list of [actual_len, F] tensors — NO padding
    activations_padded = [] # list of [128, F] tensors — for stacking into acts_tensor
    token_lengths = []      # actual number of valid tokens per sentence
    lang_tags = []
    token_cache = []
    sentences = []

    for batch in dataloader:
        texts = batch["text"]
        backbone_acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=conf.layer_idx,
            max_length=128,
            encoder_only=conf.encoder_only)

        x = backbone_acts["token_activations"]
        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)
        with torch.no_grad():
            z, _ = sae.encode(x)

        z = z.cpu().half().squeeze()
        actual_len = z.shape[0]

        activations.append(z)
        token_lengths.append(actual_len)

        z_padded = torch.nn.functional.pad(z, (0, 0, 0, 128 - actual_len))
        activations_padded.append(z_padded)

        token_cache.append(get_tokens(backbone, backbone_acts, batch['lang']))
        sentences.extend(texts)
        lang_tags.extend(batch['lang'])
        step += 1
        if step >= num_sentences:
            break

    return activations, activations_padded, token_lengths, token_cache, sentences, lang_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--output_dir",    default="./cached_embeddings")
    parser.add_argument("--num_sentences", type=int, default=500,
                        help="Sentences per language")
    parser.add_argument("--topk", type=int, default=20, help="Num tokens for each feature to keep")
    parser.add_argument("--threshold_scale", type=float, default=5e-1, help="Cuttoff for feature activations by mean, larger is more restrictive")
    args = parser.parse_args()

    with open(args.config_path) as stream:
        cfg = json.load(stream)
        conf = TrainingConf(**cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone_name = conf.backbone_name
    backbone = MLLMBackbone(device, backbone_name)  

    sae = sae_constructors[conf.sae_type](conf.model_hidden_size, conf.sae_hidden_size).to(device)

    weight_path = conf.weight_path
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device, weights_only=True)
        sae.load_state_dict(state)
    else:
        print(f"WARNING: weight file not found at {weight_path}; using random SAE weights.")
    sae.eval()

    batch_size = 1
    layer_idx = conf.layer_idx

    dataset = BalancedFLORESDataset(langs=conf.langs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    activations, activations_padded, token_lengths, token_cache, sentences, lang_tags = \
        get_sentence_activations(backbone, sae, dataloader, args.num_sentences, conf)

    acts_tensor = torch.stack(activations_padded, dim=0)  # [S, 128, F]

    max_per_feature = acts_tensor.amax(dim=(0, 1))
    alive_mask = max_per_feature > max_per_feature.mean() * args.threshold_scale
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]

    acts_alive_padded = acts_tensor[:, :, alive_mask]  # [S, 128, F_alive]

    sentence_profiles = []
    for i, z_unpadded in enumerate(activations):
        z_alive = z_unpadded[:, alive_mask]            # [actual_len, F_alive]
        sentence_profiles.append(z_alive.amax(dim=0)) # [F_alive]
    sentence_activation_profile = torch.stack(sentence_profiles, dim=0)  # [S, F_alive]

    S, T, F_alive = acts_alive_padded.shape
    acts_flat = acts_alive_padded.reshape(S * T, F_alive)  # [S*T, F_alive]

    flat_topk = torch.topk(acts_flat, dim=0, k=args.topk)  # [topk, F_alive]
    flat_indices = flat_topk.indices

    sent_indices_2d = flat_indices // T  # [topk, F_alive]
    tok_indices_2d  = flat_indices % T   # [topk, F_alive]

    topk_sentence_lookup = []
    for feat_idx in range(F_alive):
        current_feat_table = []
        seen_sentences = set()
        for rank in range(args.topk):
            sent_idx = int(sent_indices_2d[rank, feat_idx].item())
            tok_pos  = int(tok_indices_2d[rank, feat_idx].item())

            if tok_pos >= token_lengths[sent_idx]:
                continue

            if sent_idx in seen_sentences:
                continue
            seen_sentences.add(sent_idx)

            token_list = token_cache[sent_idx]
            top_token  = token_list[tok_pos] if tok_pos < len(token_list) else "[PAD]"
            current_feat_table.append((sent_idx, top_token))

        topk_sentence_lookup.append(current_feat_table)

    # ── Projections ───────────────────────────────────────────────────────────
    sentence_topk_res = torch.topk(acts_tensor, dim=-1, k=args.topk)

    feature_embeddings = sentence_activation_profile.T.float().numpy()  # [F_alive, S]
    feature_embeddings = normalize(feature_embeddings, norm='l2')

    feature_reducer  = umap.UMAP(n_neighbors=100, min_dist=0.01, metric="cosine")
    sentence_reducer = umap.UMAP(n_neighbors=30,  min_dist=0.05, metric="cosine")

    feature_projections  = feature_reducer.fit_transform(feature_embeddings)
    sentence_projections = sentence_reducer.fit_transform(
        rearrange(sentence_topk_res.values, "s t f -> s (t f)").numpy()
    )

    np.save(out_dir / "sentence_projections.npy", sentence_projections)
    np.save(out_dir / "feature_projections.npy",  feature_projections)
    np.save(out_dir / "alive_feature_indices.npy", alive_indices.numpy())

    with open(out_dir / "sentences.json", "w", encoding="utf-8") as stream:
        json.dump(sentences, stream, ensure_ascii=False, indent=2)

    with open(out_dir / "topk_sentence_lookup.json", "w", encoding="utf-8") as stream:
        json.dump(topk_sentence_lookup, stream, indent=2)

    metadata = {
        "config": cfg,
        "lang_tags": lang_tags,
        "num_sentences": len(sentences),
        "layer_idx": layer_idx,
    }
    with open(out_dir / "metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)


if __name__ == "__main__":
    main()
