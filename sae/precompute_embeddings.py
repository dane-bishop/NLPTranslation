import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

from sae import GatedSparseAutoEncoder, SAE
from train import TrainingConf
from backbone import MLLMBackbone
from dataset import BalancedFLORESDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_constructors = {"vanilla": SAE, "gated": GatedSparseAutoEncoder}

def pool_hidden_states(
    hidden: torch.Tensor,        
    attention_mask: torch.Tensor, 
    reduction: str = "mean",
) -> torch.Tensor:                
    if reduction == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    elif reduction == "cls":
        return hidden[:, 0, :]
    elif reduction == "last":
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), lengths]
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--output_dir",    default="./cached_embeddings")
    parser.add_argument("--num_sentences", type=int, default=500,
                        help="Sentences per language")
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

    batch_size = conf.batch_size 
    layer_idx = conf.layer_idx

    all_activations  = []
    sentences = []
    lang_tags = []

    dataset = BalancedFLORESDataset(langs=conf.langs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for step, batch in enumerate(dataloader):
        texts = batch["text"]
        backbone_acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=layer_idx,
            max_length=128,
            encoder_only=conf.encoder_only)
        
        if conf.pool_features:
            with torch.no_grad():
                x = reduce(backbone_acts["layer_tensor"], "b s f -> b f", conf.reduction)
        else:
            x = backbone_acts["token_activations"]
        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)
        with torch.no_grad():
            z,_ = sae.encode(x)

        all_activations.append(z.cpu().float().numpy())
        sentences.extend(texts)
        lang_tags.extend(batch['lang'])

    activations  = np.concatenate(all_activations,  axis=0) 
    np.save(out_dir / "activations.npy",  activations)
    with open(out_dir / "sentences.json", "w", encoding="utf-8") as stream:
        json.dump(sentences, stream, ensure_ascii=False, indent=2)

    metadata = {
        "config": cfg,
        "lang_tags": lang_tags, 
        "num_sentences": len(sentences),
        "layer_idx": layer_idx,
        #"reduction": reduction,
    }
    with open(out_dir / "metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)

if __name__ == "__main__":
    main()
