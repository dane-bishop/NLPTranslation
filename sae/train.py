import os
import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from einops import reduce

import numpy as np
from tqdm import tqdm

from sae import SAE, GatedSparseAutoEncoder
from backbone import MLLMBackbone
from dataset import BalancedNLLBDataset

@dataclass
class TrainingConf:
    backbone_name: str
    model_hidden_size: int
    sae_type: int
    sae_hidden_size: int
    pairs: list[str]
    langs: list[str]
    batch_size: int
    lr: float
    weight_decay: float
    max_steps: int
    layer_idx: int
    encoder_only: bool
    pool_features: bool
    reduction: str
    sparsity_weight: float
    print_every: int
    weight_path: str


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_constructors = {"vanilla": SAE, "gated": GatedSparseAutoEncoder}


def collate_records(batch):
    return {
        "texts": [item["text"] for item in batch],
        "langs": [item["lang"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }

def update_sae(autoencoder: SAE | GatedSparseAutoEncoder, embeddings, optim, conf):
    optim.zero_grad()
    outputs = autoencoder.loss(embeddings, conf.sparsity_weight)
    outputs["loss"].backward()
    optim.step()
    return outputs

def train(conf: TrainingConf):
    backbone_name = conf.backbone_name #"facebook/nllb-200-distilled-600M"
    backbone = MLLMBackbone(device, backbone_name)
    model_constructor = sae_constructors[conf.sae_type]

    autoencoder = model_constructor(conf.model_hidden_size,conf.sae_hidden_size).to(device)

    pair_configs = conf.pairs
    langs = conf.langs

    dataset = BalancedNLLBDataset(pair_configs, langs)

    loader = DataLoader(dataset, conf.batch_size, collate_fn=collate_records)

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    max_steps = conf.max_steps

    pbar = tqdm(total=max_steps)
    for step, batch in enumerate(loader):
        if step > max_steps:
            break

        texts = batch["texts"]

        acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=conf.layer_idx,
            max_length=128,
            encoder_only=conf.encoder_only            
        )

        if conf.pool_features:
            with torch.no_grad():
                x = reduce(acts["layer_tensor"], "b s f -> b f", conf.reduction)
        else:
            x = acts["token_activations"]

        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)

        updates = update_sae(autoencoder, x, optimizer, conf)
        
        if step % conf.print_every == 0:
            pbar.set_description(f"loss is {updates["loss"].item()}")
        pbar.update(1)

    torch.save(autoencoder.state_dict(), conf.weight_path)

def main():
    import argparse
    import json
    parser = argparse.ArgumentParser("Train GSA",description="Train gated sparse autoencoder on NLLB")
    parser.add_argument("config_path",help="path to training config. Required.")
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        conf = TrainingConf(**json.load(stream))

    train(conf)
    
if __name__ == "__main__":
    main()
