import json
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

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
    max_length: int
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
    activation: str = "gelu"
    topk: int = None


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


def masked_mean_pool(layer_tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.unsqueeze(-1).float()
    summed = (layer_tensor * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def train(conf: TrainingConf):
    torch.manual_seed(37)

    # -------------------------
    # Load MLLM backbone
    # -------------------------
    backbone_name = conf.backbone_name #"facebook/nllb-200-distilled-600M"
    backbone = MLLMBackbone(device, backbone_name)

    # -------------------------
    # Initialize SAE or Gated SAE
    # -------------------------
    model_constructor: SAE | GatedSparseAutoEncoder = sae_constructors[conf.sae_type]

    autoencoder = model_constructor(
        d_act=conf.model_hidden_size,
        d_hidden=conf.sae_hidden_size,
    ).to(device)

    # -------------------------
    # Create NLLB Dataset
    # -------------------------
    pair_configs = conf.pairs
    langs = conf.langs

    dataset = BalancedNLLBDataset(pair_configs, langs)
    loader = DataLoader(dataset, conf.batch_size, collate_fn=collate_records)

    # -------------------------
    # Create optimizer / begin training
    # -------------------------
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    max_steps = conf.max_steps

    pbar = tqdm(total=max_steps)
    for step, batch in enumerate(loader):
        if step > max_steps:
            break

        # -- Extract MLLM backbone activations
        texts = batch["texts"]

        acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=conf.layer_idx,
            max_length=conf.max_length,
            encoder_only=conf.encoder_only            
        )

        # -- Optionally pool token features (e.g., average for sentence embedding)
        if conf.pool_features:
            if conf.reduction != "mean":
                raise ValueError("Pooled SAE training currently requires reduction='mean'")
            with torch.no_grad():
                x = masked_mean_pool(acts["layer_tensor"], acts["valid_mask"])
        else:
            x = acts["token_activations"]

        if x.numel() == 0:
            continue

        # -- Shift embeddings to zero-mean
        x = x - x.mean(dim=0, keepdim=True)

        updates = update_sae(autoencoder, x, optimizer, conf)
        
        if step % conf.print_every == 0:
            pbar.set_description(f"loss is {updates['loss'].item()}")
        pbar.update(1)

    # -------------------------
    # Save model weights / training config
    # -------------------------
    weights_dir = Path(conf.weight_path).parent
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(autoencoder.state_dict(), conf.weight_path)

    config_path = weights_dir / "config.json"
    with open(config_path, "w") as stream:
        json.dump(asdict(conf), stream, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser("Train GSA",description="Train gated sparse autoencoder on NLLB")
    parser.add_argument("config_path",help="path to training config. Required.")
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        conf = TrainingConf(**json.load(stream))

    train(conf)
    
if __name__ == "__main__":
    main()
