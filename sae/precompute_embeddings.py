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
from datasets import load_dataset
from tqdm import tqdm

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
    activations = []
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
            z,_ = sae.encode(x)
            
        
        z = z.cpu().float().squeeze()
        z = torch.nn.functional.pad(z, (0, 0, 0, 128 - z.shape[0]))

        token_cache.append(get_tokens(backbone, backbone_acts, batch['lang']))
        activations.append(z)
        sentences.extend(texts)
        lang_tags.extend(batch['lang'])
        step += 1
        if step >= num_sentences:
            break

    return activations, token_cache, sentences, lang_tags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--output_dir",    default="./cached_embeddings")
    parser.add_argument("--num_sentences", type=int, default=500,
                        help="Sentences per language")
    parser.add_argument("--topk", type=int, default=20, help="Num tokens for each feature to keep")
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

    activations, token_cache, sentences, lang_tags = get_sentence_activations(backbone, sae, dataloader, args.num_sentences, conf)
         
    acts_tensor = torch.stack(activations, dim=0)

    max_per_feature = acts_tensor.amax(dim=(0, 1))  
    alive_mask = max_per_feature > max_per_feature.mean() * 0.1        
    alive_indices = alive_mask.nonzero(as_tuple=True)[0]  

    acts_alive = acts_tensor[:, :, alive_mask]

    feature_topk_res = torch.topk(acts_alive, dim=0, k=args.topk)

    feat_topk_indices = feature_topk_res.indices  
    feat_topk_values  = feature_topk_res.values   

    num_features = acts_tensor.shape[2]
    topk_sentence_lookup = []

    for feat_idx in range(acts_alive.shape[2]):
        feat_values  = feat_topk_values[:, :, feat_idx]   
        feat_indices = feat_topk_indices[:, :, feat_idx]  
        current_feat_table = []
        for rank in range(args.topk):
            token_list = token_cache[feat_indices[rank, 0].item()]  
            n_real_tokens = len(token_list)

            real_feat_values = feat_values[rank, :n_real_tokens]
            if real_feat_values.numel() == 0:
                continue

            best_token_pos = real_feat_values.argmax().item()
            sentence_idx   = feat_indices[rank, best_token_pos].item()

            token_list = token_cache[sentence_idx]
            top_token  = token_list[best_token_pos] if best_token_pos < len(token_list) else "[PAD]"

            current_feat_table.append((int(sentence_idx), top_token))


        topk_sentence_lookup.append(current_feat_table)
    

    sentence_topk_res = torch.topk(acts_tensor, dim=-1, k=args.topk)

    sentence_projections = TSNE(perplexity=30, metric="cosine", random_state=seed, verbose=True).fit(rearrange(sentence_topk_res.values, "s t f -> s (t f)").numpy())
    feature_projections = TSNE(perplexity=30, metric="cosine", random_state=seed, verbose=True).fit(rearrange(feat_topk_values, "s t f -> f (s t)").numpy())
    np.save(out_dir / "sentence_projections.npy",  sentence_projections)
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
       #"reduction": reduction,
    }
    with open(out_dir / "metadata.json", "w") as stream:
        json.dump(metadata, stream, indent=2)

if __name__ == "__main__":
    main()
