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

    token_cache = []
    all_activations = []
    feature_representations = []
    reduced_activations = []
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
        
        x = backbone_acts["token_activations"]
        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)
        with torch.no_grad():
            z,_ = sae.encode(x)
            
        
        z = z.cpu().float().squeeze()
        if z.shape[0] > args.topk:
            fr = torch.topk(z, args.topk, 0).values.numpy()
        else:
            continue


        token_cache.append(get_tokens(backbone, backbone_acts, batch['lang']))
        #all_activations.append(z.numpy)
        feature_representations.append(fr)
        reduced_activations.append(reduce(z, "t f -> f", "mean").numpy())
        sentences.extend(texts)
        lang_tags.extend(batch['lang'])

    reduced_acts = np.stack(reduced_activations, axis=0)
    feature_representations = np.stack(feature_representations, axis = 0)
    topk_sent = np.argpartition(reduced_acts, -args.topk, axis=0)[-args.topk:].T
    topk_sentence_lookup = []
    for n in range(topk_sent.shape[0]):
        sentence_indices = topk_sent[n]
        current_feat_table = []
        for i in sentence_indices:
            sentence_feats = feature_representations[i,...]
            tk_idx = np.argmax(sentence_feats[:,n], axis=0)
            top_token = token_cache[i][np.argmax(sentence_feats[:,n], axis=0)]
            current_feat_table.append((int(i), top_token))
        topk_sentence_lookup.append(current_feat_table)
    
    #nonzero_mask   = feature_representations > 0            # [N, F]
    #fire_counts    = nonzero_mask.sum(axis=0)    # [F]  how many sentences each feature fires on
    #fire_freq      = fire_counts / feature_representations.shape[0]
    #active_mask    = (fire_counts >= 5) & (fire_freq >= 0.01)
    #active_indices = np.where(active_mask)[0]    # feature indices that pass

    sentence_projections = TSNE(perplexity=30, metric="cosine", random_state=seed, verbose=True).fit(reduced_acts)

    #min_sentences  = args.min_active_sentences   # e.g. 5, add to argparse
    #freq_threshold = args.freq_threshold         # e.g. 0.01 (1% of sentences)

    feature_representations = torch.topk(torch.tensor(feature_representations),args.num_sentences,dim=0).values.numpy()
    #print(feature_representations.shape)

    feature_projections = TSNE(perplexity=30, metric="cosine", random_state=seed, verbose=True).fit(rearrange(feature_representations,"s n f -> f (n s)"))
    np.save(out_dir / "sentence_projections.npy",  sentence_projections)
    np.save(out_dir / "feature_projections.npy",  feature_projections)
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
