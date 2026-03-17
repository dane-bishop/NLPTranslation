import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BalancedNLLBDataset
from backbone import MLLMBackbone
from sae import SAE


def collate_records(batch):
    return {
        "texts": [item["text"] for item in batch],
        "langs": [item["lang"] for item in batch],
        "pairs": [item["pair"] for item in batch],
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pair_configs = [
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

    langs = [
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

    dataset = BalancedNLLBDataset(pair_configs, langs)

    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_records,
        num_workers=0,
    )

    backbone = MLLMBackbone(device)
    sae = SAE(in_dim=768, latent_dim=4096).to(device)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-4)

    layer_idx = 6
    l1_coef = 2.0
    max_steps = 1000

    sae.train()

    for step, batch in tqdm(enumerate(loader), total=max_steps):
        if step >= max_steps:
            break

        texts = batch["texts"]

        acts = backbone.extract_layer_activations(
            texts=texts,
            layer_idx=layer_idx,
            max_length=128,
        )

        x = acts["token_activations"]   # (N, 768)

        if x.numel() == 0:
            continue

        x = x - x.mean(dim=0, keepdim=True)

        optimizer.zero_grad()
        out = sae.loss(x, l1_coef=l1_coef)
        out["loss"].backward()
        optimizer.step()

        with torch.no_grad():
            active_frac = (out["z"] > 0).float().mean().item()
            feature_firing = (out["z"] > 0).float().mean(dim=0)
            dead_features = (feature_firing < 1e-6).sum().item()


        if step % 20 == 0:
            print(
                f"step={step} "
                f"loss={out['loss'].item():.6f} "
                f"recon={out['recon_loss'].item():.6f} "
                f"l1={out['l1_loss'].item():.6f} "
                f"active_frac={active_frac:.6f} "
                f"dead_features={dead_features}"
            )

    torch.save(sae.state_dict(), "sae_layer6.pt")
    print("Saved SAE weights to sae_layer6.pt")


if __name__ == "__main__":
    main()