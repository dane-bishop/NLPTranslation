import torch
from transformers import AutoModel, AutoTokenizer


class MLLMBackbone:
    def __init__(self, device: torch.device):
        self.device = device

        self.model = AutoModel.from_pretrained(
            "microsoft/mdeberta-v3-base", use_safetensors=True, dtype=torch.float32
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")

    @torch.no_grad()
    def extract_layer_activations(
        self, texts: list[str], layer_idx: int, max_length: int = 128
    ) -> dict[str, torch.Tensor]:
        """ """
        # -------------------------
        # Tokenize texts / put onto device
        # -------------------------
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # -------------------------
        # Pass batch through model
        # -------------------------
        outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        # -------------------------
        # Extract hidden states from specified layer_idx
        # -------------------------
        x = hidden_states[layer_idx]  # (B, T, H)

        # -------------------------
        # Mask out special tokens / attention mask
        # -------------------------
        attention_mask = batch["attention_mask"].bool()  # (B, T)

        input_ids = batch["input_ids"]
        special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

        for special_id in self.tokenizer.all_special_ids:
            special_mask |= input_ids == special_id

        valid_mask = attention_mask & (~special_mask)

        # -------------------------
        # Flatten valid tokens: (N, H)
        # -------------------------
        x_valid = x[valid_mask]

        return {
            "token_activations": x_valid,       # (N, 768)
            "layer_tensor": x,
            "valid_mask": valid_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
