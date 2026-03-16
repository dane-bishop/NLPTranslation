import torch
from transformers import AutoModel


model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base", use_safetensors=True, dtype=torch.float32)

