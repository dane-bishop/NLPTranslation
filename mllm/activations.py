import torch
import torch.nn as nn

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        assert k is not None, "Batch Top K was instantiated without argument for k; check your config."
        self.k = k

    def forward(self, z: torch.Tensor):
        z_topk = torch.topk(z, self.k, dim=-1)
        return torch.zeros_like(z).scatter(-1, z_topk.indices, z_topk.values)

class BatchTopK(TopK):
    def forward(self, z: torch.Tensor):
        z_topk = torch.topk(z.flatten(), self.k * z.shape[0], dim=-1)
        return torch.zeros_like(z.flatten()).scatter(-1, z_topk.indices, z_topk.values).reshape(z.shape) 