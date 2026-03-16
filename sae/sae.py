import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()

        self.fc = nn.Linear(in_ch, in_ch)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
    