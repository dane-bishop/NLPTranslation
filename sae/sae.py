import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAE(nn.Module):
    """
    Learns a sparse representation of MLLM token activations of shape (in_dim,)
    - Encodes the activations into a higher-dimensional latent space
    - Decodes that spare represntation back into the original activation space
    """
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Linear(in_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, in_dim, bias=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input activations into a higher dimensional sparse representation
        """
        z_pre = self.encoder(x)
        z = F.relu(z_pre)
        return z, z_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes spare representation back into original activation space
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode / decode input MLLM activations
        """
        # -- Higher-dim sparse representation
        z, z_pre = self.encode(x)
        # -- Predicted activations reconstruction
        x_hat = self.decode(z)
        return {
            "x_hat": x_hat,     # activation reconstruction
            "z": z,             # sparse latent
            "z_pre": z_pre,     # pre-activation latent
        }

    def loss(self, x: torch.Tensor, l1_coef: float = 1e-4) -> dict[str, torch.Tensor]:
        """

        """
        out = self.forward(x)
        recon_loss = F.mse_loss(out["x_hat"], x)
        l1_loss = out["z"].abs().mean()
        loss = recon_loss + l1_coef * l1_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            **out,
        }


class GatedSparseAutoEncoder(nn.Module):
    """
    SAE with added gating step
    Encoder returns both post-gate latents and pre-gate latents for aux term computation.
    """
    def __init__(self,
                 d_act: int,
                 d_hidden: int,
                 activation: Callable = nn.GELU()):
        super().__init__()
        self.decoder_bias = nn.Parameter(torch.zeros(d_act))
        self.encoder = nn.Linear(d_act,d_hidden,bias=False)
        self.decoder = nn.Linear(d_hidden, d_act, bias=False)
        self.hidden_bias = nn.Parameter(torch.zeros(d_hidden))
        self.scale = nn.Parameter(torch.zeros(d_hidden))
        self.gate_bias = nn.Parameter(torch.zeros(d_hidden))
        self.scale_bias = nn.Parameter(torch.zeros(d_hidden))


        self.activation = activation

    def encode(self, x):
        x = F.linear(x-self.decoder_bias, self.encoder.weight, self.hidden_bias)
        pi = x + self.gate_bias
        z = (pi > 0).float()
        pi = self.scale.exp() * x + self.scale_bias
        z_pre = self.activation(pi) #path with no heaviside; we need to keep it to preserve gradients
        z = z * z_pre
        return z, z_pre
    

    def decode(self, z):
        y = F.linear(z, self.decoder.weight, self.decoder_bias)
        return y

    def forward(self,x):
        z, z_pre = self.encode(x)
        return {"x_hat": self.decode(z),
                "z": z,
                "z_pre": z_pre}

    def loss(self, x, sparsity_weight = 1e-4):
        """
        Computes 3-term gated sae loss for given input x
        Includes auxilary term which is computed based on pre-gate activations
        """
        outputs = self.forward(x)

        with torch.no_grad():
            x_hat_frozen = self.decode(outputs["z_pre"])

        recon_term = F.mse_loss(outputs["x_hat"], x)
        l1_term = outputs["z_pre"].norm(p=1,dim=-1).mean()
        aux_term = F.mse_loss(x_hat_frozen, x)
        loss = recon_term + sparsity_weight * l1_term + aux_term
        return {"loss": loss,
                "recon_loss": recon_term,
                "l1_loss": l1_term,
                "auxilary_loss": aux_term,
                **outputs}
