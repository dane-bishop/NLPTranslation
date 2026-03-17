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
    
