"""
Layer-Conditioned Variational Autoencoder (VAE)

A VAE that is conditioned on which LLM layer the hidden state comes from.
This allows the model to learn the unique "normal" manifold for each layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConditionedVAE(nn.Module):
    """
    Layer-Conditioned Variational Autoencoder.

    The VAE takes as input:
    - hidden_state: A vector from an LLM layer (size: hidden_size)
    - layer_id: Which layer this came from (one-hot encoded)

    It learns to reconstruct the hidden_state while being aware of which
    layer it came from, allowing layer-specific modeling.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        latent_dim: int = 128,
        encoder_hidden_dims: Tuple[int, ...] = (512, 256),
        decoder_hidden_dims: Tuple[int, ...] = (256, 512)
    ):
        """
        Initialize the Layer-Conditioned VAE.

        Args:
            hidden_size: Dimensionality of LLM hidden states
            num_layers: Number of different layers we're conditioning on
            latent_dim: Dimensionality of the latent space
            encoder_hidden_dims: Hidden layer sizes for encoder MLP
            decoder_hidden_dims: Hidden layer sizes for decoder MLP
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Input size: hidden_state + one-hot layer_id
        encoder_input_size = hidden_size + num_layers

        # Build encoder MLP
        encoder_layers = []
        prev_dim = encoder_input_size
        for hidden_dim in encoder_hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Encoder output: mu and log_var for latent distribution
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Build decoder MLP
        # Input: latent_sample + one-hot layer_id
        decoder_input_size = latent_dim + num_layers

        decoder_layers = []
        prev_dim = decoder_input_size
        for hidden_dim in decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Final layer: reconstruct original hidden_state
        decoder_layers.append(nn.Linear(prev_dim, hidden_size))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor, layer_condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Hidden states (batch_size, hidden_size)
            layer_condition: One-hot encoded layer IDs (batch_size, num_layers)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Concatenate hidden state with layer condition
        encoder_input = torch.cat([x, layer_condition], dim=1)

        # Pass through encoder
        h = self.encoder(encoder_input)

        # Get mu and log_var
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, var) using N(0,1).

        Args:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)

        Returns:
            Sampled latent vector (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, layer_condition: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector back to hidden state space.

        Args:
            z: Latent vector (batch_size, latent_dim)
            layer_condition: One-hot encoded layer IDs (batch_size, num_layers)

        Returns:
            Reconstructed hidden states (batch_size, hidden_size)
        """
        # Concatenate latent vector with layer condition
        decoder_input = torch.cat([z, layer_condition], dim=1)

        # Pass through decoder
        reconstruction = self.decoder(decoder_input)

        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
        layer_condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Hidden states (batch_size, hidden_size)
            layer_condition: One-hot encoded layer IDs (batch_size, num_layers)

        Returns:
            reconstruction: Reconstructed hidden states (batch_size, hidden_size)
            mu: Latent distribution mean (batch_size, latent_dim)
            logvar: Latent distribution log variance (batch_size, latent_dim)
        """
        # Encode
        mu, logvar = self.encode(x, layer_condition)

        # Sample latent vector
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decode(z, layer_condition)

        return reconstruction, mu, logvar


def vae_loss(
    reconstruction: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + beta * KL Divergence.

    Args:
        reconstruction: Reconstructed hidden states (batch_size, hidden_size)
        x: Original hidden states (batch_size, hidden_size)
        mu: Latent distribution mean (batch_size, latent_dim)
        logvar: Latent distribution log variance (batch_size, latent_dim)
        beta: Weight for KL divergence term (default: 0.01)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstruction, x, reduction='mean')

    # KL divergence loss
    # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def compute_reconstruction_error(
    model: ConditionedVAE,
    x: torch.Tensor,
    layer_condition: torch.Tensor
) -> torch.Tensor:
    """
    Compute reconstruction error (MSE) for each sample.

    Args:
        model: Trained VAE model
        x: Hidden states (batch_size, hidden_size)
        layer_condition: One-hot encoded layer IDs (batch_size, num_layers)

    Returns:
        Per-sample reconstruction errors (batch_size,)
    """
    model.eval()
    with torch.no_grad():
        reconstruction, _, _ = model(x, layer_condition)

        # Compute MSE per sample (not averaged across batch)
        errors = F.mse_loss(reconstruction, x, reduction='none')
        errors = errors.mean(dim=1)  # Average across hidden_size dimension

    return errors
