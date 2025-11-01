"""
VAE-Based Anomaly Detector for Prompt Injection

Uses a Layer-Conditioned VAE to learn the normal manifold of safe prompts
and detect anomalies based on reconstruction error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json

from llm_feature_extractor import LLMFeatureExtractor
from conditioned_vae import ConditionedVAE, vae_loss, compute_reconstruction_error


class AnomalyDetector:
    """
    VAE-based anomaly detector for prompt injection attacks.

    The detector:
    1. Trains a layer-conditioned VAE on safe prompts only
    2. Learns per-layer statistics of reconstruction errors
    3. Detects anomalies when reconstruction errors exceed thresholds
    4. Uses voting across layers for robust detection
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_indices: Optional[List[int]] = None,
        latent_dim: int = 128,
        threshold_k: float = 2.0,
        min_anomalous_layers: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize the anomaly detector.

        Args:
            model_name: HuggingFace model to use for feature extraction
            layer_indices: Which layers to extract (None = auto-select)
            latent_dim: Dimensionality of VAE latent space
            threshold_k: Threshold multiplier (anomaly if error > mean + k*std)
            min_anomalous_layers: Minimum layers that must be anomalous
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.threshold_k = threshold_k
        self.min_anomalous_layers = min_anomalous_layers
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize feature extractor
        print("Initializing feature extractor...")
        self.feature_extractor = LLMFeatureExtractor(
            model_name=model_name,
            layer_indices=layer_indices,
            device=self.device
        )

        self.layer_indices = self.feature_extractor.layer_indices
        self.hidden_size = self.feature_extractor.get_hidden_size()
        self.num_selected_layers = self.feature_extractor.get_num_selected_layers()

        # Initialize VAE (will be trained in fit())
        self.vae = None
        self.layer_statistics = {}  # Store mean/std of reconstruction errors per layer
        self.is_fitted = False

    def fit(
        self,
        safe_prompts: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        validation_split: float = 0.1
    ):
        """
        Train the VAE on safe prompts only and compute baseline statistics.

        Args:
            safe_prompts: List of safe (non-attack) prompts
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            beta: KL divergence weight in VAE loss
            validation_split: Fraction of data to use for validation
        """
        print(f"\n{'='*80}")
        print("TRAINING VAE ON SAFE PROMPTS")
        print(f"{'='*80}")
        print(f"Number of safe prompts: {len(safe_prompts)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}, Beta: {beta}")

        # Extract hidden states from all layers for safe prompts
        print("\nExtracting hidden states from safe prompts...")
        hidden_states_by_layer = self.feature_extractor.extract_hidden_states(
            safe_prompts,
            batch_size=batch_size
        )

        # Prepare training data
        # Combine all layers into a single dataset with layer conditioning
        all_hidden_states = []
        all_layer_ids = []

        for layer_idx in self.layer_indices:
            layer_hidden = hidden_states_by_layer[layer_idx]  # (num_prompts, hidden_size)
            layer_id = self.layer_indices.index(layer_idx)

            all_hidden_states.append(layer_hidden)
            all_layer_ids.extend([layer_id] * len(layer_hidden))

        # Stack all data
        all_hidden_states = np.vstack(all_hidden_states)  # (num_samples, hidden_size)
        all_layer_ids = np.array(all_layer_ids)  # (num_samples,)

        print(f"Total training samples: {len(all_hidden_states)} (prompts × layers)")

        # Convert to tensors
        X = torch.FloatTensor(all_hidden_states)
        layer_ids = torch.LongTensor(all_layer_ids)

        # One-hot encode layer IDs
        layer_conditions = torch.zeros(len(layer_ids), self.num_selected_layers)
        layer_conditions[torch.arange(len(layer_ids)), layer_ids] = 1.0

        # Split into train/validation
        num_samples = len(X)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val

        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        X_train, layer_cond_train = X[train_indices], layer_conditions[train_indices]
        X_val, layer_cond_val = X[val_indices], layer_conditions[val_indices]

        # Create data loaders
        train_dataset = TensorDataset(X_train, layer_cond_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, layer_cond_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize VAE
        print(f"\nInitializing VAE (latent_dim={self.latent_dim})...")
        self.vae = ConditionedVAE(
            hidden_size=self.hidden_size,
            num_layers=self.num_selected_layers,
            latent_dim=self.latent_dim
        ).to(self.device)

        # Optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)

        # Training loop
        print("\nTraining VAE...")
        for epoch in range(epochs):
            # Train
            self.vae.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0

            for batch_x, batch_layer_cond in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_x = batch_x.to(self.device)
                batch_layer_cond = batch_layer_cond.to(self.device)

                # Forward pass
                reconstruction, mu, logvar = self.vae(batch_x, batch_layer_cond)

                # Compute loss
                loss, recon_loss, kl_loss = vae_loss(reconstruction, batch_x, mu, logvar, beta)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()

            # Validation
            self.vae.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_layer_cond in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_layer_cond = batch_layer_cond.to(self.device)

                    reconstruction, mu, logvar = self.vae(batch_x, batch_layer_cond)
                    loss, recon_loss, kl_loss = vae_loss(reconstruction, batch_x, mu, logvar, beta)

                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_loss.item()

            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f} "
                  f"(Recon: {train_recon_loss/len(train_loader):.4f}, "
                  f"KL: {train_kl_loss/len(train_loader):.4f})")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f} "
                  f"(Recon: {val_recon_loss/len(val_loader):.4f}, "
                  f"KL: {val_kl_loss/len(val_loader):.4f})")

        # Compute baseline statistics for each layer
        print("\nComputing baseline statistics per layer...")
        self._compute_baseline_statistics(hidden_states_by_layer)

        self.is_fitted = True
        print("\nTraining complete!")

    def _compute_baseline_statistics(self, hidden_states_by_layer: Dict[int, np.ndarray]):
        """
        Compute mean and std of reconstruction errors for each layer.

        Args:
            hidden_states_by_layer: Dict mapping layer_idx -> hidden states array
        """
        self.vae.eval()

        for layer_idx in self.layer_indices:
            layer_id = self.layer_indices.index(layer_idx)
            hidden_states = hidden_states_by_layer[layer_idx]

            # Convert to tensors
            X = torch.FloatTensor(hidden_states).to(self.device)
            layer_cond = torch.zeros(len(X), self.num_selected_layers).to(self.device)
            layer_cond[:, layer_id] = 1.0

            # Compute reconstruction errors
            errors = compute_reconstruction_error(self.vae, X, layer_cond)
            errors = errors.cpu().numpy()

            # Store statistics
            self.layer_statistics[layer_idx] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'threshold': float(np.mean(errors) + self.threshold_k * np.std(errors))
            }

            print(f"Layer {layer_idx}: mean={self.layer_statistics[layer_idx]['mean']:.6f}, "
                  f"std={self.layer_statistics[layer_idx]['std']:.6f}, "
                  f"threshold={self.layer_statistics[layer_idx]['threshold']:.6f}")

    def score_prompt(self, prompt: str) -> Tuple[bool, Dict[int, float], float]:
        """
        Score a single prompt for anomaly detection.

        Args:
            prompt: Text prompt to score

        Returns:
            is_anomalous: Whether the prompt is flagged as anomalous
            scores_per_layer: Reconstruction error for each layer
            aggregated_score: Overall anomaly score
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring. Call fit() first.")

        # Extract hidden states
        hidden_states_by_layer = self.feature_extractor.extract_hidden_states([prompt], batch_size=1)

        scores_per_layer = {}
        anomalous_layer_count = 0

        self.vae.eval()

        for layer_idx in self.layer_indices:
            layer_id = self.layer_indices.index(layer_idx)
            hidden_state = hidden_states_by_layer[layer_idx][0]  # Get single sample

            # Convert to tensor
            X = torch.FloatTensor(hidden_state).unsqueeze(0).to(self.device)  # (1, hidden_size)
            layer_cond = torch.zeros(1, self.num_selected_layers).to(self.device)
            layer_cond[0, layer_id] = 1.0

            # Compute reconstruction error
            error = compute_reconstruction_error(self.vae, X, layer_cond)
            error = float(error.cpu().item())

            scores_per_layer[layer_idx] = error

            # Check if anomalous for this layer
            threshold = self.layer_statistics[layer_idx]['threshold']
            if error > threshold:
                anomalous_layer_count += 1

        # Voting: is anomalous if enough layers agree
        is_anomalous = anomalous_layer_count >= self.min_anomalous_layers

        # Aggregated score: average normalized error across layers
        normalized_errors = []
        for layer_idx in self.layer_indices:
            error = scores_per_layer[layer_idx]
            mean = self.layer_statistics[layer_idx]['mean']
            std = self.layer_statistics[layer_idx]['std']
            normalized_error = (error - mean) / (std + 1e-8)
            normalized_errors.append(normalized_error)

        aggregated_score = float(np.mean(normalized_errors))

        return is_anomalous, scores_per_layer, aggregated_score

    def save_model(self, path: str):
        """Save the trained model and statistics."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        checkpoint = {
            'model_state_dict': self.vae.state_dict(),
            'layer_statistics': self.layer_statistics,
            'layer_indices': self.layer_indices,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_selected_layers,
            'latent_dim': self.latent_dim,
            'threshold_k': self.threshold_k,
            'min_anomalous_layers': self.min_anomalous_layers,
            'model_name': self.model_name
        }

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a trained model and statistics."""
        checkpoint = torch.load(path, map_location=self.device)

        # Restore configuration
        self.layer_statistics = checkpoint['layer_statistics']
        self.layer_indices = checkpoint['layer_indices']
        self.hidden_size = checkpoint['hidden_size']
        self.num_selected_layers = checkpoint['num_layers']
        self.latent_dim = checkpoint['latent_dim']
        self.threshold_k = checkpoint['threshold_k']
        self.min_anomalous_layers = checkpoint['min_anomalous_layers']

        # Recreate VAE
        self.vae = ConditionedVAE(
            hidden_size=self.hidden_size,
            num_layers=self.num_selected_layers,
            latent_dim=self.latent_dim
        ).to(self.device)

        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True

        print(f"Model loaded from {path}")
