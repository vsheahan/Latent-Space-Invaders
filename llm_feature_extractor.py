"""
LLM Feature Extractor

Loads a pre-trained LLM and extracts hidden states from specific layers.
Designed to work with TinyLlama for efficiency, but compatible with any
transformers model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import numpy as np


class LLMFeatureExtractor:
    """
    Extract hidden states from specific layers of a pre-trained LLM.

    This class handles:
    - Loading the LLM model and tokenizer
    - Tokenizing prompts with proper padding/truncation
    - Extracting hidden states from specified layers
    - Getting the last non-padding token's representation
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        layer_indices: Optional[List[int]] = None,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize the feature extractor.

        Args:
            model_name: HuggingFace model identifier
            layer_indices: Which layers to extract from (None = auto-select)
            max_length: Maximum sequence length for tokenization
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Determine layer indices
        num_layers = self.model.config.num_hidden_layers
        if layer_indices is None:
            # Auto-select evenly spaced layers (first, 20%, 40%, 60%, 80%, last)
            self.layer_indices = [
                0,
                num_layers // 5,
                2 * num_layers // 5,
                3 * num_layers // 5,
                4 * num_layers // 5,
                num_layers - 1
            ]
        else:
            self.layer_indices = layer_indices

        self.hidden_size = self.model.config.hidden_size

        print(f"Model loaded with {num_layers} layers")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Extracting from layers: {self.layer_indices}")

    def extract_hidden_states(
        self,
        prompts: List[str],
        batch_size: int = 8
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden states from specified layers for a list of prompts.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping layer_idx -> array of hidden states
            Shape: {layer_idx: (num_prompts, hidden_size)}
        """
        all_hidden_states = {layer_idx: [] for layer_idx in self.layer_indices}

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_hidden_states = self._extract_batch(batch_prompts)

            for layer_idx, hidden_states in batch_hidden_states.items():
                all_hidden_states[layer_idx].append(hidden_states)

        # Concatenate batches
        result = {}
        for layer_idx in self.layer_indices:
            result[layer_idx] = np.concatenate(all_hidden_states[layer_idx], axis=0)

        return result

    def _extract_batch(self, prompts: List[str]) -> Dict[int, np.ndarray]:
        """
        Extract hidden states for a single batch of prompts.

        Args:
            prompts: Batch of text prompts

        Returns:
            Dictionary mapping layer_idx -> array of hidden states for this batch
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Extract hidden states for target layers
        # outputs.hidden_states is a tuple: (layer_0, layer_1, ..., layer_n)
        # Each element has shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states

        batch_hidden = {}
        for layer_idx in self.layer_indices:
            # Get hidden states for this layer
            layer_hidden = hidden_states[layer_idx]  # (batch_size, seq_len, hidden_size)

            # Extract last non-padding token for each sequence
            last_token_hidden = self._get_last_token_hidden(layer_hidden, attention_mask)

            # Move to CPU and convert to numpy
            batch_hidden[layer_idx] = last_token_hidden.cpu().numpy()

        return batch_hidden

    def _get_last_token_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the hidden state of the last non-padding token for each sequence.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        batch_size = hidden_states.shape[0]

        # Find the index of the last non-padding token for each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        seq_lengths = attention_mask.sum(dim=1)  # (batch_size,)

        # Get last token indices (subtract 1 because of 0-indexing)
        last_token_indices = seq_lengths - 1  # (batch_size,)

        # Gather the hidden states at these indices
        # Using torch.gather or indexing
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_hidden_states = hidden_states[batch_indices, last_token_indices, :]

        return last_hidden_states  # (batch_size, hidden_size)

    def get_num_selected_layers(self) -> int:
        """Get the number of selected layers being extracted."""
        return len(self.layer_indices)

    def get_total_llm_layers(self) -> int:
        """Get the total number of layers in the LLM model."""
        return self.model.config.num_hidden_layers

    def get_hidden_size(self) -> int:
        """Get the hidden size of the model."""
        return self.hidden_size
