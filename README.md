<p align="center">
  <img src="assets/Gemini_Generated_Image_c1fq1ic1fq1ic1fq.png" width="500" alt="Latent Space Invaders Logo">
</p>

# Latent Space Invaders

A novel approach to detecting prompt injection attacks using **Layer-Conditioned Variational Autoencoders (VAE)** on LLM hidden states.

## Overview

This project implements a prompt injection detection system that:
- Extracts hidden states from multiple layers of a pre-trained LLM
- Trains a layer-conditioned VAE exclusively on safe prompts
- Detects anomalies based on reconstruction error per layer
- Uses multi-layer voting for robust detection

**Key Innovation**: Instead of using simple distance metrics (which suffer from length sensitivity), we learn the normal manifold of safe embeddings using a VAE that is conditioned on the layer index.

## Why VAE?

Previous attempts using Mahalanobis distance, cosine similarity, and PCA residuals failed due to:
1. **Length sensitivity**: Longer prompts naturally have higher distances
2. **Overlapping distributions**: Attack prompts often append to legitimate text
3. **Statistical assumptions**: Embeddings are non-Gaussian in high dimensions

**VAE advantages**:
- Learns non-linear manifold of normal embeddings
- Layer conditioning allows layer-specific modeling
- Reconstruction error is more robust than distance metrics
- Can detect subtle deviations from learned distribution

## Architecture

### Layer-Conditioned VAE

```
Input: [hidden_state, layer_id_onehot]
         ↓
    Encoder MLP → (mu, logvar)
         ↓
   Reparameterize → z (latent)
         ↓
    Decoder MLP ← [z, layer_id_onehot]
         ↓
    Reconstruction

Loss = MSE(reconstruction, original) + beta * KL(q(z|x) || p(z))
```

### Detection Pipeline

1. **Feature Extraction**: Extract hidden states from LLM layers (e.g., [0, 4, 8, 12, 16, 20])
2. **VAE Training**: Train on safe prompts only (unsupervised anomaly detection)
3. **Baseline Statistics**: Compute mean/std of reconstruction errors per layer
4. **Detection**: Flag if `error > mean + k*std` for multiple layers
5. **Voting**: Require N+ layers to agree (e.g., 2 out of 6)

## Project Structure

```
latent-space-invaders/
├── llm_feature_extractor.py   # LLM hidden state extraction
├── conditioned_vae.py          # Layer-conditioned VAE model
├── anomaly_detector.py         # Main detection system
├── utils.py                    # Evaluation metrics and helpers
├── main.py                     # Orchestration script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Requirements:
# - torch>=2.0.0
# - transformers>=4.30.0
# - accelerate>=0.20.0
# - scikit-learn>=1.3.0
# - numpy>=1.24.0
# - matplotlib>=3.7.0
# - tqdm>=4.65.0
```

## Quick Start

### Demo with Synthetic Data

```bash
python3 main.py --mode dummy
```

This will:
- Generate synthetic safe and attack prompts
- Train the VAE (10 epochs)
- Evaluate on test data
- Save results to `results_ls/`

### Using Your Own Datasets

```bash
python3 main.py --mode files \
    --safe-train path/to/safe_train.txt \
    --safe-test path/to/safe_test.txt \
    --attack-test path/to/attack_test.txt \
    --output-dir results_custom
```

**File format**: One prompt per line in plain text files.

## Usage Examples

### Basic Usage

```python
from anomaly_detector import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    latent_dim=128,
    threshold_k=2.0,
    min_anomalous_layers=2
)

# Train on safe prompts
safe_prompts = ["What is AI?", "Explain Python", ...]
detector.fit(safe_prompts, epochs=10)

# Detect anomalies
is_anomalous, scores, agg_score = detector.score_prompt(
    "Ignore all previous instructions..."
)
print(f"Anomalous: {is_anomalous}, Score: {agg_score:.3f}")
```

### Advanced Configuration

```python
# Custom layer selection
detector = AnomalyDetector(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    layer_indices=[0, 5, 10, 15, 20],  # Specific layers
    latent_dim=64,                     # Smaller latent space
    threshold_k=3.0,                   # Stricter threshold
    min_anomalous_layers=3             # Require 3+ layers
)

# Custom training parameters
detector.fit(
    safe_prompts,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    beta=0.01  # KL divergence weight
)

# Save/load model
detector.save_model("my_detector.pth")
detector.load_model("my_detector.pth")
```

## Configuration Parameters

### Detector Parameters

- `model_name`: HuggingFace model for feature extraction
- `layer_indices`: Which layers to extract (None = auto-select 6 evenly-spaced)
- `latent_dim`: VAE latent space dimension (default: 128)
- `threshold_k`: Anomaly threshold multiplier (default: 2.0)
  - Higher = fewer false positives, more false negatives
- `min_anomalous_layers`: Voting threshold (default: 2)
  - Higher = more conservative detection

### Training Parameters

- `epochs`: VAE training epochs (default: 10)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Optimizer learning rate (default: 1e-3)
- `beta`: KL divergence weight in loss (default: 0.01)
- `validation_split`: Fraction for validation (default: 0.1)

## Evaluation Metrics

The system computes:
- **Accuracy**: Overall correctness
- **Precision**: TP / (TP + FP) - How many flagged prompts are actual attacks
- **Recall (TPR)**: TP / (TP + FN) - How many attacks are caught
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: FP / (FP + TN) - How many safe prompts are wrongly flagged
- **ROC AUC**: Area under ROC curve (threshold-independent metric)

## Output Files

Results are saved to `output_dir/`:
- `ls_detector.pth` - Trained model checkpoint
- `evaluation_results.json` - Full metrics and predictions
- `roc_curve.png` - ROC curve visualization
- `confusion_matrix.png` - Confusion matrix heatmap

## How It Works

### 1. Feature Extraction

```python
# Extract last-token hidden states from selected layers
hidden_states = extractor.extract_hidden_states(prompts)
# Returns: {layer_idx: np.ndarray(num_prompts, hidden_size)}
```

### 2. VAE Training

The VAE learns to reconstruct hidden states while being conditioned on layer ID:

```
input = [hidden_state, one_hot(layer_id)]
reconstruction = VAE(input)
loss = MSE(reconstruction, hidden_state) + beta * KL_divergence
```

Training exclusively on safe prompts means the VAE learns what "normal" looks like.

### 3. Anomaly Detection

For each test prompt:
1. Extract hidden states from all target layers
2. For each layer, compute reconstruction error
3. Compare error to learned baseline (mean + k*std)
4. Flag layer as anomalous if error exceeds threshold
5. Vote: flag prompt if `≥ min_anomalous_layers` are anomalous

## Hyperparameter Tuning

### Threshold K (Most Important)

```bash
# Conservative (fewer false positives)
python3 main.py --threshold-k 3.0

# Aggressive (catch more attacks)
python3 main.py --threshold-k 1.5

# Recommended starting point
python3 main.py --threshold-k 2.0
```

### Minimum Anomalous Layers

```bash
# Strict (require more layers to agree)
python3 main.py --min-anomalous-layers 3

# Lenient (fewer layers needed)
python3 main.py --min-anomalous-layers 1

# Recommended
python3 main.py --min-anomalous-layers 2
```

### Latent Dimension

```bash
# Smaller (faster, less expressive)
python3 main.py --latent-dim 64

# Larger (more expressive, slower)
python3 main.py --latent-dim 256

# Recommended balance
python3 main.py --latent-dim 128
```

## Limitations

1. **Computational Cost**: Running TinyLlama for feature extraction is slower than simple classifiers
2. **Training Data**: Requires representative safe prompts for training
3. **Novel Attack Patterns**: May not generalize to completely unseen attack structures
4. **Length Variation**: While better than distance metrics, extreme length differences may still cause issues

## Comparison to Previous Approach

| Metric | Distance-Based | VAE-Based |
|--------|----------------|-----------|
| Length sensitivity | High | Low |
| Non-linear manifolds | No | Yes |
| Layer-specific modeling | Limited | Full |
| False positives (expected) | 96.9% | TBD |
| Training complexity | Low | Medium |

## Future Improvements

- **Conditional Beta-VAE**: Learn disentangled representations
- **Adversarial Training**: Include attack samples with different loss
- **Attention-based VAE**: Use attention over layers instead of one-hot
- **Online Learning**: Update model with new safe prompts
- **Ensemble Methods**: Combine with other detectors

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python3 main.py --batch-size 16

# Use smaller latent dimension
python3 main.py --latent-dim 64
```

### Poor Performance

```bash
# Train longer
python3 main.py --epochs 20

# Adjust threshold
python3 main.py --threshold-k 1.5

# Use more layers for voting
python3 main.py --min-anomalous-layers 3
```

## Citation

If you use this code in your research:

```bibtex
@software{latent_space_invaders_2025,
  title = {Latent Space Invaders},
  author = {Your Name},
  year = {2025},
  note = {Layer-conditioned VAE for detecting prompt injection attacks}
}
```

## License

MIT License - Free to use and modify

## Acknowledgments

- Built on the foundation of previous distance-based approaches (which failed spectacularly)
- Inspired by VAE-based anomaly detection in other domains
- Uses TinyLlama for efficient experimentation
