# Latent Space Invaders - Critical Fixes Applied

## Summary
Fixed dimension mismatch bug and implemented consistency improvements across the codebase.

## Critical Fix: Dimension Mismatch (HIGHEST PRIORITY)

### Problem
The `num_layers` parameter was ambiguous - it could refer to either:
1. Total number of layers in the LLM (e.g., 22 for TinyLlama)
2. Number of selected layers for extraction (e.g., 6 auto-selected layers)

This caused a dimension mismatch when creating one-hot encoded layer conditions.

### Solution
Renamed to `num_selected_layers` throughout `anomaly_detector.py` to explicitly refer to the number of layers being used for conditioning the VAE.

### Files Modified

#### 1. `llm_feature_extractor.py`
- **Renamed**: `get_num_layers()` → `get_num_selected_layers()`
- **Added**: `get_total_llm_layers()` method for clarity
- **Impact**: Clear distinction between selected layers vs total LLM layers

```python
def get_num_selected_layers(self) -> int:
    """Get the number of selected layers being extracted."""
    return len(self.layer_indices)

def get_total_llm_layers(self) -> int:
    """Get the total number of layers in the LLM model."""
    return self.model.config.num_hidden_layers
```

#### 2. `anomaly_detector.py`
- **Changed**: All instances of `self.num_layers` → `self.num_selected_layers`
- **Locations**:
  - Line 68: Instance variable initialization
  - Line 131: One-hot encoding tensor size
  - Line 157: VAE initialization parameter
  - Line 241: Baseline statistics layer condition
  - Line 288: Score prompt layer condition
  - Line 328: save_model checkpoint
  - Line 346: load_model checkpoint
  - Line 354: VAE reconstruction in load_model

**Critical**: The VAE now receives the correct number of layers for one-hot encoding (6 selected layers, not 22 total layers).

## Additional Improvements

### 3. Batch Size Consistency
**File**: `anomaly_detector.py`

Changed feature extraction to use the `batch_size` parameter from `fit()` instead of hardcoded value:

```python
# Before:
hidden_states_by_layer = self.feature_extractor.extract_hidden_states(
    safe_prompts,
    batch_size=8  # Hardcoded
)

# After:
hidden_states_by_layer = self.feature_extractor.extract_hidden_states(
    safe_prompts,
    batch_size=batch_size  # Uses parameter from fit()
)
```

### 4. Rich Results Output
**File**: `utils.py`

Added `scores_per_layer` to the results dictionary for detailed analysis:

```python
# Convert all_scores_per_layer to JSON-serializable format
scores_per_layer_serializable = [
    {str(k): float(v) for k, v in scores.items()}
    for scores in all_scores_per_layer
]

results = {
    # ... existing fields ...
    'scores_per_layer': scores_per_layer_serializable,  # NEW
    # ... rest of fields ...
}
```

**Benefit**: Users can now analyze per-layer reconstruction errors for each prompt in the test set.

### 5. Beta Parameter
**File**: `anomaly_detector.py`

**Verified**: The `beta` parameter is correctly passed to `vae_loss()` on lines 181 and 204. No changes needed.

## Testing Recommendations

After these changes, you should:

1. **Run a quick test** to verify dimensions match:
```bash
cd /Users/vincentsheahan/latent-space-invaders
python3 main.py --mode dummy --epochs 1 --batch-size 16
```

2. **Check the output** for:
   - No dimension mismatch errors
   - VAE successfully initializes with correct num_layers
   - One-hot encoding shapes match expectations
   - Results include 'scores_per_layer' field

3. **Verify saved results**:
```bash
cat results_ls/evaluation_results.json | grep "scores_per_layer"
```

## Impact

### Before
- One-hot encoding: `torch.zeros(N, 22)` (total LLM layers)
- VAE expecting: `(batch_size, hidden_size + 22)`
- **MISMATCH**: If only 6 layers selected, layer_id > 6 would be invalid

### After
- One-hot encoding: `torch.zeros(N, 6)` (selected layers)
- VAE expecting: `(batch_size, hidden_size + 6)`
- **CORRECT**: layer_id ranges from 0-5, matching tensor size

## Backward Compatibility

**Breaking Change**: Saved models from before this fix will not load correctly due to the `num_selected_layers` rename in checkpoints.

**Migration**: If you have old saved models, they will need to be retrained.

## Files Modified Summary

1. ✅ `llm_feature_extractor.py` - Method rename + new method
2. ✅ `anomaly_detector.py` - Critical dimension fix + batch_size improvement
3. ✅ `utils.py` - Enhanced results output
4. ⚪ `conditioned_vae.py` - No changes (already correct)
5. ⚪ `main.py` - No changes needed

## Date
Applied: 2025-10-31
