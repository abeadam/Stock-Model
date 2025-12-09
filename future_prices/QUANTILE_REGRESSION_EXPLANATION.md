# Quantile Regression: Explanation and Application

## What is Quantile Regression?

**Quantile regression** predicts specific percentiles (quantiles) of the target distribution instead of just the mean. Instead of asking "what's the average value?", it asks "what's the 10th percentile? 50th? 90th?"

### Key Concepts:

1. **Quantiles**: Percentiles of a distribution
   - 50th quantile (median) = middle value
   - 10th quantile = 10% of values are below this
   - 90th quantile = 90% of values are below this

2. **Quantile Loss (Pinball Loss)**:
   ```
   For quantile τ (tau) and error e = target - prediction:
   
   If e ≥ 0 (underprediction):
     loss = τ * e
   
   If e < 0 (overprediction):
     loss = (1 - τ) * |e|
   ```
   
   - For τ=0.5 (median): symmetric, like MAE
   - For τ=0.9 (90th percentile): penalizes underprediction more
   - For τ=0.1 (10th percentile): penalizes overprediction more

3. **Multiple Quantiles**: Predict several quantiles simultaneously
   - e.g., predict 10th, 50th, 90th quantiles
   - Gives you a full picture of the distribution

## Current Approach vs Quantile Regression

### Your Current Approach:
- **Center + Range**: Predicts `center = (high + low) / 2` and `range = high - low`
- **Derived**: `high = center + range/2`, `low = center - range/2`
- **Assumption**: Symmetric distribution around center
- **Loss**: MSE/Huber/Focal loss on high and low separately

### Quantile Regression Approach:
- **Direct Quantiles**: Predict 10th quantile (low) and 90th quantile (high) directly
- **No Symmetry Assumption**: Can handle asymmetric distributions
- **Loss**: Quantile loss (pinball loss) for each quantile

## Would Quantile Regression Help?

### ✅ **YES, it could help significantly:**

1. **Direct High/Low Prediction**:
   - Currently: Predict center + range, derive high/low
   - Quantile: Predict high (90th) and low (10th) directly
   - **Benefit**: No need to ensure center is between high/low (your misalignment loss becomes unnecessary)

2. **Better for Extreme Values**:
   - Quantile loss naturally emphasizes tail behavior
   - 90th quantile loss heavily penalizes underprediction of high values
   - **Benefit**: Directly addresses your "values collapsing to zero" problem

3. **No Distribution Assumptions**:
   - Current: Assumes symmetric distribution (center ± range/2)
   - Quantile: Makes no assumptions about distribution shape
   - **Benefit**: Can handle skewed, fat-tailed distributions common in finance

4. **Robust to Outliers**:
   - Quantile loss is less sensitive to outliers than MSE
   - **Benefit**: More stable training

5. **Natural Uncertainty Quantification**:
   - Predict multiple quantiles (e.g., 10th, 50th, 90th) to get full distribution
   - **Benefit**: Better uncertainty estimates

### ⚠️ **Potential Challenges:**

1. **Architecture Change Needed**:
   - Replace `center_head` and `range_head` with `quantile_10_head` and `quantile_90_head`
   - Or predict multiple quantiles with separate heads

2. **Loss Function Change**:
   - Replace MSE/Huber/Focal with quantile loss
   - Need to specify which quantile each head predicts

3. **Initialization**:
   - Need to initialize quantile heads to appropriate values
   - Can use empirical quantiles from training data

## Implementation Strategy

### Option 1: Replace Center+Range with Direct Quantiles
```python
# Instead of:
center_head = nn.Linear(prev_dim, 1)
range_head = nn.Linear(prev_dim, 1)
high = center + range/2
low = center - range/2

# Use:
quantile_10_head = nn.Linear(prev_dim, 1)  # Low (10th percentile)
quantile_90_head = nn.Linear(prev_dim, 1)  # High (90th percentile)
low = quantile_10_head(features)
high = quantile_90_head(features)
```

### Option 2: Predict Multiple Quantiles
```python
# Predict 10th, 50th, 90th quantiles
quantile_heads = nn.ModuleList([
    nn.Linear(prev_dim, 1),  # 10th quantile (low)
    nn.Linear(prev_dim, 1),  # 50th quantile (median/center)
    nn.Linear(prev_dim, 1),  # 90th quantile (high)
])
```

### Quantile Loss Function:
```python
def quantile_loss(pred, target, quantile):
    """
    Quantile loss (pinball loss)
    
    Args:
        pred: Predicted quantile value
        target: True target value
        quantile: Quantile level (0.0 to 1.0)
    
    Returns:
        Quantile loss
    """
    error = target - pred
    loss = torch.max(
        quantile * error,           # Penalty for underprediction
        (quantile - 1) * error       # Penalty for overprediction
    )
    return loss.mean()
```

## Recommendation

**YES, quantile regression would likely help** because:

1. **Directly addresses your problem**: Predicting high/low as quantiles naturally prevents collapse
2. **Better for extreme values**: Quantile loss emphasizes tail behavior
3. **Simpler structure**: No need for center+range derivation
4. **More appropriate for finance**: Financial data often has fat tails and asymmetry

### Suggested Implementation:

1. **Replace center+range with quantile heads**:
   - `quantile_10_head` → predicts low (10th percentile)
   - `quantile_90_head` → predicts high (90th percentile)
   - Optional: `quantile_50_head` → predicts median/center

2. **Use quantile loss**:
   - For 10th quantile: `quantile_loss(pred_low, target_low, quantile=0.1)`
   - For 90th quantile: `quantile_loss(pred_high, target_high, quantile=0.9)`

3. **Initialize from data**:
   - Initialize `quantile_10_head` bias to 10th percentile of training data
   - Initialize `quantile_90_head` bias to 90th percentile of training data

4. **Keep variance heads** (optional):
   - Can still predict variance for uncertainty quantification
   - Or derive variance from quantile spread

## Comparison Table

| Aspect | Current (Center+Range) | Quantile Regression |
|--------|------------------------|---------------------|
| **Prediction** | Derived: high = center + range/2 | Direct: high = quantile_90 |
| **Symmetry** | Assumes symmetric | No assumption |
| **Extreme Values** | Struggles (collapses) | Natural emphasis |
| **Loss** | MSE/Huber/Focal | Quantile (pinball) |
| **Complexity** | Center + Range + Derivation | Direct quantile heads |
| **Robustness** | Sensitive to outliers | More robust |

## Conclusion

Quantile regression is **well-suited for your problem** because:
- It directly predicts high/low without assuming symmetry
- Quantile loss naturally handles extreme values
- It's commonly used in finance for this exact reason
- It eliminates the need for center misalignment penalties

The main trade-off is changing the architecture, but the benefits likely outweigh the implementation effort.

