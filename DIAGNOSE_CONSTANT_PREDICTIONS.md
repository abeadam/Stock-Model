# Diagnosing Constant Predictions (Model Collapse)

## Problem
The model is predicting the same value (0.0001%) for all samples, which indicates the model has collapsed to predicting the mean.

## Likely Causes

### 1. **Target Values Too Small After Normalization**
- Percentage changes are very small (0.001% to 0.01%)
- After StandardScaler normalization, the signal might be too weak
- The model can't distinguish patterns in such small normalized values

### 2. **HuberLoss Delta Too Large**
- Current: `delta=1.0`
- For normalized targets with std ~1, this might be too large
- Consider reducing to `delta=0.1` or `delta=0.01`

### 3. **Learning Rate Issues**
- Too high: Model overshoots and collapses
- Too low: Model doesn't learn at all
- Current: 0.001 with warmup

### 4. **Vanishing Gradients**
- Model might be too deep
- Activation functions might be saturating
- Need to check gradient norms

### 5. **Model Initialization**
- Poor initialization can cause early collapse
- Check if model starts with reasonable outputs

## Solutions to Try

### Solution 1: Don't Normalize Targets
Since percentage changes are already in a reasonable range, try NOT normalizing them:

```python
# Instead of:
targets_scaled = scaler_targets.fit_transform(targets)

# Try:
targets_scaled = targets  # Use raw percentage changes
scaler_targets = None  # Or create a dummy scaler
```

### Solution 2: Reduce HuberLoss Delta
```python
criterion = nn.HuberLoss(delta=0.1)  # Or even 0.01
```

### Solution 3: Use MSE Instead of HuberLoss
For very small values, MSE might work better:
```python
criterion = nn.MSELoss()
```

### Solution 4: Scale Targets Before Normalization
Multiply by 100 to make them larger:
```python
targets = targets * 100  # Convert 0.001% to 0.1
targets_scaled = scaler_targets.fit_transform(targets)
```

### Solution 5: Check Model Output Range
Add diagnostics to see what the model is actually outputting:
- Check first batch output range
- Check gradient norms
- Check if outputs are constant

## Quick Fix to Try First

The most likely issue is that the normalized targets are too small. Try:

1. **Don't normalize targets** - percentage changes are already in a good range
2. **Reduce HuberLoss delta** to 0.1 or 0.01
3. **Check if model outputs vary** in the first epoch

