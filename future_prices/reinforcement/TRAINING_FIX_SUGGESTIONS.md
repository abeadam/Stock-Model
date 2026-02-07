# Training Instability Fix Suggestions

Based on the observed training divergence (R² going negative and getting worse), here are prioritized suggestions:

## 🔴 **CRITICAL FIXES (Do These First)**

### 1. **Remove Loss Scaling** ✅ DONE
- **Problem**: 5x loss scaling amplifies gradients, causing explosion during warmup
- **Fix**: Removed the `* 5.0` multiplier from mean-only loss
- **Why**: Different loss scales are fine - the optimizer adapts. Scaling amplifies gradients too much, especially during warmup when LR is increasing.

### 2. **Lower Base Learning Rate**
- **Current**: `LEARNING_RATE = 0.0002`
- **Suggested**: `LEARNING_RATE = 0.00005` (4x lower) or `0.0001` (2x lower)
- **Why**: With warmup starting at 2% and increasing, even 0.0002 might be too high. Lower base LR = more stable training.

### 3. **More Conservative Warmup**
- **Current**: Starts at 2% of base LR, 50 epochs warmup
- **Suggested**: 
  - Start at 1% of base LR (even more conservative)
  - Or reduce warmup epochs to 20-30 (faster ramp-up, less time at low LR)
- **Why**: Current warmup is very slow, and combined with loss scaling was causing issues.

### 4. **Stricter Mean-Only Switching Condition**
- **Current**: Switches when `train_r2 > 0.0 and val_r2 > 0.0` (very low threshold)
- **Suggested**: Switch when `train_r2 > 0.1 and val_r2 > 0.1` (original was better)
- **Why**: Switching too early (R² barely positive) means model hasn't learned basic patterns yet.

## 🟡 **IMPORTANT FIXES (Do These Next)**

### 5. **Reduce Focal Loss Gamma**
- **Current**: `gamma=2.0`
- **Suggested**: `gamma=1.0` or `gamma=0.5`
- **Why**: Lower gamma = less aggressive down-weighting of easy samples. High gamma can cause tiny gradients for small errors, preventing learning.

### 6. **Increase Gradient Clipping**
- **Current**: `max_norm=0.5`
- **Suggested**: `max_norm=0.25` or `max_norm=0.1`
- **Why**: More aggressive clipping prevents gradient explosion, especially important during warmup.

### 7. **Add Learning Rate Scaling Based on Loss Scale**
- **Idea**: Instead of scaling loss, scale learning rate for mean-only mode
- **Implementation**: Use different LR for mean-only vs full mode
- **Why**: Matches the effective gradient scale without amplifying gradients.

### 8. **Monitor Prediction Statistics**
- **Add**: Print prediction mean/std every epoch during mean-only mode
- **Why**: Early detection of predictions going crazy (mean=204, std=355) before R² explodes.

## 🟢 **OPTIONAL IMPROVEMENTS**

### 9. **Use Simpler Loss During Mean-Only**
- **Current**: Focal loss with Huber base
- **Alternative**: Simple MSE or Huber loss (no focal)
- **Why**: Focal loss adds complexity. Simple loss might be more stable during initial learning.

### 10. **Reduce Warmup Epochs**
- **Current**: 50 epochs
- **Suggested**: 20-30 epochs
- **Why**: Faster ramp-up means less time at very low LR, but still provides stability.

### 11. **Add Exponential Moving Average (EMA) for Model Weights**
- **Idea**: Maintain EMA of model weights, use for validation
- **Why**: EMA weights are more stable and often perform better on validation.

### 12. **Batch Normalization or Layer Normalization**
- **Check**: Does the model have normalization layers?
- **Why**: Normalization helps stabilize training, especially with varying loss scales.

## 📊 **Configuration Recommendations**

### Recommended Settings:
```python
LEARNING_RATE = 0.00005  # 4x lower than current
WARMUP_EPOCHS = 20  # Faster warmup
warmup_start_multiplier = 0.01  # Start at 1% (more conservative)
grad_clip_max_norm = 0.25  # More aggressive clipping
mean_only_switch_threshold = 0.1  # Both train and val R² > 0.1
focal_loss_gamma = 1.0  # Less aggressive
```

### Testing Order:
1. ✅ Remove loss scaling (DONE)
2. Lower base LR to 0.00005
3. Increase gradient clipping to 0.25
4. Change mean-only switch to R² > 0.1
5. Reduce focal loss gamma to 1.0
6. Reduce warmup epochs to 20

## 🔍 **Root Cause Analysis**

The training is diverging because:
1. **Loss scaling (5x)** amplifies gradients
2. **Learning rate warmup** increases LR over time
3. **Combined effect**: Large gradients × increasing LR = gradient explosion
4. **Result**: Predictions explode (mean=204, std=355), R² goes very negative

The fix is to remove loss scaling and use a lower base LR. The optimizer will naturally adapt to different loss scales.

