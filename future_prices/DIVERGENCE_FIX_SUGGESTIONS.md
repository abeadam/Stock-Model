# Divergence Fix Suggestions (R² < -1.0)

## Problem Analysis

When divergence is detected (R² = -1.21), the learning rate is reduced to 0.000003, but the model may:
1. Continue diverging (bad gradients already in optimizer state)
2. Recover too slowly (LR too small)
3. Get stuck in a bad local minimum

## 🔴 **CRITICAL FIXES (Implement First)**

### 1. **Reset Optimizer State on Divergence**
**Why it works**: Adam optimizer maintains momentum/state. When divergence occurs, this state contains bad gradient information that continues to push the model in the wrong direction. Resetting clears this bad state.

**Implementation**:
```python
if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
    # ... existing code ...
    # Reset optimizer state to clear bad momentum
    optimizer.state = defaultdict(dict)  # Clear all optimizer state
    print(f"  Optimizer state reset to clear bad momentum")
```

**Why it works**: Clears Adam's running averages of gradients, allowing fresh start with reduced LR.

### 2. **Stop Warmup Permanently on Divergence**
**Why it works**: Warmup continues increasing LR even after divergence. Once diverged, we should stop warmup entirely and use fixed reduced LR.

**Implementation**:
```python
# Add flag to track if warmup should continue
warmup_active = True

# In divergence detection:
if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
    warmup_active = False  # Stop warmup permanently
    # ... rest of divergence handling ...

# In scheduler step:
if epoch < warmup_epochs and warmup_active:
    warmup_scheduler.step()
```

**Why it works**: Prevents LR from continuing to increase after divergence, keeping it at the reduced level.

### 3. **Rollback to Previous Best Model**
**Why it works**: When divergence occurs, the model is already in a bad state. Rolling back to the last good checkpoint gives a better starting point for recovery.

**Implementation**:
```python
if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
    if best_model_state is not None:
        print(f"  Rolling back to best model from epoch {best_epoch}")
        model.load_state_dict(best_model_state)
        # Reset optimizer state
        optimizer.state = defaultdict(dict)
```

**Why it works**: Returns model to a known good state before applying reduced LR, rather than trying to recover from a bad state.

### 4. **More Aggressive Gradient Clipping**
**Why it works**: Current clipping (0.5) might not be aggressive enough. Tighter clipping prevents large gradient updates that cause divergence.

**Implementation**:
```python
# Reduce from 0.5 to 0.1 or even 0.05
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
```

**Why it works**: Limits maximum gradient magnitude, preventing sudden large parameter updates that cause predictions to explode.

## 🟡 **IMPORTANT FIXES (Implement Next)**

### 5. **Freeze Backbone on Divergence**
**Why it works**: When diverging, the backbone (feature extractor) might be learning bad features. Freezing it and only training output heads can stabilize training.

**Implementation**:
```python
if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
    # Freeze backbone, only train output heads
    for name, param in model.named_parameters():
        if 'quantile' in name or 'variance_head' in name or 'mean_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"  Backbone frozen, only output heads will train")
```

**Why it works**: Prevents backbone from learning bad features, allows output heads to adapt to current features.

### 6. **Reduce LR More Gradually**
**Why it works**: 10x reduction might be too aggressive. Gradual reduction (e.g., 2x) might allow better recovery.

**Implementation**:
```python
# Instead of 0.1 (10x reduction), use 0.5 (2x reduction)
param_group['lr'] *= 0.5
```

**Why it works**: Smaller LR reductions allow model to adapt gradually rather than sudden shock.

### 7. **Add Weight Constraints**
**Why it works**: Prevents weights from growing too large, which causes predictions to explode.

**Implementation**:
```python
# After optimizer.step(), clamp weights
with torch.no_grad():
    for param in model.parameters():
        param.clamp_(-10.0, 10.0)  # Constrain weights to reasonable range
```

**Why it works**: Hard constraint prevents weights from exploding, keeping predictions in reasonable range.

### 8. **Monitor and Clip Predictions**
**Why it works**: If predictions explode (mean=204, std=355), clip them during training to prevent bad gradients.

**Implementation**:
```python
# After forward pass, before loss calculation
if predict_both:
    mean_pred_high = torch.clamp(mean_pred_high, min=-100.0, max=100.0)
    mean_pred_low = torch.clamp(mean_pred_low, min=-100.0, max=100.0)
```

**Why it works**: Prevents extreme predictions from generating extreme gradients.

## 🟢 **OPTIONAL IMPROVEMENTS**

### 9. **Use Gradient Accumulation**
**Why it works**: Simulates larger batch size with smaller actual batches, leading to more stable gradients.

**Implementation**:
```python
accumulation_steps = 4
loss = loss / accumulation_steps  # Scale loss
loss.backward()

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### 10. **Switch to SGD on Divergence**
**Why it works**: SGD (without momentum) is more stable than Adam when recovering from divergence.

**Implementation**:
```python
if val_loss_exploded or r2_diverged or r2_rapidly_worsening:
    # Switch to SGD with very low LR
    optimizer = optim.SGD(model.parameters(), lr=current_lr * 0.1, momentum=0.0)
```

### 11. **Reduce Batch Size**
**Why it works**: Smaller batches = more frequent updates = more stable training.

**Implementation**:
```python
BATCH_SIZE = 128  # Instead of 256
```

### 12. **Add Learning Rate Floor**
**Why it works**: Prevents LR from getting too small (0.000003 is very small), which causes extremely slow learning.

**Implementation**:
```python
min_lr = 1e-5  # Don't let LR go below this
if current_lr < min_lr:
    current_lr = min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = min_lr
```

## 📊 **Recommended Implementation Order**

1. ✅ **Reset optimizer state** (Critical - clears bad momentum)
2. ✅ **Stop warmup on divergence** (Critical - prevents LR increase)
3. ✅ **Rollback to best model** (Critical - better starting point)
4. ✅ **More aggressive gradient clipping** (Important - prevents explosion)
5. ✅ **Add weight constraints** (Important - prevents weight explosion)
6. ✅ **Monitor and clip predictions** (Important - prevents bad gradients)

## 🔍 **Root Cause**

The divergence happens because:
1. **Bad gradients accumulate** in optimizer state (Adam momentum)
2. **Warmup continues** increasing LR even after divergence
3. **Model is in bad state** - trying to recover from bad state is harder than starting from good state
4. **Gradients are too large** - even with clipping at 0.5, some gradients might be problematic

The fixes address these by:
- Clearing bad state (reset optimizer)
- Stopping warmup (prevent further LR increase)
- Rolling back (start from good state)
- Tighter constraints (prevent explosion)

