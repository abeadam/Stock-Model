# Value Function Learning Fixes - Implementation Summary

## Changes Implemented

Three key fixes have been implemented to address the value loss bottleneck:

### 1. ✅ Separate Learning Rates for Actor/Critic

**Implementation:**
- Added `actor_lr` and `critic_lr` parameters to `PPOAgent.__init__()`
- Default behavior: `actor_lr = lr`, `critic_lr = lr * 2.0` (critic learns 2x faster)
- Optimizer now uses separate parameter groups:
  - **Actor group**: Feature layers + Actor head + Risk head → uses `actor_lr`
  - **Critic group**: Critic head only → uses `critic_lr`
- Learning rate scheduler applies to both groups proportionally
- Checkpoint saving/loading preserves separate learning rates

**Rationale:**
- Value function was struggling to learn (stuck at ~1.0 loss)
- Giving critic a higher learning rate allows it to catch up faster
- Actor and critic can now learn at different rates, which is common in PPO

**Configuration:**
```python
actor_lr=5e-4,   # Actor learning rate
critic_lr=1e-3   # Critic learning rate (2x actor for faster value learning)
```

---

### 2. ✅ Increased Value Loss Coefficient to 2.0

**Implementation:**
- Changed `value_coef` from `1.0` to `2.0` in `run_ppo_example.py`
- Value loss now has 2x weight in total loss calculation
- Updated resume mode to use `value_coef=2.0` if not in checkpoint

**Rationale:**
- Value loss was dominating total loss but not decreasing
- Increasing `value_coef` gives value function learning more importance
- Forces the optimizer to prioritize reducing value loss

**Configuration:**
```python
value_coef=2.0  # Value loss has 2x weight in total loss
```

---

### 3. ✅ Normalized Value Targets (Returns)

**Implementation:**
- Returns (value targets) are now normalized before being used as targets
- Normalization: `returns_normalized = (returns - mean) / (std + 1e-8)`
- Returns are clipped to ±10 standard deviations after normalization
- Value predictions are also normalized using the same stats when computing loss
- Normalization stats (`returns_mean`, `returns_std`) are stored for consistency

**Rationale:**
- Returns can vary wildly (e.g., -1000 to +1000), making value function learning unstable
- Normalization helps the value network learn more consistently
- Both targets and predictions are normalized, ensuring they're in the same space

**Code Location:**
- Normalization happens in `PPOAgent.update()` method
- After computing GAE and returns, returns are normalized
- Value predictions are normalized when computing value loss

**Details:**
```python
# Normalize returns (targets)
returns_mean = returns.mean()
returns_std = returns.std() + 1e-8
returns_normalized = (returns - returns_mean) / returns_std
returns_normalized = np.clip(returns_normalized, -10.0, 10.0)

# Normalize value predictions to match
values_normalized = (values - returns_mean) / (returns_std + 1e-8)

# Compute loss between normalized values
value_loss = mse_loss(values_normalized, returns_normalized)
```

---

## Expected Impact

### Before Fixes:
- Value loss stuck at ~1.0 after 2000 episodes
- Total loss increasing (0.974 → 0.992)
- P&L getting worse (-$6,560 → -$7,778)

### After Fixes:
- ✅ **Value loss should decrease faster** (target: <0.8 by episode 2000)
- ✅ **Total loss should decrease consistently** (not increase)
- ✅ **P&L should improve** and become positive
- ✅ **Training should be more stable** (lower variance)

---

## Configuration Summary

**Fresh Training:**
```python
PPOAgent(
    network_size='xlarge',
    lr=5e-4,              # Base LR (used for actor)
    actor_lr=5e-4,        # Actor learning rate
    critic_lr=1e-3,       # Critic learning rate (2x actor)
    value_coef=2.0,       # Value loss coefficient (2x weight)
    # ... other params
)
```

**Resume Training:**
- Automatically loads `actor_lr` and `critic_lr` from checkpoint if available
- Uses `value_coef=2.0` if not in checkpoint
- Preserves all other hyperparameters

---

## Testing Recommendations

1. **Run training for 2000-5000 episodes** with these fixes
2. **Monitor value loss closely** - should decrease to <0.8
3. **Monitor total loss** - should decrease consistently
4. **Check P&L** - should improve and become positive
5. **Compare with previous run** - should see significant improvement

---

## Technical Notes

### Normalization Consistency
- Returns are normalized per update batch
- Value predictions are normalized using the same batch stats
- This ensures value function learns to predict normalized returns
- During rollout, value function will output normalized values (which is fine for bootstrapping)

### Learning Rate Scheduler
- CosineAnnealingLR applies to both parameter groups
- Both actor and critic learning rates decay proportionally
- Minimum learning rate floor (`min_lr`) applies to both groups

### Checkpoint Compatibility
- Old checkpoints (without `actor_lr`/`critic_lr`) will use default behavior:
  - `actor_lr = lr` (from checkpoint)
  - `critic_lr = lr * 2.0` (computed)
- New checkpoints include separate learning rates for full compatibility

---

## Next Steps

1. **Run training** with these fixes: `python3 run_ppo_example.py`
2. **Monitor logs** for value loss trends
3. **If value loss still stuck**, consider:
   - Further increasing `critic_lr` (e.g., 3x actor LR)
   - Further increasing `value_coef` (e.g., 3.0)
   - Checking reward scaling (rewards might be too large/small)
4. **If training is unstable**, consider:
   - Reducing `critic_lr` (e.g., 1.5x actor LR)
   - Reducing learning rates overall
   - Adding more gradient clipping

---

## Files Modified

1. **`futures_renforcement_ppo.py`**:
   - Added `actor_lr` and `critic_lr` parameters
   - Modified optimizer to use separate parameter groups
   - Enhanced value target normalization
   - Added normalization of value predictions
   - Updated checkpoint saving/loading

2. **`run_ppo_example.py`**:
   - Updated `value_coef` to 2.0
   - Added `actor_lr` and `critic_lr` configuration
   - Updated resume mode to handle separate learning rates
   - Updated checkpoint config helper

---

## References

- PPO Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Common practice: Separate learning rates for actor/critic when value function struggles
- Value function normalization: Standard technique for stable RL training
