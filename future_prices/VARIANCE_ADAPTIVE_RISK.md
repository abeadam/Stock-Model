# Variance-Adaptive Risk Control

## Overview

The system now supports **variance-adaptive risk control**, which dynamically adjusts stop-loss thresholds based on the model's predicted uncertainty (variance). This allows the agent to be more conservative when predictions are uncertain and more aggressive when predictions are confident.

## How It Works

### 1. **Variance Prediction**
The model predicts both:
- **Mean predictions**: `pred_mean_high`, `pred_mean_low` (expected price movements)
- **Variance predictions**: `pred_std_high`, `pred_std_low` (uncertainty in predictions)

These are already in the state vector, so the agent can see them.

### 2. **Dynamic Stop-Loss Adjustment**

When `variance_adaptive_risk=True`:
- **High variance (uncertainty)** → **Tighter stop-loss** (more conservative)
- **Low variance (confidence)** → **Looser stop-loss** (more aggressive)

**Formula:**
```
max_pred_std = max(|pred_std_high|, |pred_std_low|)

If max_pred_std < 0.001:  # Very confident
    variance_factor = 1.5  # Allow 1.5x larger losses
Else if max_pred_std > 0.1:  # Very uncertain
    variance_factor = 0.5  # Tighten to 0.5x (half the base stop-loss)
Else:
    # Linear interpolation between 0.001 and 0.1
    variance_factor = 1.5 - (1.5 - 0.5) * normalized_std

# Apply sensitivity
adjustment = 1.0 + (variance_factor - 1.0) * sensitivity
adjusted_stop_loss = base_stop_loss * adjustment

# Clamp to reasonable range (100-1000)
final_stop_loss = clamp(adjusted_stop_loss, 100, 1000)
```

### 3. **Sensitivity Parameter**

The `variance_risk_sensitivity` parameter (0.0-1.0) controls how much variance affects risk:
- **0.0**: No adjustment (fixed stop-loss)
- **0.5**: Moderate adjustment (default)
- **1.0**: Full adjustment (maximum variance-based risk control)

## Usage

### Enable Variance-Adaptive Risk

```bash
# Default sensitivity (0.5)
python futures_renforcement.py --variance-adaptive-risk

# Custom sensitivity (0.7 = more aggressive adjustment)
python futures_renforcement.py --variance-adaptive-risk 0.7

# Combine with other options
python futures_renforcement.py --variance-adaptive-risk --large-network
```

### Example Output

When enabled, you'll see:
```
🛡️ VARIANCE-ADAPTIVE RISK CONTROL ENABLED
   Stop-loss will adjust based on predicted variance (uncertainty)
   Sensitivity: 0.50 (0.0=off, 1.0=full adjustment)
   High variance → Tighter stop-loss (more conservative)
   Low variance → Looser stop-loss (more aggressive)
```

## Benefits

### 1. **Adaptive Risk Management**
- Automatically adjusts to market conditions
- More conservative when predictions are uncertain
- More aggressive when predictions are confident

### 2. **Reduced Variance**
- Should help reduce P&L variance by being more conservative in uncertain times
- Better risk-adjusted returns

### 3. **Leverages Model Uncertainty**
- Uses the model's own confidence estimates
- No additional computation needed (variance already predicted)

## Example Scenarios

### Scenario 1: High Uncertainty (High Variance)
- **Predicted std**: 0.08 (high uncertainty)
- **Base stop-loss**: $500
- **Adjusted stop-loss**: ~$300 (tighter, more conservative)
- **Result**: Position closed sooner if it moves against you

### Scenario 2: Low Uncertainty (Low Variance)
- **Predicted std**: 0.002 (high confidence)
- **Base stop-loss**: $500
- **Adjusted stop-loss**: ~$750 (looser, more aggressive)
- **Result**: Allow larger moves before closing position

## Configuration

### Parameters

```python
SPXTradingEnv(
    ...
    stop_loss_per_contract=500.0,  # Base stop-loss
    variance_adaptive_risk=True,    # Enable adaptive risk
    variance_risk_sensitivity=0.5   # Adjustment strength (0.0-1.0)
)
```

### Tuning Sensitivity

- **Low sensitivity (0.2-0.3)**: Subtle adjustments, more stable
- **Medium sensitivity (0.5)**: Balanced (default)
- **High sensitivity (0.7-1.0)**: Aggressive adjustments, more responsive

## Monitoring

The current adaptive stop-loss is stored in:
- `env.current_stop_loss_per_contract` - Current effective stop-loss
- `env.current_pred_std_high` - Current high variance prediction
- `env.current_pred_std_low` - Current low variance prediction

You can log these during training to see how stop-loss adjusts.

## When to Use

### ✅ Use Variance-Adaptive Risk When:
- Model provides reliable variance estimates
- High P&L variance is a problem
- You want more conservative risk management in uncertain times
- You want to leverage model confidence

### ❌ Don't Use When:
- Model variance predictions are unreliable
- You need fixed, predictable stop-loss for evaluation
- Testing stop-loss optimization (use fixed values)

## Implementation Details

### State Vector
The variance predictions are already in the state:
```
state = [
    ...features...,           # 42 features
    position,                 # 1 value
    pred_mean_high,          # 1 value
    pred_std_high,           # 1 value ← Used for risk control
    pred_mean_low,           # 1 value
    pred_std_low,            # 1 value ← Used for risk control
    stop_loss_normalized     # 1 value (updated based on variance)
]
```

### Stop-Loss Check
The stop-loss check uses the adaptive value:
```python
effective_stop_loss = env.current_stop_loss_per_contract  # Updated each step
if unrealized_loss_per_contract > effective_stop_loss:
    # Trigger stop-loss
```

## Expected Impact

### On Performance:
- **Variance**: Should decrease (more conservative in uncertain times)
- **Average P&L**: May increase or decrease (depends on market conditions)
- **Win Rate**: May improve (better risk management)
- **Max Drawdown**: Should decrease (tighter stops in uncertain times)

### On Training:
- Agent learns to adapt to changing risk levels
- State includes current stop-loss, so agent knows the risk level
- May need retraining to fully leverage adaptive risk

## Testing

To test variance-adaptive risk:

1. **Compare with fixed stop-loss:**
   ```bash
   # Fixed stop-loss
   python futures_renforcement.py --evaluate 50
   
   # Adaptive stop-loss
   python futures_renforcement.py --variance-adaptive-risk --evaluate 50
   ```

2. **Monitor stop-loss adjustments:**
   - Check logs for stop-loss trigger events
   - Compare effective stop-loss values during high/low variance periods

3. **Tune sensitivity:**
   - Start with 0.5 (default)
   - Increase if you want more aggressive adjustment
   - Decrease if adjustments are too volatile

## Future Enhancements

Potential improvements:
- Position sizing based on variance (smaller positions in uncertain times)
- Dynamic transaction cost adjustment
- Multi-factor risk adjustment (variance + volatility + drawdown)
- Learning optimal sensitivity through hyperparameter optimization


