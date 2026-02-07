# How Variance-Adaptive Risk and Learnable Risk Work Together

## Overview

When both `--variance-adaptive-risk` and `--learnable-risk` are enabled, they work together in a **multiplicative** fashion to provide two layers of risk adjustment.

## Interaction Flow

### Step-by-Step Process

```
1. _get_state() is called:
   ├─ Model predicts variance (pred_std_high, pred_std_low)
   ├─ Variance-adaptive calculates adjustment factor (0.5-1.5)
   └─ Sets: current_stop_loss = base_stop_loss × variance_factor
   
2. Agent.act() is called:
   ├─ Network outputs: (Q-values, learned_risk_multiplier)
   └─ Returns: (action, learned_risk_multiplier)
   
3. env.step(action, risk_multiplier) is called:
   ├─ Learned risk multiplier further adjusts stop-loss
   └─ Final: current_stop_loss = (base × variance_factor) × learned_multiplier
   
4. Stop-loss check uses final adjusted value
```

## Mathematical Formula

When both are enabled:

```
Final Stop-Loss = Base Stop-Loss × Variance Factor × Learned Multiplier

Where:
- Base Stop-Loss: $500 (default)
- Variance Factor: 0.5-1.5 (from variance-adaptive, based on pred_std)
- Learned Multiplier: 0.5-2.0 (from agent network, learned from data)
```

### Example Calculations

#### Example 1: High Variance + Conservative Learned Risk
- **Base stop-loss**: $500
- **Predicted std**: 0.08 (high uncertainty)
- **Variance factor**: 0.5 (tighter due to uncertainty)
- **Learned multiplier**: 0.6 (agent learned to be conservative)
- **Final stop-loss**: $500 × 0.5 × 0.6 = **$150** (very tight)

#### Example 2: Low Variance + Aggressive Learned Risk
- **Base stop-loss**: $500
- **Predicted std**: 0.002 (high confidence)
- **Variance factor**: 1.5 (looser due to confidence)
- **Learned multiplier**: 1.8 (agent learned to be aggressive)
- **Final stop-loss**: $500 × 1.5 × 1.8 = **$1,350** (very loose, clamped to $1,000)

#### Example 3: Medium Variance + Neutral Learned Risk
- **Base stop-loss**: $500
- **Predicted std**: 0.05 (medium uncertainty)
- **Variance factor**: 0.75 (slightly tighter)
- **Learned multiplier**: 1.0 (agent learned neutral risk)
- **Final stop-loss**: $500 × 0.75 × 1.0 = **$375** (moderately tight)

## Why This Combination Works Well

### 1. **Two-Layer Risk Management**
- **Variance-adaptive**: Quick, heuristic adjustment based on model confidence
- **Learnable**: Fine-tuned adjustment based on actual trading outcomes

### 2. **Complementary Information**
- **Variance-adaptive**: Uses model's uncertainty estimate (forward-looking)
- **Learnable**: Uses learned patterns from historical outcomes (backward-looking)

### 3. **Adaptive to Both Sources**
- **Model uncertainty**: Variance-adaptive responds immediately
- **Market patterns**: Learnable risk adapts to what actually works

## Benefits of Combining

### 1. **More Sophisticated Risk Control**
- Variance-adaptive provides initial adjustment
- Learnable risk fine-tunes based on outcomes
- Both adjustments are context-aware

### 2. **Better Risk-Return Tradeoff**
- Variance-adaptive prevents large losses in uncertain times
- Learnable risk optimizes for long-term profitability
- Combined: Conservative when needed, aggressive when safe

### 3. **Reduced Variance**
- Two layers of risk adjustment
- More stable P&L
- Better risk-adjusted returns

## How the Agent Learns

### State Information

The agent sees:
- **Variance predictions**: `pred_std_high`, `pred_std_low` (uncertainty)
- **Current stop-loss**: Normalized value (includes variance adjustment)
- **All market features**: Price, indicators, etc.

### Learning Process

1. **Agent observes**:
   - High variance → variance-adaptive tightens stop-loss
   - Agent sees tight stop-loss in state
   - Agent learns: "In high variance states, I should also be conservative"

2. **Agent learns**:
   - If variance-adaptive is too conservative → agent learns to loosen (multiplier > 1.0)
   - If variance-adaptive is too aggressive → agent learns to tighten (multiplier < 1.0)
   - Agent learns optimal multiplier for each state

3. **Result**:
   - Variance-adaptive provides base adjustment
   - Learnable risk fine-tunes the adjustment
   - Combined: Optimal risk level for each situation

## Example Scenarios

### Scenario 1: High Variance, Agent Learns to Be Even More Conservative
- **Variance-adaptive**: Tightens to 0.5× (due to high uncertainty)
- **Agent learns**: "Even tighter is better" → outputs 0.6× multiplier
- **Final**: 0.5 × 0.6 = 0.3× base stop-loss (very conservative)

### Scenario 2: Low Variance, Agent Learns Variance-Adaptive Is Too Aggressive
- **Variance-adaptive**: Loosens to 1.5× (due to high confidence)
- **Agent learns**: "This is too loose, I've lost money" → outputs 0.8× multiplier
- **Final**: 1.5 × 0.8 = 1.2× base stop-loss (moderately loose)

### Scenario 3: Medium Variance, Agent Learns Optimal Level
- **Variance-adaptive**: Adjusts to 1.0× (neutral)
- **Agent learns**: "Slightly tighter works better" → outputs 0.9× multiplier
- **Final**: 1.0 × 0.9 = 0.9× base stop-loss (slightly conservative)

## Implementation Details

### Code Flow

```python
# In _get_state() - called before each step
if variance_adaptive_risk:
    variance_factor = calculate_from_predicted_variance()
    current_stop_loss = base_stop_loss * variance_factor

# In step() - called with action
if risk_multiplier is not None:  # From learnable risk
    # Multiply on top of variance-adjusted stop-loss
    current_stop_loss = current_stop_loss * risk_multiplier
    # Final stop-loss used for risk checks
```

### Key Points

1. **Order matters**: Variance-adaptive runs first (in `_get_state()`), learnable runs second (in `step()`)
2. **Multiplicative**: Both adjustments multiply together
3. **Clamped**: Final value clamped to [100, 1000] range
4. **State includes both**: Agent sees variance predictions AND current stop-loss

## When to Use Both

### ✅ Use Both When:
- You want maximum risk control sophistication
- Model variance predictions are reliable
- You have enough training data for learnable risk
- You want both heuristic and learned risk management

### ⚠️ Use One or the Other When:
- Limited training data (use variance-adaptive only)
- Need interpretable risk rules (use variance-adaptive only)
- Want pure data-driven approach (use learnable only)
- Testing specific risk strategies (use one at a time)

## Expected Behavior

### Training Phase

1. **Early Training**:
   - Variance-adaptive provides base adjustments
   - Learnable risk explores (random multipliers)
   - Combined adjustments vary widely

2. **Mid Training**:
   - Variance-adaptive continues providing base adjustments
   - Learnable risk starts converging
   - Combined adjustments become more consistent

3. **Late Training**:
   - Variance-adaptive provides context-aware base
   - Learnable risk fine-tunes optimally
   - Combined: Optimal risk for each situation

### Performance

- **Variance**: Should be lowest (two layers of risk control)
- **Average P&L**: Should improve (optimal risk-return tradeoff)
- **Win Rate**: Should improve (better stop-loss timing)
- **Max Drawdown**: Should decrease (conservative when needed)

## Monitoring

To monitor both working together:

1. **Check variance-adaptive adjustment**:
   ```python
   variance_factor = env.current_stop_loss_per_contract / env.base_stop_loss_per_contract
   # Before learnable risk is applied
   ```

2. **Check learned risk multiplier**:
   - Log the risk_multiplier output from agent
   - Should correlate with variance but also learn patterns

3. **Check final stop-loss**:
   ```python
   final_stop_loss = env.current_stop_loss_per_contract
   # After both adjustments
   ```

4. **Compare scenarios**:
   - High variance + learned multiplier
   - Low variance + learned multiplier
   - Should see different patterns

## Summary

**Variance-Adaptive Risk** (heuristic):
- Quick adjustment based on model uncertainty
- Provides base risk level
- Runs in `_get_state()`

**Learnable Risk** (data-driven):
- Fine-tunes risk based on outcomes
- Multiplies on top of variance adjustment
- Runs in `step()`

**Combined**:
- Final stop-loss = base × variance_factor × learned_multiplier
- Two layers of risk control
- More sophisticated and adaptive
- Better risk-return tradeoff


