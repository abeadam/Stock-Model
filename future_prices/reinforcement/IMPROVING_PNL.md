# Improving Negative P&L: Strategies and Fixes

## Problem
P&L is still negative despite loss decreasing and P&L improving. This document outlines fixes implemented and additional strategies.

## Fixes Implemented

### 1. **Reward Shaping for Profitable Trading**
Added three key reward modifications:

#### Profit Bonus (5% of profit)
- **When**: Closing a profitable position
- **Effect**: Encourages agent to take profits on winning trades
- **Formula**: `profit_bonus = 0.05 * profit_amount / 100.0`

#### Hold Bonus (1% of unrealized profit)
- **When**: Holding a profitable position
- **Effect**: Encourages agent to let winners run
- **Formula**: `hold_bonus = 0.01 * unrealized_pnl / 100.0`

#### Overtrading Penalty
- **When**: Trading within 5 steps of previous trade
- **Effect**: Discourages excessive trading (reduces transaction costs)
- **Formula**: `penalty = -0.5 * (5 - steps_since_last_trade) / 5.0`

### 2. **Learning Rate Scheduler (PPO)**
- Added `StepLR` scheduler to PPO
- Decays LR by 2% every 20 updates
- Helps stabilize training and improve convergence

## Additional Strategies to Try

### Strategy 1: Increase Profit Bonus Weight
If agent is still not taking profits, increase the profit bonus:

```python
# In futures_reinforcement_utils.py, line ~560
profit_bonus = 0.10 * profit_amount / 100.0  # Increase from 0.05 to 0.10
```

### Strategy 2: Adjust Transaction Cost Penalty
If overtrading is still an issue, increase the penalty:

```python
# In futures_reinforcement_utils.py, line ~580
if steps_since_last_trade < 10:  # Increase from 5 to 10
    overtrading_penalty = -1.0 * (10 - steps_since_last_trade) / 10.0  # Increase penalty
```

### Strategy 3: Increase Unrealized P&L Weight
If agent needs more immediate feedback on profitable positions:

```python
# In futures_reinforcement_utils.py, line ~590
reward += unrealized_pnl * 0.2  # Increase from 0.1 to 0.2
```

### Strategy 4: Adjust Stop-Loss
If stop-loss is too tight, it might be cutting winners short:

```python
# When creating environment
env = SPXTradingEnv(
    'es_with_indicators.csv',
    'futures_model.pt',
    stop_loss_per_contract=750.0  # Increase from 500.0
)
```

### Strategy 5: Reduce Transaction Costs (for testing)
To see if transaction costs are the main issue:

```python
# In SPXTradingEnv.__init__, line ~142
self.transaction_cost_per_contract = 1.0  # Reduce from 2.5 to 1.0 for testing
```

### Strategy 6: Add Win Rate Bonus
Encourage higher win rate:

```python
# Track win rate and add bonus
if profitable_trade:
    win_rate_bonus = 0.1  # Small bonus for each win
    reward += win_rate_bonus
```

### Strategy 7: Adjust Network Architecture
If agent is underfitting, try larger network:

```python
# When creating agent
agent = PPOAgent(
    state_size=state_size,
    action_size=5,
    network_size='large'  # Use 'large' instead of 'medium'
)
```

### Strategy 8: Increase Training Episodes
Agent might need more training to learn profitable patterns:

```python
train_agent(env, agent, num_episodes=3000)  # Increase from 2000
```

### Strategy 9: Fine-Tune Learning Rate
If learning is too slow or unstable:

```python
# For PPO
agent = PPOAgent(..., lr=5e-4)  # Increase from 3e-4

# For DQN
agent = DQNAgent(..., lr=0.0015)  # Increase from 0.001
```

### Strategy 10: Add Position Sizing
Encourage smaller positions initially:

```python
# Modify action space to prefer smaller positions
# Or add penalty for large positions
if abs(self.position) > 1:
    position_penalty = -0.01 * (abs(self.position) - 1)
    reward += position_penalty
```

## Monitoring Progress

### Key Metrics to Watch:
1. **Win Rate**: Should increase over time
2. **Average Profit per Trade**: Should be positive
3. **Transaction Costs vs. Gross P&L**: Transaction costs should be < 20% of gross P&L
4. **Hold Time**: Profitable positions should be held longer
5. **Trade Frequency**: Should decrease (fewer, better trades)

### When to Adjust:
- **If win rate < 40%**: Increase profit bonus, adjust stop-loss
- **If transaction costs > 30% of gross P&L**: Increase overtrading penalty
- **If average hold time < 10 steps**: Increase hold bonus
- **If P&L still negative after 500 episodes**: Try larger network or more training

## Expected Timeline

- **Episodes 0-200**: Exploration phase, P&L may be negative
- **Episodes 200-500**: Learning phase, P&L should start improving
- **Episodes 500-1000**: Refinement phase, P&L should be positive
- **Episodes 1000+**: Fine-tuning, optimizing win rate and profit per trade

## Quick Fixes Summary

1. ✅ **Implemented**: Profit bonus, hold bonus, overtrading penalty
2. 🔄 **Try Next**: Increase profit bonus weight (0.05 → 0.10)
3. 🔄 **Try Next**: Increase unrealized P&L weight (0.1 → 0.2)
4. 🔄 **Try Next**: Adjust stop-loss (500 → 750)
5. 🔄 **Try Next**: More training episodes (2000 → 3000)

## Notes

- Reward shaping is a powerful tool but needs careful tuning
- Too much reward shaping can destabilize training
- Monitor loss - if it increases, reduce reward shaping weights
- P&L improvement is gradual - be patient with training

