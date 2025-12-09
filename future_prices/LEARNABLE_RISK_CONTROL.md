# Learnable Risk Control

## Overview

The system now supports **learnable risk control**, where the agent learns the optimal risk level (stop-loss multiplier) directly from trading data, rather than using fixed or heuristically-adjusted values.

## How It Works

### Architecture

The Dueling DQN network is extended with a **risk control head** that outputs a risk multiplier:

```
Input: State (48 dimensions)
  │
  ▼
[Feature Layers] (shared)
  │
  ├──► [Value Stream] ──► V(s)
  ├──► [Advantage Stream] ──► A(s,a)
  └──► [Risk Stream] ──► Risk Multiplier (0.5-2.0)
  │
  ▼
Output: (Q-values, Risk Multiplier)
```

### Risk Multiplier

- **Range**: 0.5 to 2.0
- **0.5**: Tighter stop-loss (more conservative) - 50% of base stop-loss
- **1.0**: Base stop-loss (no adjustment)
- **2.0**: Looser stop-loss (more aggressive) - 200% of base stop-loss

### Learning Mechanism

The agent learns the risk multiplier through **implicit learning**:

1. **Network outputs** both trading action AND risk multiplier
2. **Risk multiplier** adjusts the stop-loss threshold
3. **Stop-loss** affects P&L (tighter = less loss but may exit too early)
4. **P&L** affects reward
5. **Reward** trains the Q-network (including risk head)

The risk head is trained through backpropagation - gradients flow from the Q-loss through the shared feature layers to the risk head. The agent learns that certain risk levels lead to better Q-values (better long-term returns).

## Usage

### Enable Learnable Risk Control

```bash
# Train with learnable risk control
python futures_renforcement.py --learnable-risk

# Combine with other options
python futures_renforcement.py --learnable-risk --large-network
python futures_renforcement.py --learnable-risk --variance-adaptive-risk
```

### Example Output

When enabled, you'll see:
```
🎯 LEARNABLE RISK CONTROL ENABLED
   Agent will learn optimal risk level (stop-loss multiplier) from data
   Network outputs both trading action AND risk multiplier (0.5-2.0)
   Risk multiplier adjusts stop-loss: 0.5=tighter (conservative), 2.0=looser (aggressive)
   Agent learns through reward signal - better risk = better P&L
```

## Benefits

### 1. **Data-Driven Risk Management**
- Learns optimal risk level from actual trading outcomes
- Adapts to market conditions automatically
- No manual tuning of risk parameters

### 2. **Context-Aware Risk**
- Risk multiplier depends on current state
- Can be conservative in uncertain states, aggressive in confident states
- Learns complex risk-return tradeoffs

### 3. **End-to-End Learning**
- Risk control optimized jointly with trading strategy
- Risk and trading decisions are coordinated
- Better overall performance

## How It Learns

### Training Process

1. **Forward Pass**:
   - Network outputs: `(Q-values, risk_multiplier)`
   - Risk multiplier: `0.5 + 1.5 * sigmoid(output)` → [0.5, 2.0]

2. **Action Selection**:
   - Choose action: `argmax(Q-values)`
   - Use risk multiplier: `adjusted_stop_loss = base_stop_loss * risk_multiplier`

3. **Environment Step**:
   - Execute action with adjusted stop-loss
   - Calculate reward based on P&L

4. **Backward Pass**:
   - Compute Q-loss: `loss = SmoothL1Loss(current_Q, target_Q)`
   - Gradients flow through:
     - Q-values → feature layers
     - Risk head → feature layers (shared)
   - Both heads learn from the same reward signal

### Learning Signal

The risk multiplier learns because:
- **Good risk multiplier** → Better stop-loss timing → Better P&L → Higher reward → Higher Q-values
- **Bad risk multiplier** → Poor stop-loss timing → Worse P&L → Lower reward → Lower Q-values

The network learns to output risk multipliers that maximize Q-values (long-term returns).

## Comparison with Other Approaches

### vs. Fixed Stop-Loss
- **Fixed**: Same stop-loss always (e.g., $500)
- **Learnable**: Adapts stop-loss based on state (e.g., $250-$1000)

### vs. Variance-Adaptive Risk
- **Variance-adaptive**: Heuristic adjustment based on predicted variance
- **Learnable**: Learned adjustment based on actual outcomes

### vs. Both Combined
You can use both:
- **Variance-adaptive**: Provides initial risk adjustment
- **Learnable**: Fine-tunes risk based on outcomes

## Expected Behavior

### During Training

1. **Early Training**:
   - Risk multiplier varies randomly (exploration)
   - Agent tries different risk levels
   - Q-values start to differentiate good vs. bad risk

2. **Mid Training**:
   - Risk multiplier becomes more consistent
   - Agent learns which states need tighter/looser stops
   - Q-values improve as risk strategy improves

3. **Late Training**:
   - Risk multiplier stabilizes
   - Agent has learned optimal risk policy
   - Risk adjustments are context-aware

### Performance Impact

- **Variance**: Should decrease (better risk management)
- **Average P&L**: Should improve (optimal risk-return tradeoff)
- **Win Rate**: May improve (better stop-loss timing)
- **Max Drawdown**: Should decrease (tighter stops when needed)

## Implementation Details

### Network Architecture

```python
class DuelingDQN:
    def __init__(self, ..., learnable_risk=False):
        # Shared feature layers
        self.feature_layers = ...
        
        # Value and advantage streams (for Q-values)
        self.value_stream = ...
        self.advantage_stream = ...
        
        # Risk control head (if enabled)
        if learnable_risk:
            self.risk_stream = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # Output [0, 1]
            )
    
    def forward(self, x):
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        if self.learnable_risk:
            risk_raw = self.risk_stream(features)  # [0, 1]
            risk_multiplier = 0.5 + 1.5 * risk_raw  # [0.5, 2.0]
            return q_values, risk_multiplier
        else:
            return q_values
```

### Action Selection

```python
def act(self, state):
    output = self.q_network(state)
    if self.learnable_risk_control:
        q_values, risk_multiplier = output
        action = q_values.argmax()
        return action, risk_multiplier
    else:
        return output.argmax()
```

### Environment Integration

```python
def step(self, action, risk_multiplier=None):
    if risk_multiplier is not None:
        # Adjust stop-loss based on learned risk multiplier
        adjusted_stop_loss = base_stop_loss * risk_multiplier
        self.current_stop_loss_per_contract = adjusted_stop_loss
    # ... rest of step logic
```

## Training Considerations

### 1. **Exploration**
- During exploration (epsilon > 0), risk multiplier is random (0.5-2.0)
- Allows agent to try different risk levels

### 2. **Learning Rate**
- Risk head shares feature layers with Q-head
- Learning rate affects both heads
- May need tuning if risk head learns too fast/slow

### 3. **Gradient Flow**
- Gradients flow from Q-loss to risk head (through shared features)
- Risk head learns indirectly from Q-learning signal
- This is sufficient - good risk → better Q-values

### 4. **State Representation**
- Current stop-loss is in state (normalized)
- Agent can see current risk level
- Helps agent learn risk adjustments

## Monitoring

To monitor risk learning:

1. **Check risk multiplier distribution**:
   - Should stabilize over training
   - Should vary by state (not constant)

2. **Check stop-loss adjustments**:
   - Monitor `env.current_stop_loss_per_contract`
   - Should adapt to market conditions

3. **Check performance**:
   - Compare P&L with/without learnable risk
   - Should see improved risk-adjusted returns

## When to Use

### ✅ Use Learnable Risk When:
- You want data-driven risk management
- Fixed or heuristic risk control isn't working well
- You have enough training data
- You want end-to-end optimization

### ❌ Don't Use When:
- You need predictable, fixed risk levels
- Training data is limited
- You want to test specific risk strategies
- You need interpretable risk rules

## Future Enhancements

Potential improvements:
- **Separate risk loss**: Add explicit loss term for risk multiplier
- **Risk constraints**: Enforce minimum/maximum risk levels
- **Multi-factor risk**: Learn position sizing, transaction cost tolerance, etc.
- **Risk regularization**: Penalize extreme risk multipliers for stability


