# Reinforcement Learning System Architecture

## Overview Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                        │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   Episode    │
    │   Starts     │
    └──────┬───────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SPXTradingEnv (Environment)                       │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │ State Vector (48 dimensions):                                │  │
    │  │  • 42 normalized market features (OHLCV, indicators)          │  │
    │  │  • Current position (-2 to +2)                               │  │
    │  │  • Model predictions: mean_high, std_high, mean_low, std_low │  │
    │  │  • Stop-loss threshold (normalized)                          │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │ Market Data:                                                  │  │
    │  │  • Historical OHLCV data                                     │  │
    │  │  • Technical indicators                                       │  │
    │  │  • Pre-trained price prediction model                        │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ state (48-dim vector)
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        DQNAgent                                      │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │ Epsilon-Greedy Policy:                                       │  │
    │  │  • ε% chance: Random action (exploration)                    │  │
    │  │  • (1-ε)% chance: Greedy action (exploitation)              │  │
    │  │  • ε decays: 1.0 → 0.01 over training                       │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │                            │                                         │
    │                            │ (if exploitation)                        │
    │                            ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │ Main Q-Network (Dueling DQN):                                │  │
    │  │                                                               │  │
    │  │  State (48) ──► [Feature Layers] ──► [Split]                │  │
    │  │                 128 → 128                                     │  │
    │  │                    │                                            │  │
    │  │        ┌──────────┴──────────┐                                 │  │
    │  │        ▼                     ▼                                 │  │
    │  │  [Value Stream]      [Advantage Stream]                       │  │
    │  │  128 → 64 → 1        128 → 64 → 5                            │  │
    │  │  V(s)                 A(s,a)                                  │  │
    │  │        │                     │                                │  │
    │  │        └──────────┬───────────┘                                │  │
    │  │                  ▼                                            │  │
    │  │         Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))             │  │
    │  │                  │                                            │  │
    │  │                  ▼                                            │  │
    │  │         Q-values for 5 actions:                               │  │
    │  │         [Hold, Buy+1, Buy+2, Sell+1, Sell+2]                  │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    │                            │                                         │
    │                            │ argmax(Q-values)                       │
    │                            ▼                                         │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ action (0-4)
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    SPXTradingEnv.step()                             │
    │                                                                      │
    │  1. Execute action: Change position by -2, -1, 0, +1, or +2        │
    │  2. Calculate P&L:                                                  │
    │     • Realized P&L: From closed positions                           │
    │     • Unrealized P&L: Mark-to-market of open positions              │
    │  3. Apply transaction costs: $2.50 per contract                     │
    │  4. Check stop-loss: Close position if loss > $500/contract         │
    │  5. Calculate reward:                                              │
    │     reward = (realized_P&L + unrealized_P&L × 0.1) / 100           │
    │     reward = clip(reward, -5.0, 5.0)                               │
    │  6. Risk penalties:                                                 │
    │     • Drawdown penalty: -0.001 × (drawdown/100)²                   │
    │     • Large loss penalty: -0.01 × |P&L|/100 (if P&L < -$1000)      │
    │                                                                      │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ (next_state, reward, done, info)
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Experience Storage                                │
    │                                                                      │
    │  Replay Buffer (max 10,000 experiences):                            │
    │  • (state, action, reward, next_state, done)                       │
    │  • FIFO queue - oldest experiences removed when full                │
    │                                                                      │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ (every step, if memory > 1000)
                            ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Training (Replay)                                  │
    │                                                                      │
    │  1. Sample batch (64 experiences) from replay buffer                │
    │  2. Normalize rewards:                                              │
    │     • Update running mean/var: μ, σ²                                 │
    │     • normalized = (reward - μ) / σ                                 │
    │     • clip to [-10, 10]                                             │
    │  3. Compute Q-targets (Double DQN):                                 │
    │     • Select action: main_network(next_state).argmax()              │
    │     • Evaluate: target_network(next_state)[selected_action]          │
    │     • target = normalized_reward + γ × next_Q × (1 - done)         │
    │     • clip target to [-100, 100]                                    │
    │  4. Compute current Q-values:                                        │
    │     • current_Q = main_network(state)[action]                       │
    │  5. Compute loss:                                                    │
    │     • loss = SmoothL1Loss(current_Q, target_Q)                       │
    │  6. Backpropagation:                                                 │
    │     • clip gradients (max norm: 0.5)                                │
    │     • optimizer.step() (Adam)                                       │
    │     • scheduler.step() (LR decay every 100k steps)                  │
    │  7. Soft update target network:                                      │
    │     • target = τ × main + (1-τ) × target                            │
    │     • τ = 0.001 (0.1% update per step)                              │
    │                                                                      │
    └───────────────────────┬─────────────────────────────────────────────┘
                            │
                            │ (if done or max_steps reached)
                            ▼
                    ┌───────────────┐
                    │ Episode Ends  │
                    └───────┬───────┘
                            │
                            │ (if more episodes)
                            ▼
                    [Loop continues...]
```

## Component Details

### 1. Environment (SPXTradingEnv)

```
┌─────────────────────────────────────────────────────────────┐
│                    SPXTradingEnv                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs:                                                    │
│  • Market data (OHLCV + indicators)                        │
│  • Pre-trained price prediction model                      │
│  • Configuration:                                           │
│    - max_steps_per_episode: 5000                           │
│    - max_loss_per_episode: -$50,000                        │
│    - stop_loss_per_contract: $500                          │
│    - transaction_cost: $2.50 per contract                 │
│                                                             │
│  State Components (48 dimensions):                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Market Features (42):                            │   │
│  │    • OHLCV normalized                               │   │
│  │    • Technical indicators (RSI, MACD, etc.)        │   │
│  │    • Volume indicators                              │   │
│  │    • All normalized to [0, 1] or [-1, 1]           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. Position (1):                                    │   │
│  │    • Current position: -2 to +2                     │   │
│  │    • Normalized to [-1, 1]                           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. Model Predictions (4):                           │   │
│  │    • mean_high: Expected high price movement         │   │
│  │    • std_high: Uncertainty in high prediction        │   │
│  │    • mean_low: Expected low price movement            │   │
│  │    • std_low: Uncertainty in low prediction           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 4. Stop-Loss (1):                                   │   │
│  │    • Current stop_loss_per_contract (normalized)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Actions (5):                                              │
│  • 0: Hold (no change)                                     │
│  • 1: Buy 1 contract                                        │
│  • 2: Buy 2 contracts                                       │
│  • 3: Sell 1 contract                                      │
│  • 4: Sell 2 contracts                                      │
│                                                             │
│  Reward Calculation:                                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ reward = 0                                            │  │
│  │ reward += realized_P&L                                │  │
│  │ reward += unrealized_P&L × 0.1                       │  │
│  │ reward -= transaction_costs                          │  │
│  │ reward -= drawdown_penalty (if drawdown > 0)         │  │
│  │ reward -= large_loss_penalty (if P&L < -$1000)       │  │
│  │ reward = reward / 100.0  # Scale down                │  │
│  │ reward = clip(reward, -5.0, 5.0)                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Risk Management:                                           │
│  • Stop-loss: Auto-close if loss > $500/contract           │
│  • Early termination: If P&L < -$50,000                    │
│  • Max steps: Episode ends at 5000 steps                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Agent (DQNAgent)

```
┌─────────────────────────────────────────────────────────────┐
│                    DQNAgent                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Architecture: Dueling DQN                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │  Input: State (48)                                │   │
│  │    │                                               │   │
│  │    ▼                                               │   │
│  │  [Linear(48 → 128) + ReLU]                        │   │
│  │    │                                               │   │
│  │    ▼                                               │   │
│  │  [Linear(128 → 128) + ReLU]                       │   │
│  │    │                                               │   │
│  │    ├──────────────────┬──────────────────┐         │   │
│  │    ▼                  ▼                  ▼         │   │
│  │  [Value]         [Advantage]                       │   │
│  │  Stream          Stream                            │   │
│  │    │                  │                            │   │
│  │  [64] + ReLU    [64] + ReLU                        │   │
│  │    │                  │                            │   │
│  │  [1]            [5]                                │   │
│  │  V(s)           A(s,a)                            │   │
│  │    │                  │                            │   │
│  │    └──────────┬───────┘                            │   │
│  │               ▼                                     │   │
│  │  Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))          │   │
│  │               │                                     │   │
│  │               ▼                                     │   │
│  │  Output: Q-values for 5 actions                   │   │
│  │                                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Two Networks:                                              │
│  • Main Q-Network: Used for action selection & training    │
│  • Target Q-Network: Used for stable Q-targets             │
│    - Soft updated: target = 0.001×main + 0.999×target        │
│                                                             │
│  Hyperparameters:                                           │
│  • Learning rate: 0.001 (decays with scheduler)            │
│  • Discount factor (γ): 0.99                               │
│  • Epsilon: 1.0 → 0.01 (decay: 0.995)                       │
│  • Batch size: 64                                           │
│  • Replay buffer: 10,000                                    │
│  • Tau (soft update): 0.001                                 │
│  • Min LR: 5e-5                                             │
│                                                             │
│  Training Features:                                          │
│  • Double DQN: Reduces overestimation bias                  │
│  • Dueling DQN: Separates value & advantage                │
│  • Reward normalization: Running mean/std                    │
│  • Gradient clipping: Max norm 0.5                           │
│  • Huber loss: Less sensitive to outliers                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Process                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each episode:                                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Reset environment                                  │   │
│  │    state = env.reset()                               │   │
│  │                                                       │   │
│  │ 2. For each step in episode (max 5000):              │   │
│  │    ┌─────────────────────────────────────────────┐   │   │
│  │    │ a. Agent selects action:                    │   │   │
│  │    │    action = agent.act(state, training=True) │   │   │
│  │    │                                             │   │   │
│  │    │ b. Environment executes action:            │   │   │
│  │    │    next_state, reward, done, info =         │   │   │
│  │    │        env.step(action)                     │   │   │
│  │    │                                             │   │   │
│  │    │ c. Store experience:                        │   │   │
│  │    │    agent.memory.append(                     │   │   │
│  │    │        (state, action, reward,              │   │   │
│  │    │         next_state, done))                  │   │   │
│  │    │                                             │   │   │
│  │    │ d. Train agent (if memory > 1000):         │   │   │
│  │    │    loss = agent.replay()                    │   │   │
│  │    │                                             │   │   │
│  │    │ e. Update state:                            │   │   │
│  │    │    state = next_state                       │   │   │
│  │    │                                             │   │   │
│  │    │ f. Check termination:                      │   │   │
│  │    │    if done: break                           │   │   │
│  │    └─────────────────────────────────────────────┘   │   │
│  │                                                       │   │
│  │ 3. Log episode statistics:                           │   │
│  │    • Total reward                                   │   │
│  │    • Total P&L                                      │   │
│  │    • Average loss                                   │   │
│  │    • Steps taken                                    │   │
│  │                                                       │   │
│  │ 4. Save checkpoint (every 100 episodes):            │   │
│  │    agent.save(f"rl_agent_ep{episode}.pt")          │   │
│  │                                                       │   │
│  │ 5. Check early stopping:                            │   │
│  │    if loss hasn't improved for 400 episodes:        │   │
│  │        stop training                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Fine-tuning Mode (episode 400+):                            │
│  • Learning rate: Reduced by 50%                            │
│  • Epsilon: Reduced to 0.001 (minimal exploration)          │
│  • Early stopping patience: 400 episodes                    │
│  • Focus: Variance reduction                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Key Algorithms

#### Double DQN (Action Selection & Evaluation)
```
Standard DQN:
  target_Q = reward + γ × max(target_network(next_state))

Double DQN:
  selected_action = argmax(main_network(next_state))
  target_Q = reward + γ × target_network(next_state)[selected_action]
  
Benefit: Reduces overestimation bias by decoupling action selection 
         from Q-value evaluation
```

#### Dueling DQN (Architecture)
```
Standard DQN:
  Q(s,a) = f(s,a)  [single stream]

Dueling DQN:
  Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
  
  Where:
  • V(s): State value (how good is this state?)
  • A(s,a): Action advantage (how much better is this action?)
  
Benefit: Agent learns state values independently from action advantages,
         leading to better value estimation
```

#### Soft Target Update (Polyak Averaging)
```
Hard Update (standard):
  target_network = main_network  [every N steps]

Soft Update (this system):
  target_network = τ × main_network + (1-τ) × target_network  [every step]
  
  Where τ = 0.001 (0.1% update per step)
  
Benefit: More stable training, smoother target Q-values
```

## Data Flow Summary

```
Market Data → Feature Extraction → State Vector (48-dim)
                                      │
                                      ▼
                            Dueling DQN Network
                                      │
                                      ▼
                            Q-values (5 actions)
                                      │
                                      ▼
                            Epsilon-Greedy Policy
                                      │
                                      ▼
                            Action Selection
                                      │
                                      ▼
                            Environment Execution
                                      │
                                      ▼
                            Reward Calculation
                                      │
                                      ▼
                            Experience Storage
                                      │
                                      ▼
                            Batch Training
                                      │
                                      ▼
                            Network Update
                                      │
                                      ▼
                            Target Network Update (soft)
```

## Performance Metrics Tracked

- **P&L Metrics:**
  - Realized P&L (closed positions)
  - Unrealized P&L (open positions)
  - Total P&L
  - Peak P&L
  - Drawdown

- **Trading Metrics:**
  - Win rate
  - Profit factor
  - Sharpe ratio
  - Average P&L per episode
  - P&L variance (stability)

- **Learning Metrics:**
  - Training loss
  - Q-value statistics
  - Epsilon (exploration rate)
  - Learning rate
  - Reward normalization stats

## Current Configuration

- **Stop-loss:** $500 per contract (optimized)
- **Max episode steps:** 5000
- **Max loss per episode:** -$50,000
- **Transaction cost:** $2.50 per contract
- **Reward scaling:** ÷100, clipped to [-5, 5]
- **Unrealized P&L weight:** 0.1
- **Fine-tuning mode:** Enabled (episode 400+)
  - LR reduced by 50%
  - Epsilon: 0.001
  - Early stopping patience: 400 episodes

