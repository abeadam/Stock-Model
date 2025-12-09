"""
Reinforcement Learning Strategy for SPX Trading using futures_model.pt

This module implements a reinforcement learning environment and agent for trading SPX
using the pre-trained futures_model.pt for price predictions.

Trading Constraints:
- Start with 0 SPX units owned
- Can buy or sell up to 2 SPX units per step
- Position can range from -2 to +2 units
- Decisions based on previous step's all non-forward-looking features
- Transaction cost: $2.5 per futures contract traded
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, List
from collections import deque
import random
from sklearn.preprocessing import StandardScaler

# Import the model class from the futures model file
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from futures_model_MVE_SNNs import MVEModel

# Import PPO agent (optional - only needed if using --ppo flag)
try:
    from futures_renforcement_ppo import PPOAgent
except ImportError:
    PPOAgent = None  # PPO not available


# SPXTradingEnv is now in futures_reinforcement_utils
# Import it after DQNAgent is defined (see below)
    """
    Reinforcement Learning Environment for SPX Trading
    
    State: Previous step's all non-forward-looking features + current position + 
           model predictions (mean_high, std_high, mean_low, std_low)
    Actions: Buy 0, 1, or 2 units OR Sell 0, 1, or 2 units (net change in position)
    Reward: Profit/loss from trading minus transaction costs ($2.5 per contract)
    """
    
    def __init__(self, data_path: str, model_path: str, device: str = 'cpu',
                 max_steps_per_episode: int = 5000, max_loss_per_episode: float = -50000.0,
                 stop_loss_per_contract: float = 500.0, 
                 variance_adaptive_risk: bool = True,
                 variance_risk_sensitivity: float = 0.5):
        """
        Initialize the trading environment
        
        Args:
            data_path: Path to es_with_indicators.csv
            model_path: Path to futures_model.pt
            device: Device to run model on ('cpu' or 'cuda')
            max_steps_per_episode: Maximum steps per episode (default: 5000, was unlimited)
            max_loss_per_episode: Maximum loss before early termination (default: -$50k)
            stop_loss_per_contract: Base stop loss threshold per contract (default: $500)
            variance_adaptive_risk: If True, adjust stop-loss based on predicted variance (default: True)
            variance_risk_sensitivity: How much variance affects risk (0.0-1.0, default: 0.5)
                                    Higher = more aggressive adjustment based on variance
        """
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
        self.max_steps_per_episode = max_steps_per_episode
        self.max_loss_per_episode = max_loss_per_episode
        self.base_stop_loss_per_contract = stop_loss_per_contract
        self.variance_adaptive_risk = variance_adaptive_risk
        self.variance_risk_sensitivity = variance_risk_sensitivity
        
        # Track current variance predictions for adaptive risk
        self.current_pred_std_high = 0.0
        self.current_pred_std_low = 0.0
        self.current_stop_loss_per_contract = stop_loss_per_contract
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in self.df.columns for col in required_cols), \
            f"Missing required columns. Found: {self.df.columns.tolist()}"
        
        # Remove rows with NaN in OHLCV
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)
        print(f"Loaded {len(self.df)} rows (removed {initial_len - len(self.df)} rows with NaN)")
        
        # Load the pre-trained model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Reconstruct model
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.scaler = checkpoint['scaler']
        self.feature_names = checkpoint['feature_names']
        
        # Determine if model predicts both high and low
        # Check if model has quantile heads (predict_both=True)
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        self.predict_both = any('quantile_10_head' in key for key in state_dict_keys)
        
        # Create model instance
        self.model = MVEModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=0.0,
            predict_both=self.predict_both
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print(f"Model compiled for faster inference")
        except Exception as e:
            print(f"Model compilation not available (PyTorch < 2.0 or error: {e}), using standard model")
        
        print(f"Model loaded. predict_both={self.predict_both}, input_dim={self.input_dim}")
        
        # Trading parameters
        self.max_position = 2  # Maximum position (long or short)
        self.min_position = -2  # Minimum position (short)
        self.initial_position = 0
        self.transaction_cost_per_contract = 2.5  # $2.5 per futures contract
        self.trade_penalty_per_contract = 0.0  # Removed - was discouraging all trading
        self.position_penalty_strength = 0.0  # Removed - was penalizing holding positions
        
        # Risk management parameters
        self.stop_loss_per_contract = stop_loss_per_contract  # Stop loss: close position if unrealized loss > threshold per contract
        self.max_drawdown_per_episode = 2000.0  # Maximum drawdown before early termination
        self.peak_pnl = 0.0  # Track peak P&L for drawdown calculation
        
        # Normalize stop_loss for state representation (typical range: 100-1000)
        self.stop_loss_normalized = (self.stop_loss_per_contract - 100.0) / 900.0  # Normalize to [0, 1] for 100-1000 range
        
        # Identify columns that don't depend on future data
        # Exclude forward-looking columns: PctChange_ToMaxHigh_5, PctChange_ToMinLow_5
        # Also exclude Date, DateTime, DateTime_ET (not useful as features)
        forward_looking_cols = ['PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
        exclude_cols = ['Date', 'DateTime', 'DateTime_ET'] + forward_looking_cols
        
        # Get all feature columns (non-forward-looking)
        self.feature_cols = [col for col in self.df.columns 
                            if col not in exclude_cols]
        
        print(f"Using {len(self.feature_cols)} non-forward-looking features for state")
        print(f"Excluded columns: {exclude_cols}")
        
        # Create scaler for all features
        # Remove rows with NaN in feature columns for fitting scaler
        feature_data = self.df[self.feature_cols].copy()
        # Fill NaN with 0 for features (some indicators may have NaN at start)
        feature_data = feature_data.fillna(0)
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(feature_data.values)
        
        print(f"Feature scaler fitted on {len(feature_data)} samples")
        
        # Action space: net change in position from -2 to +2
        # Actions: -2, -1, 0, 1, 2 (sell 2, sell 1, hold, buy 1, buy 2)
        self.action_space_size = 5
        self.action_map = {-2: -2, -1: -1, 0: 0, 1: 1, 2: 2}
        
        # Episode tracking
        self.step_count = 0
        self.episode_start_pnl = 0.0
        
        # Reset environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial state vector
        """
        # Start at a random point in the data (but not too early to have previous step)
        # Ensure we have enough room for max_steps_per_episode
        max_start = max(1, len(self.df) - self.max_steps_per_episode - 2)
        self.current_step = random.randint(1, max_start)
        self.position = self.initial_position
        self.cash = 0.0  # Track cash (negative means we owe money)
        self.total_pnl = 0.0
        self.avg_entry_price = 0.0  # Average entry price for current position
        self.position_value = 0.0  # Value of current position
        self.peak_pnl = 0.0  # Track peak P&L for drawdown calculation
        self.initial_pnl = 0.0  # Track starting P&L
        self.step_count = 0
        self.episode_start_pnl = 0.0
        
        # Get initial state (use step 0 for previous step data)
        state = self._get_state()
        return state
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        State includes:
        - Previous step's all non-forward-looking features (normalized)
        - Current position (normalized)
        - Model predictions: mean_high, std_high, mean_low, std_low
          (if model predicts both, otherwise high and low use same values)
        """
        if self.current_step == 0:
            # Use current step if no previous step available
            prev_idx = 0
        else:
            prev_idx = self.current_step - 1
        
        # Get previous step's data
        prev_row = self.df.iloc[prev_idx]
        
        # Extract all non-forward-looking features
        feature_values = []
        for col in self.feature_cols:
            val = prev_row[col]
            if pd.isna(val):
                feature_values.append(0.0)
            else:
                feature_values.append(float(val))
        
        # Normalize features
        feature_array = np.array(feature_values).reshape(1, -1)
        features_normalized = self.feature_scaler.transform(feature_array)[0]
        
        # Get model prediction using previous step's data
        # Extract features that the model expects (from feature_names)
        model_features = []
        for feat_name in self.feature_names:
            if feat_name in self.df.columns:
                val = prev_row[feat_name]
                if pd.isna(val):
                    model_features.append(0.0)
                else:
                    model_features.append(float(val))
            else:
                model_features.append(0.0)
        
        model_features = np.array(model_features).reshape(1, -1)
        model_features_scaled = self.scaler.transform(model_features)
        
        # Get prediction from model
        with torch.no_grad():
            # Keep tensor on device for faster computation
            features_tensor = torch.FloatTensor(model_features_scaled).to(self.device)
            model_output = self.model(features_tensor)
            
            if self.predict_both:
                (mean_high, mean_low), (log_var_high, log_var_low) = model_output
                # Compute std on GPU (faster), then transfer once
                pred_std_high = (log_var_high * 0.5).exp().cpu().numpy()[0, 0]  # exp(0.5*log_var) = sqrt(exp(log_var))
                pred_std_low = (log_var_low * 0.5).exp().cpu().numpy()[0, 0]
                # Transfer means
                pred_mean_high = mean_high.cpu().numpy()[0, 0]
                pred_mean_low = mean_low.cpu().numpy()[0, 0]
            else:
                mean_pred, log_var_pred = model_output
                # Compute std on GPU (faster)
                pred_std_high = (log_var_pred * 0.5).exp().cpu().numpy()[0, 0]
                pred_mean_high = mean_pred.cpu().numpy()[0, 0]
                # If model doesn't predict both, use same values for low
                pred_mean_low = pred_mean_high
                pred_std_low = pred_std_high
        
        # Store current variance predictions for adaptive risk control
        self.current_pred_std_high = pred_std_high
        self.current_pred_std_low = pred_std_low
        
        # Calculate adaptive stop-loss based on predicted variance
        if self.variance_adaptive_risk:
            # Higher variance (uncertainty) = tighter stop-loss (more conservative)
            # Lower variance (confidence) = looser stop-loss (more aggressive)
            # Use the maximum of high/low variance to be conservative
            max_pred_std = max(abs(pred_std_high), abs(pred_std_low))
            
            # Normalize variance to a scale factor (0.5 to 2.0)
            # Typical std values are in range 0.001-0.1 (percentage changes)
            # Map to risk adjustment: low std (0.001) → 1.5x (looser), high std (0.1) → 0.5x (tighter)
            # Using exponential mapping for smooth adjustment
            if max_pred_std < 0.001:
                variance_factor = 1.5  # Very confident: allow larger losses
            elif max_pred_std > 0.1:
                variance_factor = 0.5  # Very uncertain: tighten stop-loss
            else:
                # Linear interpolation between 0.001 and 0.1
                # At 0.001: factor = 1.5, at 0.1: factor = 0.5
                normalized_std = (max_pred_std - 0.001) / (0.1 - 0.001)
                variance_factor = 1.5 - (1.5 - 0.5) * normalized_std
            
            # Apply sensitivity: 0.0 = no adjustment, 1.0 = full adjustment
            adjustment = 1.0 + (variance_factor - 1.0) * self.variance_risk_sensitivity
            self.current_stop_loss_per_contract = self.base_stop_loss_per_contract * adjustment
            
            # Clamp to reasonable range (100-1000)
            self.current_stop_loss_per_contract = max(100.0, min(1000.0, self.current_stop_loss_per_contract))
        else:
            self.current_stop_loss_per_contract = self.base_stop_loss_per_contract
        
        # Normalize position to [-1, 1]
        position_normalized = self.position / self.max_position
        
        # Update stop_loss_normalized (using current adaptive stop-loss)
        self.stop_loss_normalized = (self.current_stop_loss_per_contract - 100.0) / 900.0  # Normalize to [0, 1] for 100-1000 range
        
        # Combine state: [all_features, position, pred_mean_high, pred_std_high, pred_mean_low, pred_std_low, stop_loss]
        # This gives the agent information about both expected high and low price movements
        # AND the current stop-loss threshold (so it can learn to adapt to different risk levels)
        state = np.concatenate([
            features_normalized,  # All non-forward-looking features
            [position_normalized],  # 1 value
            [pred_mean_high],  # 1 value - expected high price movement
            [pred_std_high],  # 1 value - uncertainty in high prediction
            [pred_mean_low],  # 1 value - expected low price movement
            [pred_std_low],  # 1 value - uncertainty in low prediction
            [self.stop_loss_normalized]  # 1 value - current stop-loss threshold (normalized)
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int, risk_multiplier: Optional[float] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return next state, reward, done, info
        
        Args:
            action: Action index (0=-2, 1=-1, 2=0, 3=1, 4=2)
            risk_multiplier: Optional risk multiplier from agent (0.5-2.0)
                           If provided, adjusts stop-loss: 0.5=tighter, 2.0=looser
        
        Returns:
            next_state: Next state vector
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Apply learned risk multiplier if provided
        # If both variance-adaptive and learnable risk are enabled, they work together:
        # 1. Variance-adaptive provides base adjustment (in _get_state)
        # 2. Learned risk multiplier further adjusts (multiplicative)
        if risk_multiplier is not None:
            # Clamp to reasonable range
            risk_multiplier = max(0.5, min(2.0, risk_multiplier))
            # Apply learned risk multiplier to current stop-loss
            # If variance-adaptive is enabled, this multiplies on top of variance adjustment
            # If not, this multiplies on base stop-loss
            self.current_stop_loss_per_contract = self.current_stop_loss_per_contract * risk_multiplier
            # Clamp to reasonable range (100-1000)
            self.current_stop_loss_per_contract = max(100.0, min(1000.0, self.current_stop_loss_per_contract))
        
        # Map action to position change
        position_change = self.action_map[action - 2]  # action: 0->-2, 1->-1, 2->0, 3->1, 4->2
        
        # Prevent actions that would exceed position limits
        if self.position <= self.min_position and position_change < 0:
            # Already at minimum position; cannot sell more
            position_change = 0
        elif self.position >= self.max_position and position_change > 0:
            # Already at maximum position; cannot buy more
            position_change = 0
        
        # Calculate new position
        new_position = self.position + position_change
        
        # Clamp position to valid range
        new_position = max(self.min_position, min(self.max_position, new_position))
        actual_change = new_position - self.position
        
        # Get current price (use Close price for execution)
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate transaction cost: $2.5 per contract traded
        contracts_traded = abs(actual_change)
        transaction_cost = contracts_traded * self.transaction_cost_per_contract
        
        # Calculate reward (profit/loss)
        reward = 0.0
        old_position = self.position
        
        # Calculate realized P&L from position change
        if old_position != 0 and self.avg_entry_price != 0:
            # Realized P&L when reducing or reversing position
            if (old_position > 0 and actual_change < 0) or (old_position < 0 and actual_change > 0):
                # Closing or reducing position
                closed_units = min(abs(old_position), abs(actual_change))
                if old_position > 0:
                    # Long position: profit when price goes up
                    realized_pnl = (current_price - self.avg_entry_price) * closed_units
                else:
                    # Short position: profit when price goes down
                    realized_pnl = (self.avg_entry_price - current_price) * closed_units
                reward += realized_pnl
                self.total_pnl += realized_pnl
        
        # Apply transaction cost to reward
        reward -= transaction_cost
        self.total_pnl -= transaction_cost
        
        # Update average entry price and position
        if actual_change > 0:
            # Buying: reduce cash, add to position (or reduce short)
            cost = actual_change * current_price
            self.cash -= cost
            
            if old_position == 0:
                # Starting new long position
                self.avg_entry_price = current_price
            elif old_position > 0:
                # Adding to long position - weighted average
                old_value = self.avg_entry_price * old_position
                new_cost = actual_change * current_price
                self.avg_entry_price = (old_value + new_cost) / new_position
            else:
                # Reducing short position (buying back)
                if abs(new_position) < abs(old_position):
                    # Still short, keep entry price
                    pass
                else:
                    # Flipped to long
                    self.avg_entry_price = current_price
        elif actual_change < 0:
            # Selling: increase cash, reduce position (or go short)
            proceeds = abs(actual_change) * current_price
            self.cash += proceeds
            
            if old_position == 0:
                # Starting new short position
                self.avg_entry_price = current_price
            elif old_position < 0:
                # Adding to short position - weighted average
                old_value = self.avg_entry_price * abs(old_position)
                new_cost = abs(actual_change) * current_price
                self.avg_entry_price = (old_value + new_cost) / abs(new_position)
            else:
                # Reducing long position (selling)
                if new_position > 0:
                    # Still long, keep entry price
                    pass
                else:
                    # Flipped to short
                    self.avg_entry_price = current_price
        
        # Update position
        self.position = new_position
        
        # Unrealized P&L (mark-to-market)
        if self.position != 0 and self.avg_entry_price != 0:
            if self.position > 0:
                # Long position
                unrealized_pnl = (current_price - self.avg_entry_price) * self.position
            else:
                # Short position
                unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.position)
        else:
            unrealized_pnl = 0.0
        
        # Calculate total P&L for risk management
        total_pnl_with_unrealized = self.total_pnl + unrealized_pnl
        
        # Update peak P&L for drawdown tracking
        if total_pnl_with_unrealized > self.peak_pnl:
            self.peak_pnl = total_pnl_with_unrealized
        
        # Calculate drawdown (how much we've fallen from peak)
        drawdown = self.peak_pnl - total_pnl_with_unrealized
        
        # RISK MANAGEMENT: Stop-loss mechanism
        # If unrealized loss per contract exceeds threshold, force close position
        # Uses adaptive stop-loss if variance_adaptive_risk is enabled
        stop_loss_triggered = False
        if self.position != 0 and self.avg_entry_price != 0:
            unrealized_loss_per_contract = -unrealized_pnl / abs(self.position) if unrealized_pnl < 0 else 0.0
            # Use current adaptive stop-loss (updated based on variance)
            effective_stop_loss = self.current_stop_loss_per_contract
            if unrealized_loss_per_contract > effective_stop_loss:
                # Force close position at stop-loss
                stop_loss_triggered = True
                if self.position > 0:
                    stop_loss_pnl = (current_price - self.avg_entry_price) * self.position
                else:
                    stop_loss_pnl = (self.avg_entry_price - current_price) * abs(self.position)
                
                # Apply transaction cost for closing
                closing_cost = abs(self.position) * self.transaction_cost_per_contract
                stop_loss_pnl -= closing_cost
                
                reward += stop_loss_pnl
                self.total_pnl += stop_loss_pnl
                
                # Reset position
                self.position = 0
                self.avg_entry_price = 0.0
                unrealized_pnl = 0.0
                total_pnl_with_unrealized = self.total_pnl
        
        # RISK MANAGEMENT: Drawdown penalty
        # Penalize large drawdowns to encourage risk management
        if drawdown > 0:
            # Penalty increases quadratically with drawdown
            drawdown_penalty = -0.001 * (drawdown / 100.0) ** 2
            reward += drawdown_penalty
        
        # RISK MANAGEMENT: Large loss penalty
        # Strongly penalize episodes with large negative P&L
        if total_pnl_with_unrealized < -1000.0:
            large_loss_penalty = -0.01 * abs(total_pnl_with_unrealized) / 100.0
            reward += large_loss_penalty
        
        # Total reward: primarily based on realized P&L changes
        # Add small unrealized P&L component for immediate feedback (reduced weight)
        # Reward = change in total P&L (realized + small unrealized component) - transaction costs
        reward += unrealized_pnl * 0.1  # Reduced from 0.5 - tighter coupling to actual P&L
        
        # Scale reward to prevent Q-value explosion (divide by 100 to normalize)
        # SPX prices are typically 3000-5000, so P&L can be large
        # Scale down to make Q-values more stable
        reward = reward / 100.0
        
        # Clip reward to reasonable range for stability
        # This prevents extreme rewards from destabilizing training
        reward = np.clip(reward, -5.0, 5.0)
        
        # Move to next step
        self.current_step += 1
        self.step_count += 1
        
        # Check if done: max steps reached, end of data, max loss exceeded, or drawdown too large
        done = False
        if self.step_count >= self.max_steps_per_episode:
            done = True
        elif self.current_step >= len(self.df) - 1:
            done = True
        elif total_pnl_with_unrealized <= self.max_loss_per_episode:
            # Early termination if loss exceeds threshold
            done = True
        elif drawdown > self.max_drawdown_per_episode:
            # Early termination if drawdown exceeds threshold
            done = True
        elif stop_loss_triggered:
            # Episode ends after stop-loss is triggered (position closed)
            done = True
        
        # Get next state
        if not done:
            next_state = self._get_state()
        else:
            # Final state: close all positions
            if self.position != 0:
                final_price = self.df.iloc[-1]['Close']
                if self.avg_entry_price != 0:
                    # Calculate final P&L
                    if self.position > 0:
                        # Long position
                        final_pnl = (final_price - self.avg_entry_price) * self.position
                    else:
                        # Short position
                        final_pnl = (self.avg_entry_price - final_price) * abs(self.position)
                    
                    # Apply transaction cost for closing position
                    closing_cost = abs(self.position) * self.transaction_cost_per_contract
                    final_pnl -= closing_cost
                    
                    reward += final_pnl
                    self.total_pnl += final_pnl
                self.position = 0
                self.avg_entry_price = 0.0
                self.position_value = 0.0
            
            # Use last valid state
            next_state = self._get_state()
        
        # total_pnl_with_unrealized already calculated above for risk management
        # Determine termination reason
        termination_reason = None
        if done:
            if stop_loss_triggered:
                termination_reason = 'stop_loss'
            elif drawdown > self.max_drawdown_per_episode:
                termination_reason = 'max_drawdown'
            elif self.step_count >= self.max_steps_per_episode:
                termination_reason = 'max_steps'
            elif self.current_step >= len(self.df) - 1:
                termination_reason = 'end_of_data'
            elif total_pnl_with_unrealized <= self.max_loss_per_episode:
                termination_reason = 'max_loss'
        
        info = {
            'step': self.current_step,
            'step_count': self.step_count,
            'position': self.position,
            'cash': self.cash,
            'total_pnl': self.total_pnl,  # Realized P&L only
            'total_pnl_with_unrealized': total_pnl_with_unrealized,  # Realized + Unrealized
            'unrealized_pnl': unrealized_pnl,
            'price': current_price,
            'transaction_cost': transaction_cost,
            'termination_reason': termination_reason,
            'drawdown': drawdown,
            'peak_pnl': self.peak_pnl,
            'stop_loss_triggered': stop_loss_triggered
        }
        
        return next_state, reward, done, info


class DQNAgent:
    """
    Deep Q-Network Agent for SPX Trading
    """
    
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu',
                 lr: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 64, 
                 tau: float = 0.01, min_lr: float = 5e-5, reward_norm_beta: float = 0.01,
                 network_size: str = 'medium', learnable_risk_control: bool = False):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            device: Device to run on
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            memory_size: Replay buffer size
            batch_size: Batch size for training
            tau: Soft update coefficient for target network (0.01 = 1% update per step)
                 - 0.01: Standard choice, good balance of stability and responsiveness
                 - 0.001: More stable but slower to adapt (use if training is very unstable)
                 - 0.1: Faster adaptation but less stable (use if targets are too stale)
            min_lr: Floor for the optimizer learning rate to avoid stalling at ~0
            reward_norm_beta: Update rate for running reward statistics (smaller = slower updates)
            network_size: Network capacity - 'small', 'medium', or 'large'
                          - 'small': 64-64 feature, 32-32 streams (faster, less capacity)
                          - 'medium': 128-128 feature, 64-64 streams (default, balanced)
                          - 'large': 256-256-128 feature, 128-128 streams (more capacity, slower)
            learnable_risk_control: If True, network outputs risk multiplier to adjust stop-loss
                                   Agent learns optimal risk level from data (default: False)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau  # Soft update coefficient for target network (0.01 = 1% update per step)
        self.min_lr = min_lr
        self.reward_beta = reward_norm_beta
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_std = 1.0
        self.network_size = network_size
        self.learnable_risk_control = learnable_risk_control
        
        # Track training steps for scheduler
        self.training_steps = 0
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Q-Network (with configurable size)
        self.q_network = self._build_network(network_size).to(device)
        self.target_network = self._build_network(network_size).to(device)
        # Initialize networks with small weights to prevent Q-value explosion
        self._initialize_weights(self.q_network)
        self._initialize_weights(self.target_network)
        self.update_target_network(soft_update=False)  # Hard update initially to copy weights
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # Learning rate scheduler to gradually reduce LR during training
        # Adjusted for longer training (2000 episodes): 
        # With ~5000 steps/episode × 2000 episodes = 10M steps total
        # Step every 100k steps (20 episodes worth) to get ~100 LR updates over full training
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100000,  # every 100k optimizer steps (~20 episodes)
            gamma=0.98  # Decay by 2% (gentle decay for long training)
        )
        
        # Loss function - using SmoothL1Loss (Huber loss) for better stability
        self.criterion = nn.SmoothL1Loss()
    
    def _build_network(self, network_size: str = 'medium') -> nn.Module:
        """
        Build the Q-network using Dueling DQN architecture.
        Dueling DQN separates value (V) and advantage (A) estimation,
        which helps the agent learn which states are valuable vs which actions are valuable.
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        
        Args:
            network_size: 'small', 'medium', or 'large' - controls network capacity
        """
        # Define network architectures based on size
        if network_size == 'small':
            feature_dims = [64, 64]
            value_dims = [32, 1]
            advantage_dims = [32, self.action_size]
        elif network_size == 'medium':
            feature_dims = [128, 128]
            value_dims = [64, 1]
            advantage_dims = [64, self.action_size]
        elif network_size == 'large':
            feature_dims = [256, 256, 128]
            value_dims = [128, 1]
            advantage_dims = [128, self.action_size]
        else:
            raise ValueError(f"network_size must be 'small', 'medium', or 'large', got '{network_size}'")
        
        class DuelingDQN(nn.Module):
            def __init__(self, state_size, action_size, feature_dims, value_dims, advantage_dims, learnable_risk=False):
                super(DuelingDQN, self).__init__()
                self.learnable_risk = learnable_risk
                # Shared feature layers
                feature_layers = []
                prev_dim = state_size
                for dim in feature_dims:
                    feature_layers.append(nn.Linear(prev_dim, dim))
                    feature_layers.append(nn.ReLU())
                    prev_dim = dim
                self.feature_layers = nn.Sequential(*feature_layers)
                
                # Value stream: estimates V(s) - how good is this state?
                value_layers = []
                prev_dim = feature_dims[-1]
                for dim in value_dims:
                    value_layers.append(nn.Linear(prev_dim, dim))
                    if dim != value_dims[-1]:  # Don't add ReLU after final layer
                        value_layers.append(nn.ReLU())
                    prev_dim = dim
                self.value_stream = nn.Sequential(*value_layers)
                
                # Advantage stream: estimates A(s,a) - how much better is this action?
                advantage_layers = []
                prev_dim = feature_dims[-1]
                for dim in advantage_dims:
                    advantage_layers.append(nn.Linear(prev_dim, dim))
                    if dim != advantage_dims[-1]:  # Don't add ReLU after final layer
                        advantage_layers.append(nn.ReLU())
                    prev_dim = dim
                self.advantage_stream = nn.Sequential(*advantage_layers)
                
                # Risk control head: learns optimal risk multiplier (0.5 to 2.0)
                # Outputs a multiplier for stop-loss: 0.5 = tighter (conservative), 2.0 = looser (aggressive)
                if learnable_risk:
                    risk_layers = []
                    prev_dim = feature_dims[-1]
                    # Use same dimensions as value stream
                    for dim in value_dims[:-1]:  # All but last
                        risk_layers.append(nn.Linear(prev_dim, dim))
                        risk_layers.append(nn.ReLU())
                        prev_dim = dim
                    # Final layer outputs single value (risk multiplier)
                    risk_layers.append(nn.Linear(prev_dim, 1))
                    risk_layers.append(nn.Sigmoid())  # Output in [0, 1]
                    self.risk_stream = nn.Sequential(*risk_layers)
            
            def forward(self, x):
                features = self.feature_layers(x)
                value = self.value_stream(features)
                advantage = self.advantage_stream(features)
                
                # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
                # This ensures that V(s) represents the average Q-value
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                
                if self.learnable_risk:
                    # Risk multiplier: sigmoid outputs [0, 1], map to [0.5, 2.0]
                    # 0.5 = tighter stop-loss (conservative), 2.0 = looser (aggressive)
                    risk_raw = self.risk_stream(features)
                    risk_multiplier = 0.5 + 1.5 * risk_raw  # Map [0, 1] to [0.5, 2.0]
                    return q_values, risk_multiplier
                else:
                    return q_values
        
        return DuelingDQN(self.state_size, self.action_size, feature_dims, value_dims, advantage_dims, 
                         learnable_risk=self.learnable_risk_control)
    
    def _initialize_weights(self, network: nn.Module):
        """Initialize network weights to prevent Q-value explosion"""
        for module in network.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with smaller scale
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Smaller gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using running statistics to keep TD targets in a stable range.
        Helps when raw rewards span tens of thousands of dollars.
        """
        with torch.no_grad():
            batch_mean = rewards.mean().item()
            batch_var = rewards.var(unbiased=False).item()
            self.reward_mean = (1.0 - self.reward_beta) * self.reward_mean + self.reward_beta * batch_mean
            self.reward_var = (1.0 - self.reward_beta) * self.reward_var + self.reward_beta * max(batch_var, 1e-6)
            self.reward_std = max(math.sqrt(self.reward_var), 1e-3)
        normalized = (rewards - self.reward_mean) / self.reward_std
        return torch.clamp(normalized, min=-10.0, max=10.0)
    
    def _enforce_lr_floor(self):
        """Prevent the optimizer LR from decaying below the configured floor."""
        if self.min_lr is None:
            return
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
    
    def update_target_network(self, soft_update: bool = True):
        """
        Update target network weights.
        
        Args:
            soft_update: If True, use soft update (Polyak averaging). 
                        If False, use hard update (copy all weights).
        """
        if soft_update:
            # Soft update: gradually blend target network with main network
            # tau = 0.01 means 1% of main network, 99% of target network
            for target_param, main_param in zip(self.target_network.parameters(), 
                                                self.q_network.parameters()):
                target_param.data.copy_(self.tau * main_param.data + 
                                       (1.0 - self.tau) * target_param.data)
        else:
            # Hard update: copy all weights (original method)
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
        
        Returns:
            If learnable_risk_control: (action_index, risk_multiplier)
            Else: action_index
        """
        if training and random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            if self.learnable_risk_control:
                # Random risk multiplier during exploration
                risk_multiplier = random.uniform(0.5, 2.0)
                return action, risk_multiplier
            return action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        network_output = self.q_network(state_tensor)
        
        if self.learnable_risk_control:
            q_values, risk_multiplier = network_output
            action = q_values.argmax().item()
            risk_multiplier = risk_multiplier.item()
            return action, risk_multiplier
        else:
            return network_output.argmax().item()
    
    def replay(self) -> Optional[float]:
        """
        Train the agent on a batch of experiences
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Require more memory before training for stability (warmup period)
        min_memory_for_training = max(self.batch_size * 2, 1000)  # At least 2x batch size or 1000
        if len(self.memory) < min_memory_for_training:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        # Convert to numpy arrays first for efficient tensor creation
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        normalized_rewards = self._normalize_rewards(rewards)
        
        # Current Q values (handle risk multiplier output if learnable risk is enabled)
        network_output = self.q_network(states)
        if self.learnable_risk_control:
            current_q_values, _ = network_output  # Extract Q-values, ignore risk multiplier
        else:
            current_q_values = network_output
        current_q = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        # This reduces overestimation bias in Q-learning
        with torch.no_grad():
            # Select best actions using main network
            main_output = self.q_network(next_states)
            if self.learnable_risk_control:
                main_q_values, _ = main_output
            else:
                main_q_values = main_output
            next_actions = main_q_values.argmax(1)
            
            # Evaluate those actions using target network
            target_output = self.target_network(next_states)
            if self.learnable_risk_control:
                target_q_values, _ = target_output
            else:
                target_q_values = target_output
            next_q = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = normalized_rewards + (self.gamma * next_q * ~dones)
        
        # Clip target Q-values to reasonable range for stability
        # Don't clip current_q - let the network learn the true values
        # Only clip targets to prevent extreme values from destabilizing training
        # Rewards are scaled by 100, so typical reward range is -10 to +10
        # With gamma=0.99, Q-values should be in roughly -1000 to +1000 range
        # Increased clipping range from [-100, 100] to [-200, 200] to allow more learning
        # If loss is oscillating, tighter clipping might be preventing learning
        target_q = torch.clamp(target_q, min=-200.0, max=200.0)
        
        # Compute loss using SmoothL1Loss (Huber loss) for better stability
        # This is less sensitive to outliers than MSE
        loss = self.criterion(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent instability
        # Relaxed from 1.0 to 2.0 to allow more learning while still preventing explosions
        # With tau=0.01 (faster target updates), moderate clipping helps balance stability vs learning
        # Track gradient norm before clipping to diagnose if clipping is too tight/loose
        grad_norm_before = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
        self.optimizer.step()
        
        # Store gradient norm for analysis (only keep last 1000 to avoid memory issues)
        if not hasattr(self, 'grad_norms'):
            self.grad_norms = []
        self.grad_norms.append(grad_norm_before.item())
        if len(self.grad_norms) > 1000:
            self.grad_norms.pop(0)
        
        # Increment training step counter
        self.training_steps += 1
        
        # Step the learning rate scheduler only every step_size steps
        # This prevents the LR from decaying too quickly
        # StepLR internally tracks how many times step() has been called,
        # but we want to step based on actual training steps, not scheduler calls
        if self.scheduler is not None:
            # Only step scheduler when we've done step_size training steps
            # This ensures LR decays at the intended rate
            if self.training_steps % self.scheduler.step_size == 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                self._enforce_lr_floor()
                # Log LR changes for debugging (only every 10k steps to avoid spam)
                if self.training_steps % 10000 == 0:
                    print(f"   📉 LR decay at step {self.training_steps}: {old_lr:.6f} → {new_lr:.6f}")
        
        # Soft update target network every training step (more stable than periodic hard updates)
        # This provides smoother, more stable targets
        # However, if loss is oscillating, we might want to update less frequently
        # For now, keep soft updates but consider reducing tau if oscillations persist
        self.update_target_network(soft_update=True)
        
        # Decay epsilon
        # If loss is oscillating, we might need more exploration to escape local minima
        # Consider increasing epsilon_min or reducing epsilon_decay if oscillations persist
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str, episode: Optional[int] = None):
        """Save agent to file"""
        save_dict = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'reward_mean': self.reward_mean,
            'reward_var': self.reward_var,
            'reward_std': self.reward_std
        }
        if episode is not None:
            save_dict['episode'] = episode
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str) -> Optional[int]:
        """
        Load agent from file
        
        Returns:
            Episode number if available in checkpoint, None otherwise
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Check if checkpoint has risk_stream (learnable risk control)
        checkpoint_has_risk = any('risk_stream' in k for k in checkpoint.get('q_network_state_dict', {}).keys())
        current_has_risk = self.learnable_risk_control
        
        # Handle architecture mismatch
        if checkpoint_has_risk != current_has_risk:
            if checkpoint_has_risk and not current_has_risk:
                print(f"   ⚠️  Checkpoint has learnable risk control, but current agent doesn't.")
                print(f"   Loading without risk head (will ignore risk_stream weights)")
            elif not checkpoint_has_risk and current_has_risk:
                print(f"   ⚠️  Checkpoint doesn't have learnable risk control, but current agent does.")
                print(f"   Risk head will be randomly initialized (not loaded from checkpoint)")
        
        # Load Q-network with partial loading to handle architecture differences
        checkpoint_state = checkpoint['q_network_state_dict']
        current_state = self.q_network.state_dict()
        
        # Filter checkpoint state to match current architecture
        filtered_checkpoint = {}
        for key in current_state.keys():
            if key in checkpoint_state:
                if checkpoint_state[key].shape == current_state[key].shape:
                    filtered_checkpoint[key] = checkpoint_state[key]
                else:
                    print(f"   ⚠️  Shape mismatch for {key}: checkpoint {checkpoint_state[key].shape} vs current {current_state[key].shape}")
                    print(f"   Using random initialization for this layer")
            else:
                print(f"   ⚠️  Missing key in checkpoint: {key} (will use random initialization)")
        
        # Load with strict=False to allow missing/extra keys
        missing_keys, unexpected_keys = self.q_network.load_state_dict(filtered_checkpoint, strict=False)
        if missing_keys:
            print(f"   ⚠️  Missing keys (using random init): {missing_keys[:3]}..." if len(missing_keys) > 3 else f"   ⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys (ignored): {unexpected_keys[:3]}..." if len(unexpected_keys) > 3 else f"   ⚠️  Unexpected keys: {unexpected_keys}")
        
        # Load target network (same architecture as Q-network)
        target_checkpoint = checkpoint['target_network_state_dict']
        target_filtered = {}
        target_current = self.target_network.state_dict()
        for key in target_current.keys():
            if key in target_checkpoint and target_checkpoint[key].shape == target_current[key].shape:
                target_filtered[key] = target_checkpoint[key]
        self.target_network.load_state_dict(target_filtered, strict=False)
        
        # Load optimizer (may have different parameter groups if architecture changed)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"   ⚠️  Could not load optimizer state: {e}")
            print(f"   Optimizer will be reinitialized (training will continue)")
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        # Load reward normalization stats if available
        if 'reward_mean' in checkpoint:
            self.reward_mean = checkpoint['reward_mean']
        if 'reward_var' in checkpoint:
            self.reward_var = checkpoint['reward_var']
        if 'reward_std' in checkpoint:
            self.reward_std = checkpoint['reward_std']
        
        # Return episode number if available
        return checkpoint.get('episode', None)


# Import shared utilities (after DQNAgent is defined to avoid circular import)
from futures_reinforcement_utils import (
    SPXTradingEnv,
    train_agent,
    evaluate_agent,
    quick_validation_test,
    optimize_stop_loss,
    analyze_episode_strategy,
    analyze_training_progress
)

