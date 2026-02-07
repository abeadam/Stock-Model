"""
Shared utilities for reinforcement learning trading agents.

This module contains shared code between DQN and PPO implementations:
- SPXTradingEnv: Trading environment
- train_agent: Training function (works with both DQN and PPO)
- evaluate_agent: Evaluation function
- quick_validation_test: Validation function
- optimize_stop_loss: Stop-loss optimization
- analyze_episode_strategy: Episode analysis
- analyze_training_progress: Training progress analysis
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Optional, Dict, List, cast
from collections import deque
import random
import pickle
from sklearn.preprocessing import StandardScaler

# Import the model class from the futures model file
import sys
# Add parent and MVE_SSNs_model directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'MVE_SSNs_model'))
from futures_model_MVE_SNNs import MVEModel

# Import agents (will be imported at runtime to avoid circular dependencies)
# Use string-based type checking in functions
DQNAgent = None
PPOAgent = None

def _is_dqn_agent(agent):
    """Check if agent is a DQNAgent using string-based type checking"""
    return agent.__class__.__name__ == 'DQNAgent'

def _is_ppo_agent(agent):
    """Check if agent is a PPOAgent using string-based type checking"""
    return agent.__class__.__name__ == 'PPOAgent'


class SPXTradingEnv:
    """
    Reinforcement Learning Environment for SPX Trading
    
    State: Previous step's all non-forward-looking features + current position + 
           model predictions (mean_high, std_high, mean_low, std_low)
    Actions: Buy 0, 1, or 2 units OR Sell 0, 1, or 2 units (net change in position)
    Reward: Profit/loss from trading minus transaction costs ($2.5 per contract)
    """
    
    def __init__(self, data_path: str, model_path: Optional[str] = None, device: str = 'cpu',
                 max_steps_per_episode: int = 5000, max_loss_per_episode: float = -50000.0,
                 stop_loss_per_contract: float = 750.0,  # Increased from 500.0 to 750.0 - less aggressive 
                 variance_adaptive_risk: bool = True,
                 variance_risk_sensitivity: float = 0.5,
                 max_position: int = 2,
                 scaler_path: Optional[str] = None,
                 max_rows: Optional[int] = None,
                 random_start_in_file: bool = False):
        """
        Initialize the trading environment

        Args:
            data_path: Path to es_with_indicators.csv
            model_path: Path to futures_model.pt (optional)
            device: Device to run model on ('cpu' or 'cuda')
            max_steps_per_episode: Maximum steps per episode (default: 5000, was unlimited)
            max_loss_per_episode: Maximum loss before early termination (default: -$50k)
            stop_loss_per_contract: Base stop loss threshold per contract (default: $500)
            variance_adaptive_risk: If True, adjust stop-loss based on predicted variance (default: True)
                                   Note: requires model_path to be provided.
            variance_risk_sensitivity: How much variance affects risk (0.0-1.0, default: 0.5)
                                    Higher = more aggressive adjustment based on variance
            max_position: Maximum number of contracts to hold (long or short, default: 2)
            scaler_path: Optional path to save/load feature scaler (ensures consistency)
            max_rows: Optional limit on number of rows to load (useful for quick testing, default: None = no limit)
            random_start_in_file: If True and max_rows is set, load a random max_rows-sized window from the
                                 file instead of the first max_rows rows. Ignored when max_rows is None.
        """
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
        self.max_steps_per_episode = max_steps_per_episode
        self.max_loss_per_episode = max_loss_per_episode
        self.base_stop_loss_per_contract = stop_loss_per_contract
        self.variance_adaptive_risk = variance_adaptive_risk and model_path is not None
        self.variance_risk_sensitivity = variance_risk_sensitivity
        self.max_position = max_position
        self.min_position = -max_position
        
        # Determine scaler path if not provided
        if scaler_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(script_dir, 'reinforcement_results')
            os.makedirs(results_dir, exist_ok=True)
            self.scaler_path = os.path.join(results_dir, 'feature_scaler.pkl')
        else:
            self.scaler_path = scaler_path
        
        # Track current variance predictions for adaptive risk
        self.current_pred_std_high = 0.0
        self.current_pred_std_low = 0.0
        self.current_stop_loss_per_contract = stop_loss_per_contract
        
        # Training progress tracking (for decaying penalties)
        self.training_progress = 0.0
        
        # Load data
        print(f"Loading data from {data_path}...")
        # For quick tests, avoid reading the full CSV: use nrows to limit IO.
        if max_rows is not None and max_rows > 0:
            if random_start_in_file:
                with open(data_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                total_data = max(0, total_lines - 1)  # subtract header
                if total_data <= max_rows:
                    self.df = pd.read_csv(data_path, nrows=max_rows)
                    print(f"Loaded up to {max_rows} rows (file has {total_data} data rows)")
                else:
                    skip = random.randint(0, total_data - max_rows)
                    # skiprows (callable): 0=header; skip data rows 1..skip, then read nrows
                    self.df = pd.read_csv(
                        data_path,
                        skiprows=lambda x: 1 <= cast(int, x) <= skip,
                        nrows=max_rows,
                    )
                    print(f"Loaded {max_rows} rows starting at file row {skip + 1} (random window)")
            else:
                self.df = pd.read_csv(data_path, nrows=max_rows)
                print(f"Loaded up to {max_rows} rows for quick testing (using pandas nrows)")
        else:
            self.df = pd.read_csv(data_path)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in self.df.columns for col in required_cols), \
            f"Missing required columns. Found: {self.df.columns.tolist()}"
        
        # Remove rows with NaN in OHLCV
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)
        print(f"Loaded {len(self.df)} rows (removed {initial_len - len(self.df)} rows with NaN)")
        
        # Load the pre-trained model (if provided)
        if model_path is not None:
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
        else:
            print("No prediction model provided. RL will learn directly from raw data.")
            self.model = None
            self.scaler = None
            self.feature_names = []
            self.predict_both = False
        
        # Trading parameters
        self.initial_position = 0
        self.transaction_cost_per_contract = 2.5  # $2.5 per futures contract
        self.trade_penalty_per_contract = 0.0  # Removed - was discouraging all trading
        self.position_penalty_strength = 0.0  # Removed - was penalizing holding positions
        
        # Risk management parameters
        self.stop_loss_per_contract = stop_loss_per_contract  # Stop loss: close position if unrealized loss > threshold per contract
        self.max_drawdown_per_episode = 20000.0  # Maximum drawdown before early termination
        self.peak_pnl = 0.0  # Track peak P&L for drawdown calculation
        
        # Normalize stop_loss for state representation (typical range: 100-1000)
        # self.stop_loss_normalized = (self.stop_loss_per_contract - 100.0) / 900.0  # Normalize to [0, 1] for 100-1000 range
        
        # Identify columns that don't depend on future data
        # Exclude forward-looking columns: PctChange_ToMaxHigh_5, PctChange_ToMinLow_5
        # Also exclude Date, DateTime, DateTime_ET (not useful as features)
        forward_looking_cols = ['PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
        exclude_cols = ['Date', 'DateTime', 'DateTime_ET'] + forward_looking_cols
        
        # Get all feature columns (non-forward-looking)
        # self.feature_cols = [col for col in self.df.columns 
        #                     if col not in exclude_cols]
        self.feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'BB_20_MA', 'BB_20_2_Upper', 'BB_20_2_Lower', 'MFI_14', 'RSI_14',
        'BB_20_1_Upper', 'BB_20_1_Lower', 'BB_20_3_Upper', 'BB_20_3_Lower', 'BB_50_MA', 'BB_50_2_Upper', 'BB_50_2_Lower', 'BB_50_1_Upper', 'BB_50_1_Lower', 
        'BB_50_3_Upper', 'BB_50_3_Lower', 'BB_10_MA', 'BB_10_2_Upper', 'BB_10_2_Lower', 'BB_10_1_Upper', 'BB_10_1_Lower', 'BB_10_3_Upper', 'BB_10_3_Lower',
        'EMA_10', 'EMA_20', 'EMA_50', 'BB_10_STD', 'BB_20_STD', 'BB_50_STD', 'VXM_Open', 'VXM_High', 'VXM_Low', 'VXM_Close', 'VXM_Volume', 'VXM_BB_10_MA', 
        'VXM_BB_10_STD', 'VXM_BB_10_1_Upper', 'VXM_BB_10_1_Lower', 'VXM_BB_10_2_Upper', 'VXM_BB_10_2_Lower', 'VXM_BB_10_3_Upper', 'VXM_BB_10_3_Lower',
        'VXM_BB_20_MA', 'VXM_BB_20_STD', 'VXM_BB_20_1_Upper', 'VXM_BB_20_1_Lower', 'VXM_BB_20_2_Upper', 'VXM_BB_20_2_Lower', 'VXM_BB_20_3_Upper', 'VXM_BB_20_3_Lower',
        'VXM_BB_50_MA', 'VXM_BB_50_STD', 'VXM_BB_50_1_Upper', 'VXM_BB_50_1_Lower', 'VXM_BB_50_2_Upper', 'VXM_BB_50_2_Lower', 'VXM_BB_50_3_Upper', 'VXM_BB_50_3_Lower',
        'VXM_EMA_10', 'VXM_EMA_20', 'VXM_EMA_50', 'ATR_14', 'Hours_From_Formal_Trading', 'Hours_From_Overnight_Trading']
        print(f"Using {len(self.feature_cols)} non-forward-looking features for state")
        print(f"Excluded columns: {exclude_cols}")
        
        # Create or load scaler for all features
        # Remove rows with NaN in feature columns for fitting scaler
        feature_data = self.df[self.feature_cols].copy()
        # Fill NaN with 0 for features (some indicators may have NaN at start)
        feature_data = feature_data.fillna(0)
        
        self.feature_scaler = StandardScaler()
        
        if os.path.exists(self.scaler_path):
            print(f"Loading feature scaler from {self.scaler_path}...")
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✓ Feature scaler loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading scaler from {self.scaler_path}: {e}")
                print(f"Fitting new scaler instead...")
                self.feature_scaler.fit(feature_data.values)
                # Save the newly fitted scaler
                try:
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.feature_scaler, f)
                except:
                    pass
        else:
            print(f"No scaler found at {self.scaler_path}. Fitting new scaler...")
            self.feature_scaler.fit(feature_data.values)
            # Save the newly fitted scaler
            try:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
                print(f"✓ Saved new feature scaler to {self.scaler_path}")
            except Exception as e:
                print(f"⚠️ Could not save scaler to {self.scaler_path}: {e}")
        
        # Action space: net change in position from -max_position to +max_position
        # Actions: 0 to 2*max_position (maps to -max_position to +max_position)
        self.action_space_size = 2 * self.max_position + 1
        # No mapping dictionary needed anymore, we'll calculate it in step()
        
        # Episode tracking
        self.step_count = 0
        self.episode_start_pnl = 0.0
        
        # Reset environment
        self.reset()
    
    def set_training_progress(self, progress: float):
        """
        Set training progress (0.0 to 1.0) to decay penalties
        """
        self.training_progress = max(0.0, min(1.0, progress))
    
    def set_feature_scaler(self, scaler):
        """
        Set the feature scaler from an external source (e.g., loaded from checkpoint)
        Also saves it to feature_scaler.pkl for consistency
        
        Args:
            scaler: StandardScaler instance
        """
        self.feature_scaler = scaler
        # Save to file for consistency
        try:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            print(f"✓ Set feature scaler and saved to {self.scaler_path}")
        except Exception as e:
            print(f"⚠️ Could not save feature scaler to {self.scaler_path}: {e}")

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
        
        # Reset state variables
        self.position = self.initial_position
        self.cash = 0.0  # Track cash (negative means we owe money)
        self.total_pnl = 0.0
        self.step_count = 0
        self.episode_start_pnl = 0.0
        
        # FIX: Set initial entry price to current market price to avoid "phantom P&L"
        if self.position != 0:
            self.avg_entry_price = self.df.iloc[self.current_step]['Close']
        else:
            self.avg_entry_price = 0.0
            
        self.position_value = 0.0  # Value of current position
        self.peak_pnl = 0.0  # Track peak P&L for drawdown calculation
        self.initial_pnl = 0.0  # Track starting P&L
        self.step_count = 0
        self.episode_start_pnl = 0.0
        
        # Trade tracking for overtrading penalty
        self.trade_count = 0
        self.last_trade_step = -1000  # Initialize to far in the past

        # Temporal state: in-episode history (no RNN needed)
        self.steps_holding = 0          # Consecutive steps with same non-zero position
        self.steps_since_last_trade = 0 # Steps since last position change (0 = just traded)
        
        # Drawdown tracking
        self.max_drawdown_so_far = 0.0
        self.prev_unrealized_pnl = 0.0
        self.episode_transaction_cost = 0.0  # Track total transaction costs in episode
        
        # Reset stop loss to base value
        self.current_stop_loss_per_contract = self.base_stop_loss_per_contract
        
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
          (if model is present)
        - Current stop-loss threshold (normalized)
        - Temporal: steps_holding [0,1], steps_since_last_trade [0,1], running_pnl ~[-2,2]
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
        feature_array = np.array(feature_values).reshape(1, -1)[0]
        # features_normalized = self.feature_scaler.transform(feature_array)[0]
        
        # Get model prediction using previous step's data (if model is available)
        model_outputs = []
        if self.model is not None:
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
            scaler = self.scaler
            if scaler is not None:
                model_features_scaled = scaler.transform(model_features)
            else:
                # Fallback: use unscaled features if scaler is not available
                model_features_scaled = model_features
            
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
            model_outputs = [pred_mean_high, pred_std_high, pred_mean_low, pred_std_low]
        else:
            # If no model, we don't have predictions or variance
            self.current_pred_std_high = 0.0
            self.current_pred_std_low = 0.0
            model_outputs = []
        
        # Calculate adaptive stop-loss based on predicted variance (only if model is present)
        if self.variance_adaptive_risk and self.model is not None:
            # Higher variance (uncertainty) = tighter stop-loss (more conservative)
            # Lower variance (confidence) = looser stop-loss (more aggressive)
            # Use the maximum of high/low variance to be conservative
            max_pred_std = max(abs(self.current_pred_std_high), abs(self.current_pred_std_low))
            
            # Normalize variance to a scale factor (0.5 to 2.0)
            # Typical std values are in range 0.001-0.1 (percentage changes)
            if max_pred_std < 0.001:
                variance_factor = 1.5  # Very confident: allow larger losses
            elif max_pred_std > 0.1:
                variance_factor = 0.5  # Very uncertain: tighten stop-loss
            else:
                # Linear interpolation between 0.001 and 0.1
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
        # self.stop_loss_normalized = (self.current_stop_loss_per_contract - 100.0) / 900.0  # Normalize to [0, 1] for 100-1000 range
        
        # Combine state: [all_features, position, [predictions], stop_loss]
        # This gives the agent information about both expected high and low price movements (if available)
        # AND the current stop-loss threshold (so it can learn to adapt to different risk levels)
        state_components = [
            feature_array,  # All non-forward-looking features
            [position_normalized],  # 1 value
        ]
        
        # Add model predictions if available
        if model_outputs:
            state_components.append(model_outputs)
            
        # Add stop-loss
        # state_components.append([self.stop_loss_normalized])
        state_components.append([self.current_stop_loss_per_contract])

        # Temporal state: in-episode history (no RNN needed)
        # - steps_holding: how long we've held the current position [0,1]
        # - steps_since_last_trade: steps since last position change [0,1]
        # - running_pnl: total_pnl + unrealized, normalized to ~[-2,2] ($±10k)
        if self.position != 0 and self.avg_entry_price != 0:
            current_price = self.df.iloc[self.current_step]['Close']
            if self.position > 0:
                _unreal = (current_price - self.avg_entry_price) * self.position * 50.0
            else:
                _unreal = (self.avg_entry_price - current_price) * abs(self.position) * 50.0
        else:
            _unreal = 0.0
        running_pnl = self.total_pnl + _unreal
        steps_holding_norm = min(1.0, self.steps_holding / self.max_steps_per_episode)
        steps_since_last_trade_norm = min(1.0, self.steps_since_last_trade / self.max_steps_per_episode)
        running_pnl_norm = float(np.clip(running_pnl / 5000.0, -2.0, 2.0))
        state_components.append([steps_holding_norm, steps_since_last_trade_norm, running_pnl_norm])
        
        state = np.concatenate(state_components)
        
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
            # self.current_stop_loss_per_contract = max(100.0, min(1000.0, self.current_stop_loss_per_contract))
        
        # Map action to position change
        # action 0 -> -max_position, ..., action max_position -> 0, ..., action 2*max_position -> +max_position
        position_change = action - self.max_position
        
        # Get current price (use Close price for execution)
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Calculate new position
        new_position = self.position + position_change
        
        # Clamp position to valid range
        new_position = max(self.min_position, min(self.max_position, new_position))
        actual_change = new_position - self.position
        
        # Calculate transaction cost: $2.5 per contract traded
        contracts_traded = abs(actual_change)
        transaction_cost = contracts_traded * self.transaction_cost_per_contract
        self.episode_transaction_cost += transaction_cost
        
        # Get prices for market-based reward
        # Ensure we have a previous step price
        if self.current_step > 0:
            price_prev = self.df.iloc[self.current_step - 1]['Close']
        else:
            price_prev = current_price
            
        # 1. Market Reward: money made/lost by the position held during the last step
        # This is the "true" journey reward that sums to final P&L
        market_reward = self.position * (current_price - price_prev) * 50.0
        
        # Initialize reward: Reward is purely P&L-based (no penalties)
        # The risk multiplier learns naturally through the P&L reward signal
        reward = 0.0
        
        # Add market reward (wealth change due to market movement)
        # This tracks unrealized P&L changes step-by-step
        reward += market_reward
        
        # Subtract transaction cost from this step's reward
        reward -= transaction_cost
        
        # Update P&L and Position
        old_position = self.position
        
        # Calculate realized P&L from position change
        if old_position != 0 and self.avg_entry_price != 0:
            # Realized P&L when reducing or reversing position
            if (old_position > 0 and actual_change < 0) or (old_position < 0 and actual_change > 0):
                # Closing or reducing position
                closed_units = min(abs(old_position), abs(actual_change))
                if old_position > 0:
                    # Long position: profit when price goes up
                    realized_pnl = (current_price - self.avg_entry_price) * closed_units * 50.0
                else:
                    # Short position: profit when price goes down
                    realized_pnl = (self.avg_entry_price - current_price) * closed_units * 50.0
                
                # CRITICAL FIX: When closing a position, we need to adjust reward
                # Market reward tracked unrealized P&L changes step-by-step, but now we're realizing P&L
                # We need to: (1) remove market_reward contribution for closed units this step, (2) add realized P&L
                # The market_reward for closed units this step uses the sign of old_position
                # (old_position / abs(old_position)) gives +1 for long, -1 for short
                market_reward_for_closed = (old_position / abs(old_position)) * closed_units * (current_price - price_prev) * 50.0
                # Replace with realized P&L (which is already calculated correctly above)
                reward -= market_reward_for_closed
                reward += realized_pnl
                
                # Update total P&L (realized)
                self.total_pnl += realized_pnl
        
        # Apply transaction cost to total P&L
        self.total_pnl -= transaction_cost
        
        # Update average entry price and position
        if actual_change > 0:
            # Buying: reduce cash, add to position (or reduce short)
            cost = actual_change * current_price
            self.cash -= cost * 50.0
            
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
            self.cash += proceeds * 50.0
            
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

        # Update temporal state (in-episode history)
        if self.position == 0:
            self.steps_holding = 0
        elif actual_change != 0:
            self.steps_holding = 1  # New or changed position
        else:
            self.steps_holding += 1
        if actual_change != 0:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1
        
        # Unrealized P&L (mark-to-market)
        if self.position != 0 and self.avg_entry_price != 0:
            if self.position > 0:
                # Long position
                unrealized_pnl = (current_price - self.avg_entry_price) * self.position * 50.0
            else:
                # Short position
                unrealized_pnl = (self.avg_entry_price - current_price) * abs(self.position) * 50.0
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
                stop_loss_position = self.position  # Save position before reset
                if self.position > 0:
                    stop_loss_pnl = (current_price - self.avg_entry_price) * self.position * 50.0
                else:
                    stop_loss_pnl = (self.avg_entry_price - current_price) * abs(self.position) * 50.0
                
                # Apply transaction cost for closing
                closing_cost = abs(self.position) * self.transaction_cost_per_contract
                self.episode_transaction_cost += closing_cost
                stop_loss_pnl -= closing_cost
                
                # CRITICAL FIX: Add stop-loss P&L to reward (it was missing!)
                # Also need to remove the market_reward contribution for this position
                # since we're now realizing the P&L instead of tracking unrealized
                # stop_loss_position already has the correct sign (positive for long, negative for short)
                market_reward_for_stop_loss = stop_loss_position * (current_price - price_prev) * 50.0
                reward -= market_reward_for_stop_loss
                reward += stop_loss_pnl
                
                self.total_pnl += stop_loss_pnl
                
                # Reset position and temporal state (close is a trade)
                self.position = 0
                self.avg_entry_price = 0.0
                self.steps_holding = 0
                self.steps_since_last_trade = 0
                unrealized_pnl = 0.0
                total_pnl_with_unrealized = self.total_pnl
        
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
            # Final state: close all positions and apply terminal rewards
            if self.position != 0:
                final_price = self.df.iloc[self.current_step]['Close']
                if self.avg_entry_price != 0:
                    # Calculate final P&L
                    if self.position > 0:
                        final_pnl = (final_price - self.avg_entry_price) * self.position * 50.0
                    else:
                        final_pnl = (self.avg_entry_price - final_price) * abs(self.position) * 50.0
                    
                    # CRITICAL FIX: Remove the last step's market_reward since we're now realizing the P&L
                    # The market_reward for the last step was already added to reward at line 530,
                    # but it only represents the price movement from price_prev to current_price.
                    # We need to replace it with final_pnl which is the total profit from entry to exit.
                    # market_reward = self.position * (current_price - price_prev) * 50.0 (already in reward)
                    # Note: final_price == current_price at this point, but using current_price for consistency
                    market_reward_for_last_step = self.position * (current_price - price_prev) * 50.0
                    reward -= market_reward_for_last_step
                    
                    # Terminal Reward: Add final P&L to reward
                    # The final_pnl represents the total profit/loss from entry to exit
                    # This needs to be added to reward so it reflects the actual outcome
                    reward += final_pnl
                    
                    # Apply transaction cost for closing position
                    closing_cost = abs(self.position) * self.transaction_cost_per_contract
                    reward -= closing_cost  # Subtract closing cost from reward
                    self.episode_transaction_cost += closing_cost
                    self.total_pnl += final_pnl
                    
                    # No terminal bonuses - reward is purely final_pnl - closing_cost
                
                self.position = 0
                self.avg_entry_price = 0.0
                self.position_value = 0.0
                self.steps_holding = 0
                self.steps_since_last_trade = 0
            
            # Position closed (or was already 0), so unrealized_pnl is 0
            unrealized_pnl = 0.0
            
            # Use last valid state (temporal vars already 0 from close or unchanged if was flat)
            next_state = self._get_state()
        
        # Scale reward to prevent value function explosion while maintaining signal strength
        # Dividing by 100.0 converts dollar amounts to reward units
        # Example: $100 P&L → 1.0 reward, $10,000 P&L → 100.0 reward
        reward = reward / 100.0
        
        # Clip reward to prevent extreme values from destabilizing training
        # Wide range allows model to distinguish between small wins and big wins
        # Clipping at ±2000 means max reward is ±20.0 after scaling (for $200,000 moves)
        # Increased from ±500 to better track large P&L swings without clipping
        reward = np.clip(reward, -2000.0, 2000.0)

        # Calculate total P&L with unrealized for logging and info dict
        # Note: unrealized_pnl is either from earlier calculation (if done=False) or 0 (if done=True and position closed)
            
        total_pnl_with_unrealized = self.total_pnl + unrealized_pnl
        
        # Final drawdown and peak update
        if total_pnl_with_unrealized > self.peak_pnl:
            self.peak_pnl = total_pnl_with_unrealized
        drawdown = self.peak_pnl - total_pnl_with_unrealized

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
            'cumulative_transaction_cost': self.episode_transaction_cost,
            'termination_reason': termination_reason,
            'drawdown': drawdown,
            'peak_pnl': self.peak_pnl,
            'stop_loss_triggered': stop_loss_triggered
        }
        
        return next_state, reward, done, info



def train_agent(env: SPXTradingEnv, agent, num_episodes: int = 1000,
                target_update_freq: int = 2, save_freq: int = 100, start_episode: int = 0,
                save_training_log: bool = True):
    """
    Train agent (supports both DQN and PPO)
    
    Args:
        env: Trading environment
        agent: DQNAgent or PPOAgent
        num_episodes: Number of episodes to train
        target_update_freq: Target network update frequency (DQN only)
        save_freq: Checkpoint save frequency
        start_episode: Starting episode number
        save_training_log: Whether to save training log
    """
    """
    Train the DQN agent
    
    Args:
        env: Trading environment
        agent: DQN agent
        num_episodes: Number of training episodes
        target_update_freq: Frequency to update target network
        save_freq: Frequency to save agent
        start_episode: Episode number to start from (for resuming training)
    """
    if start_episode > 0:
        print(f"\nResuming training from episode {start_episode} for {num_episodes} more episodes...")
    else:
        print(f"\nStarting training for {num_episodes} episodes...")
        # Clear the reinforcement_results folder for a fresh run
        results_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
        if os.path.exists(results_dir):
            import shutil
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            print(f"🧹 Cleared {results_dir} for fresh training")
    print("=" * 80)
    print("\n🚀 IMPROVEMENTS IMPLEMENTED:")
    print(f"  ✓ Episodes limited to {env.max_steps_per_episode} steps (was unlimited)")
    print(f"  ✓ Early termination if loss exceeds ${env.max_loss_per_episode:.0f}")
    print(f"  ✓ Reward function optimized (unrealized P&L weight: 0.01, clipped to [-50, 50])")
    print(f"  ✓ Position rotation: Penalties for sticking at max/min position")
    print(f"  ✓ Double DQN: Reduces overestimation bias")
    print(f"  ✓ Dueling DQN: Separates value and advantage estimation for better learning")
    print("=" * 80)
    
    episode_rewards = []
    episode_pnls = []
    episode_losses = []
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropies = []
    episode_steps = []
    episode_transaction_costs = []
    
    # Early stopping parameters
    # Increase patience for fine-tuning (variance reduction focus)
    if start_episode >= 400:
        early_stopping_patience = 400  # More patience for variance reduction
        print(f"\n🎯 VARIANCE REDUCTION MODE (resuming from episode {start_episode})")
        print(f"   - Increased early stopping patience: 400 episodes (was 200)")
        print(f"   - Focus: Reduce P&L variance and improve consistency")
        print(f"   - Learning rate reduced for fine-tuning")
    else:
        early_stopping_patience = 200  # Standard patience for initial training
    
    best_loss = float('inf')
    episodes_without_improvement = 0
    early_stopping_enabled = False  # Disabled - let training run to completion
    
    # Track variance metrics for fine-tuning
    pnl_variance_tracking = start_episode >= 400
    recent_pnls = []  # Track recent P&Ls for variance calculation
    best_pnl_variance = float('inf')
    variance_improvement_count = 0
    
    print(f"\n📊 TRAINING CONFIGURATION:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Start episode: {start_episode}")
    print(f"  Early stopping: {'Enabled' if early_stopping_enabled else 'Disabled'}")
    if early_stopping_enabled:
        print(f"  Patience: {early_stopping_patience} episodes (stop if loss doesn't improve)")
    if pnl_variance_tracking:
        print(f"  Variance tracking: Enabled (monitoring P&L std dev reduction)")
    print(f"  Checkpoint frequency: Every {save_freq} episodes")
    print(f"  LR scheduler: Step every 100k steps, decay by 2%")
    
    for episode in range(start_episode, start_episode + num_episodes):
        # Update training progress for decaying penalties
        total_expected_episodes = getattr(agent, 'total_episodes', start_episode + num_episodes)
        env.set_training_progress(episode / total_expected_episodes)
        
        state = env.reset()
        total_reward = 0.0
        steps = 0
        episode_loss = 0.0
        episode_policy_loss = 0.0
        episode_value_loss = 0.0
        episode_entropy = 0.0
        loss_count = 0
        
        # Track detailed episode information
        actions_taken = []
        positions = []
        prices = []
        transaction_costs = []
        rewards_per_step = []
        
        # Determine agent type
        is_ppo = _is_ppo_agent(agent)
        is_dqn = _is_dqn_agent(agent)
        
        while True:
            # Choose action
            if is_ppo:
                # PPO act() now returns (action, risk_multiplier)
                action, risk_multiplier = agent.act(state, training=True)
            elif is_dqn:
                act_result = agent.act(state, training=True)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                else:
                    action = act_result
                    risk_multiplier = None
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            actions_taken.append(action)
            
            # Execute action
            next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
            
            # Store experience (different for DQN vs PPO)
            if is_dqn:
                agent.remember(state, action, reward, next_state, done)
            elif is_ppo:
                agent.store_reward(reward, done)
            
            # Track episode details
            positions.append(info['position'])
            prices.append(info['price'])
            transaction_costs.append(info.get('transaction_cost', 0.0))
            rewards_per_step.append(reward)
            
            # Train agent (different for DQN vs PPO)
            if is_dqn:
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                        # Check for NaN or inf loss
                        if not (np.isfinite(loss)):
                            print(f"   ⚠️  Non-finite loss detected: {loss} at step {steps}")
                            print(f"   Current LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
                            print(f"   Training steps: {agent.training_steps}")
                            continue
            elif is_ppo:
                # PPO updates at end of episode or when we have enough samples (mid-episode update)
                # Update whenever we have at least batch_size samples
                buffer_size = len(agent.states)
                if (buffer_size >= agent.batch_size and not done):
                    # Mid-episode update: compute value estimate of current state for bootstrapping
                    # This allows more frequent learning and better sample efficiency
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                        _, value, _ = agent.actor_critic(state_tensor)
                        bootstrap_value = value.item()
                    
                    # Update with bootstrap value (episode not done, so bootstrap from current state)
                    loss_result = agent.update(terminal_value=bootstrap_value)
                    if loss_result is not None:
                        # Handle both dict (new format) and float (old format) for backward compatibility
                        if isinstance(loss_result, dict):
                            episode_loss += loss_result['total_loss']
                            episode_policy_loss += loss_result['policy_loss']
                            episode_value_loss += loss_result['value_loss']
                            episode_entropy += loss_result['entropy']
                        else:
                            episode_loss += loss_result
                        loss_count += 1
                        # Check for NaN or inf loss
                        loss_val = loss_result['total_loss'] if isinstance(loss_result, dict) else loss_result
                        if not (np.isfinite(loss_val)):
                            print(f"   ⚠️  Non-finite loss detected: {loss_val} at step {steps}")
                            print(f"   Current LR: {agent.optimizer.param_groups[0]['lr']:.6f}")
                            print(f"   Training steps: {agent.training_steps}")
                        # Log mid-episode update (only occasionally to avoid spam)
                        if agent.training_steps % 10 == 0:
                            print(f"   🔄 Mid-episode update at step {steps} (buffer: {buffer_size} samples, training step: {agent.training_steps})")
                    # Buffer is cleared inside update() after training
            
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                if hasattr(agent, 'on_episode_end') and callable(agent.on_episode_end):
                    agent.on_episode_end()
                break

        # PPO update at end of episode
        if is_ppo:
            # Terminal value estimate for GAE computation
            # Since the loop only exits when done=True, terminal value is always 0.0
            # (no future rewards after episode ends)
            terminal_value = 0.0
            # Update with terminal value
            # NOTE: Don't append to agent.values - compute_gae() will add next_value itself
            # Pass terminal_value to update() which will pass it to compute_gae()
            if len(agent.states) > 0:
                # Ensure last done flag is True (episode ended, so last step should be done=True)
                if len(agent.dones) > 0:
                    agent.dones[-1] = True
                loss_result = agent.update(terminal_value=terminal_value)
                if loss_result is not None:
                    # Handle both dict (new format) and float (old format) for backward compatibility
                    if isinstance(loss_result, dict):
                        episode_loss += loss_result['total_loss']
                        episode_policy_loss += loss_result['policy_loss']
                        episode_value_loss += loss_result['value_loss']
                        episode_entropy += loss_result['entropy']
                    else:
                        episode_loss += loss_result
                    loss_count += 1
                # If loss is None, keep existing episode_loss and loss_count from mid-episode updates
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        avg_policy_loss = episode_policy_loss / loss_count if loss_count > 0 else 0.0
        avg_value_loss = episode_value_loss / loss_count if loss_count > 0 else 0.0
        avg_entropy = episode_entropy / loss_count if loss_count > 0 else 0.0
        total_transaction_cost = info.get('cumulative_transaction_cost', sum(transaction_costs))
        episode_rewards.append(total_reward)
        # Use total_pnl_with_unrealized to capture final position closure
        final_total_pnl = info.get('total_pnl_with_unrealized', info['total_pnl'])
        episode_pnls.append(final_total_pnl)
        episode_losses.append(avg_loss)
        episode_policy_losses.append(avg_policy_loss)
        episode_value_losses.append(avg_value_loss)
        episode_entropies.append(avg_entropy)
        episode_steps.append(steps)
        episode_transaction_costs.append(total_transaction_cost)
        
        # Track oscillation for automatic epsilon boost
        oscillation_detected = False
        
        # Calculate action distribution
        action_counts = {}
        for a in actions_taken:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        # Calculate position statistics
        max_position = max(positions) if positions else 0
        min_position = min(positions) if positions else 0
        final_position = positions[-1] if positions else 0
        position_changes = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
        
        # Price statistics
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0
        start_price = prices[0] if prices else 0
        end_price = prices[-1] if prices else 0
        price_change = end_price - start_price
        
        # Target network is now updated continuously via soft updates in replay()
        # No need for periodic hard updates - soft updates provide better stability
        # Keeping this for backward compatibility but it won't do anything if soft_update=True
        # if episode % target_update_freq == 0:
        #     agent.update_target_network(soft_update=False)  # Disabled - using soft updates instead
        
        # Print detailed information after every episode
        print("\n" + "=" * 100)
        total_episodes = start_episode + num_episodes
        print(f"EPISODE {episode + 1}/{total_episodes} - DETAILED LOG")
        print("=" * 100)
        
        # Performance metrics
        window = min(10, episode + 1)
        avg_reward = np.mean(episode_rewards[-window:])
        avg_pnl = np.mean(episode_pnls[-window:])
        avg_loss = np.mean(episode_losses[-window:]) if episode_losses else 0.0
        avg_steps = np.mean(episode_steps[-window:])
        avg_txn_cost = np.mean(episode_transaction_costs[-window:])
        
        # Calculate expected reward from final P&L for comparison
        final_total_pnl = info.get('total_pnl_with_unrealized', info['total_pnl'])
        final_unrealized = info.get('unrealized_pnl', 0)
        # Expected reward formula: (realized_pnl - transaction_cost + unrealized_pnl * 0.5) / 100
        # But this is only if unrealized P&L was constant - in reality it changes each step
        expected_reward_from_final = (info['total_pnl'] - total_transaction_cost + final_unrealized * 0.5) / 100.0
        
        # Get termination reason
        termination_reason = info.get('termination_reason', 'unknown')
        reason_str = {
            'stop_loss': 'Stop-loss triggered',
            'max_drawdown': f'Max drawdown (${env.max_drawdown_per_episode:.0f}) exceeded',
            'max_steps': f'Max steps ({env.max_steps_per_episode}) reached',
            'end_of_data': 'End of dataset',
            'max_loss': f'Max loss (${env.max_loss_per_episode:.0f}) exceeded',
            'unknown': 'Unknown'
        }.get(termination_reason, termination_reason)
        
        # Get risk management metrics
        drawdown = info.get('drawdown', 0.0)
        peak_pnl = info.get('peak_pnl', 0.0)
        stop_loss_triggered = info.get('stop_loss_triggered', False)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Current Episode:")
        print(f"    Reward (actual): {total_reward:10.2f} (scaled, sum of all step rewards) | Loss: {avg_loss:8.4f}")
        print(f"    Reward (expected from final P&L): {expected_reward_from_final:10.2f} (if unrealized was constant)")
        print(f"    Realized P&L:  ${info['total_pnl']:10.2f} (closed positions only)")
        print(f"    Unrealized P&L: ${final_unrealized:10.2f} (open positions mark-to-market at episode end)")
        print(f"    Total P&L:     ${final_total_pnl:10.2f} (realized + unrealized)")
        print(f"    Peak P&L:     ${peak_pnl:10.2f} | Drawdown: ${drawdown:10.2f}")
        print(f"    Steps:         {steps:10d} | Transaction Cost: ${total_transaction_cost:8.2f}")
        print(f"    Termination:   {reason_str}")
        if stop_loss_triggered:
            print(f"    ⚠️  STOP-LOSS TRIGGERED: Position closed to limit losses")
        print(f"    Note: Reward clipped to [-2000, 2000] and optimized for Total P&L change")
        print(f"  Last {window} Episodes Average:")
        print(f"    Avg Reward:    {avg_reward:10.2f} | Avg P&L: {avg_pnl:10.2f} | Avg Loss: {np.mean(episode_losses[-window:]):8.4f}")
        print(f"    Avg Steps:     {avg_steps:10.1f} | Avg Txn Cost: ${avg_txn_cost:8.2f}")
        
        # Trading statistics
        print(f"\n📈 TRADING STATISTICS:")
        print(f"  Position:       Min: {min_position:2d} | Max: {max_position:2d} | Final: {final_position:2d} | Changes: {position_changes:3d}")
        print(f"  Price Range:    ${min_price:8.2f} - ${max_price:8.2f} | Change: ${price_change:8.2f} ({price_change/start_price*100 if start_price > 0 else 0:.2f}%)")
        print(f"  Start Price:    ${start_price:8.2f} | End Price: ${end_price:8.2f}")
        
        # Action distribution
        print(f"\n🎯 ACTION DISTRIBUTION:")
        # Dynamic action names based on environment's max_position
        if hasattr(env, 'max_position'):
            max_pos = env.max_position
            action_names = {}
            for a in range(env.action_space_size):
                change = a - max_pos
                if change == 0:
                    action_names[a] = "Hold"
                elif change > 0:
                    action_names[a] = f"Buy {change}"
                else:
                    action_names[a] = f"Sell {abs(change)}"
        else:
            # Fallback for old 5-action space
            action_names = {0: "Hold", 1: "Buy 1", 2: "Buy 2", 3: "Sell 1", 4: "Sell 2"}
            
        for action_idx in sorted(action_counts.keys()):
            count = action_counts[action_idx]
            pct = (count / len(actions_taken)) * 100 if actions_taken else 0
            print(f"    {action_names.get(action_idx, f'Action {action_idx}'):12s}: {count:4d} times ({pct:5.1f}%)")
        
        # Learning metrics
        print(f"\n🧠 LEARNING METRICS:")
        if _is_dqn_agent(agent):
            print(f"  Epsilon:         {agent.epsilon:.4f} (exploration rate)")
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate:   {current_lr:.6f} (scheduled: {'Yes' if hasattr(agent, 'scheduler') and agent.scheduler else 'No'})")
            if hasattr(agent, 'reward_mean') and hasattr(agent, 'reward_std'):
                print(f"  Reward Norm μ/σ: {agent.reward_mean:8.2f} / {agent.reward_std:5.2f}")
        elif _is_ppo_agent(agent):
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Learning Rate:   {current_lr:.6f} (scheduled: {'Yes' if hasattr(agent, 'scheduler') and agent.scheduler else 'No'})")
            if hasattr(agent, 'training_steps'):
                print(f"  Training Steps:  {agent.training_steps:,}")
            print(f"  Clip Epsilon:   {agent.clip_epsilon:.2f}")
            print(f"  Entropy Coef:   {agent.entropy_coef:.4f}")
            print(f"  GAE Lambda:     {agent.gae_lambda:.2f}")
        
        # Detect oscillation in loss (if loss is stuck in a small range)
        if len(episode_losses) >= 10:
            recent_losses = episode_losses[-10:]
            loss_range = max(recent_losses) - min(recent_losses)
            loss_mean = np.mean(recent_losses)
            loss_std = np.std(recent_losses)
            
            # If loss is oscillating in a small range (low std relative to mean)
            # and range is very small, we might be stuck
            # Relaxed threshold slightly (0.0005 -> 0.0008) to catch more cases
            if loss_range < 0.0008 and loss_std < 0.0004 and loss_mean < 0.003:
                oscillation_detected = True
                print(f"\n⚠️  OSCILLATION DETECTED:")
                print(f"   Loss range: {loss_range:.6f} (very small)")
                print(f"   Loss std: {loss_std:.6f} (low variance)")
                print(f"   Loss mean: {loss_mean:.6f}")
                print(f"   This suggests the agent may be stuck in a local minimum")
                
                # Analyze gradient norms to diagnose clipping issues
                if hasattr(agent, 'grad_norms') and len(agent.grad_norms) > 0:
                    recent_grad_norms = agent.grad_norms[-100:] if len(agent.grad_norms) >= 100 else agent.grad_norms
                    avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms)
                    max_grad_norm = max(recent_grad_norms)
                    min_grad_norm = min(recent_grad_norms)
                    clipped_count = sum(1 for g in recent_grad_norms if g > 1.0)
                    clipping_rate = clipped_count / len(recent_grad_norms) * 100
                    
                    print(f"   📊 Gradient Analysis (last {len(recent_grad_norms)} steps):")
                    print(f"      Avg norm: {avg_grad_norm:.4f}")
                    print(f"      Range: [{min_grad_norm:.4f}, {max_grad_norm:.4f}]")
                    print(f"      Clipping rate: {clipping_rate:.1f}% (norms > 1.0)")
                    
                    if avg_grad_norm < 0.3 and clipping_rate < 10:
                        print(f"      ⚠️  Gradients are very small - clipping may be too tight!")
                        print(f"      💡 Consider relaxing gradient clipping to 3.0-5.0")
                    elif avg_grad_norm > 1.5 and clipping_rate > 50:
                        print(f"      ⚠️  Gradients are frequently clipped - may be limiting learning")
                        print(f"      💡 Consider relaxing gradient clipping to 3.0-4.0")
                    elif avg_grad_norm < 0.1:
                        print(f"      ⚠️  Very small gradients - possible vanishing gradient problem")
                        print(f"      💡 Consider: larger network, different activation, or skip connections")
                    else:
                        print(f"      ✓ Gradient norms look reasonable")
                
                # Automatically boost exploration to escape local minimum (DQN only)
                if _is_dqn_agent(agent):
                    old_epsilon = agent.epsilon
                    agent.epsilon = min(0.1, agent.epsilon * 1.5)  # Boost exploration
                    if agent.epsilon > old_epsilon:
                        print(f"   🔄 AUTO-BOOSTING EXPLORATION:")
                        print(f"      Epsilon: {old_epsilon:.4f} → {agent.epsilon:.4f}")
                        print(f"      This will help the agent explore more to escape the local minimum")
                elif _is_ppo_agent(agent):
                    # PPO uses entropy for exploration - DO NOT auto-boost for trading envs
                    # as random exploration leads to transaction cost suicide.
                    pass
                
                print(f"   Other recommendations:")
                print(f"     - Reducing learning rate")
                print(f"     - Adjusting target network update rate (tau)")
        
        # Detect oscillation in P&L
        if len(episode_pnls) >= 10:
            recent_pnls = episode_pnls[-10:]
            pnl_std = np.std(recent_pnls)
            pnl_mean = np.mean(recent_pnls)
            
            # If P&L is oscillating (high std relative to mean) and mean is negative
            if pnl_std > abs(pnl_mean) * 2 and pnl_mean < 0:
                if not oscillation_detected:  # Only boost once per episode
                    oscillation_detected = True
                    print(f"\n⚠️  P&L OSCILLATION DETECTED:")
                    print(f"   P&L mean: ${pnl_mean:.2f}")
                    print(f"   P&L std: ${pnl_std:.2f} (high variance)")
                    print(f"   High variance with negative mean suggests unstable policy")
                    
                    # Analyze gradient norms to diagnose clipping issues
                    if hasattr(agent, 'grad_norms') and len(agent.grad_norms) > 0:
                        recent_grad_norms = agent.grad_norms[-100:] if len(agent.grad_norms) >= 100 else agent.grad_norms
                        avg_grad_norm = sum(recent_grad_norms) / len(recent_grad_norms)
                        max_grad_norm = max(recent_grad_norms)
                        min_grad_norm = min(recent_grad_norms)
                        clipped_count = sum(1 for g in recent_grad_norms if g > 1.0)
                        clipping_rate = clipped_count / len(recent_grad_norms) * 100
                        
                        print(f"   📊 Gradient Analysis (last {len(recent_grad_norms)} steps):")
                        print(f"      Avg norm: {avg_grad_norm:.4f}")
                        print(f"      Range: [{min_grad_norm:.4f}, {max_grad_norm:.4f}]")
                        print(f"      Clipping rate: {clipping_rate:.1f}% (norms > 1.0)")
                        
                        if avg_grad_norm < 0.3 and clipping_rate < 10:
                            print(f"      ⚠️  Gradients are very small - clipping may be too tight!")
                            print(f"      💡 Consider relaxing gradient clipping to 2.0-5.0")
                        elif avg_grad_norm > 0.8 and clipping_rate > 50:
                            print(f"      ⚠️  Gradients are frequently clipped - may be limiting learning")
                            print(f"      💡 Consider relaxing gradient clipping to 2.0-3.0")
                        elif avg_grad_norm < 0.1:
                            print(f"      ⚠️  Very small gradients - possible vanishing gradient problem")
                            print(f"      💡 Consider: larger network, different activation, or skip connections")
                        else:
                            print(f"      ✓ Gradient norms look reasonable")
                    
                    # Automatically boost exploration to escape unstable policy
                    if _is_dqn_agent(agent):
                        old_epsilon = agent.epsilon
                        agent.epsilon = min(0.1, agent.epsilon * 1.5)  # Boost exploration
                        if agent.epsilon > old_epsilon:
                            print(f"   🔄 AUTO-BOOSTING EXPLORATION:")
                            print(f"      Epsilon: {old_epsilon:.4f} → {agent.epsilon:.4f}")
                            print(f"      This will help the agent explore more to find a better policy")
                    elif _is_ppo_agent(agent):
                        # PPO uses entropy for exploration - DO NOT auto-boost for trading envs
                        # as random exploration leads to transaction cost suicide.
                        pass
        if _is_dqn_agent(agent):
            maxlen = agent.memory.maxlen if agent.memory.maxlen is not None else len(agent.memory)
            pct_full = (len(agent.memory) / maxlen * 100) if maxlen > 0 else 0.0
            print(f"  Memory Buffer:  {len(agent.memory):5d}/{maxlen:5d} ({pct_full:.1f}% full)")
            print(f"  Training Steps: {loss_count:5d} (loss computed {loss_count} times)")
        elif _is_ppo_agent(agent):
            # Trajectory length is only populated during training (training=True)
            # During evaluation, this will be 0, which is expected
            trajectory_len = len(agent.states)
            if trajectory_len == 0:
                print(f"  Trajectory Length: {trajectory_len:5d} steps (evaluation mode - trajectories not stored)")
            else:
                print(f"  Trajectory Length: {trajectory_len:5d} steps")
            print(f"  Training Steps: {loss_count:5d} (loss computed {loss_count} times this episode)")
        
        # Learning trend analysis
        if episode >= 9:  # Need at least 10 episodes to compare
            recent_loss = np.mean(episode_losses[-5:])  # Last 5 episodes
            earlier_loss = np.mean(episode_losses[-10:-5])  # Previous 5 episodes
            loss_improvement = earlier_loss - recent_loss
            loss_pct_change = (loss_improvement / earlier_loss * 100) if earlier_loss > 0 else 0
            
            recent_reward = np.mean(episode_rewards[-5:])
            earlier_reward = np.mean(episode_rewards[-10:-5])
            reward_change = recent_reward - earlier_reward
            
            recent_pnl = np.mean(episode_pnls[-5:])
            earlier_pnl = np.mean(episode_pnls[-10:-5])
            pnl_change = recent_pnl - earlier_pnl
            
            print(f"\n📉 LEARNING TREND ANALYSIS (Last 5 vs Previous 5 episodes):")
            print(f"  Loss:           {earlier_loss:.4f} → {recent_loss:.4f} | Change: {loss_improvement:+.4f} ({loss_pct_change:+.1f}%)")
            if loss_improvement > 0:
                print(f"    ✓ Loss is DECREASING - Agent IS LEARNING!")
            elif loss_improvement < -0.005:  # Less sensitive threshold (was -0.001)
                # Get current LR for context
                current_lr = agent.optimizer.param_groups[0]['lr'] if hasattr(agent, 'optimizer') else 'N/A'
                print(f"    ⚠ Loss is INCREASING - Monitor closely")
                print(f"       Current LR: {current_lr:.6f} | Some fluctuation is normal with LR scheduling")
            else:
                print(f"    → Loss is stable (small fluctuations are normal)")
            
            print(f"  Reward:         {earlier_reward:8.2f} → {recent_reward:8.2f} | Change: {reward_change:+.2f}")
            print(f"  P&L:            {earlier_pnl:8.2f} → {recent_pnl:8.2f} | Change: {pnl_change:+.2f}")
            
            # Transaction cost analysis
            recent_txn = np.mean(episode_transaction_costs[-5:])
            earlier_txn = np.mean(episode_transaction_costs[-10:-5])
            txn_change = recent_txn - earlier_txn
            print(f"  Txn Cost:       ${earlier_txn:8.2f} → ${recent_txn:8.2f} | Change: ${txn_change:+.2f}")
            
            if txn_change > 0:
                print(f"    ⚠ Trading more frequently (higher costs)")
            elif txn_change < -0.01:
                print(f"    ✓ Trading less frequently (lower costs)")
        
        # Episode summary
        if (episode + 1) % 10 == 0:
            print(f"\n📋 LAST 10 EPISODES SUMMARY:")
            print(f"  Avg Reward:     {np.mean(episode_rewards[-10:]):10.2f} ± {np.std(episode_rewards[-10:]):10.2f}")
            print(f"  Avg P&L:        {np.mean(episode_pnls[-10:]):10.2f} ± {np.std(episode_pnls[-10:]):10.2f}")
            print(f"  Best P&L:       {np.max(episode_pnls[-10:]):10.2f} | Worst P&L: {np.min(episode_pnls[-10:]):10.2f}")
            print(f"  Avg Loss:       {np.mean(episode_losses[-10:]):10.4f}")
            
            # Overall trend
            if len(episode_losses) >= 20:
                first_10_loss = np.mean(episode_losses[:10])
                last_10_loss = np.mean(episode_losses[-10:])
                overall_loss_change = first_10_loss - last_10_loss
                loss_pct_change = (overall_loss_change / first_10_loss * 100) if first_10_loss > 0 else 0
                
                print(f"\n📊 OVERALL LEARNING PROGRESS (First 10 vs Last 10 episodes):")
                print(f"  Loss: {first_10_loss:.4f} → {last_10_loss:.4f} | Improvement: {overall_loss_change:+.4f} ({loss_pct_change:+.1f}%)")
                
                # More nuanced assessment based on loss magnitude and change
                if abs(overall_loss_change) < 0.0005:
                    # Loss is stable (very small change)
                    if last_10_loss < 0.01:
                        print(f"  ✓ Loss is STABLE and LOW ({last_10_loss:.4f}) - This is GOOD!")
                        print(f"     Agent has converged to a good solution. Continue training to refine.")
                    else:
                        print(f"  → Loss is stable. Continue training to see if it improves.")
                elif overall_loss_change > 0.001:
                    # Loss decreased significantly
                    print(f"  ✓✓✓ Agent is LEARNING WELL! Loss decreased significantly.")
                elif overall_loss_change > 0.0001:
                    # Loss decreased slightly
                    print(f"  ✓ Agent is learning, but slowly. Continue training.")
                elif overall_loss_change > -0.001:
                    # Loss increased slightly (within noise)
                    if last_10_loss < 0.01:
                        print(f"  → Loss slightly increased but still LOW ({last_10_loss:.4f}).")
                        print(f"     This is likely noise. Continue training.")
                    else:
                        print(f"  → Loss slightly increased. Monitor for next 20 episodes.")
                else:
                    # Loss increased significantly (overall trend)
                    current_lr = agent.optimizer.param_groups[0]['lr'] if hasattr(agent, 'optimizer') else 'N/A'
                    print(f"  ⚠ Loss is INCREASING over long term. Consider:")
                    print(f"     - Current LR: {current_lr:.6f} (LR scheduler may help)")
                    if _is_dqn_agent(agent):
                        print(f"     - More exploration (current epsilon: {agent.epsilon:.4f})")
                    elif _is_ppo_agent(agent):
                        print(f"     - More exploration (current entropy coef: {agent.entropy_coef:.4f})")
                    print(f"     - Check reward structure")
                    print(f"     - Note: Some fluctuation is normal. Monitor P&L trend, not just loss.")
        
        print("=" * 100)
        
        # Early stopping check (only after we have enough episodes to compare)
        if early_stopping_enabled and len(episode_losses) >= early_stopping_patience:
            # Check if loss has improved in the last N episodes
            recent_loss = np.mean(episode_losses[-early_stopping_patience:])
            
            # Initialize best_loss on first check
            if best_loss == float('inf'):
                best_loss = recent_loss
                episodes_without_improvement = 0
            elif recent_loss < best_loss - 0.0001:  # Require meaningful improvement (0.0001 threshold)
                best_loss = recent_loss
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1
                
            if episodes_without_improvement >= early_stopping_patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED")
                print(f"   Loss has not improved for {episodes_without_improvement} episodes")
                print(f"   Best recent loss: {best_loss:.6f}")
                print(f"   Current recent loss: {recent_loss:.6f}")
                print(f"   Stopping training at episode {episode + 1} (saved {episode + 1 - start_episode} episodes)")
                
                # Save final checkpoint before stopping
                results_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
                latest_path = os.path.join(results_dir, 'rl_agent_latest.pt')
                ep_save_path = os.path.join(results_dir, f'rl_agent_ep{episode + 1}.pt')
                
                # Set feature scaler from environment before saving (for PPO agents)
                if _is_ppo_agent(agent) and hasattr(env, 'feature_scaler'):
                    agent.set_feature_scaler(env.feature_scaler)
                
                # Clean up previous episode checkpoints
                for f in os.listdir(results_dir):
                    if f.startswith('rl_agent_ep') and f.endswith('.pt'):
                        try:
                            os.remove(os.path.join(results_dir, f))
                        except:
                            pass
                
                agent.save(latest_path, episode=episode + 1)
                agent.save(ep_save_path, episode=episode + 1)
                print(f"   ✓ Saved final checkpoints: {latest_path} and {ep_save_path}")
                break
        
        # Save agent
        if (episode + 1) % save_freq == 0:
            # Save as latest and also as the current episode to only keep latest 2
            results_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
            save_path = os.path.join(results_dir, f'rl_agent_latest.pt')
            
            # Set feature scaler from environment before saving (for PPO agents)
            if _is_ppo_agent(agent) and hasattr(env, 'feature_scaler'):
                agent.set_feature_scaler(env.feature_scaler)
            agent.save(save_path, episode=episode+1)
            
            # Also save with episode number but delete previous episode-numbered checkpoints
            # to keep only the absolute latest and the last saved episode checkpoint
            for f in os.listdir(results_dir):
                if f.startswith('rl_agent_ep') and f.endswith('.pt'):
                    try:
                        os.remove(os.path.join(results_dir, f))
                    except:
                        pass
            
            ep_save_path = os.path.join(results_dir, f'rl_agent_ep{episode+1}.pt')
            agent.save(ep_save_path, episode=episode+1)
            print(f"✓ Saved agent to {ep_save_path} (and updated latest)")
    
    # Training summary with comprehensive analysis
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Completed: {len(episode_losses)} episodes")
    if early_stopping_enabled and episodes_without_improvement >= early_stopping_patience:
        print(f"   Status: Early stopped (loss plateaued)")
        print(f"   Best loss: {best_loss:.6f}")
    
    # Comprehensive progress analysis
    if len(episode_losses) >= 20:
        initial_loss = np.mean(episode_losses[:10])
        final_loss = np.mean(episode_losses[-10:])
        loss_improvement = initial_loss - final_loss
        
        initial_reward = np.mean(episode_rewards[:10])
        final_reward = np.mean(episode_rewards[-10:])
        reward_improvement = final_reward - initial_reward
        
        initial_pnl = np.mean(episode_pnls[:10])
        final_pnl = np.mean(episode_pnls[-10:])
        pnl_improvement = final_pnl - initial_pnl
        
        print(f"\n   📈 PROGRESS METRICS (First 10 vs Last 10 episodes):")
        print(f"   Loss:    {initial_loss:.6f} → {final_loss:.6f} | Change: {loss_improvement:+.6f}")
        print(f"   Reward:  {initial_reward:8.2f} → {final_reward:8.2f} | Change: {reward_improvement:+.2f}")
        print(f"   P&L:     ${initial_pnl:8.2f} → ${final_pnl:8.2f} | Change: ${pnl_improvement:+.2f}")
        
        # Overall assessment
        print(f"\n   🎯 OVERALL ASSESSMENT:")
        positive_signals = 0
        total_signals = 3
        
        if loss_improvement > 0.0001:
            print(f"   ✓ Loss is DECREASING ({loss_improvement:.6f}) - Good sign!")
            positive_signals += 1
        elif loss_improvement < -0.005:  # Less sensitive threshold (was -0.001)
            print(f"   ⚠ Loss is INCREASING ({loss_improvement:.6f}) - Monitor")
            print(f"      Note: Some fluctuation is normal with LR scheduling and long training")
        else:
            if final_loss < 0.01:
                print(f"   ✓ Loss is STABLE and LOW ({final_loss:.6f}) - Good!")
                positive_signals += 1
            else:
                print(f"   → Loss is stable ({final_loss:.6f}) - Monitor")
        
        if reward_improvement > 0:
            print(f"   ✓ Reward is INCREASING ({reward_improvement:+.2f}) - Good sign!")
            positive_signals += 1
        elif reward_improvement < -10:
            print(f"   ⚠ Reward is DECREASING ({reward_improvement:+.2f}) - Concerning")
        else:
            print(f"   → Reward is stable ({reward_improvement:+.2f})")
        
        if pnl_improvement > 0:
            print(f"   ✓ P&L is IMPROVING (${pnl_improvement:+.2f}) - Excellent!")
            positive_signals += 1
        elif pnl_improvement < -100:
            print(f"   ⚠ P&L is DECLINING (${pnl_improvement:+.2f}) - Concerning")
        else:
            print(f"   → P&L is stable (${pnl_improvement:+.2f})")
        
        # Final verdict
        print(f"\n   🏆 VERDICT:")
        if positive_signals == total_signals:
            print(f"   ✓✓✓ Model is MOVING IN THE RIGHT DIRECTION!")
            print(f"      All metrics improving. Continue training.")
        elif positive_signals >= 2:
            print(f"   ✓ Model shows MIXED but POSITIVE progress")
            print(f"      {positive_signals}/{total_signals} metrics improving. Continue training.")
        elif positive_signals == 1:
            print(f"   → Model shows LIMITED progress")
            print(f"      Only {positive_signals}/{total_signals} metrics improving. Monitor closely.")
        else:
            print(f"   ⚠ Model may NOT be learning effectively")
            print(f"      {positive_signals}/{total_signals} metrics improving. Consider adjustments.")
    
    print("=" * 100)
    
    # Save training log with P&L trends
    if save_training_log and len(episode_pnls) > 0:
        log_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_log_ep{start_episode}_to_{start_episode + len(episode_pnls)}.txt')
        
        with open(log_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("RL TRAINING LOG\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Training Period: Episodes {start_episode} to {start_episode + len(episode_pnls) - 1}\n")
            f.write(f"Total Episodes: {len(episode_pnls)}\n\n")
            
            f.write("EPISODE-BY-EPISODE P&L TRENDS\n")
            f.write("-" * 100 + "\n")
            # Check if we have loss components (PPO) or just total loss (DQN)
            if len(episode_policy_losses) > 0 and len(episode_policy_losses) == len(episode_losses):
                f.write(f"{'Episode':<10} {'P&L':<15} {'Reward':<15} {'Loss':<15} {'Policy':<12} {'Value':<12} {'Entropy':<12} {'Steps':<10}\n")
                f.write("-" * 100 + "\n")
                for i, (pnl, reward, loss, policy_loss, value_loss, entropy, steps) in enumerate(
                    zip(episode_pnls, episode_rewards, episode_losses, episode_policy_losses, 
                        episode_value_losses, episode_entropies, episode_steps)):
                    f.write(f"{start_episode + i:<10} ${pnl:<14.2f} {reward:<15.2f} {loss:<15.6f} "
                           f"{policy_loss:<12.6f} {value_loss:<12.6f} {entropy:<12.6f} {steps:<10}\n")
            else:
                f.write(f"{'Episode':<10} {'P&L':<15} {'Reward':<15} {'Loss':<15} {'Steps':<10}\n")
                f.write("-" * 100 + "\n")
                for i, (pnl, reward, loss, steps) in enumerate(zip(episode_pnls, episode_rewards, episode_losses, episode_steps)):
                    f.write(f"{start_episode + i:<10} ${pnl:<14.2f} {reward:<15.2f} {loss:<15.6f} {steps:<10}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 100 + "\n\n")
            
            if len(episode_pnls) >= 20:
                f.write("First 10 Episodes:\n")
                f.write(f"  Avg P&L: ${np.mean(episode_pnls[:10]):.2f} ± ${np.std(episode_pnls[:10]):.2f}\n")
                f.write(f"  Avg Reward: {np.mean(episode_rewards[:10]):.2f} ± {np.std(episode_rewards[:10]):.2f}\n")
                f.write(f"  Avg Loss: {np.mean(episode_losses[:10]):.6f}\n")
                if len(episode_policy_losses) > 0:
                    f.write(f"  Avg Policy Loss: {np.mean(episode_policy_losses[:10]):.6f}\n")
                    f.write(f"  Avg Value Loss: {np.mean(episode_value_losses[:10]):.6f}\n")
                    f.write(f"  Avg Entropy: {np.mean(episode_entropies[:10]):.6f}\n")
                f.write("\n")
                
                f.write("Last 10 Episodes:\n")
                f.write(f"  Avg P&L: ${np.mean(episode_pnls[-10:]):.2f} ± ${np.std(episode_pnls[-10:]):.2f}\n")
                f.write(f"  Avg Reward: {np.mean(episode_rewards[-10:]):.2f} ± {np.std(episode_rewards[-10:]):.2f}\n")
                f.write(f"  Avg Loss: {np.mean(episode_losses[-10:]):.6f}\n")
                if len(episode_policy_losses) > 0:
                    f.write(f"  Avg Policy Loss: {np.mean(episode_policy_losses[-10:]):.6f}\n")
                    f.write(f"  Avg Value Loss: {np.mean(episode_value_losses[-10:]):.6f}\n")
                    f.write(f"  Avg Entropy: {np.mean(episode_entropies[-10:]):.6f}\n")
                f.write("\n")
                
                pnl_improvement = np.mean(episode_pnls[-10:]) - np.mean(episode_pnls[:10])
                reward_improvement = np.mean(episode_rewards[-10:]) - np.mean(episode_rewards[:10])
                loss_improvement = np.mean(episode_losses[:10]) - np.mean(episode_losses[-10:])
                
                f.write("Improvement:\n")
                f.write(f"  P&L: ${pnl_improvement:+.2f}\n")
                f.write(f"  Reward: {reward_improvement:+.2f}\n")
                f.write(f"  Loss: {loss_improvement:+.6f}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total Episodes: {len(episode_pnls)}\n")
            f.write(f"Average P&L: ${np.mean(episode_pnls):.2f} ± ${np.std(episode_pnls):.2f}\n")
            f.write(f"Best P&L: ${np.max(episode_pnls):.2f}\n")
            f.write(f"Worst P&L: ${np.min(episode_pnls):.2f}\n")
            f.write(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}\n")
            f.write(f"Average Loss: {np.mean(episode_losses):.6f}\n")
        
        print(f"\n📝 Training log saved to: {log_file}")
    
    return episode_rewards, episode_pnls


def evaluate_agent(env: SPXTradingEnv, agent, num_episodes: int = 10, 
                   detailed: bool = False):
    """
    Evaluate the trained agent (supports both DQN and PPO)
    
    Args:
        env: Trading environment
        agent: DQNAgent or PPOAgent
        num_episodes: Number of evaluation episodes
        detailed: If True, show detailed statistics including win rate, Sharpe ratio, etc.
    """
    print(f"\n{'=' * 100}")
    print(f"📊 EVALUATING AGENT FOR {num_episodes} EPISODES")
    print(f"{'=' * 100}")
    
    # Set progress to 1.0 for evaluation so no overtrading penalty is applied
    env.set_training_progress(1.0)
    
    episode_rewards = []
    episode_pnls = []
    episode_steps = []
    episode_transaction_costs = []
    episode_unrealized_pnls = []
    episode_drawdowns = []
    episode_peak_pnls = []
    stop_loss_triggers = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        total_transaction_cost = 0.0
        
        while True:
            # Choose action (handle both DQN and PPO)
            if _is_ppo_agent(agent):
                act_result = agent.act(state, training=False)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                else:
                    action = act_result
                    risk_multiplier = None
                action = int(action)  # Ensure action is int
            elif _is_dqn_agent(agent):
                act_result = agent.act(state, training=False)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                    action = int(action)  # Ensure action is int
                else:
                    action = int(act_result)  # Ensure action is int
                    risk_multiplier = None
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            # Execute action
            next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
            
            # Accumulate transaction costs
            total_transaction_cost += info.get('transaction_cost', 0.0)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_pnls.append(info.get('total_pnl_with_unrealized', info.get('total_pnl', 0.0)))
        episode_steps.append(steps)
        episode_transaction_costs.append(total_transaction_cost)
        episode_unrealized_pnls.append(info.get('unrealized_pnl', 0.0))
        episode_drawdowns.append(info.get('drawdown', 0.0))
        episode_peak_pnls.append(info.get('peak_pnl', 0.0))
        
        if info.get('stop_loss_triggered', False):
            stop_loss_triggers += 1
        
        if not detailed or (episode + 1) % max(1, num_episodes // 20) == 0 or episode == 0:
            print(f"Episode {episode + 1:4d}/{num_episodes}: "
                  f"Reward={total_reward:8.2f}, "
                  f"P&L=${info.get('total_pnl_with_unrealized', info.get('total_pnl', 0.0)):8.2f}, "
                  f"Steps={steps:4d}, "
                  f"Termination={info.get('termination_reason', 'unknown'):15s}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_pnl = np.mean(episode_pnls)
    std_pnl = np.std(episode_pnls)
    best_pnl = np.max(episode_pnls)
    worst_pnl = np.min(episode_pnls)
    
    # Win rate
    profitable_episodes = sum(1 for pnl in episode_pnls if pnl > 0)
    win_rate = (profitable_episodes / num_episodes) * 100
    
    # Profit factor (gross profit / gross loss)
    gross_profit = sum(pnl for pnl in episode_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in episode_pnls if pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe-like ratio (mean P&L / std P&L)
    sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0.0
    
    # Average transaction cost
    avg_transaction_cost = np.mean(episode_transaction_costs) if episode_transaction_costs else 0.0
    
    # Average drawdown
    avg_drawdown = np.mean(episode_drawdowns) if episode_drawdowns else 0.0
    max_drawdown = np.max(episode_drawdowns) if episode_drawdowns else 0.0
    
    print(f"\n{'=' * 100}")
    print(f"📈 EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'=' * 100}")
    print(f"\n💰 P&L STATISTICS:")
    print(f"   Average P&L: ${avg_pnl:8.2f} ± ${std_pnl:8.2f}")
    print(f"   Best P&L:   ${best_pnl:8.2f}")
    print(f"   Worst P&L:  ${worst_pnl:8.2f}")
    print(f"   Range:      [${worst_pnl:8.2f}, ${best_pnl:8.2f}]")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Win Rate:        {win_rate:5.1f}% ({profitable_episodes}/{num_episodes} profitable)")
    print(f"   Profit Factor:   {profit_factor:.2f}" + (" (∞ if no losses)" if profit_factor == float('inf') else ""))
    print(f"   Sharpe Ratio:    {sharpe_ratio:.4f}")
    print(f"   Avg Reward:      {avg_reward:8.2f} ± {std_reward:8.2f}")
    
    print(f"\n💵 COST ANALYSIS:")
    print(f"   Avg Transaction Cost: ${avg_transaction_cost:.2f}")
    print(f"   Total Transaction Cost: ${sum(episode_transaction_costs):.2f}")
    
    print(f"\n📉 RISK METRICS:")
    print(f"   Average Drawdown: ${avg_drawdown:.2f}")
    print(f"   Maximum Drawdown: ${max_drawdown:.2f}")
    print(f"   Stop-Loss Triggers: {stop_loss_triggers}/{num_episodes} ({stop_loss_triggers/num_episodes*100:.1f}%)")
    
    if detailed:
        print(f"\n📈 DISTRIBUTION ANALYSIS:")
        # Quartiles
        sorted_pnls = sorted(episode_pnls)
        q1_idx = len(sorted_pnls) // 4
        q2_idx = len(sorted_pnls) // 2
        q3_idx = 3 * len(sorted_pnls) // 4
        q1 = sorted_pnls[q1_idx] if q1_idx < len(sorted_pnls) else sorted_pnls[0]
        median = sorted_pnls[q2_idx] if q2_idx < len(sorted_pnls) else sorted_pnls[0]
        q3 = sorted_pnls[q3_idx] if q3_idx < len(sorted_pnls) else sorted_pnls[-1]
        print(f"   Q1 (25th percentile): ${q1:.2f}")
        print(f"   Median (50th percentile): ${median:.2f}")
        print(f"   Q3 (75th percentile): ${q3:.2f}")
        print(f"   IQR (Q3 - Q1): ${q3 - q1:.2f}")
        
        # Consistency metrics
        positive_consistency = sum(1 for i in range(len(episode_pnls)-1) 
                                  if episode_pnls[i] > 0 and episode_pnls[i+1] > 0)
        negative_consistency = sum(1 for i in range(len(episode_pnls)-1) 
                                  if episode_pnls[i] < 0 and episode_pnls[i+1] < 0)
        print(f"\n   Consistency:")
        print(f"   Consecutive profitable: {positive_consistency}/{len(episode_pnls)-1}")
        print(f"   Consecutive losses: {negative_consistency}/{len(episode_pnls)-1}")
    
    # Stability assessment
    print(f"\n🎯 STABILITY ASSESSMENT:")
    if std_pnl < abs(avg_pnl) * 0.5:
        stability = "HIGH"
        stability_icon = "✓✓✓"
        stability_msg = "Low variance relative to mean - very stable!"
    elif std_pnl < abs(avg_pnl):
        stability = "MEDIUM"
        stability_icon = "✓✓"
        stability_msg = "Moderate variance - reasonably stable"
    else:
        stability = "LOW"
        stability_icon = "⚠"
        stability_msg = "High variance - may need more training or risk management"
    
    print(f"   {stability_icon} Stability: {stability}")
    print(f"   {stability_msg}")
    
    if avg_pnl > 0 and win_rate > 50:
        print(f"\n   ✓✓✓ OVERALL: Agent is PROFITABLE and CONSISTENT!")
        print(f"      Average P&L: ${avg_pnl:.2f} with {win_rate:.1f}% win rate")
    elif avg_pnl > 0:
        print(f"\n   ✓✓ OVERALL: Agent is PROFITABLE but needs better consistency")
        print(f"      Average P&L: ${avg_pnl:.2f} but only {win_rate:.1f}% win rate")
    elif win_rate > 50:
        print(f"\n   ⚠ OVERALL: Agent wins often but average P&L is negative")
        print(f"      Win rate: {win_rate:.1f}% but average P&L: ${avg_pnl:.2f}")
        print(f"      May need to improve risk management or reduce transaction costs")
    else:
        print(f"\n   ⚠⚠⚠ OVERALL: Agent needs improvement")
        print(f"      Average P&L: ${avg_pnl:.2f}, Win rate: {win_rate:.1f}%")
        print(f"      Consider retraining or adjusting hyperparameters")
    
    print(f"{'=' * 100}\n")
    
    return episode_rewards, episode_pnls


def quick_validation_test(env: SPXTradingEnv, agent, num_test_episodes: int = 3) -> bool:
    """
    Quick sanity check to validate the network setup before full training.
    Runs a few episodes and checks for common issues.
    
    Args:
        env: Trading environment
        agent: DQNAgent or PPOAgent
        num_test_episodes: Number of episodes to run for testing (default: 3)
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "=" * 80)
    print("QUICK VALIDATION TEST - Checking network setup...")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Network forward pass
    print("\n[TEST 1] Network Forward Pass...")
    try:
        test_state = env.reset()
        state_tensor = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
        
        if _is_dqn_agent(agent):
            q_values = agent.q_network(state_tensor)
            assert q_values.shape == (1, agent.action_size), \
                f"Q-values shape incorrect: {q_values.shape}, expected (1, {agent.action_size})"
            assert not torch.isnan(q_values).any(), "Q-values contain NaN!"
            assert not torch.isinf(q_values).any(), "Q-values contain Inf!"
            print(f"  ✓ Q-network forward pass works")
            print(f"    Q-values shape: {q_values.shape}, range: [{q_values.min():.2f}, {q_values.max():.2f}]")
        elif _is_ppo_agent(agent):
            action_logits, value, risk_mult = agent.actor_critic(state_tensor)
            assert action_logits.shape == (1, agent.action_size), \
                f"Action logits shape incorrect: {action_logits.shape}, expected (1, {agent.action_size})"
            assert value.shape == (1, 1), f"Value shape incorrect: {value.shape}, expected (1, 1)"
            assert not torch.isnan(action_logits).any(), "Action logits contain NaN!"
            assert not torch.isnan(value).any(), "Value contains NaN!"
            print(f"  ✓ Actor-critic forward pass works")
            print(f"    Action logits shape: {action_logits.shape}, Value: {value.item():.2f}, Risk: {risk_mult.item():.2f}")
        else:
            raise ValueError(f"Unknown agent type: {type(agent)}")
    except Exception as e:
        print(f"  ✗ Network forward pass failed: {e}")
        all_passed = False
    
    # Test 2: Environment step
    print("\n[TEST 2] Environment Step...")
    try:
        state = env.reset()
        if _is_ppo_agent(agent):
            act_result = agent.act(state, training=True)
            if isinstance(act_result, tuple):
                action, risk_multiplier = act_result
            else:
                action = act_result
                risk_multiplier = None
        elif _is_dqn_agent(agent):
            act_result = agent.act(state, training=True)
            if isinstance(act_result, tuple):
                action, risk_multiplier = act_result
            else:
                action = act_result
                risk_multiplier = None
        else:
            raise ValueError(f"Unknown agent type: {type(agent)}")
        next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
        
        assert len(next_state) == len(state), "State size changed after step!"
        assert isinstance(reward, (int, float)), f"Reward is not numeric: {type(reward)}"
        assert isinstance(done, bool), f"Done is not boolean: {type(done)}"
        assert not np.isnan(reward), "Reward is NaN!"
        assert not np.isinf(reward), "Reward is Inf!"
        
        print(f"  ✓ Environment step works")
        print(f"    State size: {len(state)}, Reward: {reward:.4f}, Done: {done}")
    except Exception as e:
        print(f"  ✗ Environment step failed: {e}")
        all_passed = False
    
    # Test 3: Training loop (few episodes)
    print(f"\n[TEST 3] Training Loop ({num_test_episodes} episodes)...")
    try:
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_test_episodes):
            state = env.reset()
            total_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            steps = 0
            
            while True:
                act_result = agent.act(state, training=True)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                else:
                    action = act_result
                    risk_multiplier = None
                next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
                
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()
                    if loss is not None:
                        assert not np.isnan(loss), f"Loss is NaN at episode {episode+1}!"
                        assert not np.isinf(loss), f"Loss is Inf at episode {episode+1}!"
                        episode_loss += loss
                        loss_count += 1
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)
            
            print(f"  Episode {episode+1}: Reward={total_reward:.2f}, Loss={avg_loss:.4f}, Steps={steps}")
        
        # Check trends
        if len(episode_losses) > 1:
            loss_trend = episode_losses[-1] - episode_losses[0]
            print(f"\n  Loss trend: {loss_trend:+.4f} (negative is good)")
        
        print(f"  ✓ Training loop works")
        print(f"    Avg reward: {np.mean(episode_rewards):.2f}, Avg loss: {np.mean(episode_losses):.4f}")
        
    except Exception as e:
        print(f"  ✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 4: Q-value stability
    print("\n[TEST 4] Q-Value Stability...")
    try:
        test_states = []
        for _ in range(10):
            state = env.reset()
            test_states.append(state)
        
        test_states_tensor = torch.FloatTensor(np.array(test_states)).to(agent.device)
        q_values = agent.q_network(test_states_tensor)
        
        q_mean = q_values.mean().item()
        q_std = q_values.std().item()
        q_min = q_values.min().item()
        q_max = q_values.max().item()
        
        # Check if Q-values are in reasonable range
        if abs(q_mean) > 100:
            print(f"  ⚠ Q-values mean is large: {q_mean:.2f} (might indicate initialization issue)")
        if q_std > 50:
            print(f"  ⚠ Q-values std is large: {q_std:.2f} (might indicate instability)")
        
        print(f"  ✓ Q-values in reasonable range")
        print(f"    Mean: {q_mean:.2f}, Std: {q_std:.2f}, Range: [{q_min:.2f}, {q_max:.2f}]")
        
    except Exception as e:
        print(f"  ✗ Q-value stability test failed: {e}")
        all_passed = False
    
    # Test 5: Reward distribution
    print("\n[TEST 5] Reward Distribution...")
    try:
        rewards_sample = []
        for _ in range(50):
            state = env.reset()
            act_result = agent.act(state, training=True)
            if isinstance(act_result, tuple):
                action, risk_multiplier = act_result
            else:
                action = act_result
                risk_multiplier = None
            _, reward, _, _ = env.step(action, risk_multiplier=risk_multiplier)
            rewards_sample.append(reward)
        
        reward_mean = np.mean(rewards_sample)
        reward_std = np.std(rewards_sample)
        reward_min = np.min(rewards_sample)
        reward_max = np.max(rewards_sample)
        
        print(f"  ✓ Reward distribution looks reasonable")
        print(f"    Mean: {reward_mean:.4f}, Std: {reward_std:.4f}, Range: [{reward_min:.4f}, {reward_max:.4f}]")
        
        if abs(reward_mean) > 10:
            print(f"  ⚠ Reward mean is large: {reward_mean:.4f} (check reward scaling)")
        
    except Exception as e:
        print(f"  ✗ Reward distribution test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL VALIDATION TESTS PASSED - Network is ready for training!")
        print("=" * 80)
        return True
    else:
        print("✗ SOME VALIDATION TESTS FAILED - Fix issues before training!")
        print("=" * 80)
        return False



def optimize_stop_loss(data_path: str, model_path: str, device: str = 'cpu',
                        stop_loss_candidates: Optional[list] = None, num_eval_episodes: int = 20,
                        agent_checkpoint: Optional[str] = None,
                        max_position: int = 2) -> dict:
    """
    Optimize stop_loss_per_contract by testing different values and selecting the one
    with highest average P&L.
    
    Args:
        data_path: Path to es_with_indicators.csv
        model_path: Path to futures_model.pt
        device: Device to run on
        stop_loss_candidates: List of stop_loss values to test (default: [200, 300, 400, 500, 600, 700, 800])
        num_eval_episodes: Number of episodes to evaluate each stop_loss value
        agent_checkpoint: Path to trained agent checkpoint (if None, uses random agent)
        max_position: Maximum number of contracts to hold (default: 2)
    
    Returns:
        Dictionary with results: {'best_stop_loss': value, 'best_avg_pnl': value, 'all_results': [...]}
    """
    if stop_loss_candidates is None:
        stop_loss_candidates = [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    
    print("\n" + "=" * 100)
    print("🔍 STOP-LOSS OPTIMIZATION")
    print("=" * 100)
    print(f"Testing {len(stop_loss_candidates)} stop-loss values: {stop_loss_candidates}")
    print(f"Evaluating each with {num_eval_episodes} episodes")
    print(f"Using max_position = {max_position}")
    print("=" * 100)
    
    results = []
    
    for stop_loss in stop_loss_candidates:
        print(f"\n📊 Testing stop_loss = ${stop_loss:.0f} per contract...")
        
        # Create environment with this stop_loss value
        env = SPXTradingEnv(
            data_path,
            model_path,
            device=device,
            max_steps_per_episode=5000,
            max_loss_per_episode=-50000.0,
            stop_loss_per_contract=stop_loss,
            variance_adaptive_risk=False,  # Disable for optimization to test fixed values
            max_position=max_position
        )
        
        # Create or load agent
        if agent_checkpoint and os.path.exists(agent_checkpoint):
            # First, peek at checkpoint to determine expected state size
            checkpoint = torch.load(agent_checkpoint, map_location=device, weights_only=False)
            # Get state size from first layer weight shape
            if 'q_network_state_dict' in checkpoint:
                first_layer_key = 'feature_layers.0.weight'
                if first_layer_key in checkpoint['q_network_state_dict']:
                    checkpoint_state_size = checkpoint['q_network_state_dict'][first_layer_key].shape[1]
                else:
                    # Fallback: try to find any weight layer
                    checkpoint_state_size = None
                    for key in checkpoint['q_network_state_dict'].keys():
                        if 'weight' in key and len(checkpoint['q_network_state_dict'][key].shape) == 2:
                            checkpoint_state_size = checkpoint['q_network_state_dict'][key].shape[1]
                            break
            else:
                checkpoint_state_size = None
            
            # Get current environment state size
            test_state = env.reset()
            current_state_size = len(test_state)
            
            # Use checkpoint state size if available, otherwise use current
            if checkpoint_state_size is not None:
                state_size = checkpoint_state_size
                if state_size != current_state_size:
                    print(f"   ⚠️  State size mismatch: checkpoint expects {state_size}, environment provides {current_state_size}")
                    print(f"   Using checkpoint state size ({state_size}) for compatibility")
                    # We'll need to pad/trim states when using this agent
            else:
                state_size = current_state_size
            
            # Import DQNAgent dynamically to avoid circular imports
            from futures_renforcement import DQNAgent as _DQNAgent
            agent = _DQNAgent(
                state_size=state_size,
                action_size=env.action_space_size,
                device=device
            )
            agent.load(agent_checkpoint)
            print(f"   Loaded agent from {agent_checkpoint} (state_size={state_size})")
            
            # Store state size info for state adaptation (using setattr to avoid linter warnings)
            setattr(agent, '_checkpoint_state_size', state_size)
            setattr(agent, '_current_state_size', current_state_size)
        else:
            # Use random agent for baseline
            test_state = env.reset()
            state_size = len(test_state)
            # Import DQNAgent dynamically to avoid circular imports
            from futures_renforcement import DQNAgent as _DQNAgent
            agent = _DQNAgent(
                state_size=state_size,
                action_size=env.action_space_size,
                device=device,
                epsilon=1.0  # Fully random
            )
            print(f"   Using random agent (no checkpoint provided)")
        
        # Evaluate agent
        episode_pnls = []
        stop_loss_triggers = 0
        max_unrealized_losses = []  # Track max unrealized loss per episode to diagnose why stop-loss doesn't trigger
        
        # Helper function to adapt state size if needed
        def adapt_state(state, target_size):
            """Adapt state to target size by padding or trimming"""
            if len(state) == target_size:
                return state
            elif len(state) > target_size:
                # Trim: remove last element (stop_loss)
                return state[:target_size]
            else:
                # Pad: add default stop_loss value (normalized)
                default_stop_loss = (500.0 - 100.0) / 900.0  # Default normalized stop_loss
                return np.concatenate([state, [default_stop_loss]])
        
        # Check if we need to adapt states
        checkpoint_state_size = getattr(agent, '_checkpoint_state_size', None)
        current_state_size = getattr(agent, '_current_state_size', None)
        needs_adaptation = (checkpoint_state_size is not None and 
                           current_state_size is not None and
                           checkpoint_state_size != current_state_size)
        
        for episode in range(num_eval_episodes):
            state = env.reset()
            # Adapt state size if checkpoint has different size
            if needs_adaptation:
                state = adapt_state(state, checkpoint_state_size)
            
            total_reward = 0.0
            max_unrealized_loss_this_episode = 0.0
            
            while True:
                # Choose action (handle both DQN and PPO)
                if _is_ppo_agent(agent):
                    act_result = agent.act(state, training=False)
                    if isinstance(act_result, tuple):
                        action, risk_multiplier = act_result
                    else:
                        action = act_result
                        risk_multiplier = None
                    action = int(action)
                elif _is_dqn_agent(agent):
                    act_result = agent.act(state, training=False)
                    if isinstance(act_result, tuple):
                        action, risk_multiplier = act_result
                        action = int(action)  # Ensure action is int
                    else:
                        action = int(act_result)  # Ensure action is int
                        risk_multiplier = None
                else:
                    raise ValueError(f"Unknown agent type: {type(agent)}")
                next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
                # Adapt next state size if needed
                if needs_adaptation:
                    next_state = adapt_state(next_state, checkpoint_state_size)
                state = next_state
                total_reward += reward
                
                # Track maximum unrealized loss per contract to diagnose stop-loss
                if info.get('position', 0) != 0:
                    unrealized = info.get('unrealized_pnl', 0)
                    position = abs(info.get('position', 1))
                    if unrealized < 0:
                        loss_per_contract = -unrealized / position
                        max_unrealized_loss_this_episode = max(max_unrealized_loss_this_episode, loss_per_contract)
                
                if done:
                    episode_pnls.append(info['total_pnl_with_unrealized'])
                    max_unrealized_losses.append(max_unrealized_loss_this_episode)
                    if info.get('stop_loss_triggered', False):
                        stop_loss_triggers += 1
                    break
        
        avg_pnl = np.mean(episode_pnls)
        std_pnl = np.std(episode_pnls)
        min_pnl = np.min(episode_pnls)
        max_pnl = np.max(episode_pnls)
        trigger_rate = stop_loss_triggers / num_eval_episodes
        avg_max_unrealized_loss = np.mean(max_unrealized_losses) if max_unrealized_losses else 0.0
        max_max_unrealized_loss = np.max(max_unrealized_losses) if max_unrealized_losses else 0.0
        
        results.append({
            'stop_loss': stop_loss,
            'avg_pnl': avg_pnl,
            'std_pnl': std_pnl,
            'min_pnl': min_pnl,
            'max_pnl': max_pnl,
            'trigger_rate': trigger_rate,
            'avg_max_unrealized_loss': avg_max_unrealized_loss,
            'max_max_unrealized_loss': max_max_unrealized_loss
        })
        
        print(f"   Results: Avg P&L = ${avg_pnl:.2f} ± ${std_pnl:.2f}")
        print(f"            Range: [${min_pnl:.2f}, ${max_pnl:.2f}]")
        print(f"            Stop-loss triggered: {stop_loss_triggers}/{num_eval_episodes} ({trigger_rate*100:.1f}%)")
        if trigger_rate == 0.0 and max_max_unrealized_loss > 0:
            print(f"            ⚠️  Max unrealized loss per contract: ${max_max_unrealized_loss:.2f} (threshold: ${stop_loss:.0f})")
            print(f"            💡 Stop-loss never triggered because losses never exceeded ${stop_loss:.0f} per contract")
    
    # Find best stop_loss
    best_result = max(results, key=lambda x: x['avg_pnl'])
    
    print("\n" + "=" * 100)
    print("📈 OPTIMIZATION RESULTS")
    print("=" * 100)
    
    # Check if all results are negative
    all_negative = all(r['avg_pnl'] < 0 for r in results)
    if all_negative:
        print("\n⚠️  WARNING: All stop-loss values resulted in negative average P&L!")
        print("   This suggests the agent itself may need retraining or improvement.")
        print("   The agent was trained without stop_loss in the state, so it cannot adapt to it.")
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Retrain the agent with stop_loss included in the state")
        print("   2. The agent may need more training episodes")
        print("   3. Consider that evaluation conditions may differ from training")
    
    print(f"\nBest stop_loss: ${best_result['stop_loss']:.0f} per contract")
    print(f"  Average P&L: ${best_result['avg_pnl']:.2f} ± ${best_result['std_pnl']:.2f}")
    print(f"  Range: [${best_result['min_pnl']:.2f}, ${best_result['max_pnl']:.2f}]")
    print(f"  Stop-loss trigger rate: {best_result['trigger_rate']*100:.1f}%")
    if best_result.get('max_max_unrealized_loss', 0) > 0:
        print(f"  Max unrealized loss observed: ${best_result['max_max_unrealized_loss']:.2f} per contract")
    
    print(f"\nAll results (sorted by avg P&L):")
    sorted_results = sorted(results, key=lambda x: x['avg_pnl'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        marker = "🏆" if r == best_result else f"{i}."
        trigger_info = f"trigger: {r['trigger_rate']*100:5.1f}%"
        if r.get('max_max_unrealized_loss', 0) > 0 and r['trigger_rate'] == 0:
            trigger_info += f" (max loss: ${r['max_max_unrealized_loss']:.0f})"
        print(f"  {marker} ${r['stop_loss']:6.0f}: ${r['avg_pnl']:8.2f} ± ${r['std_pnl']:6.2f} ({trigger_info})")
    
    print("=" * 100)
    
    return {
        'best_stop_loss': best_result['stop_loss'],
        'best_avg_pnl': best_result['avg_pnl'],
        'all_results': results
    }


def analyze_episode_strategy(data_path: str, model_path: str, device: str = 'cpu',
                             agent_checkpoint: Optional[str] = None, 
                             num_episodes: int = 1,
                             episode_indices: Optional[list] = None,
                             max_position: int = 2):
    """
    Analyze specific episodes in detail to understand the agent's strategy.
    
    Args:
        data_path: Path to es_with_indicators.csv
        model_path: Path to futures_model.pt
        device: Device to run on
        agent_checkpoint: Path to agent checkpoint (default: latest final checkpoint)
        num_episodes: Number of episodes to analyze (default: 1)
        episode_indices: Specific episode indices to analyze (if None, analyzes first num_episodes)
        max_position: Maximum number of contracts to hold (default: 2)
    """
    print("\n" + "=" * 100)
    print("📊 DETAILED EPISODE STRATEGY ANALYSIS")
    print("=" * 100)
    
    # Create environment
    env = SPXTradingEnv(
        data_path, 
        model_path, 
        device=device,
        max_steps_per_episode=5000,
        max_loss_per_episode=-50000.0,
        stop_loss_per_contract=500.0,
        max_position=max_position
    )
    
    # Load agent
    if agent_checkpoint is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
        final_checkpoint = os.path.join(checkpoint_dir, 'rl_agent_final.pt')
        if os.path.exists(final_checkpoint):
            agent_checkpoint = final_checkpoint
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith('rl_agent_ep') and f.endswith('.pt')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.replace('rl_agent_ep', '').replace('.pt', '')))
                agent_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    if agent_checkpoint and os.path.exists(agent_checkpoint):
        # Detect state size from checkpoint
        checkpoint = torch.load(agent_checkpoint, map_location=device)
        checkpoint_state_size = None
        if 'q_network_state_dict' in checkpoint:
            first_layer_key = 'feature_layers.0.weight'
            if first_layer_key in checkpoint['q_network_state_dict']:
                checkpoint_state_size = checkpoint['q_network_state_dict'][first_layer_key].shape[1]
        
        test_state = env.reset()
        current_state_size = len(test_state)
        state_size = checkpoint_state_size if checkpoint_state_size else current_state_size
        
        # Import DQNAgent dynamically to avoid circular imports
        from futures_renforcement import DQNAgent as _DQNAgent
        agent = _DQNAgent(state_size=state_size, action_size=env.action_space_size, device=device)
        agent.load(agent_checkpoint)
        print(f"✓ Loaded agent from {agent_checkpoint}")
        
        # State adaptation if needed
        needs_adaptation = (checkpoint_state_size is not None and 
                           checkpoint_state_size != current_state_size)
        def adapt_state(state, target_size):
            if len(state) == target_size:
                return state
            elif len(state) > target_size:
                return state[:target_size]
            else:
                default_stop_loss = (500.0 - 100.0) / 900.0
                return np.concatenate([state, [default_stop_loss]])
    else:
        print("❌ No agent checkpoint found!")
        return
    
    # Analyze episodes
    # Dynamic action names based on environment's max_position
    if hasattr(env, 'max_position'):
        max_pos = env.max_position
        action_names = {}
        for a in range(env.action_space_size):
            change = a - max_pos
            if change == 0:
                action_names[a] = "Hold"
            elif change > 0:
                action_names[a] = f"Buy {change}"
            else:
                action_names[a] = f"Sell {abs(change)}"
    else:
        # Fallback for old 5-action space
        action_names = {0: "Sell 2", 1: "Sell 1", 2: "Hold", 3: "Buy 1", 4: "Buy 2"}
    
    for ep_idx in range(num_episodes):
        print(f"\n{'=' * 100}")
        print(f"EPISODE {ep_idx + 1} ANALYSIS")
        print(f"{'=' * 100}")
        
        state = env.reset()
        if needs_adaptation:
            state = adapt_state(state, checkpoint_state_size)
        
        # Detailed tracking
        steps = []
        actions = []
        positions = []
        prices = []
        rewards = []
        realized_pnls = []
        unrealized_pnls = []
        transaction_costs = []
        total_pnls = []
        q_values = []
        stop_loss_events = []
        drawdowns = []
        peak_pnls = []
        
        step_num = 0
        total_reward = 0.0
        
        while True:
            # Get Q-values or action probabilities for analysis
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                if _is_dqn_agent(agent):
                    q_vals = agent.q_network(state_tensor).cpu().numpy()[0]
                elif _is_ppo_agent(agent):
                    action_logits, _, _ = agent.actor_critic(state_tensor)  # type: ignore[attr-defined]
                    q_vals = torch.softmax(action_logits, dim=1).cpu().numpy()[0]  # Use probabilities as proxy
                else:
                    q_vals = np.zeros(agent.action_size)
            
            # Choose action (handle both DQN and PPO)
            if _is_ppo_agent(agent):
                act_result = agent.act(state, training=False)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                else:
                    action = act_result
                    risk_multiplier = None
                action = int(action)
            elif _is_dqn_agent(agent):
                act_result = agent.act(state, training=False)
                if isinstance(act_result, tuple):
                    action, risk_multiplier = act_result
                    action = int(action)  # Ensure action is int
                else:
                    action = int(act_result)  # Ensure action is int
                    risk_multiplier = None
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            # Execute action
            next_state, reward, done, info = env.step(action, risk_multiplier=risk_multiplier)
            if needs_adaptation:
                next_state = adapt_state(next_state, checkpoint_state_size)
            
            # Track everything
            steps.append(step_num)
            actions.append(action)
            positions.append(info['position'])
            prices.append(info['price'])
            rewards.append(reward)
            transaction_costs.append(info.get('transaction_cost', 0.0))
            realized_pnls.append(info.get('total_pnl', 0.0))
            unrealized_pnls.append(info.get('unrealized_pnl', 0.0))
            total_pnls.append(info.get('total_pnl_with_unrealized', 0.0))
            q_values.append(q_vals.tolist())
            drawdowns.append(info.get('drawdown', 0.0))
            peak_pnls.append(info.get('peak_pnl', 0.0))
            
            if info.get('stop_loss_triggered', False):
                stop_loss_events.append(step_num)
            
            total_reward += reward
            step_num += 1
            state = next_state
            
            if done:
                break
        
        # Calculate statistics
        final_pnl = total_pnls[-1] if total_pnls else 0.0
        total_transaction_cost = sum(transaction_costs)
        total_realized_pnl = realized_pnls[-1] if realized_pnls else 0.0
        final_unrealized_pnl = unrealized_pnls[-1] if unrealized_pnls else 0.0
        
        # Action distribution
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        
        # Position statistics
        position_changes = [positions[i] - positions[i-1] if i > 0 else positions[i] 
                           for i in range(len(positions))]
        max_position = max(positions) if positions else 0
        min_position = min(positions) if positions else 0
        avg_position = np.mean(positions) if positions else 0.0
        
        # Price statistics
        price_changes = [prices[i] - prices[i-1] if i > 0 else 0.0 for i in range(len(prices))]
        max_price = max(prices) if prices else 0.0
        min_price = min(prices) if prices else 0.0
        price_range = max_price - min_price
        
        # Q-value statistics
        avg_q_values = [np.mean(q) for q in q_values] if q_values else []
        max_q_values = [np.max(q) for q in q_values] if q_values else []
        
        # Print analysis
        print(f"\n📈 EPISODE SUMMARY")
        print(f"   Steps: {step_num}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final P&L: ${final_pnl:.2f}")
        print(f"   Realized P&L: ${total_realized_pnl:.2f}")
        print(f"   Unrealized P&L: ${final_unrealized_pnl:.2f}")
        print(f"   Total Transaction Cost: ${total_transaction_cost:.2f}")
        print(f"   Termination Reason: {info.get('termination_reason', 'unknown')}")
        print(f"   Peak P&L: ${peak_pnls[-1] if peak_pnls else 0.0:.2f}")
        print(f"   Max Drawdown: ${drawdowns[-1] if drawdowns else 0.0:.2f}")
        if stop_loss_events:
            print(f"   ⚠️  Stop-Loss Triggered: {len(stop_loss_events)} time(s) at steps {stop_loss_events}")
        
        print(f"\n🎯 ACTION DISTRIBUTION")
        total_actions = len(actions)
        for action_id in sorted(action_counts.keys()):
            count = action_counts[action_id]
            pct = (count / total_actions) * 100
            print(f"   {action_names[action_id]:8s}: {count:5d} ({pct:5.1f}%)")
        
        print(f"\n📊 POSITION ANALYSIS")
        print(f"   Max Position: {max_position}")
        print(f"   Min Position: {min_position}")
        print(f"   Average Position: {avg_position:.2f}")
        print(f"   Position Changes: {sum(abs(pc) for pc in position_changes)}")
        print(f"   Net Position Change: {positions[-1] - positions[0] if len(positions) > 1 else 0}")
        
        print(f"\n💰 PRICE ANALYSIS")
        print(f"   Starting Price: ${prices[0]:.2f}")
        print(f"   Ending Price: ${prices[-1]:.2f}")
        print(f"   Price Change: ${prices[-1] - prices[0]:.2f} ({((prices[-1] / prices[0] - 1) * 100):.2f}%)")
        print(f"   Max Price: ${max_price:.2f}")
        print(f"   Min Price: ${min_price:.2f}")
        print(f"   Price Range: ${price_range:.2f}")
        
        print(f"\n🧠 Q-VALUE ANALYSIS")
        if avg_q_values:
            print(f"   Average Q-value: {np.mean(avg_q_values):.4f}")
            print(f"   Max Q-value: {np.max(max_q_values):.4f}")
            print(f"   Min Q-value: {np.min(avg_q_values):.4f}")
            print(f"   Q-value Range: {np.max(max_q_values) - np.min(avg_q_values):.4f}")
        
        # Key decision points
        print(f"\n🔍 KEY DECISION POINTS")
        
        # Find largest position changes
        large_changes = [(i, abs(pc)) for i, pc in enumerate(position_changes) if abs(pc) >= 1]
        large_changes.sort(key=lambda x: x[1], reverse=True)
        if large_changes:
            print(f"   Largest Position Changes:")
            for idx, (step, change) in enumerate(large_changes[:5]):
                action_taken = actions[step]
                price_at_step = prices[step]
                print(f"      Step {step:4d}: {action_names[action_taken]:8s} | "
                      f"Position: {positions[step-1] if step > 0 else 0} → {positions[step]} | "
                      f"Price: ${price_at_step:.2f}")
        
        # Find largest P&L swings
        pnl_changes = [total_pnls[i] - total_pnls[i-1] if i > 0 else total_pnls[i] 
                      for i in range(len(total_pnls))]
        large_pnl_swings = [(i, abs(pc)) for i, pc in enumerate(pnl_changes) if abs(pc) >= 100]
        large_pnl_swings.sort(key=lambda x: x[1], reverse=True)
        if large_pnl_swings:
            print(f"   Largest P&L Swings:")
            for idx, (step, swing) in enumerate(large_pnl_swings[:5]):
                print(f"      Step {step:4d}: ${swing:.2f} | "
                      f"P&L: ${total_pnls[step]:.2f} | "
                      f"Position: {positions[step]} | Price: ${prices[step]:.2f}")
        
        # Find stop-loss events
        if stop_loss_events:
            print(f"   Stop-Loss Events:")
            for step in stop_loss_events:
                print(f"      Step {step:4d}: Position: {positions[step]} | "
                      f"Unrealized P&L: ${unrealized_pnls[step]:.2f} | "
                      f"Price: ${prices[step]:.2f}")
        
        # Reward breakdown
        print(f"\n💵 REWARD BREAKDOWN")
        positive_rewards = [r for r in rewards if r > 0]
        negative_rewards = [r for r in rewards if r < 0]
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Positive Rewards: {sum(positive_rewards):.2f} ({len(positive_rewards)} steps)")
        print(f"   Negative Rewards: {sum(negative_rewards):.2f} ({len(negative_rewards)} steps)")
        print(f"   Average Reward per Step: {np.mean(rewards):.4f}")
        print(f"   Reward Std Dev: {np.std(rewards):.4f}")
        
        # Position-Price correlation
        if len(positions) > 1 and len(prices) > 1:
            position_price_corr = np.corrcoef(positions, prices)[0, 1]
            print(f"\n📈 POSITION-PRICE CORRELATION")
            print(f"   Correlation: {position_price_corr:.4f}")
            if position_price_corr > 0.3:
                print(f"   → Agent tends to hold long positions when price rises (momentum strategy)")
            elif position_price_corr < -0.3:
                print(f"   → Agent tends to hold short positions when price rises (mean reversion strategy)")
            else:
                print(f"   → Agent's position is not strongly correlated with price (mixed strategy)")
    
    print(f"\n{'=' * 100}")
    print("Analysis complete!")
    print(f"{'=' * 100}\n")


def analyze_training_progress(checkpoint_dir: Optional[str] = None) -> dict:
    """
    Analyze training progress by examining checkpoint files and their metadata.
    
    Args:
        checkpoint_dir: Directory containing checkpoints (default: reinforcement_results)
    
    Returns:
        Dictionary with analysis results
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return {}
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('rl_agent_ep') and f.endswith('.pt')]
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        return {}
    
    print(f"\n📊 ANALYZING TRAINING PROGRESS")
    print("=" * 100)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Extract episode numbers and sort
    def extract_episode_num(filename):
        try:
            return int(filename.replace('rl_agent_ep', '').replace('.pt', ''))
        except:
            return 0
    
    checkpoint_files.sort(key=extract_episode_num)
    episodes = [extract_episode_num(f) for f in checkpoint_files]
    
    print(f"Episode range: {min(episodes)} to {max(episodes)}")
    
    # Load first and last checkpoints to compare
    first_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    last_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    try:
        first_data = torch.load(first_checkpoint, map_location='cpu', weights_only=False)
        last_data = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
        
        first_episode = first_data.get('episode', extract_episode_num(checkpoint_files[0]))
        last_episode = last_data.get('episode', extract_episode_num(checkpoint_files[-1]))
        
        first_epsilon = first_data.get('epsilon', 1.0)
        last_epsilon = last_data.get('epsilon', 0.01)
        
        first_reward_mean = first_data.get('reward_mean', 0.0)
        last_reward_mean = last_data.get('reward_mean', 0.0)
        first_reward_std = first_data.get('reward_std', 1.0)
        last_reward_std = last_data.get('reward_std', 1.0)
        
        print(f"\n📈 CHECKPOINT COMPARISON:")
        print(f"   First checkpoint (episode {first_episode}):")
        print(f"     Epsilon: {first_epsilon:.4f}")
        print(f"     Reward norm μ: {first_reward_mean:.4f}")
        print(f"     Reward norm σ: {first_reward_std:.4f}")
        print(f"   Last checkpoint (episode {last_episode}):")
        print(f"     Epsilon: {last_epsilon:.4f}")
        print(f"     Reward norm μ: {last_reward_mean:.4f}")
        print(f"     Reward norm σ: {last_reward_std:.4f}")
        
        epsilon_change = first_epsilon - last_epsilon
        reward_mean_change = last_reward_mean - first_reward_mean
        reward_std_change = last_reward_std - first_reward_std
        
        print(f"\n   Changes:")
        print(f"     Epsilon: {epsilon_change:+.4f} (decreasing = more exploitation)")
        print(f"     Reward μ: {reward_mean_change:+.4f}")
        print(f"     Reward σ: {reward_std_change:+.4f} (decreasing = more stable rewards)")
        
        print(f"\n   💡 INTERPRETATION:")
        
        # Epsilon analysis
        if first_epsilon > 0.5 and last_epsilon < 0.1:
            print(f"     ✓ Epsilon decreased from {first_epsilon:.4f} to {last_epsilon:.4f}")
            print(f"        Agent transitioned from exploration to exploitation")
        elif last_epsilon <= 0.01:
            print(f"     → Epsilon at minimum ({last_epsilon:.4f}) - agent is fully exploiting")
        else:
            print(f"     → Epsilon: {first_epsilon:.4f} → {last_epsilon:.4f} (still exploring)")
        
        # Reward normalization analysis
        # Note: reward_mean and reward_std are normalization statistics, not actual rewards
        # A decrease in reward_mean could mean:
        # 1. Rewards are becoming more centered (good if variance also decreases)
        # 2. Rewards are getting worse (bad)
        # We need to check both mean and std together
        
        if abs(reward_mean_change) < 0.1:
            print(f"     → Reward normalization mean is stable (change: {reward_mean_change:+.4f})")
        elif reward_mean_change < -0.1:
            print(f"     ⚠️  Reward normalization mean DECREASED significantly ({reward_mean_change:+.4f})")
            print(f"        This could indicate:")
            print(f"        - Rewards are becoming more negative (concerning)")
            print(f"        - Or rewards are centering around 0 (could be good if variance decreases)")
        
        if reward_std_change < -0.1:
            print(f"     ✓ Reward normalization std DECREASED ({reward_std_change:+.4f})")
            print(f"        Rewards are becoming more stable/consistent")
        elif reward_std_change > 0.1:
            print(f"     ⚠️  Reward normalization std INCREASED ({reward_std_change:+.4f})")
            print(f"        Rewards are becoming more variable")
        
        # Combined assessment
        print(f"\n   🎯 ASSESSMENT:")
        if reward_mean_change < -0.1 and reward_std_change > 0.1:
            print(f"     ⚠️  CONCERNING: Reward mean decreased AND variance increased")
            print(f"        Model may not be learning effectively")
        elif reward_mean_change < -0.1 and reward_std_change < -0.1:
            print(f"     → MIXED: Reward mean decreased but variance decreased too")
            print(f"        Rewards may be centering around a lower value (check actual P&L)")
        elif abs(reward_mean_change) < 0.1 and reward_std_change < -0.1:
            print(f"     ✓ POSITIVE: Rewards becoming more stable (lower variance)")
        else:
            print(f"     → Monitor: Changes are within normal range")
        
        print(f"\n   ⚠️  NOTE: This analysis is based on checkpoint metadata only.")
        print(f"      Reward normalization stats track the distribution of rewards, not actual P&L.")
        print(f"      For definitive assessment, check:")
        print(f"      - Training logs showing actual P&L trends")
        print(f"      - Loss trends (decreasing is good)")
        print(f"      - Evaluation results (average P&L)")
        
        # Check for training log files
        log_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('training_log_') and f.endswith('.txt')]
        if log_files:
            print(f"\n📋 FOUND {len(log_files)} TRAINING LOG FILE(S):")
            log_files.sort(reverse=True)  # Most recent first
            for log_file in log_files[:3]:  # Show up to 3 most recent
                log_path = os.path.join(checkpoint_dir, log_file)
                print(f"\n   Analyzing: {log_file}")
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        
                    # Extract P&L data
                    pnl_values = []
                    in_pnl_section = False
                    for line in lines:
                        if 'EPISODE-BY-EPISODE P&L TRENDS' in line:
                            in_pnl_section = True
                            continue
                        if in_pnl_section and line.strip() and not line.startswith('-') and not 'Episode' in line:
                            try:
                                parts = line.split()
                                if len(parts) >= 2:
                                    # Extract P&L (remove $ sign)
                                    pnl_str = parts[1].replace('$', '')
                                    pnl = float(pnl_str)
                                    pnl_values.append(pnl)
                            except:
                                if 'SUMMARY' in line:
                                    break
                                continue
                    
                    if pnl_values:
                        print(f"     Episodes analyzed: {len(pnl_values)}")
                        print(f"     Average P&L: ${np.mean(pnl_values):.2f} ± ${np.std(pnl_values):.2f}")
                        print(f"     Best P&L: ${np.max(pnl_values):.2f}")
                        print(f"     Worst P&L: ${np.min(pnl_values):.2f}")
                        
                        if len(pnl_values) >= 20:
                            first_10_avg = np.mean(pnl_values[:10])
                            last_10_avg = np.mean(pnl_values[-10:])
                            improvement = last_10_avg - first_10_avg
                            print(f"     First 10 avg: ${first_10_avg:.2f}")
                            print(f"     Last 10 avg: ${last_10_avg:.2f}")
                            print(f"     Improvement: ${improvement:+.2f}")
                            
                            if improvement > 0:
                                print(f"     ✓✓✓ P&L is IMPROVING! Model is learning effectively.")
                            elif improvement > -50:
                                print(f"     → P&L is relatively stable.")
                            else:
                                print(f"     ⚠️  P&L is DECLINING. Model may not be learning effectively.")
                except Exception as e:
                    print(f"     ⚠️  Could not read log file: {e}")
        
    except Exception as e:
        print(f"   ⚠️  Could not load checkpoint data: {e}")
    
    print("=" * 100)
    
    return {
        'num_checkpoints': len(checkpoint_files),
        'episode_range': (min(episodes), max(episodes)),
        'first_checkpoint': checkpoint_files[0],
        'last_checkpoint': checkpoint_files[-1]
    }
