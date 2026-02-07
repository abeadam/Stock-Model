"""
Model Tester for SPX Trading

This script:
- Loads SPX data for the last day from daily_data
- Calculates all necessary technical indicators
- Uses the latest RL agent from reinforcement_results to make trading decisions
- Simulates trading with 1 million starting capital
- Prints results after every trade
"""

import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, Optional
import pytz

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import indicator calculation functions
from futures_price import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_rsi,
    calculate_mfi,
    calculate_current_percent_changes,
    calculate_trading_hours,
    extract_cyclical_time_features
)

# Import RL classes
from reinforcement.futures_renforcement import SPXTradingEnv, DQNAgent
from reinforcement.futures_renforcement_ppo import PPOAgent


def load_spx_data_for_last_n_days(data_dir: str = 'daily_data', n_days: int = 5) -> list:
    """Load SPX data for the last N available days
    
    Returns:
        List of tuples: (date_string, dataframe) for each day
    """
    print(f"Loading SPX data for the last {n_days} days...")
    
    # Convert relative path to absolute path if needed
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
        data_dir = os.path.normpath(data_dir)
    
    # Get all ES files (E-mini S&P 500)
    es_files = glob.glob(os.path.join(data_dir, "*_ES.txt"))
    
    if not es_files:
        # Try _SPX.txt as fallback
        es_files = glob.glob(os.path.join(data_dir, "*_SPX.txt"))
    
    if not es_files:
        raise ValueError(f"No ES or SPX files found in {data_dir}")
    
    # Sort files and get the last N
    es_files = sorted(es_files)
    last_n_files = es_files[-n_days:]
    
    days_data = []
    for es_path in last_n_files:
        # Extract date from filename (e.g., "2025-07-08_ES.txt" -> "2025-07-08")
        filename = os.path.basename(es_path)
        date_str = filename.split('_')[0]
        
        print(f"Loading ES data for {date_str} from: {filename}")
        
        # Load ES data
        df = pd.read_csv(es_path)
        
        # Ensure we have required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: {filename} missing required columns, skipping")
            continue
            
        # Try to load and merge VXM data for the same date
        vxm_path = os.path.join(data_dir, f"{date_str}_VXM.txt")
        if os.path.exists(vxm_path):
            print(f"  - Loading and merging VXM data for {date_str}...")
            vxm_df = pd.read_csv(vxm_path)
            
            # Ensure required columns
            vxm_req = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in vxm_df.columns for col in vxm_req):
                # Rename VXM columns
                vxm_rename = {col: f'VXM_{col}' for col in vxm_df.columns if col != 'Date'}
                vxm_df = vxm_df.rename(columns=vxm_rename)
                
                # Merge with main df
                df = pd.merge(df, vxm_df[['Date'] + list(vxm_rename.values())], on='Date', how='left')
                # Fill VXM NaNs if any (e.g., if timestamps don't match perfectly)
                vxm_cols = list(vxm_rename.values())
                df[vxm_cols] = df[vxm_cols].ffill().bfill().fillna(0)
                print(f"  - ✓ VXM data merged successfully")
            else:
                print(f"  - Warning: VXM file for {date_str} missing required columns, skipping merge")
        else:
            print(f"  - Warning: No VXM data found for {date_str} ({vxm_path})")
        
        # Remove rows with NaN in OHLCV
        initial_len = len(df)
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        print(f"  Loaded {len(df)} rows (removed {initial_len - len(df)} rows with NaN)")
        
        days_data.append((date_str, df))
    
    print(f"\nSuccessfully loaded {len(days_data)} days of data")
    return days_data


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators needed for trading"""
    print("\nCalculating technical indicators...")
    
    # Calculate trading hours
    print("  - Trading hours...")
    df = calculate_trading_hours(df)
    
    # Extract cyclical time features (hour_sin, hour_cos)
    print("  - Cyclical time features (hour_sin, hour_cos)...")
    df = extract_cyclical_time_features(df)
    
    # Calculate Bollinger Bands
    print("  - Bollinger Bands...")
    for period in [10, 20, 50]:
        for std_dev in [1.0, 2.0, 3.0]:
            df = calculate_bollinger_bands(df, period=period, std_dev=std_dev)
    
    # Calculate EMA
    print("  - Exponential Moving Averages...")
    for period in [10, 20, 50]:
        df = calculate_ema(df, period=period)
    
    # Calculate RSI
    print("  - RSI...")
    for period in [7, 14, 28]:
        df = calculate_rsi(df, period=period)
    
    # Calculate MFI
    print("  - MFI...")
    for period in [7, 14, 28]:
        df = calculate_mfi(df, period=period)
    
    # Calculate current percent changes
    print("  - Current percent changes...")
    df = calculate_current_percent_changes(df)
    
    # Note: We do NOT calculate forward-looking indicators (PctChange_ToMaxHigh_5, PctChange_ToMinLow_5)
    # as they require future data which is not available in real trading scenarios
    
    # Fill NaN values with 0 (for indicators that need history)
    df = df.fillna(0)
    
    # Skip initial rows needed for indicator calculations
    # The longest rolling window is 50 (Bollinger Bands 50), so we skip first 50 rows
    # This ensures all indicators have valid values
    skip_rows = 50
    if len(df) > skip_rows:
        initial_len = len(df)
        df = df.iloc[skip_rows:].reset_index(drop=True)
        print(f"  - Skipped first {skip_rows} rows for indicator warm-up (remaining: {len(df)} rows)")
    else:
        print(f"  - Warning: Only {len(df)} rows available, cannot skip {skip_rows} rows")
    
    print("✓ All indicators calculated")
    return df


def find_latest_model(reinforcement_results_dir: str) -> str:
    """Find the latest RL agent model"""
    if not os.path.isabs(reinforcement_results_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reinforcement_results_dir = os.path.join(script_dir, reinforcement_results_dir)
        reinforcement_results_dir = os.path.normpath(reinforcement_results_dir)
    
    # Check for rl_agent_final.pt first
    final_model = os.path.join(reinforcement_results_dir, 'rl_agent_final.pt')
    if os.path.exists(final_model):
        print(f"Using final model: rl_agent_final.pt")
        return final_model
    
    # Otherwise, find the model with highest episode number
    model_files = glob.glob(os.path.join(reinforcement_results_dir, 'rl_agent_ep*.pt'))
    if not model_files:
        raise ValueError(f"No RL agent models found in {reinforcement_results_dir}")
    
    # Extract episode numbers and find the highest
    def extract_episode(filename):
        basename = os.path.basename(filename)
        try:
            # Extract number from rl_agent_epXXXX.pt
            ep_str = basename.replace('rl_agent_ep', '').replace('.pt', '')
            return int(ep_str)
        except:
            return 0
    
    model_files = sorted(model_files, key=extract_episode, reverse=True)
    latest_model = model_files[0]
    episode = extract_episode(latest_model)
    print(f"Using latest model: {os.path.basename(latest_model)} (episode {episode})")
    return latest_model


def create_trading_simulator(df: pd.DataFrame, agent_path: str, 
                            initial_capital: float = 1000000.0, device: str = 'cpu',
                            max_steps: Optional[int] = None):
    """
    Create a trading simulator for a single day of data
    
    Returns:
        A function that runs the trading simulation
    """
    print(f"\nInitializing trading simulator...")
    print(f"  - Initial capital: ${initial_capital:,.2f}")
    print(f"  - Device: {device}")
    
    # Identify feature columns (non-forward-looking)
    # IMPORTANT: Use the SAME feature list as the training environment
    # The training environment uses a hardcoded list of 45 features, not all features
    forward_looking_cols = ['PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
    exclude_cols = ['Date', 'DateTime', 'DateTime_ET'] + forward_looking_cols
    
    # Use the exact same feature list as training environment (futures_reinforcement_utils.py)
    training_feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                             'BB_20_MA', 'BB_20_2_Upper', 'BB_20_2_Lower', 'MFI_14', 'RSI_14',
                             'BB_20_1_Upper', 'BB_20_1_Lower', 'BB_20_3_Upper', 'BB_20_3_Lower', 
                             'BB_50_MA', 'BB_50_2_Upper', 'BB_50_2_Lower', 'BB_50_1_Upper', 
                             'BB_50_1_Lower', 'BB_50_3_Upper', 'BB_50_3_Lower',
                             'BB_10_MA', 'BB_10_2_Upper', 'BB_10_2_Lower', 'BB_10_1_Upper', 
                             'BB_10_1_Lower', 'BB_10_3_Upper', 'BB_10_3_Lower',
                             'EMA_10', 'EMA_20', 'EMA_50', 'BB_10_STD', 'BB_20_STD', 'BB_50_STD',
                             'VXM_Open', 'VXM_High', 'VXM_Low', 'VXM_Close', 'VXM_Volume', 'ATR_14']
    
    # Only use features that exist in the dataframe
    feature_cols = [col for col in training_feature_cols if col in df.columns]
    
    if len(feature_cols) != len(training_feature_cols):
        missing = set(training_feature_cols) - set(feature_cols)
        print(f"  - Warning: Missing {len(missing)} features: {missing}")
        print(f"  - Using {len(feature_cols)} features (expected {len(training_feature_cols)})")
    else:
        print(f"  - Using {len(feature_cols)} features for state (matches training)")
    
    # Load RL agent checkpoint to detect agent type and get metadata
    print(f"  - Loading RL agent from {agent_path}...")
    agent_checkpoint = torch.load(agent_path, map_location=device, weights_only=False)
    
    # Initialize feature_scaler (will be set from checkpoint or created from test data)
    feature_scaler = None
    
    # Detect agent type (PPO or DQN)
    is_ppo = 'actor_critic_state_dict' in agent_checkpoint
    is_dqn = 'q_network_state_dict' in agent_checkpoint
    
    if not is_ppo and not is_dqn:
        raise ValueError(f"Unknown checkpoint format. Expected either 'actor_critic_state_dict' (PPO) or 'q_network_state_dict' (DQN)")
    
    # Infer state_size and action_size from the saved network architecture
    if is_ppo:
        # PPO checkpoint structure
        state_size = agent_checkpoint.get('state_size', None)
        action_size = agent_checkpoint.get('action_size', None)
        network_size = agent_checkpoint.get('network_size', 'medium')
        
        # If not in checkpoint, infer from network architecture
        if state_size is None or action_size is None:
            first_layer_key = 'feature_layers.0.weight'
            actor_output_key = 'actor.1.weight'  # Last layer of actor head
            
            if first_layer_key in agent_checkpoint['actor_critic_state_dict']:
                state_size = agent_checkpoint['actor_critic_state_dict'][first_layer_key].shape[1]
            else:
                # Fallback: calculate from feature count (without MVE predictions)
                # State = features + position + stop_loss = features + 2
                state_size = len(feature_cols) + 2
            
            if actor_output_key in agent_checkpoint['actor_critic_state_dict']:
                action_size = agent_checkpoint['actor_critic_state_dict'][actor_output_key].shape[0]
            else:
                action_size = 5  # Default: -2, -1, 0, 1, 2
        
        print(f"  - Detected PPO agent (network_size={network_size}, state_size={state_size}, action_size={action_size})")
        
        # Create PPO agent
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            device=device,
            network_size=network_size,
            lr=agent_checkpoint.get('lr', 3e-4),
            gamma=agent_checkpoint.get('gamma', 0.99),
            clip_epsilon=agent_checkpoint.get('clip_epsilon', 0.2),
            value_coef=agent_checkpoint.get('value_coef', 0.5),
            entropy_coef=agent_checkpoint.get('entropy_coef', 0.01),
            gae_lambda=agent_checkpoint.get('gae_lambda', 0.95)
        )
        
        # Load agent weights (this will also load the scaler if it exists in checkpoint)
        agent.load(agent_path)
        
        # Use the scaler from the loaded agent if available, otherwise try loading from file
        if agent.feature_scaler is not None:
            feature_scaler = agent.feature_scaler
            print(f"  - ✓ Using feature scaler from checkpoint (training data statistics)")
        else:
            # Try loading from external file first to avoid distribution shift
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_scaler_path = os.path.join(script_dir, 'reinforcement', 'reinforcement_results', 'feature_scaler.pkl')
            
            if os.path.exists(potential_scaler_path):
                print(f"  - Loading feature scaler from {potential_scaler_path} for PPO agent")
                try:
                    with open(potential_scaler_path, 'rb') as f:
                        feature_scaler = pickle.load(f)
                    print(f"  - ✓ Feature scaler loaded successfully from file")
                except Exception as e:
                    print(f"  - ⚠️ Error loading scaler from file: {e}")
            
            if feature_scaler is None:
                # Fallback: create scaler from test data (not ideal but works)
                print(f"  - ⚠️  No feature scaler in checkpoint or file, creating new scaler from test data")
                print(f"  - Note: This may cause distribution shift. For best results, retrain with scaler saving enabled.")
                feature_data = df[feature_cols].copy().fillna(0)
                from sklearn.preprocessing import StandardScaler
                feature_scaler = StandardScaler()
                feature_scaler.fit(feature_data.values)
                print(f"  - Feature scaler fitted on {len(feature_data)} test samples")
        
        # Set to evaluation mode
        agent.actor_critic.eval()
        
        print(f"  - PPO agent loaded successfully")
        
    else:  # DQN
        # DQN checkpoint structure
        first_layer_key = 'feature_layers.0.weight'
        advantage_output_key = 'advantage_stream.2.weight'
        
        if first_layer_key in agent_checkpoint['q_network_state_dict']:
            state_size = agent_checkpoint['q_network_state_dict'][first_layer_key].shape[1]
        else:
            # Fallback: calculate from feature count (without MVE predictions)
            # State = features + position + stop_loss = features + 2
            state_size = len(feature_cols) + 2
        
        if advantage_output_key in agent_checkpoint['q_network_state_dict']:
            action_size = agent_checkpoint['q_network_state_dict'][advantage_output_key].shape[0]
        else:
            action_size = 5  # Default: -2, -1, 0, 1, 2
        
        # Get saved epsilon (but we'll override to 0.0 for testing)
        saved_epsilon = agent_checkpoint.get('epsilon', 0.0)
        
        print(f"  - Detected DQN agent (state_size={state_size}, action_size={action_size})")
        
        # Create DQN agent with inferred architecture
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device,
            epsilon=0.0,  # Override to 0.0 for testing (no exploration)
            epsilon_min=0.0,
            epsilon_decay=0.0
            # Other parameters use defaults from DQNAgent.__init__
        )
        
        # Load agent weights and training state
        agent.q_network.load_state_dict(agent_checkpoint['q_network_state_dict'])
        agent.target_network.load_state_dict(agent_checkpoint['target_network_state_dict'])
        
        # Load optimizer state if available (not needed for inference, but good to have)
        if 'optimizer_state_dict' in agent_checkpoint:
            agent.optimizer.load_state_dict(agent_checkpoint['optimizer_state_dict'])
        
        # Load reward normalization stats if available
        if 'reward_mean' in agent_checkpoint:
            agent.reward_mean = agent_checkpoint['reward_mean']
        if 'reward_var' in agent_checkpoint:
            agent.reward_var = agent_checkpoint['reward_var']
        if 'reward_std' in agent_checkpoint:
            agent.reward_std = agent_checkpoint['reward_std']
        
        # Set to evaluation mode
        agent.q_network.eval()
        agent.target_network.eval()
        
        # For DQN, try loading scaler from file, otherwise create from test data
        if feature_scaler is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_scaler_path = os.path.join(script_dir, 'reinforcement', 'reinforcement_results', 'feature_scaler.pkl')
            
            if os.path.exists(potential_scaler_path):
                print(f"  - Loading feature scaler from {potential_scaler_path} for DQN agent")
                try:
                    with open(potential_scaler_path, 'rb') as f:
                        feature_scaler = pickle.load(f)
                    print(f"  - ✓ Feature scaler loaded successfully")
                except Exception as e:
                    print(f"  - ⚠️ Error loading scaler: {e}, fitting new one")
            
            if feature_scaler is None:
                print(f"  - Creating feature scaler from test data for DQN agent")
                feature_data = df[feature_cols].copy().fillna(0)
                from sklearn.preprocessing import StandardScaler
                feature_scaler = StandardScaler()
                feature_scaler.fit(feature_data.values)
                print(f"  - Feature scaler fitted on {len(feature_data)} test samples")
        
        print(f"  - DQN agent loaded (saved_epsilon={saved_epsilon:.4f}, using epsilon=0.0 for testing)")
    
    # Trading parameters
    # Infer max_position from action_size: action_space_size = 2 * max_position + 1
    # So max_position = (action_size - 1) / 2
    if action_size % 2 == 1:  # action_size should be odd (5, 7, 9, etc.)
        inferred_max_position = (action_size - 1) // 2
    else:
        # Fallback if action_size is unexpected
        inferred_max_position = 2
        print(f"  - Warning: action_size={action_size} is not odd, using default max_position=2")
    
    max_position = inferred_max_position
    min_position = -max_position
    transaction_cost_per_contract = 2.5
    stop_loss_per_contract = 2000.0
    
    print(f"  - Trading parameters: max_position={max_position}, action_size={action_size}")
    
    # Initialize trading state
    position = 0
    cash = initial_capital
    cumulative_realized_pnl = 0.0
    cumulative_transaction_costs = 0.0
    avg_entry_price = 0.0
    trades = []
    
    def get_state(step: int) -> np.ndarray:
        """Get state representation for current step
        
        If checkpoint was trained with MVE predictions, we pad with zeros.
        """
        if step == 0:
            prev_idx = 0
        else:
            prev_idx = step - 1
        
        prev_row = df.iloc[prev_idx]
        
        # Extract features
        feature_values = []
        for col in feature_cols:
            val = prev_row[col]
            if pd.isna(val):
                feature_values.append(0.0)
            else:
                feature_values.append(float(val))
        
        # Normalize features
        feature_array = np.array(feature_values).reshape(1, -1)
        features_normalized = feature_scaler.transform(feature_array)[0]
        
        # Normalize position
        position_normalized = position / max_position
        stop_loss_normalized = (stop_loss_per_contract - 100.0) / 900.0
        
        # Combine base state (without MVE model predictions)
        base_state = np.concatenate([
            features_normalized,
            [position_normalized],
            [stop_loss_normalized]
        ])
        
        # Construct state to match training environment format
        # Training state: [features (45)] + [position] + [MVE predictions (4)] + [stop_loss]
        # Total: 45 + 1 + 4 + 1 = 51
        base_state_size = len(feature_cols) + 2  # features + position + stop_loss
        expected_state_size = state_size
        
        if expected_state_size > base_state_size:
            # Checkpoint was trained with MVE predictions, pad with zeros
            # MVE predictions: [pred_mean_high, pred_std_high, pred_mean_low, pred_std_low] = 4 values
            padding_size = expected_state_size - base_state_size
            if padding_size == 4:
                # Standard case: 4 MVE prediction values (model was trained with MVE model)
                mve_predictions = np.zeros(4, dtype=np.float32)
                state = np.concatenate([
                    features_normalized,
                    [position_normalized],
                    mve_predictions,  # [pred_mean_high, pred_std_high, pred_mean_low, pred_std_low]
                    [stop_loss_normalized]
                ])
            else:
                # Unexpected padding size, just pad with zeros
                padding = np.zeros(padding_size, dtype=np.float32)
                state = np.concatenate([
                    features_normalized,
                    [position_normalized],
                    padding,
                    [stop_loss_normalized]
                ])
        elif expected_state_size < base_state_size:
            # Model expects fewer features than we have - truncate to match
            # This can happen if model was trained with fewer features
            print(f"  - Warning: Checkpoint state_size ({expected_state_size}) < base state size ({base_state_size})")
            # Try to match by removing features (keep position and stop_loss)
            features_to_use = expected_state_size - 2  # position + stop_loss
            if features_to_use > 0 and features_to_use <= len(features_normalized):
                state = np.concatenate([
                    features_normalized[:features_to_use],
                    [position_normalized],
                    [stop_loss_normalized]
                ])
            else:
                # Fallback: just truncate
                state = base_state[:expected_state_size]
        else:
            # Perfect match: no MVE predictions expected
            state = base_state
        
        return state.astype(np.float32)
    
    def execute_trade(step: int, action: int, risk_multiplier: float = 1.0) -> Dict:
        """Execute a trade and return trade information"""
        nonlocal position, cash, cumulative_realized_pnl, cumulative_transaction_costs, avg_entry_price, stop_loss_per_contract
        
        # Apply learned risk multiplier to stop-loss if it changed
        # This allows the agent to dynamically adjust its risk threshold
        if risk_multiplier != 1.0:
            # We use a base stop loss (e.g. 750) and multiply it by the learned multiplier [0.5, 2.0]
            base_stop_loss = 750.0 # Standard base stop loss
            stop_loss_per_contract = base_stop_loss * risk_multiplier
            # Clamp to reasonable range
            stop_loss_per_contract = max(100.0, min(1500.0, stop_loss_per_contract))
        
        # Map action to position change
        # action 0 -> -max_position, ..., action max_position -> 0, ..., action 2*max_position -> +max_position
        position_change = action - max_position
        
        # Prevent actions that would exceed position limits
        if position <= min_position and position_change < 0:
            position_change = 0
        elif position >= max_position and position_change > 0:
            position_change = 0
        
        # Calculate new position
        old_position = position
        new_position = position + position_change
        new_position = max(min_position, min(max_position, new_position))
        actual_change = new_position - position
        
        # Get current price
        current_price = df.iloc[step]['Close']
        
        # Calculate transaction cost
        contracts_traded = abs(actual_change)
        transaction_cost = contracts_traded * transaction_cost_per_contract
        
        # Calculate realized P&L
        realized_pnl = 0.0
        if position != 0 and avg_entry_price != 0:
            if (position > 0 and actual_change < 0) or (position < 0 and actual_change > 0):
                closed_units = min(abs(position), abs(actual_change))
                if position > 0:
                    realized_pnl = (current_price - avg_entry_price) * closed_units
                else:
                    realized_pnl = (avg_entry_price - current_price) * closed_units
        
        # Update position and cash
        # Note: avg_entry_price is always positive, position sign indicates long/short
        if actual_change > 0:
            # Buying: reduce cash, add to position (or reduce short)
            cost = actual_change * current_price
            cash -= cost
            
            if old_position == 0:
                # Starting new long position
                avg_entry_price = current_price
            elif old_position > 0:
                # Adding to long position - weighted average
                old_value = avg_entry_price * old_position
                new_cost = actual_change * current_price
                avg_entry_price = (old_value + new_cost) / new_position
            else:
                # Reducing short position (buying back)
                if abs(new_position) < abs(old_position):
                    # Still short, keep entry price
                    pass
                else:
                    # Flipped to long
                    avg_entry_price = current_price
        elif actual_change < 0:
            # Selling: increase cash, reduce position (or go short)
            proceeds = abs(actual_change) * current_price
            cash += proceeds
            
            if old_position == 0:
                # Starting new short position
                avg_entry_price = current_price
            elif old_position < 0:
                # Adding to short position - weighted average
                old_value = avg_entry_price * abs(old_position)
                new_cost = abs(actual_change) * current_price
                avg_entry_price = (old_value + new_cost) / abs(new_position)
            else:
                # Reducing long position (selling)
                if new_position > 0:
                    # Still long, keep entry price
                    pass
                else:
                    # Flipped to short
                    avg_entry_price = current_price
        
        position = new_position
        
        # Reset avg_entry_price if position is closed
        if position == 0:
            avg_entry_price = 0.0
        
        # Apply transaction cost
        cash -= transaction_cost
        cumulative_realized_pnl += realized_pnl
        cumulative_transaction_costs += transaction_cost
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        if position != 0 and avg_entry_price != 0:
            if position > 0:
                unrealized_pnl = (current_price - avg_entry_price) * position
            else:
                unrealized_pnl = (avg_entry_price - current_price) * abs(position)
        
        # Total portfolio value (cash + position value)
        portfolio_value = cash + (position * current_price if position != 0 else 0)
        
        # Total P&L = portfolio_value - initial_capital
        # This correctly accounts for all realized P&L, unrealized P&L, and transaction costs
        # (transaction costs are already deducted from cash, so they're included in portfolio_value)
        total_pnl = portfolio_value - initial_capital
        
        trade_info = {
            'step': step,
            'timestamp': df.iloc[step]['Date'],
            'price': current_price,
            'action': action,
            'position_change': actual_change,
            'new_position': position,
            'realized_pnl': realized_pnl,
            'cumulative_realized_pnl': cumulative_realized_pnl,
            'transaction_cost': transaction_cost,
            'cumulative_transaction_costs': cumulative_transaction_costs,
            'unrealized_pnl': unrealized_pnl,
            'cash': cash,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl  # Net P&L after all costs
        }
        
        return trade_info
    
    def run_simulation():
        """Run the trading simulation"""
        print(f"\n{'='*80}")
        print("STARTING TRADING SIMULATION")
        print(f"{'='*80}")
        print(f"Total data points: {len(df)}")
        print(f"Starting capital: ${initial_capital:,.2f}")
        print(f"{'='*80}\n")
        
        # Determine simulation length
        total_steps = len(df)
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
            
        # Start from step 1 (need previous step for state)
        for step in range(1, total_steps):
            # Get state
            state = get_state(step)
            
            # Get action from agent (no exploration)
            act_result = agent.act(state, training=False)
            # Handle tuple return (action, risk_multiplier)
            if isinstance(act_result, tuple):
                action = int(act_result[0])
                learned_risk_multiplier = float(act_result[1])
            else:
                action = int(act_result)
                learned_risk_multiplier = 1.0  # Default if not provided
            
            # Clamp action to valid range [0, action_size-1] to prevent out-of-bounds actions
            original_action = action
            if action < 0 or action >= action_size:
                print(f"  - Warning: Invalid action {action} (valid range: 0-{action_size-1}), clamping")
                action = max(0, min(action_size - 1, action))
            
            # Calculate what position change this action would result in
            # Action mapping: action 0 → -max_position, action max_position → 0, action 2*max_position → +max_position
            intended_position_change = action - max_position
            hold_action = max_position  # Action that results in no position change
            
            # Execute trade if action is not hold OR if we have a position to manage (to update unrealized P&L)
            # Note: We always execute to update unrealized P&L, but only print if there's actual trading activity
            trade_info = execute_trade(step, action, risk_multiplier=learned_risk_multiplier)
            
            # Debug: Log if action was ineffective due to position limits
            if trade_info['position_change'] == 0 and intended_position_change != 0:
                if (position >= max_position and intended_position_change > 0) or \
                   (position <= min_position and intended_position_change < 0):
                    # This is expected - position limit reached, but model should learn to close positions
                    # Only print occasionally to avoid spam
                    if step % 100 == 0:
                        print(f"  - Note: Action {action} (intended: {intended_position_change:+d}) blocked by position limit (pos: {position}, max: {max_position})")
            
            # Only print if there was actual trading activity or if we have a position (to track unrealized P&L)
            if trade_info['position_change'] != 0 or trade_info['new_position'] != 0:
                
                # Only print if there was an actual trade (position changed) or if we have a position
                # This avoids printing on every step when holding with no position
                if trade_info['position_change'] != 0 or trade_info['new_position'] != 0:
                    # Print trade results
                    print(f"Step {step:5d} | Price: ${trade_info['price']:8.2f} | "
                          f"Action: {trade_info['action']} ({trade_info['position_change']:+2d}) | "
                          f"Risk: {learned_risk_multiplier:.2f} | "
                          f"Position: {trade_info['new_position']:+2d} | "
                          f"Realized P&L: ${trade_info['realized_pnl']:8.2f} | "
                          f"Cum Realized: ${trade_info['cumulative_realized_pnl']:8.2f} | "
                          f"Unrealized P&L: ${trade_info['unrealized_pnl']:8.2f} | "
                          f"Transaction Cost: ${trade_info['transaction_cost']:6.2f} | "
                          f"Total P&L: ${trade_info['total_pnl']:10.2f} | "
                          f"Portfolio Value: ${trade_info['portfolio_value']:12.2f} | "
                          f"Cash: ${trade_info['cash']:12.2f}")
                
                trades.append(trade_info)
            else:
                # No position and action is hold - just track portfolio value for final summary
                # Calculate current portfolio value without executing a trade
                current_price = df.iloc[step]['Close']
                portfolio_value = cash + (position * current_price if position != 0 else 0)
                total_pnl = portfolio_value - initial_capital
                
                # Only track if this is meaningful (we'll use final value anyway)
                pass
        
        # Final summary
        final_price = df.iloc[-1]['Close']
        final_unrealized = 0.0
        if position != 0 and avg_entry_price != 0:
            if position > 0:
                final_unrealized = (final_price - avg_entry_price) * position
            else:
                final_unrealized = (avg_entry_price - final_price) * abs(position)
        
        final_portfolio_value = cash + (position * final_price if position != 0 else 0)
        total_return = final_portfolio_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Calculate breakdown for diagnostics
        total_realized = cumulative_realized_pnl
        total_transaction_costs = cumulative_transaction_costs
        gross_pnl = total_realized + final_unrealized
        net_pnl = gross_pnl - total_transaction_costs
        
        print(f"\n{'='*80}")
        print("TRADING SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total trades executed: {len(trades)}")
        print(f"Final position: {position}")
        print(f"Final cash: ${cash:,.2f}")
        print(f"Final unrealized P&L: ${final_unrealized:,.2f}")
        print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
        print(f"Total return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
        print(f"\nP&L Breakdown:")
        print(f"  Cumulative Realized P&L: ${total_realized:,.2f}")
        print(f"  Final Unrealized P&L: ${final_unrealized:,.2f}")
        print(f"  Gross P&L (Realized + Unrealized): ${gross_pnl:,.2f}")
        print(f"  Total Transaction Costs: ${total_transaction_costs:,.2f}")
        print(f"  Net P&L (Gross - Costs): ${net_pnl:,.2f}")
        print(f"  Portfolio Value Method: ${final_portfolio_value:,.2f}")
        print(f"  Difference: ${abs(final_portfolio_value - (initial_capital + net_pnl)):,.2f}")
        print(f"{'='*80}\n")
        
        return {
            'trades': trades,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'final_position': position
        }
    
    return run_simulation


def main():
    """Main function"""
    print("="*80)
    print("SPX MODEL TESTER - LAST 5 DAYS")
    print("="*80)
    
    # Configuration
    DATA_DIR = '../daily_data'
    REINFORCEMENT_RESULTS_DIR = 'reinforcement/reinforcement_results'
    INITIAL_CAPITAL = 1000000.0  # $1 million
    NUM_DAYS = 5
    MAX_STEPS_PER_DAY = 256  # Set to a number (e.g. 500) to limit steps per day
    SHUFFLE_DAYS = True  # Set to True to mix the order of days
    # Detect best available device
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    print(f"Using device: {DEVICE}")
    
    try:
        # Load SPX data for last N days
        days_data = load_spx_data_for_last_n_days(DATA_DIR, n_days=NUM_DAYS)
        
        if not days_data:
            raise ValueError("No valid days of data loaded")
        
        # Shuffle days if requested
        if SHUFFLE_DAYS:
            import random
            print(f"🎲 Shuffling the order of {len(days_data)} trading days...")
            random.shuffle(days_data)
        
        # Find latest model (once, reuse for all days)
        agent_path = find_latest_model(REINFORCEMENT_RESULTS_DIR)
        
        # Process each day
        all_results = []
        total_initial_capital = INITIAL_CAPITAL
        
        for day_idx, (date_str, df) in enumerate(days_data, 1):
            print(f"\n{'='*80}")
            print(f"PROCESSING DAY {day_idx}/{len(days_data)}: {date_str}")
            print(f"{'='*80}")
            
            # Calculate all indicators for this day
            df = calculate_all_indicators(df)
            
            # Check if we have enough data after skipping warm-up rows
            if len(df) < 2:
                print(f"⚠️  Skipping {date_str}: Not enough data after indicator warm-up ({len(df)} rows)")
                continue
            
            # Create and run simulator for this day
            # Use the final portfolio value from previous day as starting capital (if available)
            current_capital = all_results[-1]['final_portfolio_value'] if all_results else INITIAL_CAPITAL
            
            simulator = create_trading_simulator(
                df=df,
                agent_path=agent_path,
                initial_capital=current_capital,
                device=DEVICE,
                max_steps=MAX_STEPS_PER_DAY
            )
            
            # Run simulation for this day
            results = simulator()
            results['date'] = date_str
            results['day_number'] = day_idx
            results['starting_capital'] = current_capital
            all_results.append(results)
        
        # Print summary across all days
        print(f"\n{'='*80}")
        print("SUMMARY - ALL DAYS")
        print(f"{'='*80}")
        print(f"Total days tested: {len(all_results)}")
        print(f"Starting capital (Day 1): ${INITIAL_CAPITAL:,.2f}")
        
        if all_results:
            final_portfolio_value = all_results[-1]['final_portfolio_value']
            total_return = final_portfolio_value - INITIAL_CAPITAL
            total_return_pct = (total_return / INITIAL_CAPITAL) * 100
            
            print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
            print(f"Total return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
            
            print(f"\nPer-day breakdown:")
            print(f"{'Day':<4} {'Date':<12} {'Start Capital':>15} {'End Value':>15} {'Return':>12} {'Return %':>10} {'Trades':>8}")
            print(f"{'-'*80}")
            for result in all_results:
                day_return = result['final_portfolio_value'] - result['starting_capital']
                day_return_pct = (day_return / result['starting_capital']) * 100 if result['starting_capital'] > 0 else 0
                print(f"{result['day_number']:<4} {result['date']:<12} "
                      f"${result['starting_capital']:>14,.2f} "
                      f"${result['final_portfolio_value']:>14,.2f} "
                      f"${day_return:>11,.2f} "
                      f"{day_return_pct:>9.2f}% "
                      f"{len(result['trades']):>8}")
        
        print(f"{'='*80}\n")
        print("All simulations completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

