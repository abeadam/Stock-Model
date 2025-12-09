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
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict
from datetime import datetime
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
    calculate_trading_hours
)

# Import RL classes
from futures_renforcement import SPXTradingEnv, DQNAgent
from futures_model_MVE_SNNs import MVEModel


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
    
    # Get all SPX files
    spx_files = glob.glob(os.path.join(data_dir, "*_SPX.txt"))
    
    if not spx_files:
        raise ValueError(f"No SPX files found in {data_dir}")
    
    # Sort files and get the last N
    spx_files = sorted(spx_files)
    last_n_files = spx_files[-n_days:]
    
    days_data = []
    for file_path in last_n_files:
        # Extract date from filename (e.g., "2025-05-01_SPX.txt" -> "2025-05-01")
        filename = os.path.basename(file_path)
        date_str = filename.replace("_SPX.txt", "")
        
        print(f"Loading data from: {filename}")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Ensure we have required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: {filename} missing required columns, skipping")
            continue
        
        # Add Volume if missing (needed for MFI)
        if 'Volume' not in df.columns:
            print(f"Warning: {filename} missing Volume column, using default volume for MFI")
            df['Volume'] = 1000000
        
        # Remove rows with NaN in OHLCV
        initial_len = len(df)
        df = df.dropna(subset=required_cols + ['Volume']).reset_index(drop=True)
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


def create_trading_simulator(df: pd.DataFrame, model_path: str, agent_path: str, 
                            initial_capital: float = 1000000.0, device: str = 'cpu'):
    """
    Create a trading simulator for a single day of data
    
    Returns:
        A function that runs the trading simulation
    """
    print(f"\nInitializing trading simulator...")
    print(f"  - Initial capital: ${initial_capital:,.2f}")
    print(f"  - Device: {device}")
    
    # Load the MVE model (needed for predictions)
    mve_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'futures_model.pt')
    if not os.path.exists(mve_model_path):
        raise FileNotFoundError(f"MVE model not found at {mve_model_path}")
    
    print(f"  - Loading MVE model from {mve_model_path}...")
    mve_checkpoint = torch.load(mve_model_path, map_location=device, weights_only=False)
    
    # Reconstruct MVE model
    input_dim = mve_checkpoint['input_dim']
    hidden_dims = mve_checkpoint['hidden_dims']
    scaler = mve_checkpoint['scaler']
    feature_names = mve_checkpoint['feature_names']
    
    # Determine if model predicts both high and low
    state_dict_keys = list(mve_checkpoint['model_state_dict'].keys())
    predict_both = any('quantile_10_head' in key for key in state_dict_keys)
    
    mve_model = MVEModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=0.0,
        predict_both=predict_both
    )
    mve_model.load_state_dict(mve_checkpoint['model_state_dict'])
    mve_model.to(device)
    mve_model.eval()
    
    print(f"  - MVE model loaded (predict_both={predict_both}, input_dim={input_dim})")
    
    # Identify feature columns (non-forward-looking)
    forward_looking_cols = ['PctChange_ToMaxHigh_5', 'PctChange_ToMinLow_5']
    exclude_cols = ['Date', 'DateTime', 'DateTime_ET'] + forward_looking_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  - Using {len(feature_cols)} features for state")
    
    # Create scaler for features
    feature_data = df[feature_cols].copy().fillna(0)
    from sklearn.preprocessing import StandardScaler
    feature_scaler = StandardScaler()
    feature_scaler.fit(feature_data.values)
    
    # Load RL agent
    print(f"  - Loading RL agent from {agent_path}...")
    agent_checkpoint = torch.load(agent_path, map_location=device, weights_only=False)
    
    # Infer state_size and action_size from the saved network architecture
    # state_size = input dimension of first layer
    # action_size = output dimension of advantage stream
    if 'q_network_state_dict' in agent_checkpoint:
        first_layer_key = 'feature_layers.0.weight'
        advantage_output_key = 'advantage_stream.2.weight'
        
        if first_layer_key in agent_checkpoint['q_network_state_dict']:
            state_size = agent_checkpoint['q_network_state_dict'][first_layer_key].shape[1]
        else:
            # Fallback: calculate from feature count
            state_size = len(feature_cols) + 6  # features + 6 additional values
        
        if advantage_output_key in agent_checkpoint['q_network_state_dict']:
            action_size = agent_checkpoint['q_network_state_dict'][advantage_output_key].shape[0]
        else:
            action_size = 5  # Default: -2, -1, 0, 1, 2
    else:
        # Fallback if checkpoint structure is unexpected
        state_size = len(feature_cols) + 6
        action_size = 5
    
    # Get saved epsilon (but we'll override to 0.0 for testing)
    saved_epsilon = agent_checkpoint.get('epsilon', 0.0)
    
    # Create DQN agent with inferred architecture
    # Other hyperparameters (lr, gamma, etc.) are not needed for inference,
    # but we need to provide them to initialize the agent
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
    
    print(f"  - RL agent loaded (state_size={state_size}, action_size={action_size}, saved_epsilon={saved_epsilon:.4f}, using epsilon=0.0 for testing)")
    
    # Trading parameters
    max_position = 2
    min_position = -2
    transaction_cost_per_contract = 2.5
    stop_loss_per_contract = 500.0
    
    # Initialize trading state
    position = 0
    cash = initial_capital
    cumulative_realized_pnl = 0.0
    cumulative_transaction_costs = 0.0
    avg_entry_price = 0.0
    trades = []
    
    def get_state(step: int) -> np.ndarray:
        """Get state representation for current step"""
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
        
        # Get model prediction
        model_features = []
        for feat_name in feature_names:
            if feat_name in df.columns:
                val = prev_row[feat_name]
                if pd.isna(val):
                    model_features.append(0.0)
                else:
                    model_features.append(float(val))
            else:
                model_features.append(0.0)
        
        model_features = np.array(model_features).reshape(1, -1)
        model_features_scaled = scaler.transform(model_features)
        
        # Get prediction from model
        with torch.no_grad():
            features_tensor = torch.FloatTensor(model_features_scaled).to(device)
            model_output = mve_model(features_tensor)
            
            if predict_both:
                (mean_high, mean_low), (log_var_high, log_var_low) = model_output
                pred_std_high = (log_var_high * 0.5).exp().cpu().numpy()[0, 0]
                pred_std_low = (log_var_low * 0.5).exp().cpu().numpy()[0, 0]
                pred_mean_high = mean_high.cpu().numpy()[0, 0]
                pred_mean_low = mean_low.cpu().numpy()[0, 0]
            else:
                mean_pred, log_var_pred = model_output
                pred_std_high = (log_var_pred * 0.5).exp().cpu().numpy()[0, 0]
                pred_mean_high = mean_pred.cpu().numpy()[0, 0]
                pred_mean_low = pred_mean_high
                pred_std_low = pred_std_high
        
        # Normalize position
        position_normalized = position / max_position
        stop_loss_normalized = (stop_loss_per_contract - 100.0) / 900.0
        
        # Combine state
        state = np.concatenate([
            features_normalized,
            [position_normalized],
            [pred_mean_high],
            [pred_std_high],
            [pred_mean_low],
            [pred_std_low],
            [stop_loss_normalized]
        ])
        
        return state.astype(np.float32)
    
    def execute_trade(step: int, action: int) -> Dict:
        """Execute a trade and return trade information"""
        nonlocal position, cash, cumulative_realized_pnl, cumulative_transaction_costs, avg_entry_price
        
        # Map action to position change (action: 0=-2, 1=-1, 2=0, 3=1, 4=2)
        action_map = {-2: -2, -1: -1, 0: 0, 1: 1, 2: 2}
        position_change = action_map[action - 2]
        
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
        
        # Total portfolio value
        portfolio_value = cash + (position * current_price if position != 0 else 0)
        
        # Total P&L (realized + unrealized - transaction costs)
        total_pnl = cumulative_realized_pnl + unrealized_pnl - cumulative_transaction_costs
        
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
            'total_pnl': total_pnl
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
        
        # Start from step 1 (need previous step for state)
        for step in range(1, len(df)):
            # Get state
            state = get_state(step)
            
            # Get action from agent (no exploration)
            act_result = agent.act(state, training=False)
            # Handle tuple return (action, risk_multiplier) from DQN with learnable_risk_control
            if isinstance(act_result, tuple):
                action = int(act_result[0])
            else:
                action = int(act_result)
            
            # Execute trade if action is not hold (action 2 = hold)
            if action != 2 or position != 0:  # Trade if not holding or if we have a position
                trade_info = execute_trade(step, action)
                
                # Print trade results
                print(f"Step {step:5d} | Price: ${trade_info['price']:8.2f} | "
                      f"Action: {trade_info['action']} ({trade_info['position_change']:+2d}) | "
                      f"Position: {trade_info['new_position']:+2d} | "
                      f"Realized P&L: ${trade_info['realized_pnl']:8.2f} | "
                      f"Cum Realized: ${trade_info['cumulative_realized_pnl']:8.2f} | "
                      f"Unrealized P&L: ${trade_info['unrealized_pnl']:8.2f} | "
                      f"Transaction Cost: ${trade_info['transaction_cost']:6.2f} | "
                      f"Total P&L: ${trade_info['total_pnl']:10.2f} | "
                      f"Portfolio Value: ${trade_info['portfolio_value']:12.2f} | "
                      f"Cash: ${trade_info['cash']:12.2f}")
                
                trades.append(trade_info)
        
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
        
        print(f"\n{'='*80}")
        print("TRADING SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total trades executed: {len(trades)}")
        print(f"Final position: {position}")
        print(f"Final cash: ${cash:,.2f}")
        print(f"Final unrealized P&L: ${final_unrealized:,.2f}")
        print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
        print(f"Total return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
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
    REINFORCEMENT_RESULTS_DIR = 'reinforcement_results'
    INITIAL_CAPITAL = 1000000.0  # $1 million
    NUM_DAYS = 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load SPX data for last N days
        days_data = load_spx_data_for_last_n_days(DATA_DIR, n_days=NUM_DAYS)
        
        if not days_data:
            raise ValueError("No valid days of data loaded")
        
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
                model_path='futures_model.pt',  # Will be resolved in function
                agent_path=agent_path,
                initial_capital=current_capital,
                device=DEVICE
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

