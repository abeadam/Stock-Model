#!/usr/bin/env python3
"""
Example script showing how to run PPO training

Usage:
    python3 run_ppo_example.py                    # Fresh training
    python3 run_ppo_example.py --evaluate 10      # Evaluate latest checkpoint
    python3 run_ppo_example.py --resume           # Resume from latest PPO checkpoint
"""

import os
import shutil
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from futures_renforcement_ppo import PPOAgent
from futures_reinforcement_utils import SPXTradingEnv, train_agent, evaluate_agent

def main():
    """Main function to train/evaluate PPO agent"""
    
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'es_with_indicators.csv')
    MODEL_PATH = None #os.path.join(os.path.dirname(__file__), '..', 'futures_model.pt')
    NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 8  # Reduced from 256 for faster feedback and more consistent strategies
    MAX_ROWS = 0  # Set to 10000 for quick testing, None for full dataset
    RANDOM_START_IN_FILE = True  # When max_rows is set, load a random max_rows-sized window from the file

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Check command line arguments
    evaluate_mode = '--evaluate' in sys.argv
    resume_mode = '--resume' in sys.argv or '--resume-latest' in sys.argv
    reset_entropy = '--reset-entropy' in sys.argv

    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')

    # Fresh training: clear reinforcement_results *before* creating env so we don't
    # load an old feature_scaler.pkl (e.g. 46 features) when the code now uses 73.
    if not evaluate_mode and not resume_mode:
        if os.path.isdir(checkpoint_dir):
            for name in os.listdir(checkpoint_dir):
                path = os.path.join(checkpoint_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            print(f"🗑️  Emptied {checkpoint_dir} for fresh training")
    
    # Create environment
    env = SPXTradingEnv(
        DATA_PATH, 
        MODEL_PATH, 
        device=device,
        max_steps_per_episode=STEPS_PER_EPISODE,
        max_loss_per_episode=-30000.0,  # Tighter limit to prevent catastrophic losses (was -50000)
        stop_loss_per_contract=500.0,  # Tighter stop loss to prevent large drawdowns (was 750.0)
        max_position=1,  # CRITICAL: Reduced from 100 to 5 to prevent catastrophic losses
        max_rows=MAX_ROWS,  # Limit rows for quick testing (None = full dataset)
        random_start_in_file=RANDOM_START_IN_FILE,  # When max_rows set, load random window from file
    )
    
    # Get state size
    test_state = env.reset()
    state_size = len(test_state)
    action_size = env.action_space_size
    print(f"State size: {state_size}, Action size: {action_size}")
    

    
    # Helper to peek at checkpoint for config
    def get_checkpoint_config(path):
        if not path or not os.path.exists(path):
            return {}
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            return {
                'network_size': checkpoint.get('network_size', 'medium'),
                'state_size': checkpoint.get('state_size', state_size),
                'action_size': checkpoint.get('action_size', action_size),
                'episode': checkpoint.get('episode', 0),
                'lr': checkpoint.get('lr', None),
                'value_coef': checkpoint.get('value_coef', None),
                'actor_lr': checkpoint.get('actor_lr', None),
                'critic_lr': checkpoint.get('critic_lr', None)
            }
        except:
            return {}

    if evaluate_mode:
        # Evaluation mode
        num_episodes = 10
        if '--evaluate' in sys.argv:
            idx = sys.argv.index('--evaluate')
            if idx + 1 < len(sys.argv):
                try:
                    num_episodes = int(sys.argv[idx + 1])
                except:
                    pass
        
        # Load latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'rl_agent_latest.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith('rl_agent_ep') and f.endswith('.pt')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.replace('rl_agent_ep', '').replace('.pt', '')))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            else:
                checkpoint_path = None
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            config = get_checkpoint_config(checkpoint_path)
            agent = PPOAgent(
                state_size=config.get('state_size', state_size),
                action_size=config.get('action_size', action_size),
                device=device,
                network_size=config.get('network_size', 'medium'),
                entropy_coef=0.0, # Zero exploration for evaluation
                entropy_min=0.0,
                entropy_decay=1.0,
                min_lr=1e-5,
                total_episodes=NUM_EPISODES
            )
            agent.load(checkpoint_path)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            evaluate_agent(env, agent, num_episodes=num_episodes, detailed=True)
        else:
            print("❌ No checkpoint found for evaluation!")
    
    elif resume_mode:
        # Resume training
        checkpoint_path = None
        start_episode = 0
        
        print("\n🔍 Looking for latest PPO checkpoint...")
        latest_path = os.path.join(checkpoint_dir, 'rl_agent_latest.pt')
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
                if 'actor_critic_state_dict' in checkpoint:
                    checkpoint_path = latest_path
                    start_episode = checkpoint.get('episode', 0)
                    print(f"✓ Found latest PPO checkpoint: rl_agent_latest.pt (episode {start_episode})")
            except:
                pass
        
        if not checkpoint_path:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith('rl_agent_ep') and f.endswith('.pt')]
            if checkpoint_files:
                ppo_checkpoints = []
                for f in checkpoint_files:
                    checkpoint_file = os.path.join(checkpoint_dir, f)
                    try:
                        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                        if 'actor_critic_state_dict' in checkpoint:
                            ppo_checkpoints.append((f, checkpoint.get('episode', 0)))
                    except:
                        continue
                if ppo_checkpoints:
                    ppo_checkpoints.sort(key=lambda x: x[1], reverse=True)
                    latest_file, latest_episode = ppo_checkpoints[0]
                    checkpoint_path = os.path.join(checkpoint_dir, latest_file)
                    start_episode = latest_episode
                    print(f"✓ Found latest PPO checkpoint: {latest_file} (episode {latest_episode})")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            config = get_checkpoint_config(checkpoint_path)
            
            # If resetting entropy, use a higher value to "shock" the model back into exploration
            resumed_entropy = 0.15 if reset_entropy else 0.08
            print(f"🔄 {'Resetting' if reset_entropy else 'Initializing'} entropy coefficient to {resumed_entropy}")
            
            agent = PPOAgent(
                state_size=config.get('state_size', state_size),
                action_size=config.get('action_size', action_size),
                device=device,
                network_size=config.get('network_size', 'medium'),
                batch_size=64,           # Reduced from 256 for more frequent updates
                lr=config.get('lr', 1e-4),  # Use checkpoint LR if available, else use reduced LR
                value_coef=config.get('value_coef', 2.0),  # Use checkpoint value_coef if available, else use 2.0
                entropy_coef=resumed_entropy,  # Cap at 0.005 for resumed training
                entropy_min=0.001,       # Reduced from 0.02 - allow policy to converge more
                entropy_decay=0.999,     # Faster decay from 0.9995 - reduce exploration faster
                min_lr=1e-6,            # Reduced from 1e-4 to allow LR to decay lower
                total_episodes=NUM_EPISODES,
                actor_lr=config.get('actor_lr', None),  # Use checkpoint actor_lr if available
                critic_lr=config.get('critic_lr', None)  # Use checkpoint critic_lr if available
            )
            agent.load(checkpoint_path)
            print(f"✓ Resuming from episode {start_episode}")
            
            # Sync environment's feature scaler with agent's scaler from checkpoint
            if hasattr(agent, 'feature_scaler') and agent.feature_scaler is not None:
                if hasattr(env, 'set_feature_scaler'):
                    env.set_feature_scaler(agent.feature_scaler)
                    print(f"✓ Synced environment feature scaler with agent's scaler from checkpoint")
                else:
                    env.feature_scaler = agent.feature_scaler
                    print(f"✓ Set environment feature scaler from agent's checkpoint")
        else:
            print("❌ No checkpoint found to resume training!")
            return

        train_agent(env, agent, num_episodes=NUM_EPISODES, start_episode=start_episode)
    
    else:
        # Fresh training (reinforcement_results was already cleared before env creation)
        print("\n🎯 Starting PPO training from scratch")
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            device=device,
            network_size='medium',   # Upgraded from 'large' to address value loss bottleneck
            lr=5e-4,                 # Base learning rate (actor will use this, critic will use 2x)
            batch_size=8,           # Reduced from 256 for more frequent updates (4x more updates)
            value_coef=1.5,        # Increased from 1.0 to 1.5 for better value function learning
            entropy_coef=0.2,     # Increased from 0.1 to maintain exploration longer
                                     # Policy loss near zero suggests over-convergence
            entropy_min=0.1,      # Increased from 0.01 to maintain more exploration
            entropy_decay=0.99995,  # Slower decay (was 0.9995) to maintain exploration longer
            min_lr=1e-6,            # Reduced from 1e-4 to allow LR to decay lower
            total_episodes=NUM_EPISODES,
            actor_lr=8e-4,        # Reduced from 1e-3 for more stable policy updates
            critic_lr=1.5e-3,     # Reduced from 2e-3 for more stable value function learning
            use_ou_risk=True,       # Ornstein-Uhlenbeck (colored) noise on risk_mult for smooth exploration
            use_action_persistence=False,  # Bias logits toward last action for correlated, smoother exploration
        )
        train_agent(env, agent, num_episodes=NUM_EPISODES, start_episode=0)

if __name__ == "__main__":
    main()
