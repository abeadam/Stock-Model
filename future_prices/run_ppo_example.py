#!/usr/bin/env python3
"""
Example script showing how to run PPO training

Usage:
    python3 run_ppo_example.py                    # Fresh training
    python3 run_ppo_example.py --evaluate 10      # Evaluate latest checkpoint
    python3 run_ppo_example.py --resume-latest    # Resume from latest PPO checkpoint
    python3 run_ppo_example.py --resume-episode 100  # Resume from specific episode
"""

import os
import sys
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from futures_renforcement_ppo import PPOAgent
from futures_reinforcement_utils import SPXTradingEnv, train_agent, evaluate_agent

def main():
    """Main function to train/evaluate PPO agent"""
    
    # Configuration
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'es_with_indicators.csv')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'futures_model.pt')
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create environment
    env = SPXTradingEnv(
        DATA_PATH, 
        MODEL_PATH, 
        device=device,
        max_steps_per_episode=5000,
        max_loss_per_episode=-50000.0,
        stop_loss_per_contract=750.0  # Increased from 500.0 to 750.0 - less aggressive
    )
    
    # Get state size
    test_state = env.reset()
    state_size = len(test_state)
    action_size = env.action_space_size
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Check command line arguments
    evaluate_mode = '--evaluate' in sys.argv
    resume_mode = '--resume-episode' in sys.argv
    resume_latest = '--resume-latest' in sys.argv or '--resume' in sys.argv
    
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
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                           if f.startswith('rl_agent_ep') and f.endswith('.pt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: int(x.replace('rl_agent_ep', '').replace('.pt', '')))
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            
            # Create PPO agent
            agent = PPOAgent(
                state_size=state_size,
                action_size=action_size,
                device=device,
                network_size='large',
                entropy_coef=0.03  # Increased for more exploration
            )
            agent.load(checkpoint_path)
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
            
            # Evaluate
            evaluate_agent(env, agent, num_episodes=num_episodes, detailed=True)
        else:
            print("❌ No checkpoint found for evaluation!")
    
    elif resume_latest or resume_mode:
        # Resume training
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'reinforcement_results')
        checkpoint_path = None
        start_episode = 0
        
        if resume_latest:
            # Find latest PPO checkpoint automatically
            print("\n🔍 Looking for latest PPO checkpoint...")
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.startswith('rl_agent_ep') and f.endswith('.pt')]
            
            if checkpoint_files:
                # Filter for PPO checkpoints only
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
                    # Sort by episode number
                    ppo_checkpoints.sort(key=lambda x: x[1], reverse=True)
                    latest_file, latest_episode = ppo_checkpoints[0]
                    checkpoint_path = os.path.join(checkpoint_dir, latest_file)
                    start_episode = latest_episode
                    print(f"✓ Found latest PPO checkpoint: {latest_file} (episode {latest_episode})")
                else:
                    print("⚠️  No PPO checkpoints found (only DQN checkpoints exist)")
            else:
                print("⚠️  No checkpoints found in reinforcement_results/")
        
        elif resume_mode:
            # Resume from specific episode
            if '--resume-episode' in sys.argv:
                idx = sys.argv.index('--resume-episode')
                if idx + 1 < len(sys.argv):
                    try:
                        start_episode = int(sys.argv[idx + 1])
                    except:
                        pass
            
            checkpoint_path = os.path.join(checkpoint_dir, f'rl_agent_ep{start_episode}.pt')
        
        # Load checkpoint if found
        if checkpoint_path and os.path.exists(checkpoint_path):
            agent = PPOAgent(
                state_size=state_size,
                action_size=action_size,
                device=device,
                network_size='large',
                entropy_coef=0.03  # Increased for more exploration
            )
            loaded_episode = agent.load(checkpoint_path)
            if loaded_episode is not None:
                start_episode = loaded_episode
            print(f"✓ Resuming from episode {start_episode}")
        else:
            if checkpoint_path:
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("Starting fresh training...")
            agent = PPOAgent(
                state_size=state_size,
                action_size=action_size,
                device=device,
                network_size='large',
                entropy_coef=0.03  # Increased for more exploration
            )
            start_episode = 0
        
        # Train
        train_agent(env, agent, num_episodes=20000, start_episode=start_episode)
    
    else:
        # Fresh training
        print("\n🎯 Starting PPO training from scratch")
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            device=device,
            network_size='medium',  # or 'small' or 'large'
            lr=3e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.03,  # Increased from 0.01 to 0.03 - more exploration
            gae_lambda=0.95,
            update_epochs=4,
            batch_size=64
        )
        
        # Train
        train_agent(env, agent, num_episodes=20000, save_freq=100)

if __name__ == "__main__":
    main()

