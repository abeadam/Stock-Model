"""
Proximal Policy Optimization (PPO) Agent for SPX Trading

This module implements a PPO agent that can be used as an alternative to DQN.
PPO is a policy gradient method that:
- Learns a policy directly (actor network)
- Estimates state values (critic network)
- Uses clipped objective to prevent large policy updates
- Handles high variance better than DQN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for SPX Trading
    
    PPO is a policy gradient method that:
    - Learns a policy directly (actor network)
    - Estimates state values (critic network)
    - Uses clipped objective to prevent large policy updates
    - Handles high variance better than DQN
    """
    
    def __init__(self, state_size: int, action_size: int, device: str = 'cpu',
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, gae_lambda: float = 0.95,
                 update_epochs: int = 4, batch_size: int = 64,
                 network_size: str = 'large'):
        """
        Initialize PPO Agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            device: Device to run on
            lr: Learning rate (default 3e-4 is standard for PPO)
            gamma: Discount factor
            clip_epsilon: PPO clip parameter (0.1-0.3, default 0.2)
            value_coef: Value loss coefficient (0.5-1.0)
            entropy_coef: Entropy bonus coefficient (0.01-0.1)
            gae_lambda: GAE lambda parameter (0.9-0.99)
            update_epochs: Number of epochs per update (3-10)
            batch_size: Batch size for updates
            network_size: Network capacity ('small', 'medium', 'large')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.network_size = network_size
        
        # Build actor-critic network
        self.actor_critic = self._build_network(network_size).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Learning rate scheduler to gradually reduce LR during training
        # PPO updates happen once per episode (when trajectory is collected)
        # With ~4000 steps/episode × 2000 episodes = 2000 updates total
        # Step every 20 updates (~20 episodes) to get ~100 LR updates over full training
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,  # every 20 updates (~20 episodes)
            gamma=0.98  # Decay by 2% (gentle decay for long training)
        )
        self.training_steps = 0  # Track number of training steps for scheduler
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def _build_network(self, network_size: str = 'medium') -> nn.Module:
        """Build actor-critic network"""
        # Define network architectures
        if network_size == 'small':
            feature_dims = [64, 64]
            actor_dims = [32, self.action_size]
            critic_dims = [32, 1]
        elif network_size == 'medium':
            feature_dims = [128, 128]
            actor_dims = [64, self.action_size]
            critic_dims = [64, 1]
        elif network_size == 'large':
            feature_dims = [256, 256, 128]
            actor_dims = [128, self.action_size]
            critic_dims = [128, 1]
        else:
            raise ValueError(f"network_size must be 'small', 'medium', or 'large', got '{network_size}'")
        
        class ActorCritic(nn.Module):
            def __init__(self, state_size, feature_dims, actor_dims, critic_dims):
                super(ActorCritic, self).__init__()
                
                # Shared feature layers
                feature_layers = []
                prev_dim = state_size
                for dim in feature_dims:
                    feature_layers.append(nn.Linear(prev_dim, dim))
                    feature_layers.append(nn.ReLU())
                    prev_dim = dim
                self.feature_layers = nn.Sequential(*feature_layers)
                
                # Actor head (policy)
                actor_layers = []
                prev_dim = feature_dims[-1]
                for dim in actor_dims:
                    actor_layers.append(nn.Linear(prev_dim, dim))
                    if dim != actor_dims[-1]:
                        actor_layers.append(nn.ReLU())
                    prev_dim = dim
                self.actor = nn.Sequential(*actor_layers)
                
                # Critic head (value)
                critic_layers = []
                prev_dim = feature_dims[-1]
                for dim in critic_dims:
                    critic_layers.append(nn.Linear(prev_dim, dim))
                    if dim != critic_dims[-1]:
                        critic_layers.append(nn.ReLU())
                    prev_dim = dim
                self.critic = nn.Sequential(*critic_layers)
            
            def forward(self, state):
                features = self.feature_layers(state)
                action_logits = self.actor(features)
                value = self.critic(features)
                return action_logits, value
            
            def get_action_and_value(self, state):
                """Get action, log_prob, and value for a state"""
                action_logits, value = self.forward(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return int(action.item()), log_prob, value.squeeze()
        
        return ActorCritic(self.state_size, feature_dims, actor_dims, critic_dims)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using current policy
        
        Args:
            state: Current state
            training: If True, store trajectory data
        
        Returns:
            Action index
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits, value = self.actor_critic(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
        
        return int(action.item())
    
    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag for current step"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            next_value: Value estimate for terminal state (0 if done)
        
        Returns:
            advantages: Advantage estimates
            returns: Discounted returns
        """
        advantages = []
        returns = []
        gae = 0
        
        # Add next_value to values for computation
        values = self.values + [next_value]
        
        # Compute advantages backwards
        for step in reversed(range(len(self.rewards))):
            if self.dones[step]:
                delta = self.rewards[step] - values[step]
                gae = delta
            else:
                delta = self.rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return np.array(advantages), np.array(returns)
    
    def update(self):
        """
        Update policy using PPO clipped objective
        
        Returns:
            Average loss over update epochs
        """
        if len(self.states) < self.batch_size:
            return None
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        total_loss = 0.0
        
        # Update for multiple epochs
        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Get current policy predictions
                action_logits, values = self.actor_critic(batch_states)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute policy ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Increment training step counter (once per update call, not per batch)
        self.training_steps += 1
        
        # Step the learning rate scheduler (StepLR counts calls to scheduler.step())
        # StepLR will automatically decay LR every step_size calls
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        new_lr = self.optimizer.param_groups[0]['lr']
        
        # Log LR changes when they occur (only every 100 updates to avoid spam)
        if old_lr != new_lr and self.training_steps % 100 == 0:
            print(f"   📉 LR decay at update {self.training_steps}: {old_lr:.6f} → {new_lr:.6f}")
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        return total_loss / (self.update_epochs * (len(states) // self.batch_size + 1))
    
    def save(self, filepath: str, episode: Optional[int] = None):
        """Save agent to file"""
        save_dict = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_steps': self.training_steps,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'network_size': self.network_size,
            'lr': self.lr,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'gae_lambda': self.gae_lambda
        }
        if episode is not None:
            save_dict['episode'] = episode
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str) -> Optional[int]:
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Check if this is a PPO checkpoint
        if 'actor_critic_state_dict' not in checkpoint:
            raise ValueError(
                f"Checkpoint {filepath} is not a PPO checkpoint. "
                f"It appears to be a DQN checkpoint (has 'q_network_state_dict'). "
                f"Please use DQN agent (without --ppo flag) to load this checkpoint, "
                f"or train a new PPO agent from scratch."
            )
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'training_steps' in checkpoint:
            self.training_steps = checkpoint['training_steps']
        return checkpoint.get('episode', None)

