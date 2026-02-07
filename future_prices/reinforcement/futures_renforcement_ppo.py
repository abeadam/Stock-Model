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
from typing import Optional, Tuple, Union
import pickle
import io


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
                 entropy_coef: float = 0.05, gae_lambda: float = 0.95,
                 update_epochs: int = 8, batch_size: int = 256,
                 network_size: str = 'large', min_lr: float = 1e-6,
                 entropy_min: float = 0.005, entropy_decay: float = 0.9995,
                 total_episodes: int = 80000, actor_lr: Optional[float] = None,
                 critic_lr: Optional[float] = None,
                 use_ou_risk: bool = False, ou_theta: float = 0.15, ou_mu: float = 0.0, ou_sigma: float = 0.2,
                 use_action_persistence: bool = False, persistence_bonus: float = 0.3, persistence_sigma: float = 15.0):
        """
        Initialize PPO Agent

        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            device: Device to run on
            lr: Learning rate (default 3e-4 is standard for PPO). Used if actor_lr/critic_lr not specified.
            gamma: Discount factor
            clip_epsilon: PPO clip parameter (0.1-0.3, default 0.2)
            value_coef: Value loss coefficient (0.5-1.0)
            entropy_coef: Initial entropy bonus coefficient (0.01-0.1)
            gae_lambda: GAE lambda parameter (0.9-0.99)
            update_epochs: Number of epochs per update (3-10)
            batch_size: Batch size for updates (also triggers mid-episode updates)
            network_size: Network capacity ('small', 'medium', 'large', 'xlarge')
            min_lr: Minimum learning rate floor (default: 1e-6)
            entropy_min: Minimum entropy coefficient floor (default: 0.005)
            entropy_decay: Rate to decay entropy every update (default: 0.9995)
            total_episodes: Total episodes for LR scheduler decay (default: 80000)
            actor_lr: Separate learning rate for actor (if None, uses lr)
            critic_lr: Separate learning rate for critic (if None, uses lr * 2.0 for faster value learning)
            use_ou_risk: If True, add Ornstein-Uhlenbeck (colored) noise to risk_multiplier for smooth exploration.
            ou_theta: OU mean-reversion speed (default 0.15).
            ou_mu: OU long-run mean (default 0).
            ou_sigma: OU volatility (default 0.2).
            use_action_persistence: If True, bias action logits toward last action (Gaussian kernel) for smoother, correlated exploration.
            persistence_bonus: Logit bonus at last action (default 0.3).
            persistence_sigma: Gaussian width over action indices (default 15 for ~201 actions).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.entropy_min = entropy_min
        self.entropy_decay = entropy_decay
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.network_size = network_size
        self.min_lr = min_lr
        self.total_episodes = total_episodes
        
        # Set separate learning rates for actor and critic
        # Default: critic learns 2x faster to address value loss bottleneck
        self.actor_lr = actor_lr if actor_lr is not None else lr
        self.critic_lr = critic_lr if critic_lr is not None else (lr * 2.0)
        
        # Build actor-critic network
        self.actor_critic = self._build_network(network_size).to(device)
        
        # Create optimizer with separate parameter groups for actor and critic
        # Feature layers are shared, so we'll include them in both groups (they'll use actor_lr)
        # Use named_parameters to filter by module name (avoids type inference issues)
        feature_params = []
        actor_params = []
        critic_params = []
        risk_params = []
        
        for name, param in self.actor_critic.named_parameters():
            if name.startswith('feature_layers'):
                feature_params.append(param)
            elif name.startswith('actor'):
                actor_params.append(param)
            elif name.startswith('critic'):
                critic_params.append(param)
            elif name.startswith('risk_head'):
                risk_params.append(param)
        
        # Group parameters: feature layers and actor use actor_lr, critic uses critic_lr
        # Note: feature layers are in actor group, but this is fine since they're shared
        self.optimizer = optim.Adam([
            {'params': feature_params + actor_params + risk_params, 'lr': self.actor_lr, 'name': 'actor'},
            {'params': critic_params, 'lr': self.critic_lr, 'name': 'critic'}
        ])
        
        # Using CosineAnnealingLR for "fast learning then slow down"
        # T_max is the number of iterations until first restart/end.
        # Note: CosineAnnealingLR will decay both parameter groups proportionally
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_episodes,
            eta_min=min_lr
        )
        self.training_steps = 0  # Track number of training steps for scheduler
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Feature scaler (for normalizing features during inference)
        # This should be set from the training environment's scaler
        # The scaler is used to normalize features to match training distribution
        self.feature_scaler = None
        
        # Normalization stats for value targets (returns)
        # Used to normalize returns and value predictions for stable learning
        self.returns_mean = 0.0
        self.returns_std = 1.0

        # Correlated (colored) noise for exploration
        self.use_ou_risk = use_ou_risk
        self.ou_theta = ou_theta
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.use_action_persistence = use_action_persistence
        self.persistence_bonus = persistence_bonus
        self.persistence_sigma = persistence_sigma
        self._ou_state = ou_mu
        self._last_action: Optional[int] = None

    def on_episode_end(self) -> None:
        """Reset exploration state at episode boundaries (for colored noise)."""
        self._ou_state = self.ou_mu
        self._last_action = None

    def set_feature_scaler(self, scaler):
        """
        Set the feature scaler from the training environment
        
        Args:
            scaler: StandardScaler instance from SPXTradingEnv.feature_scaler
        """
        self.feature_scaler = scaler
        
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
        elif network_size == 'xlarge':
            feature_dims = [512, 512, 256, 128]
            actor_dims = [256, 128, self.action_size]
            critic_dims = [256, 128, 1]
        else:
            raise ValueError(f"network_size must be 'small', 'medium', 'large', or 'xlarge', got '{network_size}'")
        
        class ActorCritic(nn.Module):
            def __init__(self, state_size, feature_dims, actor_dims, critic_dims):
                super(ActorCritic, self).__init__()
                
                # Shared feature layers
                feature_layers = []
                prev_dim = state_size
                for dim in feature_dims:
                    feature_layers.append(nn.Linear(prev_dim, dim))
                    feature_layers.append(nn.GELU())
                    prev_dim = dim
                self.feature_layers = nn.Sequential(*feature_layers)
                
                # Actor head (policy)
                actor_layers = []
                prev_dim = feature_dims[-1]
                for dim in actor_dims:
                    actor_layers.append(nn.Linear(prev_dim, dim))
                    if dim != actor_dims[-1]:
                        actor_layers.append(nn.GELU())
                    prev_dim = dim
                self.actor = nn.Sequential(*actor_layers)
                
                # Critic head (value)
                critic_layers = []
                prev_dim = feature_dims[-1]
                for dim in critic_dims:
                    critic_layers.append(nn.Linear(prev_dim, dim))
                    if dim != critic_dims[-1]:
                        critic_layers.append(nn.GELU())
                    prev_dim = dim
                self.critic = nn.Sequential(*critic_layers)
                
                # Risk head (Continuous output for risk multiplier)
                # Maps to a single value that we will scale to [0.5, 2.0]
                self.risk_head = nn.Sequential(
                    nn.Linear(feature_dims[-1], 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid() # Scale to [0, 1] then transform to [0.5, 2.0]
                )
            
            def forward(self, state):
                features = self.feature_layers(state)
                action_logits = self.actor(features)
                value = self.critic(features)
                
                # Transform sigmoid output [0, 1] to risk range [0.5, 2.0]
                risk_raw = self.risk_head(features)
                risk_multiplier = 0.5 + (risk_raw * 1.5)
                
                return action_logits, value, risk_multiplier
            
            def get_action_and_value(self, state):
                """Get action, log_prob, value, and risk for a state"""
                action_logits, value, risk_mult = self.forward(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return int(action.item()), log_prob, value.squeeze(), risk_mult.item()
        
        return ActorCritic(self.state_size, feature_dims, actor_dims, critic_dims)
    
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """
        Select action using current policy

        Args:
            state: Current state
            training: If True, sample from distribution and store trajectory data.
                     If False, use deterministic action (mode) for evaluation.

        Returns:
            (action_index, risk_multiplier)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value, risk_multiplier = self.actor_critic(state_tensor)

            # Correlated (colored) noise: bias logits toward last action (smoother exploration)
            if self.use_action_persistence and training and self._last_action is not None:
                indices = torch.arange(
                    self.action_size, device=action_logits.device, dtype=action_logits.dtype
                )
                kernel = torch.exp(
                    -((indices - self._last_action) ** 2)
                    / (2.0 * self.persistence_sigma ** 2)
                )
                action_logits = action_logits + self.persistence_bonus * kernel.unsqueeze(0)

            dist = Categorical(logits=action_logits)

            if training:
                # During training: sample from distribution for exploration
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # During evaluation: use deterministic action (most likely)
                action = action_logits.argmax(dim=-1)
                log_prob = dist.log_prob(action)

        risk_val = float(risk_multiplier.item())
        # Correlated (colored) noise: Ornstein-Uhlenbeck on risk_multiplier (smooth exploration)
        if self.use_ou_risk and training:
            self._ou_state = (
                self._ou_state
                + self.ou_theta * (self.ou_mu - self._ou_state)
                + self.ou_sigma * np.random.randn()
            )
            risk_val = float(np.clip(risk_val + self._ou_state, 0.5, 2.0))

        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
            # Note: dones will be set by store_reward() after step() is called

        self._last_action = int(action.item())
        return int(action.item()), risk_val
    
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
        # Safety check: ensure all arrays have the same length
        n_rewards = len(self.rewards)
        n_values = len(self.values)
        n_dones = len(self.dones)
        if n_rewards != n_values or n_rewards != n_dones:
            raise ValueError(
                f"Array length mismatch in compute_gae: "
                f"rewards={n_rewards}, values={n_values}, dones={n_dones}. "
                f"All arrays must have the same length."
            )
        
        advantages = []
        returns = []
        gae = 0
        
        # Build values array with proper terminal values at episode boundaries
        # For each episode ending (done=True), we need terminal_value=0.0
        # Only the very last step uses the provided next_value parameter
        # Note: next_value should be normalized if returns are normalized
        values = list(self.values)  # Copy values
        values.append(next_value)  # Add terminal value for the last step
        
        # Compute advantages backwards
        for step in reversed(range(len(self.rewards))):
            if self.dones[step]:
                # Episode ended: don't bootstrap from future (use terminal value 0.0)
                # For intermediate episodes, the "next value" is implicitly 0.0
                # For the last step, we use the provided next_value (which is 0.0 when done=True)
                if step == len(self.rewards) - 1:
                    # Last step of trajectory: use provided next_value
                    delta = self.rewards[step] - values[step]
                else:
                    # Intermediate episode boundary: terminal value is 0.0 (episode ended)
                    # Don't bootstrap from next episode's value
                    delta = self.rewards[step] - values[step]
                gae = delta  # Reset GAE at episode boundary
            else:
                # Normal step: bootstrap from next step's value
                delta = self.rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return np.array(advantages), np.array(returns)
    
    def update(self, terminal_value: float = 0.0):
        """
        Update policy using PPO clipped objective
        
        Args:
            terminal_value: Value estimate for terminal state (0 if done)
        
        Returns:
            Average loss over update epochs, or None if insufficient samples
        """
        # Need at least 1 sample to update (batch_size is just for batching, not a minimum)
        if len(self.states) == 0:
            return None
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value=terminal_value)
        
        # Normalize advantages (standard practice in PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalize value targets (returns) for stability (critical for consistent learning)
        # Returns can vary wildly, making value function learning unstable
        # Normalization helps the value network learn more consistently
        # Store normalization stats for denormalization if needed (currently not used)
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        returns_normalized = (returns - returns_mean) / returns_std
        
        # Also clip normalized returns to prevent extreme outliers from destabilizing training
        # After normalization, clip to reasonable range (e.g., ±10 standard deviations)
        returns_normalized = np.clip(returns_normalized, -3.0, 3.0)
        
        # Store normalization stats for potential future use (e.g., denormalizing predictions)
        self.returns_mean = returns_mean
        self.returns_std = returns_std
        
        # Use normalized returns as value targets
        returns = returns_normalized
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        batch_count = 0
        
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
                action_logits, values, _ = self.actor_critic(batch_states)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute policy ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Normalize value predictions to match normalized returns
                # This ensures value function learns to predict normalized returns
                values_normalized = (values.view(-1) - self.returns_mean) / (self.returns_std + 1e-8)
                
                # Value loss: Huber (smooth L1) is more robust to outlier returns than MSE.
                # With small datasets or high reward variance, MSE can blow up and increase total loss.
                value_loss = nn.functional.huber_loss(values_normalized, batch_returns, delta=1.0)
                # Clip value loss to prevent extreme values from destabilizing training
                value_loss = torch.clamp(value_loss, max=10000.0)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Check for NaN/Inf before backprop
                if not torch.isfinite(loss):
                    print(f"   ⚠️  Non-finite loss detected: policy={policy_loss.item():.2f}, "
                          f"value={value_loss.item():.2f}, entropy={entropy.item():.2f}")
                    # Skip this batch if loss is invalid
                    continue
                
                # Clip total loss to prevent extreme values (safety net)
                loss = torch.clamp(loss, min=-10000.0, max=10000.0)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients and stabilize training
                # Reduced from 2.0 to 1.0 for more conservative clipping to prevent instability
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
                self.optimizer.step()
                
                # Accumulate loss components for logging
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                batch_count += 1
        
        # Increment training step counter (once per update call, not per batch)
        self.training_steps += 1
        
        # Step the learning rate scheduler (StepLR counts calls to scheduler.step())
        # StepLR will automatically decay LR every step_size calls
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        
        # Enforce learning rate floor for both parameter groups
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
        
        # Get learning rates for logging (actor and critic may differ)
        actor_lr = self.optimizer.param_groups[0]['lr']
        critic_lr = self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else actor_lr
        new_lr = actor_lr  # For backward compatibility
        
        # Decay entropy coefficient
        old_entropy = self.entropy_coef
        self.entropy_coef = max(self.entropy_min, self.entropy_coef * self.entropy_decay)
        
        # Log LR and Entropy changes
        if self.training_steps % 100 == 0:
            if old_lr != new_lr:
                if len(self.optimizer.param_groups) > 1:
                    print(f"   📉 LR decay at update {self.training_steps}: Actor {actor_lr:.6f}, Critic {critic_lr:.6f}")
                else:
                    print(f"   📉 LR decay at update {self.training_steps}: {old_lr:.6f} → {new_lr:.6f}")
            if old_entropy != self.entropy_coef:
                print(f"   🧠 Entropy decay: {old_entropy:.4f} → {self.entropy_coef:.4f}")
        
        # Clear trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Return loss components as dictionary
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_policy_loss = total_policy_loss / batch_count
            avg_value_loss = total_value_loss / batch_count
            avg_entropy = total_entropy / batch_count
            return {
                'total_loss': avg_loss,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'entropy': avg_entropy
            }
        else:
            return None
    
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
            'entropy_min': self.entropy_min,
            'entropy_decay': self.entropy_decay,
            'total_episodes': self.total_episodes,
            'gae_lambda': self.gae_lambda,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'use_ou_risk': self.use_ou_risk,
            'ou_theta': self.ou_theta,
            'ou_mu': self.ou_mu,
            'ou_sigma': self.ou_sigma,
            'use_action_persistence': self.use_action_persistence,
            'persistence_bonus': self.persistence_bonus,
            'persistence_sigma': self.persistence_sigma,
        }
        if episode is not None:
            save_dict['episode'] = episode
        
        # Save feature scaler if available
        if self.feature_scaler is not None:
            # Serialize the scaler using pickle
            scaler_buffer = io.BytesIO()
            pickle.dump(self.feature_scaler, scaler_buffer)
            save_dict['feature_scaler'] = scaler_buffer.getvalue()
            
            # Also save to feature_scaler.pkl file for easy access
            try:
                import os
                results_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'
                scaler_file = os.path.join(results_dir, 'feature_scaler.pkl')
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
            except Exception as e:
                # Non-critical error, just warn
                pass
        
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
        
        # Robust loading: filter state_dict to match current architecture
        checkpoint_state = checkpoint['actor_critic_state_dict']
        current_state = self.actor_critic.state_dict()
        
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
        
        # Load with strict=False to allow partial matches
        self.actor_critic.load_state_dict(filtered_checkpoint, strict=True)
        
        # Load optimizer and scheduler if available
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"   ⚠️  Could not load optimizer state: {e}")
            
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"   ⚠️  Could not load scheduler state: {e}")
                
        if 'training_steps' in checkpoint:
            self.training_steps = checkpoint['training_steps']
        
        if 'total_episodes' in checkpoint:
            self.total_episodes = checkpoint['total_episodes']
            # Re-initialize scheduler with the correct T_max if it changed
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_episodes,
                eta_min=self.min_lr,
                last_epoch=self.training_steps - 1
            )

        # Restore colored-noise (OU / action persistence) settings if present
        for key in ('use_ou_risk', 'ou_theta', 'ou_mu', 'ou_sigma',
                    'use_action_persistence', 'persistence_bonus', 'persistence_sigma'):
            if key in checkpoint:
                setattr(self, key, checkpoint[key])
        self._ou_state = self.ou_mu
        self._last_action = None

        # Load feature scaler if available
        if 'feature_scaler' in checkpoint:
            scaler_buffer = io.BytesIO(checkpoint['feature_scaler'])
            self.feature_scaler = pickle.load(scaler_buffer)
            print(f"   ✓ Loaded feature scaler from checkpoint")
        else:
            print(f"   ⚠️  No feature scaler found in checkpoint (will need to fit new scaler for testing)")
            
        return checkpoint.get('episode', None)

