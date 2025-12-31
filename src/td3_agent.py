"""
TD3 Agent for Hedging using Stable-Baselines3
==============================================

This module provides a wrapper around Stable-Baselines3's TD3 implementation
for options hedging. It maintains API compatibility with the custom implementation
while leveraging SB3's robust and tested algorithms.

For the original custom implementation, see td3_agent_custom.py
"""

import numpy as np
import torch
import os
from typing import Optional, Dict, Any, Union, Callable

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.logger import configure

from config import CONFIG, get_model_config, get_exploration_config


# Set device
device = torch.device("cuda" if torch.cuda.is_available() and CONFIG.get("use_gpu", False) else "cpu")


class TD3Agent:
    """
    TD3 Agent wrapper using Stable-Baselines3
    
    This class provides a simplified interface to SB3's TD3 implementation
    while maintaining compatibility with the existing training infrastructure.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        env=None
    ):
        """
        Initialize TD3 Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Optional configuration override
            env: Gymnasium environment (required for SB3)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Load configuration
        model_config = get_model_config()
        exploration_config = get_exploration_config()
        
        if config is not None:
            model_config.update(config)
        
        # Store key parameters
        self.max_action = model_config["max_action"]
        self.gamma = model_config["gamma"]
        self.tau = model_config["tau"]
        self.batch_size = model_config["batch_size"]
        self.buffer_size = model_config["replay_buffer_size"]
        
        # Exploration parameters
        self.initial_noise = exploration_config["initial_noise"]
        self.final_noise = exploration_config["final_noise"]
        self.noise_decay_steps = exploration_config["noise_decay_steps"]
        self.current_noise = self.initial_noise
        
        # Training counters
        self.total_steps = 0
        self.update_counter = 0
        
        # Store losses for compatibility
        self.actor_losses = []
        self.critic_losses = []
        
        # Initialize model if environment is provided
        self.model = None
        self.env = env
        
        if env is not None:
            self._init_model(env, model_config, exploration_config)
    
    def _init_model(self, env, model_config: Dict, exploration_config: Dict):
        """Initialize the SB3 TD3 model"""
        
        # Setup action noise
        n_actions = env.action_space.shape[0]
        
        # Use Ornstein-Uhlenbeck noise for exploration
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=self.initial_noise * np.ones(n_actions),
            theta=CONFIG.get("ou_theta", 0.15)
        )
        
        # Policy kwargs for network architecture
        policy_kwargs = dict(
            net_arch=dict(
                pi=[model_config["hidden_dim"], model_config["hidden_dim"]],
                qf=[model_config["hidden_dim"], model_config["hidden_dim"]]
            )
        )
        
        # Get warmup steps from config (random exploration before learning starts)
        warmup_steps = CONFIG.get("warmup_steps", 5000)
        
        # Get seed for reproducibility
        seed = CONFIG.get("seed", 101)
        
        # Create TD3 model
        self.model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=model_config["actor_lr"],
            buffer_size=model_config["replay_buffer_size"],
            learning_starts=warmup_steps,  # Use warmup from config for proper exploration
            batch_size=model_config["batch_size"],
            tau=model_config["tau"],
            gamma=model_config["gamma"],
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=model_config["policy_freq"],
            target_policy_noise=model_config["policy_noise"],
            target_noise_clip=model_config["noise_clip"],
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            seed=seed  # Set seed for SB3 reproducibility
        )
        
        # Configure logger for SB3 (required for train() method)
        if not hasattr(self.model, '_logger') or self.model._logger is None:
            from stable_baselines3.common.logger import configure as sb3_configure
            logger = sb3_configure(folder=None, format_strings=[])
            self.model.set_logger(logger)
        
        self.action_noise = action_noise
    
    def set_env(self, env):
        """Set or update the environment"""
        self.env = env
        if self.model is None:
            model_config = get_model_config()
            exploration_config = get_exploration_config()
            self._init_model(env, model_config, exploration_config)
        else:
            self.model.set_env(env)
            # Ensure logger is configured after setting new env
            if not hasattr(self.model, '_logger') or self.model._logger is None:
                from stable_baselines3.common.logger import configure as sb3_configure
                logger = sb3_configure(folder=None, format_strings=[])
                self.model.set_logger(logger)
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
        
        Returns:
            Selected action
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set environment first.")
        
        # Get action from model
        action, _ = self.model.predict(state, deterministic=not add_noise)
        
        # Ensure action is numpy array with correct shape
        if isinstance(action, (int, float)):
            action = np.array([action])
        
        return action.flatten()
    
    def get_q_values(self, state: np.ndarray, action: np.ndarray) -> Dict[str, Any]:
        """
        Get Q-values for logging (compatibility method)
        
        Args:
            state: Current state
            action: Action taken
        
        Returns:
            Dictionary with Q-values and metadata
        """
        if self.model is None:
            return {'q1': 0, 'q2': 0, 'q_mean': 0, 'action_raw': action[0], 'state': state}
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.model.device)
        
        with torch.no_grad():
            # Get Q-values from both critics
            q1 = self.model.critic.q1_forward(state_tensor, action_tensor)
            q2_input = torch.cat([state_tensor, action_tensor], dim=1)
            # Access q2 through the critic's forward
            q_values = self.model.critic(state_tensor, action_tensor)
            q1_val = q_values[0].cpu().item()
            q2_val = q_values[1].cpu().item()
        
        return {
            'q1': q1_val,
            'q2': q2_val,
            'q_mean': (q1_val + q2_val) / 2,
            'action_raw': action[0] if len(action) > 0 else action,
            'state': state
        }
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer
        
        Note: SB3 handles this internally during learn(), but we provide
        this method for compatibility with custom training loops.
        """
        if self.model is not None and hasattr(self.model, 'replay_buffer'):
            # Reshape for SB3's buffer format
            self.model.replay_buffer.add(
                obs=np.array([state]),
                next_obs=np.array([next_state]),
                action=np.array([action]),
                reward=np.array([reward]),
                done=np.array([done]),
                infos=[{}]
            )
    
    def train_step(self):
        """
        Perform one training step
        
        Returns:
            tuple: (actor_loss, critic_loss) or (None, None) if buffer not ready
        """
        if self.model is None:
            return None, None
        
        # Check if we have enough samples
        if self.model.replay_buffer.size() < self.batch_size:
            return None, None
        
        self.total_steps += 1
        self.update_counter += 1
        
        # SB3's train performs one gradient step
        self.model.train(gradient_steps=1, batch_size=self.batch_size)
        
        # Update noise (decay)
        self._update_noise()
        
        # Return placeholder losses (SB3 doesn't expose these directly)
        # For actual loss tracking, use TensorBoard logging
        return 0.0, 0.0
    
    def _update_noise(self):
        """Decay exploration noise"""
        if self.total_steps < self.noise_decay_steps:
            decay_ratio = self.total_steps / self.noise_decay_steps
            self.current_noise = self.initial_noise - (self.initial_noise - self.final_noise) * decay_ratio
        else:
            self.current_noise = self.final_noise
        
        # Update the action noise sigma
        if self.action_noise is not None:
            self.action_noise._sigma = self.current_noise * np.ones(self.action_dim)
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
        progress_bar: bool = True
    ):
        """
        Train the agent using SB3's native training loop
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for logging/early stopping
            log_interval: Interval for logging
            progress_bar: Whether to show progress bar
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set environment first.")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            progress_bar=progress_bar
        )
        
        self.total_steps = self.model.num_timesteps
    
    def reset_noise(self):
        """Reset exploration noise for new episode"""
        if self.action_noise is not None:
            self.action_noise.reset()
    
    def save(self, filepath: str):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Remove .pth extension if present (SB3 uses .zip)
        if filepath.endswith('.pth'):
            filepath = filepath[:-4]
        
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'current_noise': self.current_noise,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        import json
        metadata_path = filepath + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"Model saved to {filepath}.zip")
    
    def load(self, filepath: str, env=None):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load the model from
            env: Environment (required if not already set)
        
        Returns:
            Configuration dictionary
        """
        # Handle path extension
        if filepath.endswith('.pth'):
            filepath = filepath[:-4]
        if not filepath.endswith('.zip'):
            zip_path = filepath + '.zip'
        else:
            zip_path = filepath
            filepath = filepath[:-4]
        
        # Load the model
        if env is not None:
            self.model = TD3.load(zip_path, env=env, device=device)
            self.env = env
        elif self.env is not None:
            self.model = TD3.load(zip_path, env=self.env, device=device)
        else:
            self.model = TD3.load(zip_path, device=device)
        
        # Load metadata if available
        import json
        metadata_path = filepath + '_metadata.json'
        config = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.total_steps = metadata.get('total_steps', 0)
            self.update_counter = metadata.get('update_counter', 0)
            self.current_noise = metadata.get('current_noise', self.final_noise)
            config = metadata
        
        print(f"Model loaded from {zip_path}")
        return config
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        buffer_size = 0
        if self.model is not None and hasattr(self.model, 'replay_buffer'):
            buffer_size = self.model.replay_buffer.size()
        
        return {
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'current_noise': self.current_noise,
            'buffer_size': buffer_size,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
        }


class HedgingCallback(BaseCallback):
    """
    Custom callback for tracking hedging-specific metrics during training
    """
    
    def __init__(
        self,
        eval_env=None,
        eval_freq: int = 1000,
        log_path: Optional[str] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        
        self.episode_rewards = []
        self.episode_pnls = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called after each step"""
        # Get episode info if available
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                if 'cumulative_pnl' in info:
                    self.episode_pnls.append(info['cumulative_pnl'])
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            if self.verbose > 0 and self.num_timesteps % 1000 == 0:
                print(f"Timesteps: {self.num_timesteps}, Mean Reward (last 10): {mean_reward:.4f}")


def create_td3_agent(env, config: Optional[Dict] = None) -> TD3Agent:
    """
    Factory function to create a TD3 agent
    
    Args:
        env: Gymnasium environment
        config: Optional configuration override
    
    Returns:
        Initialized TD3Agent
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    return TD3Agent(state_dim, action_dim, config=config, env=env)
