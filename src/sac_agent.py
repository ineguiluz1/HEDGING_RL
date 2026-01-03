"""
SAC Agent for Hedging using Stable-Baselines3
==============================================

This module provides a wrapper around Stable-Baselines3's SAC implementation
for options hedging. It maintains API compatibility with the TD3 agent
while leveraging SB3's robust and tested algorithms.

SAC (Soft Actor-Critic) is an off-policy actor-critic algorithm that
incorporates entropy maximization for improved exploration.
"""

import numpy as np
import torch
import os
from typing import Optional, Dict, Any, Union, Callable

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from config import CONFIG, get_model_config, get_exploration_config


# Set device
device = torch.device("cuda" if torch.cuda.is_available() and CONFIG.get("use_gpu", False) else "cpu")


class SACAgent:
    """
    SAC Agent wrapper using Stable-Baselines3
    
    This class provides a simplified interface to SB3's SAC implementation
    while maintaining compatibility with the existing training infrastructure.
    
    SAC uses entropy regularization for exploration, so it doesn't need
    explicit action noise like TD3.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        env=None
    ):
        """
        Initialize SAC Agent
        
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
        
        # SAC-specific parameters
        self.ent_coef = CONFIG.get("sac_ent_coef", "auto")  # Automatic entropy tuning
        self.target_entropy = CONFIG.get("sac_target_entropy", "auto")
        
        # For compatibility with TD3 interface
        self.current_noise = exploration_config["initial_noise"]
        
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
        """Initialize the SB3 SAC model"""
        
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
        
        # SAC-specific learning rate (can be different from TD3)
        sac_lr = CONFIG.get("sac_learning_rate", model_config["actor_lr"])
        
        # Create SAC model
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=sac_lr,
            buffer_size=model_config["replay_buffer_size"],
            learning_starts=warmup_steps,
            batch_size=model_config["batch_size"],
            tau=model_config["tau"],
            gamma=model_config["gamma"],
            train_freq=1,
            gradient_steps=1,
            ent_coef=self.ent_coef,  # Automatic entropy coefficient
            target_update_interval=1,
            target_entropy=self.target_entropy,
            use_sde=CONFIG.get("sac_use_sde", False),  # State Dependent Exploration
            sde_sample_freq=CONFIG.get("sac_sde_sample_freq", -1),
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            seed=seed
        )
        
        # Configure logger for SB3
        if not hasattr(self.model, '_logger') or self.model._logger is None:
            from stable_baselines3.common.logger import configure as sb3_configure
            logger = sb3_configure(folder=None, format_strings=[])
            self.model.set_logger(logger)
    
    def set_env(self, env):
        """Set or update the environment"""
        self.env = env
        if self.model is None:
            model_config = get_model_config()
            exploration_config = get_exploration_config()
            self._init_model(env, model_config, exploration_config)
        else:
            self.model.set_env(env)
            if not hasattr(self.model, '_logger') or self.model._logger is None:
                from stable_baselines3.common.logger import configure as sb3_configure
                logger = sb3_configure(folder=None, format_strings=[])
                self.model.set_logger(logger)
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action given state
        
        Args:
            state: Current state
            add_noise: Whether to use stochastic policy (SAC always samples from policy)
        
        Returns:
            Selected action
        """
        if self.model is None:
            raise ValueError("Model not initialized. Set environment first.")
        
        # SAC uses stochastic policy - deterministic=False samples from distribution
        # deterministic=True uses mean of the distribution
        action, _ = self.model.predict(state, deterministic=not add_noise)
        
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
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.model.device)
        
        with torch.no_grad():
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
        """
        if self.model is not None and hasattr(self.model, 'replay_buffer'):
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
        
        if self.model.replay_buffer.size() < self.batch_size:
            return None, None
        
        self.total_steps += 1
        self.update_counter += 1
        
        # SB3's train performs one gradient step
        self.model.train(gradient_steps=1, batch_size=self.batch_size)
        
        return 0.0, 0.0
    
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
        """Reset exploration - SAC doesn't use explicit noise"""
        pass  # SAC uses entropy-based exploration
    
    def save(self, filepath: str):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        # Remove .pth extension if present (SB3 uses .zip)
        if filepath.endswith('.pth'):
            filepath = filepath[:-4]
        
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'model_type': 'SAC'
        }
        
        import json
        metadata_path = filepath + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        print(f"SAC Model saved to {filepath}.zip")
    
    def load(self, filepath: str, env=None):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to load the model from
            env: Environment (required if not already set)
        
        Returns:
            Configuration dictionary
        """
        if filepath.endswith('.pth'):
            filepath = filepath[:-4]
        if not filepath.endswith('.zip'):
            zip_path = filepath + '.zip'
        else:
            zip_path = filepath
            filepath = filepath[:-4]
        
        if env is not None:
            self.model = SAC.load(zip_path, env=env, device=device)
            self.env = env
        elif self.env is not None:
            self.model = SAC.load(zip_path, env=self.env, device=device)
        else:
            self.model = SAC.load(zip_path, device=device)
        
        import json
        metadata_path = filepath + '_metadata.json'
        config = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.total_steps = metadata.get('total_steps', 0)
            self.update_counter = metadata.get('update_counter', 0)
            config = metadata
        
        print(f"SAC Model loaded from {zip_path}")
        return config
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        buffer_size = 0
        if self.model is not None and hasattr(self.model, 'replay_buffer'):
            buffer_size = self.model.replay_buffer.size()
        
        # Get entropy coefficient if available
        ent_coef = None
        if self.model is not None and hasattr(self.model, 'ent_coef'):
            if isinstance(self.model.ent_coef, torch.Tensor):
                ent_coef = self.model.ent_coef.item()
            else:
                ent_coef = self.model.ent_coef
        
        return {
            'total_steps': self.total_steps,
            'update_counter': self.update_counter,
            'buffer_size': buffer_size,
            'entropy_coef': ent_coef,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
        }


def create_sac_agent(env, config: Optional[Dict] = None) -> SACAgent:
    """
    Factory function to create a SAC agent
    
    Args:
        env: Gymnasium environment
        config: Optional configuration override
    
    Returns:
        Initialized SACAgent
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    return SACAgent(state_dim, action_dim, config=config, env=env)
