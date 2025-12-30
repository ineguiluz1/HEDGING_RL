"""
Training Script for TD3 Hedging Agent using Stable-Baselines3
==============================================================

This module provides training utilities for the TD3 agent using SB3.
It supports both the native SB3 training loop and a custom training
loop for compatibility with the original implementation.

For the original custom implementation, see trainer_custom.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CONFIG, get_model_config, get_environment_config
from td3_agent import TD3Agent, HedgingCallback, create_td3_agent, device
from data_loader import (
    create_environments_for_training,
    create_train_environments_list,
    load_hedging_data,
    split_data_by_years,
    create_environment
)
from benchmark import run_benchmark, delta_hedging


class TrainingMetrics:
    """Track and store training metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_lengths = []
        self.validation_rewards = []
        self.validation_pnls = []
        self.actor_losses = []
        self.critic_losses = []
        self.exploration_noise = []
        self.timestamps = []
    
    def add_episode(self, reward, pnl, length, actor_loss=0, critic_loss=0, noise=0):
        self.episode_rewards.append(reward)
        self.episode_pnls.append(pnl)
        self.episode_lengths.append(length)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.exploration_noise.append(noise)
        self.timestamps.append(datetime.now())
    
    def add_validation(self, reward, pnl):
        self.validation_rewards.append(reward)
        self.validation_pnls.append(pnl)
    
    def to_dict(self):
        return {
            'episode_rewards': self.episode_rewards,
            'episode_pnls': self.episode_pnls,
            'episode_lengths': self.episode_lengths,
            'validation_rewards': self.validation_rewards,
            'validation_pnls': self.validation_pnls,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'exploration_noise': self.exploration_noise
        }


class TrainingCallback(BaseCallback):
    """Custom callback for detailed training metrics"""
    
    def __init__(
        self,
        metrics: TrainingMetrics,
        val_env=None,
        validation_interval: int = 10,
        save_path: str = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.metrics = metrics
        self.val_env = val_env
        self.validation_interval = validation_interval
        self.save_path = save_path
        self.best_val_reward = float('-inf')
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_pnl = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Called after each environment step"""
        self.current_episode_length += 1
        
        # Get reward from locals
        if 'rewards' in self.locals:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check for episode end
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_count += 1
            
            # Get P&L from info if available
            pnl = 0
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                pnl = self.locals['infos'][0].get('cumulative_pnl', 0)
            
            # Get noise level from model
            noise = 0.1  # Default
            if hasattr(self.model, 'action_noise') and self.model.action_noise is not None:
                noise = float(np.mean(self.model.action_noise._sigma))
            
            # Record metrics
            self.metrics.add_episode(
                reward=self.current_episode_reward,
                pnl=pnl,
                length=self.current_episode_length,
                actor_loss=0,  # SB3 doesn't expose these directly
                critic_loss=0,
                noise=noise
            )
            
            # Print progress
            if self.verbose > 0 and self.episode_count % CONFIG.get("training_report_interval", 10) == 0:
                print(f"Episode {self.episode_count} | "
                      f"Reward: {self.current_episode_reward:.4f} | "
                      f"P&L: {pnl:.4f} | "
                      f"Steps: {self.current_episode_length} | "
                      f"Total Steps: {self.num_timesteps}")
            
            # Validation
            if self.val_env is not None and self.episode_count % self.validation_interval == 0:
                val_stats = self._evaluate(self.val_env)
                self.metrics.add_validation(val_stats['total_reward'], val_stats['total_pnl'])
                
                if self.verbose > 0:
                    print(f"  → Validation: Reward={val_stats['total_reward']:.4f}, "
                          f"P&L={val_stats['total_pnl']:.4f}")
                
                # Save best model
                if val_stats['total_reward'] > self.best_val_reward and self.save_path:
                    self.best_val_reward = val_stats['total_reward']
                    best_path = self.save_path.replace('.pth', '_best').replace('.zip', '_best')
                    self.model.save(best_path)
                    if self.verbose > 0:
                        print(f"  → New best model saved! (Reward: {self.best_val_reward:.4f})")
            
            # Reset episode tracking
            self.current_episode_reward = 0
            self.current_episode_pnl = 0
            self.current_episode_length = 0
        
        return True
    
    def _evaluate(self, env) -> Dict[str, float]:
        """Evaluate the model on the given environment"""
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        total_reward = 0
        total_pnl = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            result = env.step(action)
            
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            
            total_reward += reward
            total_pnl += info.get('step_pnl', 0)
            steps += 1
        
        return {
            'total_reward': total_reward,
            'total_pnl': info.get('cumulative_pnl', total_pnl),
            'steps': steps
        }


def train_episode(agent: TD3Agent, env, update_every: int = 1, log_q_values: bool = False, show_progress: bool = False) -> Dict:
    """
    Train agent for one episode using custom loop (compatibility with original code)
    
    Args:
        agent: TD3Agent instance
        env: HedgingEnv instance
        update_every: Perform training update every N steps
        log_q_values: Whether to log Q-values
        show_progress: Whether to show tqdm progress bar
    
    Returns:
        dict: Episode statistics
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    agent.reset_noise()
    
    episode_reward = 0
    episode_pnl = 0
    step_count = 0
    actor_losses = []
    critic_losses = []
    
    # Create progress bar if requested
    pbar = tqdm(total=None, desc="Training steps", disable=not show_progress, leave=False)
    
    done = False
    while not done:
        # Select action with exploration noise
        action = agent.select_action(state, add_noise=True)
        
        # Get Q-values for logging if requested
        q_values = None
        if log_q_values and CONFIG.get("log_q_values_training", False):
            q_values = agent.get_q_values(state, action)
        
        # Take step in environment
        result = env.step(action, q_values=q_values) if hasattr(env.step, '__code__') and 'q_values' in env.step.__code__.co_varnames else env.step(action)
        
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, float(done))
        
        # Train agent
        if step_count % update_every == 0:
            a_loss, c_loss = agent.train_step()
            if a_loss is not None:
                actor_losses.append(a_loss)
            if c_loss is not None:
                critic_losses.append(c_loss)
        
        state = next_state
        episode_reward += reward
        episode_pnl += info.get('step_pnl', 0)
        step_count += 1
        
        if show_progress:
            pbar.update(1)
            pbar.set_postfix({'reward': f"{episode_reward:.2f}", 'pnl': f"{episode_pnl:.2f}"})
    
    if show_progress:
        pbar.close()
    
    return {
        'reward': episode_reward,
        'pnl': episode_pnl,
        'steps': step_count,
        'avg_actor_loss': np.mean(actor_losses) if actor_losses else 0,
        'avg_critic_loss': np.mean(critic_losses) if critic_losses else 0,
        'final_cumulative_pnl': info.get('cumulative_pnl', 0),
        'final_cumulative_reward': info.get('cumulative_reward', 0)
    }


def evaluate_agent(agent: TD3Agent, env, verbose: bool = True) -> Dict:
    """
    Evaluate agent without exploration noise
    
    Args:
        agent: TD3Agent instance
        env: HedgingEnv instance
        verbose: Whether to print results
    
    Returns:
        dict: Evaluation statistics
    """
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    episode_reward = 0
    episode_pnl = 0
    step_count = 0
    actions = []
    rewards = []
    pnls = []
    
    done = False
    while not done:
        # Select action without noise
        action = agent.select_action(state, add_noise=False)
        
        # Get Q-values for logging
        q_values = None
        if CONFIG.get("log_q_values", True):
            q_values = agent.get_q_values(state, action)
        
        # Take step - handle both old and new gymnasium API
        try:
            result = env.step(action, q_values=q_values)
        except TypeError:
            result = env.step(action)
        
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        state = next_state
        episode_reward += reward
        episode_pnl += info.get('step_pnl', 0)
        step_count += 1
        
        actions.append(action[0] if hasattr(action, '__len__') else action)
        rewards.append(reward)
        pnls.append(info.get('step_pnl', 0))
    
    # Calculate statistics
    actions_arr = np.array(actions)
    rewards_arr = np.array(rewards)
    pnls_arr = np.array(pnls)
    
    stats = {
        'total_reward': episode_reward,
        'total_pnl': episode_pnl,
        'steps': step_count,
        'cumulative_pnl': info.get('cumulative_pnl', 0),
        'cumulative_reward': info.get('cumulative_reward', 0),
        'mean_action': np.mean(actions_arr),
        'std_action': np.std(actions_arr),
        'mean_reward': np.mean(rewards_arr),
        'std_reward': np.std(rewards_arr),
        'sharpe_ratio': np.mean(pnls_arr) / (np.std(pnls_arr) + 1e-8) * np.sqrt(252),
        'actions': actions_arr,
        'rewards': rewards_arr,
        'pnls': pnls_arr
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {episode_reward:.4f}")
        print(f"Total P&L: {episode_pnl:.4f}")
        print(f"Cumulative P&L: {stats['cumulative_pnl']:.4f}")
        print(f"Mean Action (Delta): {stats['mean_action']:.4f} ± {stats['std_action']:.4f}")
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
        print(f"{'='*50}")
    
    return stats


def train_td3(
    train_env,
    val_env=None,
    num_episodes: int = None,
    total_timesteps: int = None,
    validation_interval: int = 10,
    save_interval: int = 50,
    save_path: str = None,
    verbose: bool = True,
    use_native_loop: bool = True
) -> Tuple[TD3Agent, TrainingMetrics]:
    """
    Main training function for TD3 agent
    
    Args:
        train_env: Training environment
        val_env: Validation environment (optional)
        num_episodes: Number of training episodes (for custom loop)
        total_timesteps: Total timesteps (for native SB3 loop)
        validation_interval: Evaluate on validation set every N episodes
        save_interval: Save model every N episodes
        save_path: Path for saving model checkpoints
        verbose: Whether to print training progress
        use_native_loop: Whether to use SB3's native training loop
    
    Returns:
        tuple: (trained_agent, training_metrics)
    """
    if num_episodes is None:
        num_episodes = CONFIG.get("num_episodes", 100)
    if save_path is None:
        save_path = CONFIG.get("model_save_path", "results/trained_model")
    
    # Estimate total timesteps if not provided
    if total_timesteps is None:
        # Estimate based on episodes and typical episode length
        estimated_steps_per_episode = 250  # Approximate
        total_timesteps = num_episodes * estimated_steps_per_episode
    
    # Initialize agent
    agent = create_td3_agent(train_env)
    metrics = TrainingMetrics()
    
    training_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING TD3 TRAINING (Stable-Baselines3)")
    print(f"{'='*70}")
    print(f"State dimension: {agent.state_dim}")
    print(f"Action dimension: {agent.action_dim}")
    print(f"Device: {device}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Replay buffer size: {CONFIG['replay_buffer_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Training mode: {'Native SB3' if use_native_loop else 'Custom loop'}")
    print(f"{'='*70}\n")
    
    if use_native_loop:
        # Use SB3's native training loop with custom callback
        callback = TrainingCallback(
            metrics=metrics,
            val_env=val_env,
            validation_interval=validation_interval,
            save_path=save_path,
            verbose=1 if verbose else 0
        )
        
        # Train
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=CONFIG.get("training_report_interval", 10),
            progress_bar=verbose
        )
    else:
        # Use custom training loop (more control, compatible with original code)
        best_val_reward = float('-inf')
        
        for episode in range(1, num_episodes + 1):
            # Train one episode
            episode_stats = train_episode(
                agent,
                train_env,
                update_every=CONFIG.get("update_every", 1),
                log_q_values=CONFIG.get("log_q_values_training", False)
            )
            
            # Track metrics
            metrics.add_episode(
                reward=episode_stats['reward'],
                pnl=episode_stats['pnl'],
                length=episode_stats['steps'],
                actor_loss=episode_stats['avg_actor_loss'],
                critic_loss=episode_stats['avg_critic_loss'],
                noise=agent.current_noise
            )
            
            # Print progress
            if verbose and (episode % CONFIG.get("training_report_interval", 10) == 0 or episode == 1):
                elapsed = time.time() - training_start
                agent_stats = agent.get_training_stats()
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_stats['reward']:.4f} | "
                      f"P&L: {episode_stats['pnl']:.4f} | "
                      f"Steps: {episode_stats['steps']} | "
                      f"Buffer: {agent_stats['buffer_size']} | "
                      f"Noise: {agent.current_noise:.4f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Validation
            if val_env is not None and episode % validation_interval == 0:
                val_stats = evaluate_agent(agent, val_env, verbose=False)
                metrics.add_validation(val_stats['total_reward'], val_stats['total_pnl'])
                
                if verbose:
                    print(f"  → Validation: Reward={val_stats['total_reward']:.4f}, "
                          f"P&L={val_stats['total_pnl']:.4f}, "
                          f"Sharpe={val_stats['sharpe_ratio']:.4f}")
                
                # Save best model
                if val_stats['total_reward'] > best_val_reward and CONFIG.get("save_best_only", True):
                    best_val_reward = val_stats['total_reward']
                    best_path = save_path.replace('.pth', '_best').replace('.zip', '_best')
                    os.makedirs(os.path.dirname(best_path) if os.path.dirname(best_path) else ".", exist_ok=True)
                    agent.save(best_path)
                    if verbose:
                        print(f"  → New best model saved! (Reward: {best_val_reward:.4f})")
            
            # Periodic save
            if episode % save_interval == 0 and CONFIG.get("save_model", True):
                checkpoint_path = save_path.replace('.pth', f'_ep{episode}').replace('.zip', f'_ep{episode}')
                os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
                agent.save(checkpoint_path)
    
    # Final save
    if CONFIG.get("save_model", True):
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        agent.save(save_path)
    
    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    if metrics.episode_rewards:
        print(f"Final episode reward: {metrics.episode_rewards[-1]:.4f}")
    if metrics.validation_rewards:
        print(f"Best validation reward: {max(metrics.validation_rewards):.4f}")
    print(f"Total training steps: {agent.total_steps}")
    print(f"{'='*70}")
    
    return agent, metrics


def train_multi_year(
    data_path: str = None,
    train_years: List[int] = None,
    validation_year: int = None,
    num_epochs: int = None,
    shuffle_years: bool = True,
    verbose: bool = True,
    use_native_loop: bool = False
) -> Tuple[TD3Agent, TrainingMetrics, Dict]:
    """
    Train agent on multiple years of data
    
    Args:
        data_path: Path to data file
        train_years: List of training years
        validation_year: Validation year
        num_epochs: Number of epochs
        shuffle_years: Whether to shuffle year order each epoch
        verbose: Whether to print progress
        use_native_loop: Whether to use native SB3 loop
    
    Returns:
        tuple: (trained_agent, training_metrics, norm_stats)
    """
    if train_years is None:
        train_years = CONFIG.get("train_years", [2005, 2006, 2007, 2008, 2009, 2010])
    if validation_year is None:
        validation_year = CONFIG.get("validation_year", 2011)
    if num_epochs is None:
        num_epochs = CONFIG.get("num_epochs", 10)
    
    # Create environments for each year
    train_envs, norm_stats = create_train_environments_list(
        data_path=data_path,
        train_years=train_years,
        normalize=CONFIG.get("normalize_data", True),
        verbose=False
    )
    
    # Create validation environment
    df = load_hedging_data(data_path)
    datetime_col = CONFIG.get("datetime_column", "timestamp")
    val_df = df[df[datetime_col].dt.year == validation_year].copy().reset_index(drop=True)
    
    val_env = None
    if len(val_df) > 0:
        val_env = create_environment(
            val_df,
            normalization_stats=norm_stats,
            normalize=CONFIG.get("normalize_data", True),
            verbose=False
        )
    
    # Initialize agent with first environment
    first_env = train_envs[0][1]
    agent = create_td3_agent(first_env)
    metrics = TrainingMetrics()
    
    best_val_reward = float('-inf')
    save_path = CONFIG.get("model_save_path", "results/trained_model")
    
    print(f"\n{'='*70}")
    print(f"MULTI-YEAR TD3 TRAINING (Stable-Baselines3)")
    print(f"{'='*70}")
    print(f"Training years: {train_years}")
    print(f"Validation year: {validation_year}")
    print(f"Epochs: {num_epochs}")
    print(f"Total episodes: {num_epochs * len(train_envs)}")
    print(f"{'='*70}\n")
    
    episode = 0
    training_start = time.time()
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Training Epochs", position=0)
    
    for epoch in epoch_pbar:
        epoch_rewards = []
        epoch_pnls = []
        
        # Optionally shuffle training order
        env_order = list(range(len(train_envs)))
        if shuffle_years:
            np.random.shuffle(env_order)
        
        # Progress bar for years within epoch
        year_pbar = tqdm(env_order, desc=f"Epoch {epoch}/{num_epochs} - Years", position=1, leave=False)
        
        for idx in year_pbar:
            year, env = train_envs[idx]
            episode += 1
            
            # Update year progress description
            year_pbar.set_description(f"Epoch {epoch}/{num_epochs} - Year {year}")
            
            # Update agent's environment
            agent.set_env(env)
            
            # Train on this year (with step progress)
            episode_stats = train_episode(
                agent,
                env,
                update_every=CONFIG.get("update_every", 1),
                show_progress=False  # Disable step-level progress to avoid clutter
            )
            
            epoch_rewards.append(episode_stats['reward'])
            epoch_pnls.append(episode_stats['pnl'])
            
            metrics.add_episode(
                reward=episode_stats['reward'],
                pnl=episode_stats['pnl'],
                length=episode_stats['steps'],
                actor_loss=episode_stats['avg_actor_loss'],
                critic_loss=episode_stats['avg_critic_loss'],
                noise=agent.current_noise
            )
            
            # Update year progress bar with metrics
            year_pbar.set_postfix({
                'reward': f"{episode_stats['reward']:.2f}",
                'pnl': f"{episode_stats['pnl']:.2f}",
                'noise': f"{agent.current_noise:.3f}"
            })
        
        year_pbar.close()
        
        # Update epoch progress bar with summary
        avg_reward = np.mean(epoch_rewards)
        avg_pnl = np.mean(epoch_pnls)
        epoch_pbar.set_postfix({
            'avg_reward': f"{avg_reward:.2f}",
            'avg_pnl': f"{avg_pnl:.2f}"
        })
        
        # Validation at end of each epoch
        if val_env is not None:
            val_stats = evaluate_agent(agent, val_env, verbose=False)
            metrics.add_validation(val_stats['total_reward'], val_stats['total_pnl'])
            
            # Print validation results (tqdm compatible)
            tqdm.write(f"  Epoch {epoch} Validation: Reward={val_stats['total_reward']:.4f}, "
                      f"P&L={val_stats['total_pnl']:.4f}, "
                      f"Sharpe={val_stats['sharpe_ratio']:.4f}")
            
            # Save best model
            if val_stats['total_reward'] > best_val_reward:
                best_val_reward = val_stats['total_reward']
                best_path = save_path.replace('.pth', '_best').replace('.zip', '_best')
                os.makedirs(os.path.dirname(best_path) if os.path.dirname(best_path) else ".", exist_ok=True)
                agent.save(best_path)
                if verbose:
                    tqdm.write(f"  → New best model saved!")
    
    # Final save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    agent.save(save_path)
    
    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Total episodes: {episode}")
    print(f"Best validation reward: {best_val_reward:.4f}")
    print(f"{'='*70}")
    
    return agent, metrics, norm_stats


def compare_with_benchmark(
    agent: TD3Agent,
    test_env,
    test_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Compare RL agent with Delta Hedging benchmark
    
    Args:
        agent: Trained TD3Agent
        test_env: Test environment
        test_df: Test DataFrame for benchmark
        verbose: Whether to print detailed comparison
    
    Returns:
        tuple: (comparison_dict, rl_stats, benchmark_df)
    """
    print(f"\n{'='*70}")
    print(f"COMPARING RL AGENT VS DELTA HEDGING BENCHMARK")
    print(f"{'='*70}")
    
    # Evaluate RL agent
    print("\n1. Evaluating RL Agent...")
    rl_stats = evaluate_agent(agent, test_env, verbose=verbose)
    
    # Run delta hedging benchmark
    print("\n2. Running Delta Hedging Benchmark...")
    test_year = CONFIG.get("test_year", 2012)
    
    # Get full data for benchmark
    data_path = CONFIG.get("data_path", "./data/historical_hedging_data.csv")
    full_df = load_hedging_data(data_path)
    
    benchmark_df = run_benchmark(
        full_df,
        start_year=test_year,
        end_year=test_year
    )
    
    benchmark_pnl = benchmark_df["Cumulative PnL"].iloc[-1]
    benchmark_reward = benchmark_df["Reward Cumulative PnL"].iloc[-1] if "Reward Cumulative PnL" in benchmark_df.columns else benchmark_pnl
    
    # Calculate benchmark Sharpe
    benchmark_returns = benchmark_df["PnL"].dropna()
    benchmark_sharpe = benchmark_returns.mean() / (benchmark_returns.std() + 1e-8) * np.sqrt(252)
    
    # Comparison
    comparison = {
        'rl_reward': rl_stats['total_reward'],
        'rl_pnl': rl_stats['cumulative_pnl'],
        'rl_sharpe': rl_stats['sharpe_ratio'],
        'benchmark_reward': benchmark_reward,
        'benchmark_pnl': benchmark_pnl,
        'benchmark_sharpe': benchmark_sharpe,
        'reward_improvement': rl_stats['total_reward'] - benchmark_reward,
        'pnl_improvement': rl_stats['cumulative_pnl'] - benchmark_pnl,
        'sharpe_improvement': rl_stats['sharpe_ratio'] - benchmark_sharpe
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'RL Agent':<15} {'Delta Hedge':<15} {'Improvement':<15}")
        print(f"{'-'*70}")
        print(f"{'Total Reward':<25} {comparison['rl_reward']:<15.4f} {comparison['benchmark_reward']:<15.4f} {comparison['reward_improvement']:<+15.4f}")
        print(f"{'Cumulative P&L':<25} {comparison['rl_pnl']:<15.4f} {comparison['benchmark_pnl']:<15.4f} {comparison['pnl_improvement']:<+15.4f}")
        print(f"{'Sharpe Ratio':<25} {comparison['rl_sharpe']:<15.4f} {comparison['benchmark_sharpe']:<15.4f} {comparison['sharpe_improvement']:<+15.4f}")
        print(f"{'='*70}")
        
        if comparison['pnl_improvement'] > 0:
            print(f"\n✅ RL Agent OUTPERFORMS Delta Hedging by {comparison['pnl_improvement']:.4f} P&L units")
        else:
            print(f"\n❌ Delta Hedging outperforms RL Agent by {-comparison['pnl_improvement']:.4f} P&L units")
    
    return comparison, rl_stats, benchmark_df


def plot_training_curves(metrics: TrainingMetrics, save_path: str = None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax1 = axes[0, 0]
    ax1.plot(metrics.episode_rewards, alpha=0.6, label='Episode Reward')
    if len(metrics.episode_rewards) > 10:
        window = min(10, len(metrics.episode_rewards) // 5)
        moving_avg = pd.Series(metrics.episode_rewards).rolling(window=window).mean()
        ax1.plot(moving_avg, 'r-', linewidth=2, label=f'{window}-ep Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode P&L
    ax2 = axes[0, 1]
    ax2.plot(metrics.episode_pnls, alpha=0.6, label='Episode P&L')
    if len(metrics.episode_pnls) > 10:
        window = min(10, len(metrics.episode_pnls) // 5)
        moving_avg = pd.Series(metrics.episode_pnls).rolling(window=window).mean()
        ax2.plot(moving_avg, 'r-', linewidth=2, label=f'{window}-ep Moving Avg')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('P&L')
    ax2.set_title('Training P&L')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Validation metrics
    ax3 = axes[1, 0]
    if metrics.validation_rewards:
        ax3.plot(metrics.validation_rewards, 'g-', marker='o', label='Validation Reward')
        ax3.plot(metrics.validation_pnls, 'b-', marker='s', label='Validation P&L')
    ax3.set_xlabel('Validation Step')
    ax3.set_ylabel('Value')
    ax3.set_title('Validation Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Exploration noise
    ax4 = axes[1, 1]
    ax4.plot(metrics.exploration_noise)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Noise Level')
    ax4.set_title('Exploration Noise Decay')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=CONFIG.get("plot_dpi", 300))
        print(f"Training curves saved to {save_path}")
    
    plt.show()
    return fig


def plot_comparison(rl_stats: Dict, benchmark_df: pd.DataFrame, test_env, save_path: str = None, test_year: int = None, output_dir: str = None):
    """
    Plot comparison between RL agent and benchmark
    
    Args:
        rl_stats: RL agent statistics
        benchmark_df: Benchmark dataframe
        test_env: Test environment
        save_path: Path to save the combined plot
        test_year: Year being tested
        output_dir: Directory for individual plots (defaults to same dir as save_path)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main plot: Cumulative P&L
    ax_main = fig.add_subplot(gs[0, :])
    
    rl_cumulative_pnl = np.cumsum(rl_stats['pnls'])
    benchmark_cumulative_pnl = benchmark_df['Cumulative PnL'].values
    
    ax_main.plot(rl_cumulative_pnl, label='RL Agent', linewidth=2.5, color='#2E86DE', alpha=0.9)
    ax_main.plot(benchmark_cumulative_pnl, label='Delta Hedging', linewidth=2.5, color='#EE5A6F', alpha=0.9)
    ax_main.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    ax_main.set_xlabel('Trading Step', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Cumulative P&L (normalized)', fontsize=12, fontweight='bold')
    
    title = 'Out-of-Sample Cumulative P&L: RL Agent vs Delta Hedging'
    if test_year is not None:
        title += f' (Year {test_year})'
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax_main.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    
    # Add final P&L values
    final_rl_pnl = rl_cumulative_pnl[-1]
    final_bench_pnl = benchmark_cumulative_pnl[-1]
    improvement = final_rl_pnl - final_bench_pnl
    
    notional = CONFIG.get('notional', 1000)
    textstr = (f'Final P&L (normalized):\n'
               f'RL Agent: {final_rl_pnl:.2f}\n'
               f'Delta Hedging: {final_bench_pnl:.2f}\n'
               f'Improvement: {improvement:+.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    # Hedge Ratio Comparison
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(rl_stats['actions'], label='RL Actions', alpha=0.8, linewidth=1.5, color='#2E86DE')
    if 'Delta' in benchmark_df.columns:
        ax1.plot(benchmark_df['Delta'].values, label='BS Delta', alpha=0.8,
                linewidth=1.5, color='#EE5A6F', linestyle='--')
    ax1.set_xlabel('Step', fontsize=10)
    ax1.set_ylabel('Hedge Ratio', fontsize=10)
    ax1.set_title('Hedge Ratio Comparison', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Step P&L comparison
    ax2 = fig.add_subplot(gs[1, 1])
    window = max(1, len(rl_stats['pnls']) // 100)
    rl_pnl_smooth = pd.Series(rl_stats['pnls']).rolling(window=window, center=True).mean()
    bench_pnl_smooth = pd.Series(benchmark_df['PnL'].values).rolling(window=window, center=True).mean()
    
    ax2.plot(rl_pnl_smooth, label='RL Step P&L (smoothed)', alpha=0.8, linewidth=1.5, color='#2E86DE')
    ax2.plot(bench_pnl_smooth, label='Benchmark Step P&L (smoothed)', alpha=0.8,
            linewidth=1.5, color='#EE5A6F', linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax2.set_xlabel('Step', fontsize=10)
    ax2.set_ylabel('Step P&L', fontsize=10)
    ax2.set_title('Step-by-Step P&L', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Distribution of hedge ratios
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(rl_stats['actions'], bins=50, alpha=0.6, label='RL Actions',
            density=True, color='#2E86DE', edgecolor='black', linewidth=0.5)
    if 'Delta' in benchmark_df.columns:
        ax3.hist(benchmark_df['Delta'].dropna().values, bins=50, alpha=0.6,
                label='BS Delta', density=True, color='#EE5A6F',
                edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Hedge Ratio', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Distribution of Hedge Ratios', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Statistics summary
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    rl_sharpe = rl_stats['sharpe_ratio']
    bench_sharpe = benchmark_df['PnL'].mean() / (benchmark_df['PnL'].std() + 1e-8) * np.sqrt(252)
    
    stats_text = f"""
    Performance Statistics:
    
    RL Agent (SB3 TD3):
      • Final P&L: {final_rl_pnl:.2f}
      • Sharpe Ratio: {rl_sharpe:.3f}
      • Mean Hedge: {rl_stats['mean_action']:.3f}
      • Std Hedge: {rl_stats['std_action']:.3f}
    
    Delta Hedging:
      • Final P&L: {final_bench_pnl:.2f}
      • Sharpe Ratio: {bench_sharpe:.3f}
    
    Improvement:
      • P&L: {improvement:+.2f}
      • Sharpe: {rl_sharpe - bench_sharpe:+.3f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        
        # Save individual plots to same directory or specified output_dir
        if output_dir is None:
            output_dir = os.path.dirname(save_path) or "./output"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        # 1. Cumulative P&L Plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(rl_cumulative_pnl, label='RL Agent', 
                linewidth=2.5, color='#2E86DE', alpha=0.9)
        ax1.plot(benchmark_cumulative_pnl, label='Delta Hedging', 
                linewidth=2.5, color='#EE5A6F', alpha=0.9)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Trading Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative P&L (normalized)', fontsize=12, fontweight='bold')
        title1 = 'Cumulative P&L: RL Agent vs Delta Hedging'
        if test_year is not None:
            title1 += f' (Year {test_year})'
        ax1.set_title(title1, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        textstr = (f'Final P&L (normalized):\n'
                   f'RL Agent: {final_rl_pnl:.2f}\n'
                   f'Delta Hedging: {final_bench_pnl:.2f}\n'
                   f'Improvement: {improvement:+.2f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        plot1_path = os.path.join(output_dir, f"{base_name}_cumulative_pnl.png")
        plt.savefig(plot1_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Hedge Ratio Comparison
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(rl_stats['actions'], label='RL Actions', alpha=0.8, linewidth=1.5, color='#2E86DE')
        if 'Delta' in benchmark_df.columns:
            ax2.plot(benchmark_df['Delta'].values, label='BS Delta', alpha=0.8, 
                    linewidth=1.5, color='#EE5A6F', linestyle='--')
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Hedge Ratio', fontsize=11)
        ax2.set_title('Hedge Ratio Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plot2_path = os.path.join(output_dir, f"{base_name}_hedge_ratio.png")
        plt.savefig(plot2_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Step P&L Comparison
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(rl_pnl_smooth, label='RL Step P&L (smoothed)', alpha=0.8, linewidth=1.5, color='#2E86DE')
        ax3.plot(bench_pnl_smooth, label='Benchmark Step P&L (smoothed)', alpha=0.8, 
                linewidth=1.5, color='#EE5A6F', linestyle='--')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Step P&L', fontsize=11)
        ax3.set_title('Step-by-Step P&L', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        plot3_path = os.path.join(output_dir, f"{base_name}_step_pnl.png")
        plt.savefig(plot3_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Hedge Ratio Distribution
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.hist(rl_stats['actions'], bins=50, alpha=0.6, label='RL Actions', 
                density=True, color='#2E86DE', edgecolor='black', linewidth=0.5)
        if 'Delta' in benchmark_df.columns:
            ax4.hist(benchmark_df['Delta'].dropna().values, bins=50, alpha=0.6, 
                    label='BS Delta', density=True, color='#EE5A6F', 
                    edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Hedge Ratio', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title('Distribution of Hedge Ratios', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        plot4_path = os.path.join(output_dir, f"{base_name}_distribution.png")
        plt.savefig(plot4_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        plt.close(fig4)
        
        print(f"Individual plots saved to {output_dir}:")
        print(f"  - {os.path.basename(plot1_path)}")
        print(f"  - {os.path.basename(plot2_path)}")
        print(f"  - {os.path.basename(plot3_path)}")
        print(f"  - {os.path.basename(plot4_path)}")
    
    plt.show()
    return fig


def evaluate_agent_multi_episode(agent: TD3Agent, test_envs: List, verbose: bool = True) -> Dict:
    """
    Evaluate agent across multiple test episodes (30-day windows).
    
    This evaluates the agent in the same paradigm as training:
    multiple short episodes simulating option hedging to expiry.
    
    Args:
        agent: TD3Agent instance
        test_envs: List of HedgingEnv instances (one per test episode)
        verbose: Whether to print results
    
    Returns:
        dict: Aggregated evaluation statistics across all episodes
    """
    all_rewards = []
    all_pnls = []
    all_cumulative_pnls = []
    all_sharpes = []
    all_actions = []
    episode_stats_list = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-EPISODE EVALUATION ({len(test_envs)} episodes)")
        print(f"{'='*70}")
    
    for i, env in enumerate(test_envs):
        # Evaluate single episode
        stats = evaluate_agent(agent, env, verbose=False)
        
        all_rewards.append(stats['total_reward'])
        # Use sum of step pnls (normalized) for consistency
        episode_pnl = np.sum(stats['pnls'])
        all_pnls.append(episode_pnl)
        all_cumulative_pnls.append(episode_pnl)  # Same as all_pnls for consistency
        all_sharpes.append(stats['sharpe_ratio'])
        all_actions.extend(stats['actions'].tolist())
        episode_stats_list.append(stats)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(test_envs)} episodes...")
    
    # Aggregate statistics (all P&L values are normalized/comparable)
    aggregated_stats = {
        # Per-episode means
        'mean_episode_reward': np.mean(all_rewards),
        'std_episode_reward': np.std(all_rewards),
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'mean_cumulative_pnl': np.mean(all_cumulative_pnls),
        'std_cumulative_pnl': np.std(all_cumulative_pnls),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        
        # Aggregated totals (sum of normalized episode P&Ls)
        'total_reward': sum(all_rewards),
        'total_pnl': sum(all_pnls),
        'total_cumulative_pnl': sum(all_pnls),  # Use normalized P&L sum
        
        # Action statistics across all episodes
        'mean_action': np.mean(all_actions),
        'std_action': np.std(all_actions),
        'min_action': np.min(all_actions),
        'max_action': np.max(all_actions),
        
        # Number of episodes
        'n_episodes': len(test_envs),
        
        # All individual episode stats
        'episode_stats': episode_stats_list,
        'all_rewards': all_rewards,
        'all_pnls': all_pnls,
        'all_cumulative_pnls': all_cumulative_pnls,
        'all_sharpes': all_sharpes,
        'all_actions': np.array(all_actions)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-EPISODE EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Number of Episodes: {aggregated_stats['n_episodes']}")
        print(f"\nPer-Episode Statistics:")
        print(f"  Mean Reward: {aggregated_stats['mean_episode_reward']:.4f} ± {aggregated_stats['std_episode_reward']:.4f}")
        print(f"  Mean P&L: {aggregated_stats['mean_episode_pnl']:.4f} ± {aggregated_stats['std_episode_pnl']:.4f}")
        print(f"  Mean Sharpe: {aggregated_stats['mean_sharpe']:.4f} ± {aggregated_stats['std_sharpe']:.4f}")
        print(f"\nAction Statistics (across all episodes):")
        print(f"  Mean Hedge Ratio: {aggregated_stats['mean_action']:.4f} ± {aggregated_stats['std_action']:.4f}")
        print(f"  Action Range: [{aggregated_stats['min_action']:.4f}, {aggregated_stats['max_action']:.4f}]")
        print(f"\nAggregated Totals:")
        print(f"  Total P&L: {aggregated_stats['total_pnl']:.4f}")
        print(f"  Total Reward: {aggregated_stats['total_reward']:.4f}")
        print(f"{'='*70}")
    
    return aggregated_stats


def plot_multi_episode_results(stats: Dict, save_path: str = None):
    """
    Plot results from multi-episode evaluation.
    
    Args:
        stats: Dictionary from evaluate_agent_multi_episode
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode P&L distribution
    ax1 = axes[0, 0]
    ax1.hist(stats['all_cumulative_pnls'], bins=30, edgecolor='black', alpha=0.7, color='#2E86DE')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(x=stats['mean_cumulative_pnl'], color='green', linestyle='-', linewidth=2, 
                label=f"Mean: {stats['mean_cumulative_pnl']:.2f}")
    ax1.set_xlabel('Cumulative P&L per Episode', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Episode P&L (30-day windows)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Sharpe distribution
    ax2 = axes[0, 1]
    ax2.hist(stats['all_sharpes'], bins=30, edgecolor='black', alpha=0.7, color='#28A745')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Sharpe')
    ax2.axvline(x=stats['mean_sharpe'], color='blue', linestyle='-', linewidth=2,
                label=f"Mean: {stats['mean_sharpe']:.2f}")
    ax2.set_xlabel('Sharpe Ratio per Episode', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Episode Sharpe Ratios', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Action distribution
    ax3 = axes[1, 0]
    ax3.hist(stats['all_actions'], bins=50, edgecolor='black', alpha=0.7, color='#FF6B6B')
    ax3.axvline(x=stats['mean_action'], color='blue', linestyle='-', linewidth=2,
                label=f"Mean: {stats['mean_action']:.3f}")
    ax3.set_xlabel('Hedge Ratio (Action)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Hedge Ratios Across All Episodes', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative P&L over episodes
    ax4 = axes[1, 1]
    cumsum_pnl = np.cumsum(stats['all_cumulative_pnls'])
    ax4.plot(cumsum_pnl, linewidth=2, color='#2E86DE')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax4.fill_between(range(len(cumsum_pnl)), 0, cumsum_pnl, 
                     where=(np.array(cumsum_pnl) > 0), alpha=0.3, color='green')
    ax4.fill_between(range(len(cumsum_pnl)), 0, cumsum_pnl, 
                     where=(np.array(cumsum_pnl) <= 0), alpha=0.3, color='red')
    ax4.set_xlabel('Episode Number', fontsize=11)
    ax4.set_ylabel('Cumulative P&L (Sum of Episodes)', fontsize=11)
    ax4.set_title('Cumulative P&L Progression Across Episodes', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        print(f"Multi-episode results plot saved to {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    # Quick test of training
    print("Testing SB3 Training Script...")
    
    # Create environments
    envs = create_environments_for_training(
        train_years=[2005],  # Quick test with one year
        validation_year=2006,
        test_year=2007,
        verbose=False
    )
    
    # Quick training test
    agent, metrics = train_td3(
        train_env=envs['train_env'],
        val_env=envs['val_env'],
        num_episodes=2,
        total_timesteps=1000,  # Very short for testing
        verbose=True,
        use_native_loop=False  # Use custom loop for more control
    )
    
    print("\nTraining test completed!")
