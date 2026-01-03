"""
Training Script for TD3 Hedging Agent
Handles training loop, evaluation, and model checkpointing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

from config import CONFIG, get_model_config, get_environment_config
from td3_agent import TD3Agent, device
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
    
    def add_episode(self, reward, pnl, length, actor_loss, critic_loss, noise):
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


def train_episode(agent, env, update_every=1, log_q_values=False):
    """
    Train agent for one episode (full trajectory through environment)
    
    Args:
        agent: TD3Agent instance
        env: HedgingEnv instance
        update_every: Perform training update every N steps
        log_q_values: Whether to log Q-values (slower)
    
    Returns:
        dict: Episode statistics
    """
    state = env.reset()
    agent.reset_noise()
    
    episode_reward = 0
    episode_pnl = 0
    step_count = 0
    actor_losses = []
    critic_losses = []
    
    done = False
    while not done:
        # Select action with exploration noise
        action = agent.select_action(state, add_noise=True)
        
        # Get Q-values for logging if requested
        q_values = None
        if log_q_values and CONFIG.get("log_q_values_training", False):
            q_values = agent.get_q_values(state, action)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action, q_values=q_values)
        
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
    
    return {
        'reward': episode_reward,
        'pnl': episode_pnl,
        'steps': step_count,
        'avg_actor_loss': np.mean(actor_losses) if actor_losses else 0,
        'avg_critic_loss': np.mean(critic_losses) if critic_losses else 0,
        'final_cumulative_pnl': info.get('cumulative_pnl', 0),
        'final_cumulative_reward': info.get('cumulative_reward', 0)
    }


def evaluate_agent(agent, env, verbose=True):
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
        
        # Take step
        next_state, reward, done, info = env.step(action, q_values=q_values)
        
        state = next_state
        episode_reward += reward
        episode_pnl += info.get('step_pnl', 0)
        step_count += 1
        
        actions.append(action[0])
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
    num_episodes=None,
    validation_interval=10,
    save_interval=50,
    save_path=None,
    verbose=True
):
    """
    Main training function for TD3 agent
    
    Args:
        train_env: Training environment
        val_env: Validation environment (optional)
        num_episodes: Number of training episodes
        validation_interval: Evaluate on validation set every N episodes
        save_interval: Save model every N episodes
        save_path: Path for saving model checkpoints
        verbose: Whether to print training progress
    
    Returns:
        tuple: (trained_agent, training_metrics)
    """
    if num_episodes is None:
        num_episodes = CONFIG.get("num_episodes", 100)
    if save_path is None:
        save_path = CONFIG.get("model_save_path", "results/trained_model.pth")
    
    # Initialize agent
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    metrics = TrainingMetrics()
    
    best_val_reward = float('-inf')
    training_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"STARTING TD3 TRAINING")
    print(f"{'='*70}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Replay buffer size: {CONFIG['replay_buffer_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"{'='*70}\n")
    
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
                best_path = save_path.replace('.pth', '_best.pth')
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                agent.save(best_path)
                if verbose:
                    print(f"  → New best model saved! (Reward: {best_val_reward:.4f})")
        
        # Periodic save
        if episode % save_interval == 0 and CONFIG.get("save_model", True):
            checkpoint_path = save_path.replace('.pth', f'_ep{episode}.pth')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            agent.save(checkpoint_path)
    
    # Final save
    if CONFIG.get("save_model", True):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
    
    total_time = time.time() - training_start
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Final episode reward: {metrics.episode_rewards[-1]:.4f}")
    print(f"Best validation reward: {best_val_reward:.4f}")
    print(f"Total training steps: {agent.total_steps}")
    print(f"{'='*70}")
    
    return agent, metrics


def train_multi_year(
    data_path=None,
    train_years=None,
    validation_year=None,
    num_epochs=None,
    shuffle_years=True,
    verbose=True
):
    """
    Train agent on multiple years of data
    Each year is treated as a separate episode
    
    Args:
        data_path: Path to data file
        train_years: List of training years
        validation_year: Validation year
        num_epochs: Number of epochs (passes through all years)
        shuffle_years: Whether to shuffle year order each epoch
        verbose: Whether to print progress
    
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
    
    # Initialize agent
    state_dim = train_envs[0][1].observation_space.shape[0]
    action_dim = train_envs[0][1].action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    metrics = TrainingMetrics()
    
    best_val_reward = float('-inf')
    save_path = CONFIG.get("model_save_path", "results/trained_model.pth")
    
    print(f"\n{'='*70}")
    print(f"MULTI-YEAR TD3 TRAINING")
    print(f"{'='*70}")
    print(f"Training years: {train_years}")
    print(f"Validation year: {validation_year}")
    print(f"Epochs: {num_epochs}")
    print(f"Total episodes: {num_epochs * len(train_envs)}")
    print(f"{'='*70}\n")
    
    episode = 0
    training_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_rewards = []
        epoch_pnls = []
        
        # Optionally shuffle training order (with controlled seed for reproducibility)
        env_order = list(range(len(train_envs)))
        if shuffle_years:
            seed = CONFIG.get("seed", 42)
            shuffle_rng = np.random.default_rng(seed + epoch)  # Different per epoch but reproducible
            shuffle_rng.shuffle(env_order)
        
        for idx in env_order:
            year, env = train_envs[idx]
            episode += 1
            
            # Train on this year
            episode_stats = train_episode(
                agent, 
                env,
                update_every=CONFIG.get("update_every", 1)
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
            
            if verbose:
                print(f"Epoch {epoch}/{num_epochs} | Year {year} | "
                      f"Reward: {episode_stats['reward']:.4f} | "
                      f"P&L: {episode_stats['pnl']:.4f} | "
                      f"Noise: {agent.current_noise:.4f}")
        
        # Epoch summary
        if verbose:
            print(f"  → Epoch {epoch} Summary: "
                  f"Avg Reward: {np.mean(epoch_rewards):.4f}, "
                  f"Avg P&L: {np.mean(epoch_pnls):.4f}")
        
        # Validation at end of each epoch
        if val_env is not None:
            val_stats = evaluate_agent(agent, val_env, verbose=False)
            metrics.add_validation(val_stats['total_reward'], val_stats['total_pnl'])
            
            if verbose:
                print(f"  → Validation: Reward={val_stats['total_reward']:.4f}, "
                      f"P&L={val_stats['total_pnl']:.4f}, "
                      f"Sharpe={val_stats['sharpe_ratio']:.4f}")
            
            # Save best model
            if val_stats['total_reward'] > best_val_reward:
                best_val_reward = val_stats['total_reward']
                best_path = save_path.replace('.pth', '_best.pth')
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                agent.save(best_path)
                if verbose:
                    print(f"  → New best model saved!")
        
        print()  # Empty line between epochs
    
    # Final save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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


def compare_with_benchmark(agent, test_env, test_df, verbose=True):
    """
    Compare RL agent with Delta Hedging benchmark
    
    Args:
        agent: Trained TD3Agent
        test_env: Test environment
        test_df: Test DataFrame for benchmark
        verbose: Whether to print detailed comparison
    
    Returns:
        dict: Comparison results
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
    
    # Get full data for benchmark (needs prior data for volatility calculation)
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


def plot_training_curves(metrics, save_path=None):
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
    
    # Losses
    ax3 = axes[1, 0]
    if metrics.critic_losses:
        ax3.plot(metrics.critic_losses, alpha=0.6, label='Critic Loss')
    if metrics.actor_losses:
        ax3.plot(metrics.actor_losses, alpha=0.6, label='Actor Loss')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Losses')
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


def plot_comparison(rl_stats, benchmark_df, test_env, save_path=None, test_year=None):
    """
    Plot comparison between RL agent and benchmark
    Creates a clean visualization similar to academic papers
    """
    # Create figure with better spacing
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main plot: Cumulative Return (%) - Large plot at top
    ax_main = fig.add_subplot(gs[0, :])
    
    # Calculate cumulative P&L (normalized by notional)
    # step_pnl is already normalized by notional, so cumulative sum gives us
    # the total P&L in units of notional.
    rl_cumulative_pnl = np.cumsum(rl_stats['pnls'])
    
    # Benchmark is also normalized
    benchmark_cumulative_pnl = benchmark_df['Cumulative PnL'].values
    
    # Plot with better styling
    ax_main.plot(rl_cumulative_pnl, label='RL Agent', 
                linewidth=2.5, color='#2E86DE', alpha=0.9)
    ax_main.plot(benchmark_cumulative_pnl, label='Delta Hedging', 
                linewidth=2.5, color='#EE5A6F', alpha=0.9)
    
    # Add zero line
    ax_main.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    # Styling
    ax_main.set_xlabel('Trading Step', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Cumulative P&L (normalized)', fontsize=12, fontweight='bold')
    
    title = 'Out-of-Sample Cumulative P&L: RL Agent vs Delta Hedging'
    if test_year is not None:
        title += f' (Year {test_year})'
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax_main.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Add final P&L values as text
    final_rl_pnl = rl_cumulative_pnl[-1]
    final_bench_pnl = benchmark_cumulative_pnl[-1]
    improvement = final_rl_pnl - final_bench_pnl
    
    # Convert to dollars for display
    notional = CONFIG.get('notional', 1000)
    textstr = (f'Final P&L (normalized):\n'
               f'RL Agent: {final_rl_pnl:.2f}\n'
               f'Delta Hedging: {final_bench_pnl:.2f}\n'
               f'Improvement: {improvement:+.2f}\n'
               f'\n'
               f'In dollars (x{notional}):\n'
               f'RL: ${final_rl_pnl*notional:.2f}\n'
               f'Bench: ${final_bench_pnl*notional:.2f}')
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
    # Use rolling average for smoother visualization
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
    
    # Calculate statistics
    rl_sharpe = rl_stats['sharpe_ratio']
    bench_sharpe = benchmark_df['PnL'].mean() / (benchmark_df['PnL'].std() + 1e-8) * np.sqrt(252)
    
    rl_mean_action = rl_stats['mean_action']
    rl_std_action = rl_stats['std_action']
    bench_mean_delta = benchmark_df['Delta'].mean()
    bench_std_delta = benchmark_df['Delta'].std()
    
    # Create statistics table
    stats_text = f"""
    Performance Statistics:
    
    RL Agent:
      • Final P&L: {final_rl_pnl:.2f}
      • Sharpe Ratio: {rl_sharpe:.3f}
      • Mean Hedge: {rl_mean_action:.3f}
      • Std Hedge: {rl_std_action:.3f}
      • Total Steps: {rl_stats['steps']}
    
    Delta Hedging:
      • Final P&L: {final_bench_pnl:.2f}
      • Sharpe Ratio: {bench_sharpe:.3f}
      • Mean Delta: {bench_mean_delta:.3f}
      • Std Delta: {bench_std_delta:.3f}
    
    Improvement:
      • P&L: {improvement:+.2f}
      • In $: ${improvement*CONFIG.get('notional', 1000):+.2f}
      • Sharpe: {rl_sharpe - bench_sharpe:+.3f}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Save complete figure
    if save_path:
        plt.savefig(save_path, dpi=CONFIG.get("plot_dpi", 300), bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        
        # Save individual plots to output folder
        output_dir = "./output"
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
        notional = CONFIG.get('notional', 1000)
        textstr = (f'Final P&L (normalized):\n'
                   f'RL Agent: {final_rl_pnl:.2f}\n'
                   f'Delta Hedging: {final_bench_pnl:.2f}\n'
                   f'Improvement: {improvement:+.2f}\n'
                   f'\n'
                   f'In dollars (x{notional}):\n'
                   f'RL: ${final_rl_pnl*notional:.2f}\n'
                   f'Bench: ${final_bench_pnl*notional:.2f}')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        plot1_path = os.path.join(output_dir, f"{base_name}_cumulative_return.png")
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
        window = max(1, len(rl_stats['pnls']) // 100)
        rl_pnl_smooth = pd.Series(rl_stats['pnls']).rolling(window=window, center=True).mean()
        bench_pnl_smooth = pd.Series(benchmark_df['PnL'].values).rolling(window=window, center=True).mean()
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


if __name__ == "__main__":
    # Quick test of training
    print("Testing Training Script...")
    
    from data_loader import create_environments_for_training
    
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
        num_episodes=2,  # Very short for testing
        verbose=True
    )
    
    print("\nTraining test completed!")
