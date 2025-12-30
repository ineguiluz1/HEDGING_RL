#!/usr/bin/env python3
"""
Main Training Script for TD3 Hedging Agent
==========================================

This script trains a TD3 agent for options hedging using:
- Monte Carlo simulated trajectories for training
- Real S&P 500 daily data for testing

Usage:
    python run_training.py                    # Run training and evaluation
    
Configuration:
    Training: mc_train_trajectories (e.g., 50) synthetic 1-year paths
    Testing: Real S&P 500 data (2004-2025)

Author: Generated for HEDGING_RL project
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, get_environment_config
from td3_agent import TD3Agent, device
from data_loader import (
    load_hedging_data,
    split_data_by_years,
    create_environment,
    create_environments_for_training,
    get_year_ranges_from_data
)
from trainer import (
    train_td3,
    train_multi_year,
    evaluate_agent,
    evaluate_agent_multi_episode,
    compare_with_benchmark,
    plot_comparison,
    plot_multi_episode_results,
    TrainingMetrics
)
from benchmark import run_benchmark_simple, delta_hedging_simple


def setup_results_dir():
    """Create results directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(CONFIG.get("results_dir", "results"), f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def save_config(results_dir):
    """Save configuration to results directory"""
    config_path = os.path.join(results_dir, "config.json")
    
    # Convert config to JSON-serializable format
    config_save = {}
    for k, v in CONFIG.items():
        if isinstance(v, (int, float, str, bool, list, dict, type(None))):
            config_save[k] = v
        else:
            config_save[k] = str(v)
    
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def run_full_training_pipeline(
    results_dir=None,
    verbose=True
):
    """
    Run the full training pipeline with Monte Carlo trajectories.
    
    1. Generate MC trajectories for training
    2. Train TD3 agent on MC trajectories (single pass, no validation)
    3. Test on real S&P 500 data
    
    Note: Each trajectory is used only ONCE to avoid overfitting.
    To train more, increase mc_train_trajectories in config.
    
    Args:
        results_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        dict: Results dictionary
    """
    if results_dir is None:
        results_dir = setup_results_dir()
    
    print(f"\n{'='*70}")
    print(f"TD3 HEDGING AGENT - FULL TRAINING PIPELINE")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {device}")
    print(f"Training episodes: {CONFIG.get('mc_train_trajectories', 50)} x {CONFIG.get('mc_episode_length', 30)} days")
    print(f"Test data: Real S&P 500 ({CONFIG.get('test_start_year', 2004)}-{CONFIG.get('test_end_year', 2025)})")
    print(f"Test mode: {'Windowed episodes' if CONFIG.get('use_windowed_test', True) else 'Single long episode'}")
    print(f"Note: Single-pass training (no epochs, no validation to avoid overfitting)")
    print(f"{'='*70}\n")
    
    # Save configuration
    save_config(results_dir)
    
    # Store results_dir in CONFIG so data_loader can access it for plotting
    CONFIG['current_results_dir'] = results_dir
    
    # =========================================================================
    # CREATE ENVIRONMENTS
    # =========================================================================
    print("Step 1: Creating environments...")
    
    envs = create_environments_for_training(verbose=verbose)
    
    train_envs = envs['train_envs']
    test_env = envs.get('test_env')
    test_envs = envs.get('test_envs', [])  # List of 30-day test windows
    norm_stats = envs['normalization_stats']
    use_windowed_test = CONFIG.get('use_windowed_test', True) and len(test_envs) > 1
    
    print(f"\n  Training environments: {len(train_envs)} ({CONFIG.get('mc_episode_length', 30)} days each)")
    if use_windowed_test:
        print(f"  Test environments: {len(test_envs)} ({CONFIG.get('test_episode_length', 30)}-day windows)")
    else:
        print(f"  Test environment: {'Ready' if test_env else 'Not available'}")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    print(f"\nStep 2: Training TD3 Agent on {len(train_envs)} trajectories...")
    
    model_save_path = os.path.join(results_dir, "td3_model.pth")
    CONFIG["model_save_path"] = model_save_path
    
    # Train using multi-environment approach (single pass, no validation)
    agent, metrics = train_multi_env(
        train_envs=train_envs,
        verbose=verbose
    )
    
    # Save model
    agent.save(model_save_path)
    print(f"\nModel saved to: {model_save_path}")
    
    # Save normalization statistics (CRITICAL for evaluation)
    norm_stats_path = os.path.join(results_dir, "normalization_stats.json")
    if norm_stats is not None:
        norm_stats_save = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                          for k, v in norm_stats.items()}
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats_save, f, indent=2)
        print(f"Normalization stats saved to: {norm_stats_path}")
    
    # Save training metrics
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    if test_env is None and len(test_envs) == 0:
        print("\n⚠ No test environment available - skipping evaluation")
        return {'agent': agent, 'metrics': metrics, 'results_dir': results_dir}
    
    print(f"\nStep 3: Evaluating on Test Data...")
    
    # Check if we're using windowed evaluation (same paradigm as training)
    if use_windowed_test and len(test_envs) > 1:
        # Multi-episode evaluation (30-day windows)
        print(f"\nEvaluating RL Agent on {len(test_envs)} test episodes ({CONFIG.get('test_episode_length', 30)} days each)...")
        rl_stats = evaluate_agent_multi_episode(agent, test_envs, verbose=True)
        
        # For windowed test, we use per-episode statistics
        rl_cumulative_pnl = rl_stats['total_cumulative_pnl']
        rl_sharpe = rl_stats['mean_sharpe']
        
        # Run benchmark on all test windows
        print(f"\nStep 4: Running Delta Hedging Benchmark on {len(test_envs)} test episodes...")
        benchmark_results = run_benchmark_multi_episode(test_envs, verbose=True)
        
        benchmark_pnl = benchmark_results['total_cumulative_pnl']
        benchmark_reward = benchmark_results['total_cumulative_reward']
        benchmark_sharpe = benchmark_results['mean_sharpe']
        benchmark_df = benchmark_results['aggregated_df']
        
    else:
        # Single long episode evaluation (legacy mode)
        print("\nEvaluating RL Agent on single test episode...")
        rl_stats = evaluate_agent(agent, test_env, verbose=True)
        rl_cumulative_pnl = np.sum(rl_stats['pnls'])
        rl_sharpe = rl_stats['sharpe_ratio']
        
        # =========================================================================
        # BENCHMARK COMPARISON
        # =========================================================================
        print(f"\nStep 4: Running Delta Hedging Benchmark...")
        
        # Run benchmark on test environment data
        benchmark_results = run_benchmark_on_env(test_env, verbose=True)
        
        benchmark_pnl = benchmark_results['cumulative_pnl']
        benchmark_reward = benchmark_results['cumulative_reward']
        benchmark_sharpe = benchmark_results['sharpe_ratio']
        benchmark_df = benchmark_results['df']
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON: RL Agent vs Delta Hedging")
    if use_windowed_test:
        print(f"  (Multi-Episode Evaluation: {len(test_envs)} x {CONFIG.get('test_episode_length', 30)} days)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RL Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    
    if use_windowed_test:
        print(f"{'Mean Episode P&L':<30} {rl_stats['mean_episode_pnl']:<20.4f} {benchmark_results['mean_episode_pnl']:<20.4f}")
        print(f"{'Total P&L (all episodes)':<30} {rl_cumulative_pnl:<20.4f} {benchmark_pnl:<20.4f}")
        print(f"{'Mean Sharpe Ratio':<30} {rl_sharpe:<20.4f} {benchmark_sharpe:<20.4f}")
        print(f"{'Mean Hedge Ratio':<30} {rl_stats['mean_action']:<20.4f} {benchmark_results['mean_delta']:<20.4f}")
    else:
        print(f"{'Total Reward':<30} {rl_stats['total_reward']:<20.4f} {benchmark_reward:<20.4f}")
        print(f"{'Cumulative P&L':<30} {rl_cumulative_pnl:<20.4f} {benchmark_pnl:<20.4f}")
        print(f"{'Sharpe Ratio':<30} {rl_sharpe:<20.4f} {benchmark_sharpe:<20.4f}")
        print(f"{'Mean Action (Hedge Ratio)':<30} {rl_stats['mean_action']:<20.4f} {benchmark_df['Delta'].mean():<20.4f}")
        print(f"{'Std Action':<30} {rl_stats['std_action']:<20.4f} {benchmark_df['Delta'].std():<20.4f}")
    
    print(f"{'='*70}")
    
    # Calculate improvements
    pnl_improvement = rl_cumulative_pnl - benchmark_pnl
    sharpe_improvement = rl_sharpe - benchmark_sharpe
    
    print(f"\nIMPROVEMENTS:")
    print(f"  P&L Improvement: {pnl_improvement:+.4f} ({pnl_improvement/abs(benchmark_pnl + 1e-8)*100:+.2f}%)")
    print(f"  Sharpe Improvement: {sharpe_improvement:+.4f}")
    
    if pnl_improvement > 0:
        print(f"\n✅ RL Agent OUTPERFORMS Delta Hedging!")
    else:
        print(f"\n❌ Delta Hedging outperforms RL Agent")
    
    # Plot results
    if CONFIG.get("save_plots", True):
        if use_windowed_test:
            # Multi-episode results plot (RL vs Benchmark comparison)
            multi_ep_path = os.path.join(results_dir, "multi_episode_results.png")
            plot_multi_episode_results(rl_stats, benchmark_results, save_path=multi_ep_path)
        else:
            # Single episode comparison plot
            comparison_path = os.path.join(results_dir, "comparison_test.png")
            plot_comparison(rl_stats, benchmark_df, test_env, save_path=comparison_path, 
                           test_year="Test", output_dir=results_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'mode': envs['mode'],
        'num_train_trajectories': len(train_envs),
        'episode_length_train': CONFIG.get('mc_episode_length', 30),
        'episode_length_test': CONFIG.get('test_episode_length', 30),
        'use_windowed_test': use_windowed_test,
        'num_test_episodes': len(test_envs) if use_windowed_test else 1,
        'rl_agent': {
            'total_pnl': float(rl_cumulative_pnl),
            'sharpe_ratio': float(rl_sharpe),
            'mean_action': float(rl_stats['mean_action']),
            'std_action': float(rl_stats['std_action']),
        },
        'benchmark': {
            'total_pnl': float(benchmark_pnl),
            'sharpe_ratio': float(benchmark_sharpe),
        },
        'improvements': {
            'pnl': float(pnl_improvement),
            'sharpe': float(sharpe_improvement)
        },
        'model_path': model_save_path,
        'results_dir': results_dir
    }
    
    # Add additional per-episode stats if windowed
    if use_windowed_test:
        results['rl_agent']['mean_episode_pnl'] = float(rl_stats['mean_episode_pnl'])
        results['rl_agent']['std_episode_pnl'] = float(rl_stats['std_episode_pnl'])
        results['benchmark']['mean_episode_pnl'] = float(benchmark_results['mean_episode_pnl'])
        results['benchmark']['mean_delta'] = float(benchmark_results['mean_delta'])
    
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Export detailed data
    if CONFIG.get("save_detailed_results", True):
        # RL agent step data
        if hasattr(test_env, 'step_data_history') and test_env.step_data_history:
            rl_steps_path = os.path.join(results_dir, "rl_agent_steps.csv")
            pd.DataFrame(test_env.step_data_history).to_csv(rl_steps_path, index=False)
            print(f"RL agent step data saved to: {rl_steps_path}")
        
        # Benchmark step data
        benchmark_path = os.path.join(results_dir, "benchmark_steps.csv")
        benchmark_df.to_csv(benchmark_path, index=False)
        print(f"Benchmark step data saved to: {benchmark_path}")
    
    return results


def train_multi_env(train_envs, verbose=True):
    """
    Train TD3 agent on multiple environments (trajectories).
    Each trajectory is used only ONCE to avoid overfitting.
    No validation set - test directly on real data.
    To train more, increase mc_train_trajectories in config.
    
    Args:
        train_envs: List of training environments
        verbose: Print progress
    
    Returns:
        tuple: (trained_agent, metrics)
    """
    if len(train_envs) == 0:
        raise ValueError("No training environments provided")
    
    # Get dimensions from first environment
    state_dim = train_envs[0].observation_space.shape[0]
    action_dim = train_envs[0].action_space.shape[0]
    
    # Create agent and initialize with first environment
    agent = TD3Agent(state_dim, action_dim)
    agent.set_env(train_envs[0])  # Initialize model with first environment
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    total_steps = 0
    n_trajectories = len(train_envs)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING ON {n_trajectories} TRAJECTORIES (single pass)")
        print(f"{'='*60}")
    
    # Shuffle environments
    env_indices = np.random.permutation(n_trajectories)
    
    all_rewards = []
    all_losses = []
    
    for i, env_idx in enumerate(env_indices):
        env = train_envs[env_idx]
        
        # Update agent's environment reference
        agent.set_env(env)
        
        # Train one episode on this environment
        episode_reward, episode_steps, episode_losses = train_single_episode(
            agent, env, total_steps
        )
        
        total_steps += episode_steps
        all_rewards.append(episode_reward)
        all_losses.extend(episode_losses)
        
        # Record metrics
        loss = episode_losses[-1] if episode_losses else 0.0
        metrics.add_episode(
            reward=episode_reward,
            pnl=0,  # PnL calculated separately
            length=episode_steps,
            actor_loss=0,
            critic_loss=loss,
            noise=agent.current_noise if hasattr(agent, 'current_noise') else 0.1
        )
        
        if verbose:
            progress = (i + 1) / n_trajectories * 100
            print(f"  [{progress:5.1f}%] Trajectory {env_idx + 1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    # Summary
    avg_reward = np.mean(all_rewards)
    avg_loss = np.mean(all_losses) if all_losses else 0.0
    
    if verbose:
        print(f"\n  Training Summary:")
        print(f"    Total Trajectories: {n_trajectories}")
        print(f"    Total Steps: {total_steps:,}")
        print(f"    Avg Reward: {avg_reward:.4f}")
    
    return agent, metrics


def train_single_episode(agent, env, global_step=0):
    """
    Train agent for a single episode (one trajectory).
    Uses SB3's built-in training through collect_rollouts and train.
    
    Args:
        agent: TD3Agent
        env: HedgingEnv
        global_step: Global step counter for logging
    
    Returns:
        tuple: (episode_reward, steps, losses)
    """
    # Handle both old-style (obs only) and new-style (obs, info) reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        state = reset_result[0]
    else:
        state = reset_result
    
    episode_reward = 0.0
    losses = []
    steps = 0
    done = False
    
    warmup_steps = CONFIG.get("warmup_steps", 1000)
    
    while not done:
        # Select action
        if global_step + steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=True)
        
        # Step environment - handle both 4-tuple and 5-tuple returns
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
        
        # Store transition in SB3's replay buffer
        agent.store_transition(state, action, reward, next_state, done)
        
        # Perform training step if past warmup
        if global_step + steps >= warmup_steps:
            loss_info = agent.train_step()
            if loss_info and loss_info[1] is not None:
                losses.append(loss_info[1])  # critic loss
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    return episode_reward, steps, losses


def run_benchmark_multi_episode(test_envs, verbose=True):
    """
    Run delta hedging benchmark on multiple test episodes.
    
    Args:
        test_envs: List of HedgingEnv instances
        verbose: Print progress
    
    Returns:
        dict: Aggregated benchmark results
    """
    all_pnls = []
    all_rewards = []
    all_sharpes = []
    all_deltas = []
    all_episode_results = []
    
    if verbose:
        print(f"\nRunning benchmark on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        results = run_benchmark_on_env(env, verbose=False)
        
        all_pnls.append(results['cumulative_pnl'])
        all_rewards.append(results['cumulative_reward'])
        all_sharpes.append(results['sharpe_ratio'])
        all_deltas.extend(results['df']['Delta'].tolist())
        all_episode_results.append(results)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(test_envs)} episodes...")
    
    # Aggregate
    aggregated = {
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'total_cumulative_pnl': sum(all_pnls),
        'total_cumulative_reward': sum(all_rewards),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        'mean_delta': np.mean(all_deltas),
        'std_delta': np.std(all_deltas),
        'n_episodes': len(test_envs),
        'all_pnls': all_pnls,
        'all_rewards': all_rewards,
        'all_sharpes': all_sharpes,
        'episode_results': all_episode_results,
        # Create aggregated DataFrame for plotting
        'aggregated_df': pd.concat([r['df'] for r in all_episode_results], ignore_index=True)
    }
    
    if verbose:
        print(f"\nBenchmark Multi-Episode Results:")
        print(f"  Episodes: {len(test_envs)}")
        print(f"  Mean Episode P&L: {aggregated['mean_episode_pnl']:.4f} ± {aggregated['std_episode_pnl']:.4f}")
        print(f"  Total P&L: {aggregated['total_cumulative_pnl']:.4f}")
        print(f"  Mean Sharpe: {aggregated['mean_sharpe']:.4f} ± {aggregated['std_sharpe']:.4f}")
        print(f"  Mean Delta: {aggregated['mean_delta']:.4f}")
    
    return aggregated


def run_benchmark_on_env(env, verbose=True):
    """
    Run delta hedging benchmark on an environment.
    Uses SAME P&L calculation as the RL environment for fair comparison.
    
    Args:
        env: HedgingEnv
        verbose: Print progress
    
    Returns:
        dict: Benchmark results
    """
    from scipy.stats import norm
    
    # Get RAW (unnormalized) data from environment
    option_prices = env.option_prices_raw
    stock_prices = env.stock_prices_raw
    moneyness = env.moneyness_raw
    ttm = env.ttm_raw
    
    # Get config
    r = CONFIG.get("risk_free_rate", 0.02)
    vol = CONFIG.get("mc_volatility", 0.20)
    tc = CONFIG.get("transaction_cost", 0.001)
    notional = CONFIG.get("notional", 1000)
    xi = CONFIG.get("risk_aversion", 0.01)
    
    pnl_list = []
    reward_list = []
    delta_list = []
    
    prev_delta = 0.0
    prev_position = 0.0  # Position = delta * notional
    
    for i in range(len(stock_prices)):
        S_now = stock_prices[i]
        O_now = option_prices[i]
        K = S_now / moneyness[i] if moneyness[i] > 0 else S_now
        T = max(ttm[i], 1e-6)
        
        # Black-Scholes delta
        d1 = (np.log(S_now/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T) + 1e-8)
        delta = norm.cdf(d1)
        delta = np.clip(delta, 0, 1)
        
        # P&L calculation - SAME as hedging_env.py
        if i > 0:
            S_prev = stock_prices[i-1]
            O_prev = option_prices[i-1]
            
            # HO_t = -1 (short option position) - always
            HO_t = -1
            
            # Option component: HO_t * (Ct - Ct-1) / notional
            option_component = HO_t * (O_now - O_prev) / notional
            
            # Hedge component: (prev_position / notional) * (S_now / S_prev - 1)
            hedge_component = (prev_position / notional) * (S_now / S_prev - 1)
            
            # Transaction component
            hedge_adjustment = (delta * notional - prev_position) / S_now
            transaction_component = tc * S_now * abs(hedge_adjustment) / notional
            
            # Step P&L (normalized, same as env)
            step_pnl = option_component + hedge_component - transaction_component
            pnl_list.append(step_pnl)
            
            # Reward (with risk aversion)
            reward = step_pnl - xi * abs(step_pnl)
            reward_list.append(reward)
        
        delta_list.append(delta)
        prev_delta = delta
        prev_position = delta * notional
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Stock_Price': stock_prices,
        'Option_Price': option_prices,
        'Delta': delta_list,
        'PnL': [0] + pnl_list,
        'Cumulative PnL': np.cumsum([0] + pnl_list),
        'Reward': [0] + reward_list,
        'Cumulative Reward': np.cumsum([0] + reward_list)
    })
    
    cum_pnl = results_df['Cumulative PnL'].iloc[-1]
    cum_reward = results_df['Cumulative Reward'].iloc[-1]
    sharpe = results_df['PnL'].mean() / (results_df['PnL'].std() + 1e-8) * np.sqrt(252)
    
    if verbose:
        print(f"\nBenchmark Results (normalized):")
        print(f"  Cumulative P&L: {cum_pnl:.4f}")
        print(f"  Cumulative Reward: {cum_reward:.4f}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Mean Delta: {results_df['Delta'].mean():.4f}")
    
    return {
        'df': results_df,
        'cumulative_pnl': cum_pnl,
        'cumulative_reward': cum_reward,
        'sharpe_ratio': sharpe
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train TD3 agent for options hedging')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    run_full_training_pipeline(verbose=not args.quiet)


if __name__ == "__main__":
    main()
