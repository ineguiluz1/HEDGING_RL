#!/usr/bin/env python3
"""
Main Training Script for TD3 Hedging Agent (v2)
================================================

This script trains a TD3 agent for options hedging using:
- Monte Carlo simulated trajectories for training/validation
- Real S&P 500 daily data for testing

Usage:
    python run_training.py                    # Use default config
    python run_training.py --test-only        # Only run test evaluation
    python run_training.py --legacy           # Use legacy mode (historical CSV)
    
Configuration:
    Monte Carlo Mode (default):
        Training: mc_train_trajectories (e.g., 50) synthetic 1-year paths
        Validation: mc_val_trajectories (e.g., 10) synthetic 1-year paths
        Testing: Real S&P 500 data (2004-2025)
    
    Legacy Mode (--legacy):
        Training years: 2005-2010 (from config.py)
        Validation year: 2011
        Test year: 2012

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
    compare_with_benchmark,
    plot_training_curves,
    plot_comparison,
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
    data_path=None,
    train_years=None,
    validation_year=None,
    test_year=None,
    results_dir=None,
    verbose=True
):
    """
    Run the full training pipeline with Monte Carlo or Legacy mode.
    
    Monte Carlo Mode (default):
    1. Generate MC trajectories for training
    2. Train TD3 agent on MC trajectories (single pass, no validation)
    3. Test on real S&P 500 data
    
    Legacy Mode:
    1. Load historical CSV data
    2. Split by years
    3. Train and test on historical data
    
    Note: Each trajectory is used only ONCE to avoid overfitting.
    To train more, increase mc_train_trajectories in config.
    
    Args:
        data_path: Path to data file (legacy mode only)
        train_years: List of training years (legacy mode only)
        validation_year: Validation year (legacy mode only)
        test_year: Test year (legacy mode only)
        results_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        dict: Results dictionary
    """
    # Set defaults from config
    use_mc = CONFIG.get("use_montecarlo_training", True)
    
    if results_dir is None:
        results_dir = setup_results_dir()
    
    mode_str = "MONTE CARLO" if use_mc else "LEGACY"
    
    print(f"\n{'='*70}")
    print(f"TD3 HEDGING AGENT - FULL TRAINING PIPELINE ({mode_str} MODE)")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Device: {device}")
    
    if use_mc:
        print(f"Training trajectories: {CONFIG.get('mc_train_trajectories', 50)}")
        print(f"Test data: Real S&P 500 ({CONFIG.get('test_start_year', 2004)}-{CONFIG.get('test_end_year', 2025)})")
        print(f"Note: Single-pass training (no epochs, no validation to avoid overfitting)")
    else:
        if train_years is None:
            train_years = CONFIG.get("train_years", [2005, 2006, 2007, 2008, 2009, 2010])
        if validation_year is None:
            validation_year = CONFIG.get("validation_year", 2011)
        if test_year is None:
            test_year = CONFIG.get("test_year", 2012)
        print(f"Training years: {train_years}")
        print(f"Validation year: {validation_year}")
        print(f"Test year: {test_year}")
    
    print(f"{'='*70}\n")
    
    # Save configuration
    save_config(results_dir)
    
    # =========================================================================
    # CREATE ENVIRONMENTS
    # =========================================================================
    print("Step 1: Creating environments...")
    
    envs = create_environments_for_training(
        data_path=data_path,
        train_years=train_years,
        validation_year=validation_year,
        test_year=test_year,
        verbose=verbose
    )
    
    train_envs = envs['train_envs']
    test_env = envs['test_env']
    norm_stats = envs['normalization_stats']
    
    print(f"\n  Training environments: {len(train_envs)}")
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
    
    # Save training metrics
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    
    # Plot training curves
    if CONFIG.get("save_plots", True):
        curves_path = os.path.join(results_dir, "training_curves.png")
        plot_training_curves(metrics, save_path=curves_path)
    
    # =========================================================================
    # TEST EVALUATION
    # =========================================================================
    if test_env is None:
        print("\n⚠ No test environment available - skipping evaluation")
        return {'agent': agent, 'metrics': metrics, 'results_dir': results_dir}
    
    print(f"\nStep 3: Evaluating on Test Data...")
    
    # Evaluate RL agent
    print("\nEvaluating RL Agent on test data...")
    rl_stats = evaluate_agent(agent, test_env, verbose=True)
    
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
    
    # Calculate RL cumulative P&L from normalized step pnls (same scale as benchmark)
    rl_cumulative_pnl = np.sum(rl_stats['pnls'])
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON: RL Agent vs Delta Hedging (Normalized P&L)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RL Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    print(f"{'Total Reward':<30} {rl_stats['total_reward']:<20.4f} {benchmark_reward:<20.4f}")
    print(f"{'Cumulative P&L':<30} {rl_cumulative_pnl:<20.4f} {benchmark_pnl:<20.4f}")
    print(f"{'Sharpe Ratio':<30} {rl_stats['sharpe_ratio']:<20.4f} {benchmark_sharpe:<20.4f}")
    print(f"{'Mean Action (Hedge Ratio)':<30} {rl_stats['mean_action']:<20.4f} {benchmark_df['Delta'].mean():<20.4f}")
    print(f"{'Std Action':<30} {rl_stats['std_action']:<20.4f} {benchmark_df['Delta'].std():<20.4f}")
    print(f"{'='*70}")
    
    # Calculate improvements
    pnl_improvement = rl_cumulative_pnl - benchmark_pnl
    sharpe_improvement = rl_stats['sharpe_ratio'] - benchmark_sharpe
    
    print(f"\nIMPROVEMENTS:")
    print(f"  P&L Improvement: {pnl_improvement:+.4f} ({pnl_improvement/abs(benchmark_pnl + 1e-8)*100:+.2f}%)")
    print(f"  Sharpe Improvement: {sharpe_improvement:+.4f}")
    
    if pnl_improvement > 0:
        print(f"\n✅ RL Agent OUTPERFORMS Delta Hedging!")
    else:
        print(f"\n❌ Delta Hedging outperforms RL Agent")
    
    # Plot comparison
    if CONFIG.get("save_plots", True):
        comparison_path = os.path.join(results_dir, "comparison_test.png")
        plot_comparison(rl_stats, benchmark_df, test_env, save_path=comparison_path, 
                       test_year="Test", output_dir=results_dir)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    results = {
        'mode': envs['mode'],
        'num_train_trajectories': len(train_envs),
        'rl_agent': {
            'total_reward': float(rl_stats['total_reward']),
            'cumulative_pnl_normalized': float(rl_cumulative_pnl),  # Normalized (same scale as benchmark)
            'cumulative_pnl_raw': float(rl_stats['cumulative_pnl']),  # Raw from environment
            'sharpe_ratio': float(rl_stats['sharpe_ratio']),
            'mean_action': float(rl_stats['mean_action']),
            'std_action': float(rl_stats['std_action']),
            'steps': int(rl_stats['steps'])
        },
        'benchmark': {
            'cumulative_pnl': float(benchmark_pnl),
            'cumulative_reward': float(benchmark_reward),
            'sharpe_ratio': float(benchmark_sharpe),
            'mean_delta': float(benchmark_df['Delta'].mean()),
            'std_delta': float(benchmark_df['Delta'].std())
        },
        'improvements': {
            'pnl': float(pnl_improvement),
            'sharpe': float(sharpe_improvement)
        },
        'model_path': model_save_path,
        'results_dir': results_dir
    }
    
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


def run_test_only(
    model_path,
    data_path=None,
    test_year=None,
    verbose=True
):
    """
    Run test evaluation only using a pre-trained model
    
    Args:
        model_path: Path to saved model
        data_path: Path to data file
        test_year: Year to test on
        verbose: Whether to print results
    
    Returns:
        dict: Test results
    """
    if data_path is None:
        data_path = CONFIG.get("data_path", "./data/historical_hedging_data.csv")
    if test_year is None:
        test_year = CONFIG.get("test_year", 2012)
    
    print(f"\n{'='*70}")
    print(f"TD3 HEDGING AGENT - TEST EVALUATION")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Test year: {test_year}")
    print(f"{'='*70}\n")
    
    # Load data
    df = load_hedging_data(data_path)
    datetime_col = CONFIG.get("datetime_column", "timestamp")
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    test_df = df[df[datetime_col].dt.year == test_year].copy().reset_index(drop=True)
    
    if len(test_df) == 0:
        raise ValueError(f"No data found for test year {test_year}")
    
    # Create test environment (without normalization stats - will compute from test data)
    test_env = create_environment(
        test_df,
        normalize=CONFIG.get("normalize_data", True),
        verbose=False
    )
    
    # Load agent
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.shape[0]
    
    agent = TD3Agent(state_dim, action_dim)
    agent.load(model_path)
    
    # Evaluate
    rl_stats = evaluate_agent(agent, test_env, verbose=verbose)
    
    # Run benchmark
    benchmark_df = run_benchmark_simple(df, year=test_year, verbose=verbose)
    
    benchmark_pnl = benchmark_df["Cumulative PnL"].iloc[-1]
    benchmark_sharpe = benchmark_df["PnL"].mean() / (benchmark_df["PnL"].std() + 1e-8) * np.sqrt(252)
    
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"RL Cumulative P&L: {rl_stats['cumulative_pnl']:.4f}")
    print(f"Benchmark Cumulative P&L: {benchmark_pnl:.4f}")
    print(f"Improvement: {rl_stats['cumulative_pnl'] - benchmark_pnl:+.4f}")
    print(f"{'='*70}")
    
    return {
        'rl_stats': rl_stats,
        'benchmark_pnl': benchmark_pnl,
        'benchmark_sharpe': benchmark_sharpe
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train TD3 agent for options hedging')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file (legacy mode only)')
    parser.add_argument('--train-years', type=int, nargs='+', default=None,
                       help='Years to use for training (legacy mode only)')
    parser.add_argument('--val-year', type=int, default=None,
                       help='Year to use for validation (legacy mode only)')
    parser.add_argument('--test-year', type=int, default=None,
                       help='Year to use for testing (legacy mode only)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run test evaluation')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for test-only mode')
    parser.add_argument('--legacy', action='store_true',
                       help='Use legacy mode (historical CSV data instead of Monte Carlo)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set mode in config
    if args.legacy:
        CONFIG["use_montecarlo_training"] = False
    
    if args.test_only:
        if args.model_path is None:
            print("Error: --model-path required for --test-only mode")
            sys.exit(1)
        
        run_test_only(
            model_path=args.model_path,
            data_path=args.data_path,
            test_year=args.test_year,
            verbose=not args.quiet
        )
    else:
        run_full_training_pipeline(
            data_path=args.data_path,
            train_years=args.train_years,
            validation_year=args.val_year,
            test_year=args.test_year,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
