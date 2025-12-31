#!/usr/bin/env python3
"""
Evaluate trained TD3 agent on REAL S&P 500 data.

This script evaluates a trained agent using real market data (S&P 500)
with the same multi-episode paradigm used in training.

Usage:
    python evaluate_model.py
    
Configuration:
    Edit MODEL_PATH and other parameters at the top of main() function
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from scipy.stats import norm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from td3_agent import TD3Agent, device
from data_loader import (
    load_hedging_data,
    split_data_by_years,
    create_environment,
    _create_windowed_test_envs
)
from trainer import (
    evaluate_agent_multi_episode
)


def create_test_environments_from_real_data(
    start_year=2004,
    end_year=2025,
    episode_length=30,
    norm_stats=None,
    verbose=True
):
    """
    Create test environments from real S&P 500 data using sliding windows.
    
    Args:
        start_year: Start year for test data
        end_year: End year for test data
        episode_length: Length of each episode in days
        norm_stats: Normalization statistics from training (CRITICAL for correct evaluation)
        verbose: Print progress
    
    Returns:
        list: List of HedgingEnv instances
    """
    from generate_contract import generate_historical_hedging_data
    
    # Load S&P 500 data
    sp500_path = CONFIG.get("sp500_data_path", "./data/sp500_data.csv")
    
    # Handle relative paths
    if not os.path.isabs(sp500_path):
        if not os.path.exists(sp500_path):
            # Try from parent directory (when running from src/)
            alt_path = os.path.join(os.path.dirname(__file__), '..', sp500_path.lstrip('./'))
            if os.path.exists(alt_path):
                sp500_path = alt_path
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"LOADING REAL S&P 500 TEST DATA")
        print(f"{'='*60}")
        print(f"  Data path: {sp500_path}")
        print(f"  Years: {start_year}-{end_year}")
        print(f"  Episode length: {episode_length} days")
        if norm_stats is not None:
            print(f"  Using normalization stats from training")
    
    # Generate hedging data from S&P 500 prices
    df = generate_historical_hedging_data(
        csv_path=sp500_path,
        r=CONFIG.get("risk_free_rate", 0.02),
        vol_window=CONFIG.get("vol_window", 20)
    )
    
    # Filter by year range
    datetime_col = 'timestamp'  # generate_historical_hedging_data uses 'timestamp'
    df['_year'] = df[datetime_col].dt.year
    df = df[(df['_year'] >= start_year) & (df['_year'] <= end_year)].copy()
    df.drop('_year', axis=1, inplace=True)
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError(f"No data found for years {start_year}-{end_year}")
    
    if verbose:
        print(f"  Total data points: {len(df)}")
        print(f"  Date range: {df[datetime_col].min().date()} to {df[datetime_col].max().date()}")
    
    # Use the same windowing logic as run_training.py
    test_envs = _create_windowed_test_envs(
        df, 
        window_length=episode_length,
        norm_stats=norm_stats,  # Use norm_stats from training!
        normalize=CONFIG.get("normalize_data", True),
        verbose=verbose
    )
    
    return test_envs


def run_benchmark_on_real_data_multi_episode(test_envs, verbose=True):
    """
    Run delta hedging benchmark on multiple test episodes from real data.
    
    Args:
        test_envs: List of HedgingEnv instances
        verbose: Print progress
    
    Returns:
        dict: Aggregated benchmark results
    """
    all_pnls = []
    all_sharpes = []
    all_deltas = []
    
    if verbose:
        print(f"\nRunning benchmark on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        reset_result = env.reset()
        
        episode_pnls = []
        episode_deltas = []
        done = False
        
        while not done:
            # Calculate Black-Scholes delta
            S = env.stock_prices_raw[env.current_step]
            K = S / env.moneyness_raw[env.current_step]
            T = max(env.ttm_raw[env.current_step], 1e-6)
            r = env.discount_rate
            sigma = max(env.realized_vol_raw[env.current_step], 0.01)
            
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            delta = norm.cdf(d1)
            episode_deltas.append(delta)
            
            action = np.array([delta])
            step_result = env.step(action)
            
            if len(step_result) == 5:
                _, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                _, reward, done, info = step_result
            
            episode_pnls.append(info.get('step_pnl', reward))
        
        episode_pnl = np.sum(episode_pnls)
        all_pnls.append(episode_pnl)
        all_deltas.extend(episode_deltas)
        
        if len(episode_pnls) > 1 and np.std(episode_pnls) > 0:
            sharpe = np.mean(episode_pnls) / np.std(episode_pnls) * np.sqrt(252)
        else:
            sharpe = 0
        all_sharpes.append(sharpe)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(test_envs)} episodes...")
    
    return {
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'total_cumulative_pnl': np.sum(all_pnls),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        'mean_delta': np.mean(all_deltas),
        'all_pnls': all_pnls
    }


def plot_real_data_comparison(rl_stats, benchmark_stats, save_path=None):
    """
    Plot comparison between RL agent and benchmark on real data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L Distribution
    ax1 = axes[0, 0]
    ax1.hist(rl_stats['all_cumulative_pnls'], bins=30, alpha=0.7, 
             label=f"RL Agent (μ={rl_stats['mean_cumulative_pnl']:.4f})", color='blue')
    ax1.hist(benchmark_stats['all_pnls'], bins=30, alpha=0.7, 
             label=f"Delta Hedge (μ={benchmark_stats['mean_episode_pnl']:.4f})", color='green')
    ax1.axvline(0, color='red', linestyle='--', label='Break-even')
    ax1.set_xlabel('Cumulative P&L per Episode')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Episode P&L (Real S&P 500 Data)')
    ax1.legend()
    
    # Action Distribution
    ax2 = axes[0, 1]
    ax2.hist(rl_stats['all_actions'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(rl_stats['mean_action'], color='blue', linestyle='-', linewidth=2, 
                label=f"RL Mean: {rl_stats['mean_action']:.3f}")
    ax2.axvline(benchmark_stats['mean_delta'], color='green', linestyle='--', linewidth=2, 
                label=f"Benchmark Delta: {benchmark_stats['mean_delta']:.3f}")
    ax2.set_xlabel('Hedge Ratio (Action)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of RL Agent Actions')
    ax2.legend()
    
    # Cumulative P&L progression
    ax3 = axes[1, 0]
    rl_cumsum = np.cumsum(rl_stats['all_cumulative_pnls'])
    bench_cumsum = np.cumsum(benchmark_stats['all_pnls'])
    ax3.plot(rl_cumsum, label='RL Agent', color='blue', linewidth=2)
    ax3.plot(bench_cumsum, label='Delta Hedge', color='green', linewidth=2)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between(range(len(rl_cumsum)), rl_cumsum, alpha=0.3, color='blue')
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Cumulative P&L')
    ax3.set_title('Cumulative P&L Progression (Real S&P 500)')
    ax3.legend()
    
    # Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    REAL S&P 500 EVALUATION RESULTS
    {'='*50}
    
    RL Agent:
      Mean Episode P&L: {rl_stats['mean_cumulative_pnl']:.4f} ± {rl_stats['std_cumulative_pnl']:.4f}
      Total P&L: {rl_stats['total_cumulative_pnl']:.4f}
      Mean Sharpe: {rl_stats['mean_sharpe']:.4f}
      Mean Hedge Ratio: {rl_stats['mean_action']:.4f}
      Action Range: [{rl_stats.get('min_action', min(rl_stats['all_actions'])):.4f}, {rl_stats.get('max_action', max(rl_stats['all_actions'])):.4f}]
    
    Delta Hedging Benchmark:
      Mean Episode P&L: {benchmark_stats['mean_episode_pnl']:.4f} ± {benchmark_stats['std_episode_pnl']:.4f}
      Total P&L: {benchmark_stats['total_cumulative_pnl']:.4f}
      Mean Sharpe: {benchmark_stats['mean_sharpe']:.4f}
      Mean Delta: {benchmark_stats['mean_delta']:.4f}
    
    {'='*50}
    P&L Improvement: {rl_stats['total_cumulative_pnl'] - benchmark_stats['total_cumulative_pnl']:+.4f}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def load_normalization_stats(model_path):
    """
    Load normalization statistics saved during training.
    These are CRITICAL for correct evaluation.
    """
    model_dir = os.path.dirname(model_path)
    norm_stats_path = os.path.join(model_dir, "normalization_stats.json")
    
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        print(f"  ✓ Loaded normalization stats from {norm_stats_path}")
        return norm_stats
    else:
        print(f"  ⚠ WARNING: normalization_stats.json not found at {norm_stats_path}")
        print(f"    Evaluation results may differ from training!")
        return None


def main():
    # =========================================================================
    # CONFIGURATION - Edit these values to change what to evaluate
    # =========================================================================
    MODEL_PATH = "results/run_20251230_144831/td3_model.zip"
    START_YEAR = 2004
    END_YEAR = 2025
    EPISODE_LENGTH = 30  # Days per episode (same as training)
    OUTPUT_PATH = None  # None = save next to model
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"REAL S&P 500 DATA EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test period: {START_YEAR}-{END_YEAR}")
    print(f"Episode length: {EPISODE_LENGTH} days")
    print(f"{'='*70}\n")
    
    # Resolve model path (handle relative paths from workspace root)
    model_path = MODEL_PATH
    
    # If path doesn't exist, try from workspace root
    if not os.path.exists(model_path):
        # Get workspace root (parent of src/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(script_dir)
        workspace_model_path = os.path.join(workspace_root, model_path)
        
        if os.path.exists(workspace_model_path):
            model_path = workspace_model_path
        else:
            # Try adding extensions
            for ext in ['.zip', '.pth', '']:
                if os.path.exists(model_path + ext):
                    model_path = model_path + ext
                    break
                elif os.path.exists(workspace_model_path + ext):
                    model_path = workspace_model_path + ext
                    break
            else:
                # Try without extension
                model_base = model_path.rsplit('.', 1)[0] if '.' in model_path else model_path
                workspace_base = os.path.join(workspace_root, model_base)
                for ext in ['.zip', '.pth']:
                    if os.path.exists(model_base + ext):
                        model_path = model_base + ext
                        break
                    elif os.path.exists(workspace_base + ext):
                        model_path = workspace_base + ext
                        break
                else:
                    print(f"Error: Model not found")
                    print(f"  Tried: {MODEL_PATH}")
                    print(f"  Tried: {workspace_model_path}")
                    return
    
    model_path_final = model_path
    
    # Load normalization stats from training (CRITICAL for correct evaluation)
    norm_stats = load_normalization_stats(model_path_final)
    
    # Create test environments from real S&P 500 data
    test_envs = create_test_environments_from_real_data(
        start_year=START_YEAR,
        end_year=END_YEAR,
        episode_length=EPISODE_LENGTH,
        norm_stats=norm_stats,  # Use norm_stats from training!
        verbose=True
    )
    
    # Create agent and load model
    print("Loading trained agent...")
    sample_env = test_envs[0]
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.shape[0]
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, env=sample_env)
    agent.load(model_path_final)
    print(f"  ✓ Model loaded from {model_path_final}")
    
    # Evaluate RL agent
    print(f"\nEvaluating RL Agent on {len(test_envs)} test episodes...")
    rl_stats = evaluate_agent_multi_episode(agent, test_envs, verbose=True)
    
    # Run benchmark
    print(f"\nRunning Delta Hedging Benchmark...")
    benchmark_stats = run_benchmark_on_real_data_multi_episode(test_envs, verbose=True)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"REAL S&P 500 RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RL Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    print(f"{'Mean Episode P&L':<30} {rl_stats['mean_cumulative_pnl']:<20.4f} {benchmark_stats['mean_episode_pnl']:<20.4f}")
    print(f"{'Total P&L':<30} {rl_stats['total_cumulative_pnl']:<20.4f} {benchmark_stats['total_cumulative_pnl']:<20.4f}")
    print(f"{'Mean Sharpe':<30} {rl_stats['mean_sharpe']:<20.4f} {benchmark_stats['mean_sharpe']:<20.4f}")
    print(f"{'Mean Hedge Ratio':<30} {rl_stats['mean_action']:<20.4f} {benchmark_stats['mean_delta']:<20.4f}")
    print(f"{'Action Std':<30} {rl_stats['std_action']:<20.4f} {'-':<20}")
    print(f"{'='*70}")
    
    pnl_diff = rl_stats['total_cumulative_pnl'] - benchmark_stats['total_cumulative_pnl']
    if pnl_diff > 0:
        print(f"\n✅ RL Agent OUTPERFORMS Delta Hedging by {pnl_diff:+.4f}")
    else:
        print(f"\n❌ Delta Hedging outperforms RL Agent by {-pnl_diff:.4f}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS:")
    print(f"{'='*70}")
    
    if rl_stats['std_action'] < 0.01:
        print(f"⚠ Agent uses constant action ({rl_stats['mean_action']:.3f}) - NOT dynamic hedging!")
        if rl_stats['mean_action'] > 0.9:
            print("   → Agent always fully hedges (1.0)")
            print("   → May have learned to exploit bullish bias in S&P 500 data")
        elif rl_stats['mean_action'] < 0.1:
            print("   → Agent never hedges (0.0)")
    else:
        print(f"✓ Agent uses variable actions (std={rl_stats['std_action']:.3f})")
        print(f"   → Range: [{rl_stats.get('min_action', min(rl_stats['all_actions'])):.3f}, {rl_stats.get('max_action', max(rl_stats['all_actions'])):.3f}]")
    
    # Plot
    output_path = OUTPUT_PATH
    if output_path is None:
        # Save next to model
        model_dir = os.path.dirname(model_path_final)
        output_path = os.path.join(model_dir, 'real_data_evaluation.png')
    
    plot_real_data_comparison(rl_stats, benchmark_stats, save_path=output_path)


if __name__ == "__main__":
    main()
