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
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from scipy.stats import norm
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    Same function as in run_training.py for consistency.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)

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
from metrics import (
    HedgingMetrics,
    evaluate_agent_with_metrics,
    evaluate_benchmark_with_metrics,
    compare_metrics,
    print_metrics_comparison,
    plot_efficient_frontier
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
    Uses the SAME implementation as run_training.py for consistency.
    
    Args:
        test_envs: List of HedgingEnv instances
        verbose: Print progress
    
    Returns:
        dict: Aggregated benchmark results
    """
    # Import run_benchmark_on_env from run_training to ensure exact same logic
    from run_training import run_benchmark_on_env
    
    all_pnls = []
    all_rewards = []
    all_sharpes = []
    all_deltas = []
    all_episode_results = []
    
    if verbose:
        print(f"\nRunning benchmark on {len(test_envs)} episodes...")
    
    for i, env in enumerate(test_envs):
        # Use exact same benchmark function as run_training.py
        results = run_benchmark_on_env(env, verbose=False)
        
        all_pnls.append(results['cumulative_pnl'])
        all_rewards.append(results['cumulative_reward'])
        all_sharpes.append(results['sharpe_ratio'])
        all_deltas.extend(results['df']['Delta'].tolist())
        all_episode_results.append(results)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(test_envs)} episodes...")
    
    # Aggregate - same format as run_training.py
    return {
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
        'episode_results': all_episode_results
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
    
    # Calculate additional metrics
    rl_variance = np.var(rl_stats['all_cumulative_pnls']) if 'all_cumulative_pnls' in rl_stats else 0.0
    bench_variance = np.var(benchmark_stats['all_pnls']) if 'all_pnls' in benchmark_stats else 0.0
    
    # Calculate Information Ratio
    if len(rl_stats.get('all_cumulative_pnls', [])) == len(benchmark_stats.get('all_pnls', [])):
        excess_returns = np.array(rl_stats['all_cumulative_pnls']) - np.array(benchmark_stats['all_pnls'])
        tracking_error = np.std(excess_returns)
        info_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0
    else:
        info_ratio = 0.0
    
    # Calculate Max Drawdown
    def calc_max_dd(pnls):
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    rl_max_dd = calc_max_dd(rl_stats.get('all_cumulative_pnls', []))
    bench_max_dd = calc_max_dd(benchmark_stats.get('all_pnls', []))
    
    summary_text = f"""
    REAL S&P 500 EVALUATION RESULTS
    {'='*50}
    
    RL Agent:
      Mean Episode P&L: {rl_stats['mean_cumulative_pnl']:.4f} ± {rl_stats['std_cumulative_pnl']:.4f}
      Total P&L: {rl_stats['total_cumulative_pnl']:.4f}
      Mean Sharpe: {rl_stats['mean_sharpe']:.4f}
      Variance: {rl_variance:.6f}
      Max Drawdown: {rl_max_dd:.4f}
      Mean Hedge Ratio: {rl_stats['mean_action']:.4f}
      Action Range: [{rl_stats.get('min_action', min(rl_stats['all_actions'])):.4f}, {rl_stats.get('max_action', max(rl_stats['all_actions'])):.4f}]
    
    Delta Hedging Benchmark:
      Mean Episode P&L: {benchmark_stats['mean_episode_pnl']:.4f} ± {benchmark_stats['std_episode_pnl']:.4f}
      Total P&L: {benchmark_stats['total_cumulative_pnl']:.4f}
      Mean Sharpe: {benchmark_stats['mean_sharpe']:.4f}
      Variance: {bench_variance:.6f}
      Max Drawdown: {bench_max_dd:.4f}
      Mean Delta: {benchmark_stats['mean_delta']:.4f}
    
    {'='*50}
    P&L Improvement: {rl_stats['total_cumulative_pnl'] - benchmark_stats['total_cumulative_pnl']:+.4f}
    Variance Reduction: {bench_variance - rl_variance:+.6f}
    Information Ratio: {info_ratio:.4f}
    Max DD Improvement: {bench_max_dd - rl_max_dd:+.4f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
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
    MODEL_PATH = "results/run_20260103_185415/td3_model.zip"
    START_YEAR = 2004
    END_YEAR = 2025
    EPISODE_LENGTH = 30  # Days per episode (same as training)
    OUTPUT_PATH = None  # None = save next to model
    # =========================================================================
    
    # Set random seed for reproducibility (same as run_training.py)
    seed = CONFIG.get("seed", 101)
    set_all_seeds(seed)
    
    print(f"\n{'='*70}")
    print(f"REAL S&P 500 DATA EVALUATION - COMPREHENSIVE METRICS")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH}")
    print(f"Test period: {START_YEAR}-{END_YEAR}")
    print(f"Episode length: {EPISODE_LENGTH} days")
    print(f"Random seed: {seed}")
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
    model_dir = os.path.dirname(model_path_final)
    
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
    
    # =========================================================================
    # COMPREHENSIVE METRICS EVALUATION
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE EVALUATION: {len(test_envs)} episodes x {EPISODE_LENGTH} days")
    print(f"{'='*70}")
    
    # Evaluate agent with comprehensive metrics
    print(f"\nEvaluating RL Agent with comprehensive metrics...")
    agent_metrics, agent_data = evaluate_agent_with_metrics(agent, test_envs, verbose=True)
    
    # Evaluate benchmark with comprehensive metrics
    print(f"\nEvaluating Delta Hedging Benchmark with comprehensive metrics...")
    benchmark_metrics, bench_data = evaluate_benchmark_with_metrics(test_envs, verbose=True)
    
    # Add comparison metrics
    agent_metrics = compare_metrics(agent_metrics, benchmark_metrics)
    
    # Print comprehensive comparison
    print_metrics_comparison(agent_metrics, benchmark_metrics, 
                            title="REAL S&P 500 EVALUATION RESULTS")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("DETAILED ANALYSIS:")
    print(f"{'='*70}")
    
    # Dynamic hedging check
    if agent_metrics.std_hedge_ratio < 0.01:
        print(f"⚠ Agent uses constant action ({agent_metrics.mean_hedge_ratio:.3f}) - NOT dynamic hedging!")
        if agent_metrics.mean_hedge_ratio > 0.9:
            print("   → Agent always fully hedges (1.0)")
            print("   → May have learned to exploit bullish bias in S&P 500 data")
        elif agent_metrics.mean_hedge_ratio < 0.1:
            print("   → Agent never hedges (0.0)")
    else:
        print(f"✓ Agent uses variable actions (std={agent_metrics.std_hedge_ratio:.3f})")
        print(f"   → Range: [{agent_metrics.min_hedge_ratio:.3f}, {agent_metrics.max_hedge_ratio:.3f}]")
    
    # Cost efficiency analysis
    print(f"\n--- Cost Efficiency Analysis ---")
    if agent_metrics.tc_savings > 0:
        print(f"✓ Agent saves {agent_metrics.tc_savings:.4f} in transaction costs ({agent_metrics.tc_savings_pct:.1f}%)")
    else:
        print(f"⚠ Agent incurs {-agent_metrics.tc_savings:.4f} MORE in transaction costs ({-agent_metrics.tc_savings_pct:.1f}%)")
    
    # Trading intensity
    turnover_ratio = agent_metrics.total_turnover / (benchmark_metrics.total_turnover + 1e-8)
    print(f"   Turnover ratio: {turnover_ratio:.2f}x benchmark")
    
    # Risk analysis
    print(f"\n--- Risk Analysis ---")
    variance_improvement = (benchmark_metrics.pnl_variance - agent_metrics.pnl_variance) / (benchmark_metrics.pnl_variance + 1e-8) * 100
    print(f"   P&L Variance change: {variance_improvement:+.1f}%")
    
    drawdown_improvement = (benchmark_metrics.max_drawdown - agent_metrics.max_drawdown) / (benchmark_metrics.max_drawdown + 1e-8) * 100
    print(f"   Max Drawdown change: {drawdown_improvement:+.1f}%")
    
    # Information Ratio interpretation
    print(f"\n--- Risk-Adjusted Analysis ---")
    print(f"   Information Ratio: {agent_metrics.information_ratio:.4f}")
    if agent_metrics.information_ratio > 0.5:
        print("   → Strong outperformance on risk-adjusted basis")
    elif agent_metrics.information_ratio > 0:
        print("   → Positive but modest risk-adjusted outperformance")
    else:
        print("   → Benchmark has better risk-adjusted returns")
    
    # =========================================================================
    # SAVE PLOTS
    # =========================================================================
    output_path = OUTPUT_PATH
    if output_path is None:
        output_path = os.path.join(model_dir, 'comprehensive_evaluation.png')
    
    # Plot efficient frontier and comparison
    frontier_path = os.path.join(model_dir, 'efficient_frontier.png')
    plot_efficient_frontier(agent_metrics, benchmark_metrics, save_path=frontier_path, show=False)
    
    # Legacy plot for backward compatibility
    # Convert metrics to old format for legacy plotting function
    rl_stats_legacy = {
        'mean_cumulative_pnl': agent_metrics.mean_episode_pnl,
        'std_cumulative_pnl': agent_metrics.std_episode_pnl,
        'total_cumulative_pnl': agent_metrics.total_pnl,
        'mean_sharpe': agent_metrics.sharpe_ratio,
        'mean_action': agent_metrics.mean_hedge_ratio,
        'std_action': agent_metrics.std_hedge_ratio,
        'min_action': agent_metrics.min_hedge_ratio,
        'max_action': agent_metrics.max_hedge_ratio,
        'all_cumulative_pnls': agent_metrics.all_episode_pnls,
        'all_actions': agent_metrics.all_hedge_ratios,
    }
    benchmark_stats_legacy = {
        'mean_episode_pnl': benchmark_metrics.mean_episode_pnl,
        'total_cumulative_pnl': benchmark_metrics.total_pnl,
        'mean_sharpe': benchmark_metrics.sharpe_ratio,
        'mean_delta': benchmark_metrics.mean_hedge_ratio,
        'std_episode_pnl': benchmark_metrics.std_episode_pnl,
        'all_pnls': benchmark_metrics.all_episode_pnls,
    }
    
    plot_real_data_comparison(rl_stats_legacy, benchmark_stats_legacy, save_path=output_path)
    
    # Save metrics to JSON
    metrics_path = os.path.join(model_dir, 'evaluation_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump({
            'agent_metrics': agent_metrics.to_dict(),
            'benchmark_metrics': benchmark_metrics.to_dict(),
            'test_period': f"{START_YEAR}-{END_YEAR}",
            'episode_length': EPISODE_LENGTH,
            'n_episodes': len(test_envs)
        }, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
