#!/usr/bin/env python3
"""
Evaluate trained agent on BEARISH market scenarios.

This script generates Monte Carlo trajectories with negative drift (prices tend to fall)
to test if the agent has truly learned dynamic hedging or is just exploiting
the bullish bias in S&P 500 data.

Usage:
    python evaluate_bearish.py --model results/run_20251230_093709/td3_model.pth
    python evaluate_bearish.py --model results/run_20251230_093709/td3_model.pth --episodes 200
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from td3_agent import TD3Agent, device
from hedging_env import HedgingEnv
from benchmark import delta_hedging_simple


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option pricing."""
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-6)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def generate_bearish_trajectory(
    S0=100.0,
    mu=-0.15,  # Negative drift - bearish market
    sigma=0.25,
    r=0.02,
    episode_length=30,
    rng=None,
    start_date=None
):
    """
    Generate a single bearish Monte Carlo trajectory for option hedging.
    
    Args:
        S0: Initial stock price
        mu: Annual drift (NEGATIVE for bearish)
        sigma: Annual volatility
        r: Risk-free rate
        episode_length: Number of trading days
        rng: Random number generator
        start_date: Start date for timestamps
    
    Returns:
        DataFrame with trajectory data
    """
    if rng is None:
        rng = np.random.default_rng()
    if start_date is None:
        start_date = pd.Timestamp('2020-01-01')
    
    steps_per_year = 252
    T_total = episode_length / steps_per_year
    dt = T_total / episode_length
    
    # Generate GBM path with negative drift
    prices = np.zeros(episode_length + 1)
    prices[0] = S0
    
    for t in range(1, episode_length + 1):
        Z = rng.standard_normal()
        prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    
    # Strike is ATM at inception
    K = S0
    
    # Calculate option prices and other data
    data = []
    for t in range(episode_length + 1):
        S = prices[t]
        T_remaining = max((episode_length - t) / steps_per_year, 1/steps_per_year)
        
        option_price = black_scholes_price(S, K, T_remaining, r, sigma)
        moneyness = S / K
        
        data.append({
            'timestamp': start_date + pd.Timedelta(days=t),
            'underlying_price': S,
            'option_price': option_price,
            'moneyness': moneyness,
            'time_to_maturity': T_remaining,
            'strike': K,
            'volatility': sigma
        })
    
    df = pd.DataFrame(data)
    return df


def generate_bearish_test_envs(
    n_episodes=183,
    mu=-0.15,  # -15% annual drift (bearish)
    sigma_range=(0.15, 0.35),
    S0_range=(80, 120),
    norm_stats=None,  # Normalization stats from training (CRITICAL!)
    seed=42,
    verbose=True
):
    """
    Generate bearish test environments.
    
    Args:
        n_episodes: Number of episodes to generate
        mu: Negative drift for bearish market
        sigma_range: Range of volatilities
        S0_range: Range of initial prices
        norm_stats: Normalization statistics from training (CRITICAL for correct evaluation)
        seed: Random seed
        verbose: Print progress
    
    Returns:
        list: List of HedgingEnv instances
    """
    rng = np.random.default_rng(seed)
    episode_length = CONFIG.get("test_episode_length", 30)
    r = CONFIG.get("risk_free_rate", 0.02)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATING BEARISH TEST SCENARIOS")
        print(f"{'='*60}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Episode length: {episode_length} days")
        print(f"  Drift (μ): {mu*100:.1f}% annual (BEARISH)")
        print(f"  Volatility range: {sigma_range[0]*100:.0f}% - {sigma_range[1]*100:.0f}%")
        print(f"  Initial price range: ${S0_range[0]} - ${S0_range[1]}")
        if norm_stats is not None:
            print(f"  Using normalization stats from training")
    
    envs = []
    
    for i in range(n_episodes):
        # Randomize parameters
        S0 = rng.uniform(S0_range[0], S0_range[1])
        sigma = rng.uniform(sigma_range[0], sigma_range[1])
        
        # Random start date
        start_year = 2015 + i % 10
        start_month = rng.integers(1, 13)
        start_day = rng.integers(1, 28)
        start_date = pd.Timestamp(f'{start_year}-{start_month:02d}-{start_day:02d}')
        
        # Generate trajectory
        df = generate_bearish_trajectory(
            S0=S0,
            mu=mu,
            sigma=sigma,
            r=r,
            episode_length=episode_length,
            rng=rng,
            start_date=start_date
        )
        
        # Calculate realized volatility
        vol_window = CONFIG.get("vol_window", 20)
        log_returns = np.diff(np.log(df['underlying_price'].values))
        log_returns = np.insert(log_returns, 0, 0)
        realized_vol = pd.Series(log_returns).rolling(window=vol_window).std().values * np.sqrt(252)
        realized_vol[np.isnan(realized_vol)] = sigma
        
        # Create environment
        env = HedgingEnv(
            option_prices=df['option_price'].values,
            stock_prices=df['underlying_price'].values,
            moneyness=df['moneyness'].values,
            ttm=df['time_to_maturity'].values,
            timestamps=df['timestamp'].values,
            normalize=CONFIG.get("normalize_data", True),
            normalization_stats=norm_stats,
            verbose=False
        )
        
        envs.append(env)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_episodes} bearish episodes...")
    
    if verbose:
        print(f"  ✓ Created {n_episodes} bearish test environments")
        print(f"{'='*60}\n")
    
    return envs


def evaluate_agent_on_envs(agent, envs, verbose=True):
    """
    Evaluate agent on a list of environments.
    
    Returns:
        dict: Evaluation statistics
    """
    all_pnls = []
    all_rewards = []
    all_sharpes = []
    all_actions = []
    
    for i, env in enumerate(envs):
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        episode_reward = 0
        episode_pnls = []
        episode_actions = []
        done = False
        
        while not done:
            action = agent.select_action(state, add_noise=False)
            episode_actions.append(action[0])
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            episode_reward += reward
            episode_pnls.append(info.get('step_pnl', reward))
            state = next_state
        
        episode_pnl = np.sum(episode_pnls)
        all_pnls.append(episode_pnl)
        all_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        
        # Calculate Sharpe
        if len(episode_pnls) > 1 and np.std(episode_pnls) > 0:
            sharpe = np.mean(episode_pnls) / np.std(episode_pnls) * np.sqrt(252)
        else:
            sharpe = 0
        all_sharpes.append(sharpe)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Evaluated {i + 1}/{len(envs)} episodes...")
    
    return {
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'total_cumulative_pnl': np.sum(all_pnls),
        'mean_reward': np.mean(all_rewards),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        'mean_action': np.mean(all_actions),
        'std_action': np.std(all_actions),
        'min_action': np.min(all_actions),
        'max_action': np.max(all_actions),
        'all_pnls': all_pnls,
        'all_actions': all_actions
    }


def run_benchmark_on_envs(envs, verbose=True):
    """
    Run delta hedging benchmark on environments.
    """
    all_pnls = []
    all_sharpes = []
    all_deltas = []
    
    for i, env in enumerate(envs):
        reset_result = env.reset()
        
        episode_pnls = []
        episode_deltas = []
        done = False
        step = 0
        
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
            step += 1
        
        episode_pnl = np.sum(episode_pnls)
        all_pnls.append(episode_pnl)
        all_deltas.extend(episode_deltas)
        
        if len(episode_pnls) > 1 and np.std(episode_pnls) > 0:
            sharpe = np.mean(episode_pnls) / np.std(episode_pnls) * np.sqrt(252)
        else:
            sharpe = 0
        all_sharpes.append(sharpe)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Benchmark evaluated {i + 1}/{len(envs)} episodes...")
    
    return {
        'mean_episode_pnl': np.mean(all_pnls),
        'std_episode_pnl': np.std(all_pnls),
        'total_cumulative_pnl': np.sum(all_pnls),
        'mean_sharpe': np.mean(all_sharpes),
        'std_sharpe': np.std(all_sharpes),
        'mean_delta': np.mean(all_deltas),
        'all_pnls': all_pnls
    }


def plot_comparison(rl_stats, benchmark_stats, save_path=None):
    """
    Plot comparison between RL agent and benchmark.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # P&L Distribution
    ax1 = axes[0, 0]
    ax1.hist(rl_stats['all_pnls'], bins=30, alpha=0.7, label=f"RL Agent (μ={rl_stats['mean_episode_pnl']:.4f})", color='blue')
    ax1.hist(benchmark_stats['all_pnls'], bins=30, alpha=0.7, label=f"Delta Hedge (μ={benchmark_stats['mean_episode_pnl']:.4f})", color='green')
    ax1.axvline(0, color='red', linestyle='--', label='Break-even')
    ax1.set_xlabel('Cumulative P&L per Episode')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Episode P&L (BEARISH Market)')
    ax1.legend()
    
    # Action Distribution
    ax2 = axes[0, 1]
    ax2.hist(rl_stats['all_actions'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(rl_stats['mean_action'], color='blue', linestyle='-', linewidth=2, label=f"Mean: {rl_stats['mean_action']:.3f}")
    ax2.axvline(benchmark_stats['mean_delta'], color='green', linestyle='--', linewidth=2, label=f"Benchmark Delta: {benchmark_stats['mean_delta']:.3f}")
    ax2.set_xlabel('Hedge Ratio (Action)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of RL Agent Actions')
    ax2.legend()
    
    # Cumulative P&L progression
    ax3 = axes[1, 0]
    rl_cumsum = np.cumsum(rl_stats['all_pnls'])
    bench_cumsum = np.cumsum(benchmark_stats['all_pnls'])
    ax3.plot(rl_cumsum, label='RL Agent', color='blue', linewidth=2)
    ax3.plot(bench_cumsum, label='Delta Hedge', color='green', linewidth=2)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.fill_between(range(len(rl_cumsum)), rl_cumsum, alpha=0.3, color='blue')
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Cumulative P&L')
    ax3.set_title('Cumulative P&L Progression (BEARISH Market)')
    ax3.legend()
    
    # Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    BEARISH MARKET EVALUATION RESULTS
    {'='*50}
    
    RL Agent:
      Mean Episode P&L: {rl_stats['mean_episode_pnl']:.4f} ± {rl_stats['std_episode_pnl']:.4f}
      Total P&L: {rl_stats['total_cumulative_pnl']:.4f}
      Mean Sharpe: {rl_stats['mean_sharpe']:.4f}
      Mean Hedge Ratio: {rl_stats['mean_action']:.4f}
      Action Range: [{rl_stats['min_action']:.4f}, {rl_stats['max_action']:.4f}]
    
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
    import json
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
    MODEL_PATH = "results/run_20251230_150834/td3_model.zip"
    N_EPISODES = 200
    DRIFT = -0.15  # Annual drift (negative for bearish market)
    SEED = 42
    OUTPUT_PATH = None  # None = save next to model
    # =========================================================================
    
    print(f"\n{'='*70}")
    print(f"BEARISH MARKET EVALUATION")
    print(f"{'='*70}")
    print(f"Model: {MODEL_PATH}")
    print(f"Episodes: {N_EPISODES}")
    print(f"Market drift: {DRIFT*100:.1f}% annual")
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
    
    # Generate bearish test environments
    bearish_envs = generate_bearish_test_envs(
        n_episodes=N_EPISODES,
        mu=DRIFT,
        norm_stats=norm_stats,  # Use norm_stats from training!
        seed=SEED,
        verbose=True
    )
    
    # Create agent and load model
    print("Loading trained agent...")
    sample_env = bearish_envs[0]
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.shape[0]
    agent = TD3Agent(state_dim=state_dim, action_dim=action_dim, env=sample_env)
    agent.load(model_path_final)
    print(f"  ✓ Model loaded from {model_path_final}")
    
    # Evaluate RL agent
    print(f"\nEvaluating RL Agent on {len(bearish_envs)} bearish episodes...")
    rl_stats = evaluate_agent_on_envs(agent, bearish_envs, verbose=True)
    
    # Run benchmark
    print(f"\nRunning Delta Hedging Benchmark...")
    # Reset environments for benchmark
    bearish_envs_bench = generate_bearish_test_envs(
        n_episodes=N_EPISODES,
        mu=DRIFT,
        norm_stats=norm_stats,  # Use same norm_stats
        seed=SEED,
        verbose=False
    )
    benchmark_stats = run_benchmark_on_envs(bearish_envs_bench, verbose=True)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"BEARISH MARKET RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'RL Agent':<20} {'Delta Hedge':<20}")
    print(f"{'-'*70}")
    print(f"{'Mean Episode P&L':<30} {rl_stats['mean_episode_pnl']:<20.4f} {benchmark_stats['mean_episode_pnl']:<20.4f}")
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
            print("   → Agent always fully hedges (1.0) regardless of market conditions")
            print("   → This is NAIVE behavior - doesn't adapt to bearish market")
        elif rl_stats['mean_action'] < 0.1:
            print("   → Agent never hedges (0.0)")
    else:
        print(f"✓ Agent uses variable actions (std={rl_stats['std_action']:.3f})")
        print(f"   → Range: [{rl_stats['min_action']:.3f}, {rl_stats['max_action']:.3f}]")
    
    # Plot
    output_path = OUTPUT_PATH
    if output_path is None:
        # Save next to model
        model_dir = os.path.dirname(model_path_final)
        output_path = os.path.join(model_dir, 'bearish_evaluation.png')
    
    plot_comparison(rl_stats, benchmark_stats, save_path=output_path)


if __name__ == "__main__":
    main()
