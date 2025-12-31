"""
Monte Carlo Data Generator for Hedging RL Training
Generates synthetic option hedging data using Monte Carlo simulations
for training, while real S&P 500 data is used for testing.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import CONFIG
import os


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes option pricing.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
    
    Returns:
        Option price
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-6)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def generate_gbm_path(S0, mu, sigma, T, steps, rng=None):
    """
    Generate a single Geometric Brownian Motion price path.
    
    Args:
        S0: Initial price
        mu: Annual drift
        sigma: Annual volatility
        T: Time horizon (years)
        steps: Number of time steps
        rng: Random number generator
    
    Returns:
        np.ndarray: Price path of shape (steps+1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    dt = T / steps
    path = np.zeros(steps + 1)
    path[0] = S0
    
    for t in range(1, steps + 1):
        Z = rng.standard_normal()
        path[t] = path[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    
    return path


def calculate_rolling_volatility(prices, window=20, annualization=252):
    """
    Calculate rolling realized volatility.
    
    Args:
        prices: Price series
        window: Rolling window size
        annualization: Annualization factor
    
    Returns:
        np.ndarray: Realized volatility series
    """
    log_returns = np.diff(np.log(prices))
    log_returns = np.insert(log_returns, 0, 0)
    
    vol = pd.Series(log_returns).rolling(window=window).std().values * np.sqrt(annualization)
    
    # Fill initial NaN values with random volatility
    mask_nan = np.isnan(vol)
    vol[mask_nan] = np.random.uniform(
        CONFIG.get("initial_vol_min", 0.02),
        CONFIG.get("initial_vol_max", 0.10),
        size=mask_nan.sum()
    )
    
    return vol


def generate_mc_hedging_trajectory(
    S0=100.0,
    mu=0.05,
    sigma=0.20,
    r=0.02,
    T=None,
    steps=None,
    vol_window=20,
    rng=None,
    start_date=None,
    simulate_option_expiry=True
):
    """
    Generate a single Monte Carlo trajectory with option hedging data.
    
    Simulates a realistic option hedging scenario:
    - At t=0: Option is sold, strike set at current spot (ATM)
    - As time passes: TTM decreases linearly toward expiry
    - At t=steps: Option expires (TTM=0)
    
    Args:
        S0: Initial stock price
        mu: Annual drift
        sigma: Annual volatility for GBM
        r: Risk-free rate
        T: Time horizon (years). If None, uses mc_episode_length from config
        steps: Number of time steps (trading days). If None, uses mc_episode_length from config
        vol_window: Window for realized volatility
        rng: Random number generator
        start_date: Start date for timestamps
        simulate_option_expiry: If True, TTM decreases from initial to 0 (realistic option)
    
    Returns:
        pd.DataFrame: DataFrame with hedging data
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get episode length from config if not provided
    if steps is None:
        steps = CONFIG.get("mc_episode_length", 30)
    if T is None:
        T = steps / CONFIG.get("mc_steps_per_year", 252)  # Convert days to years
    
    if start_date is None:
        start_date = pd.Timestamp('2020-01-01')
    
    # Generate stock price path
    S = generate_gbm_path(S0, mu, sigma, T, steps, rng)
    
    # Generate timestamps (business days)
    timestamps = pd.date_range(start=start_date, periods=steps+1, freq='B')
    
    # Calculate realized volatility
    realized_vol = calculate_rolling_volatility(S, window=vol_window, annualization=252)
    
    # Fixed strike price: ATM at inception (t=0) with small noise
    # This simulates selling an option at the start of the episode
    initial_moneyness_noise = rng.normal(0, 0.02)  # ±2% around ATM
    K = np.full(steps + 1, S[0] * (1 + initial_moneyness_noise))  # Fixed strike throughout
    
    # Time to maturity: decreases linearly from initial TTM to near-zero at expiry
    if simulate_option_expiry:
        initial_ttm = steps / 252.0  # Initial TTM in years (e.g., 30/252 ≈ 0.119 years)
        ttm = np.linspace(initial_ttm, 1/252.0, steps + 1)  # Decrease to ~1 day at expiry
    else:
        # Legacy behavior: rolling contracts with constant ~30 day TTM
        target_ttm = 30 / 365.0
        noise_ttm = rng.normal(0, 0.001, steps+1)
        ttm = np.clip(target_ttm + noise_ttm, 0.01, 0.2)
    
    # Option prices using Black-Scholes
    option_prices = black_scholes_price(S, K, ttm, r, realized_vol)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'underlying_price': S,
        'strike': K,
        'time_to_maturity': ttm,
        'option_price': option_prices,
        'risk_free_rate': r,
        'realized_volatility': realized_vol,
        'moneyness': S / K
    })
    
    return df


def generate_mc_training_data(
    n_trajectories=50,
    S0=None,
    mu=None,
    sigma=None,
    r=None,
    episode_length=None,
    vol_window=None,
    seed=None,
    verbose=True,
    plot_path=None
):
    """
    Generate Monte Carlo trajectories for training.
    Each trajectory simulates one option hedging episode (e.g., 30 days until expiry).
    
    Args:
        n_trajectories: Number of episode trajectories to generate
        S0: Initial price (uses config default if None)
        mu: Annual drift (uses config default if None)
        sigma: Annual volatility (uses config default if None)
        r: Risk-free rate (uses config default if None)
        episode_length: Episode length in trading days (uses config default if None)
        vol_window: Volatility window (uses config default if None)
        seed: Random seed (uses config default if None)
        verbose: Whether to print progress
        plot_path: Optional path to save trajectory plot (e.g., 'results/run_*/mc_trajectories.png')
    
    Returns:
        list: List of DataFrames, one per trajectory
    """
    # Use config defaults
    if S0 is None:
        S0 = CONFIG.get("mc_initial_price", 100.0)
    if mu is None:
        mu = CONFIG.get("mc_drift", 0.05)
    if sigma is None:
        sigma = CONFIG.get("mc_volatility", 0.20)
    if r is None:
        r = CONFIG.get("risk_free_rate", 0.02)
    if episode_length is None:
        episode_length = CONFIG.get("mc_episode_length", 30)
    if vol_window is None:
        vol_window = CONFIG.get("vol_window", 20)
    if seed is None:
        # Use main seed for consistency, fallback to mc_seed
        seed = CONFIG.get("seed", CONFIG.get("mc_seed", 42))
    
    # Calculate T (time horizon in years) from episode length
    steps_per_year = CONFIG.get("mc_steps_per_year", 252)
    T = episode_length / steps_per_year
    
    rng = np.random.default_rng(seed)
    trajectories = []
    
    # Check if using mixed drift scenarios
    use_mixed_drift = CONFIG.get("mc_use_mixed_drift", False)
    drift_distribution = CONFIG.get("mc_drift_distribution", {
        "bullish": 0.33, "neutral": 0.34, "bearish": 0.33
    })
    drift_ranges = CONFIG.get("mc_drift_ranges", {
        "bullish": (0.05, 0.20),
        "neutral": (-0.05, 0.05),
        "bearish": (-0.20, -0.05)
    })
    
    # Curriculum learning info
    use_curriculum = CONFIG.get("use_curriculum_learning", False)
    curriculum_neutral_ratio = CONFIG.get("curriculum_neutral_ratio", 0.4)
    
    if verbose:
        print(f"\nGenerating {n_trajectories} Monte Carlo trajectories...")
        print(f"  Episode Length: {episode_length} trading days ({T*252:.1f} days)")
        print(f"  Initial Price: ${S0:.2f}")
        if use_curriculum:
            curriculum_count = int(n_trajectories * curriculum_neutral_ratio)
            print(f"  CURRICULUM LEARNING ENABLED:")
            print(f"    - Phase 1: {curriculum_count} trajectories with NEUTRAL drift only (learn hedging)")
            print(f"    - Phase 2: {n_trajectories - curriculum_count} trajectories with mixed drift")
        if use_mixed_drift:
            print(f"  Drift Mode: MIXED (bullish/neutral/bearish)")
            print(f"    - Bullish ({drift_distribution['bullish']*100:.0f}%): μ in [{drift_ranges['bullish'][0]*100:.0f}%, {drift_ranges['bullish'][1]*100:.0f}%]")
            print(f"    - Neutral ({drift_distribution['neutral']*100:.0f}%): μ in [{drift_ranges['neutral'][0]*100:.0f}%, {drift_ranges['neutral'][1]*100:.0f}%]")
            print(f"    - Bearish ({drift_distribution['bearish']*100:.0f}%): μ in [{drift_ranges['bearish'][0]*100:.0f}%, {drift_ranges['bearish'][1]*100:.0f}%]")
        else:
            print(f"  Drift (μ): {mu*100:.1f}%")
        print(f"  Volatility (σ): {sigma*100:.1f}%")
        print(f"  Simulates: Option sold at t=0, expires at t={episode_length}")
    
    # Track drift distribution for verification
    drift_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    
    # Curriculum learning configuration
    use_curriculum = CONFIG.get("use_curriculum_learning", False)
    curriculum_neutral_ratio = CONFIG.get("curriculum_neutral_ratio", 0.4)
    curriculum_boundary = int(n_trajectories * curriculum_neutral_ratio) if use_curriculum else 0
    
    for i in range(n_trajectories):
        # Vary initial price significantly for each trajectory
        # This helps the agent generalize across different price levels
        S0_varied = S0 * rng.uniform(0.7, 1.3)
        
        # Vary volatility to simulate different market conditions
        sigma_varied = sigma * rng.uniform(0.6, 1.5)
        
        # CURRICULUM LEARNING: First N trajectories are neutral drift only
        # This forces the agent to learn hedging before it can exploit directional moves
        if use_curriculum and i < curriculum_boundary:
            # Phase 1: Neutral drift only (learn to hedge, not trade)
            mu_varied = rng.uniform(-0.02, 0.02)  # Very small drift (~0)
            drift_counts["neutral"] += 1
        elif use_mixed_drift:
            # Phase 2 (or no curriculum): Sample drift scenario
            rand_val = rng.random()
            if rand_val < drift_distribution["bullish"]:
                scenario = "bullish"
                mu_varied = rng.uniform(drift_ranges["bullish"][0], drift_ranges["bullish"][1])
            elif rand_val < drift_distribution["bullish"] + drift_distribution["neutral"]:
                scenario = "neutral"
                mu_varied = rng.uniform(drift_ranges["neutral"][0], drift_ranges["neutral"][1])
            else:
                scenario = "bearish"
                mu_varied = rng.uniform(drift_ranges["bearish"][0], drift_ranges["bearish"][1])
            drift_counts[scenario] += 1
        else:
            mu_varied = mu
        
        # Random start date for diversity (doesn't affect simulation, just timestamps)
        start_year = 2010 + i % 10
        start_month = rng.integers(1, 13)
        start_day = rng.integers(1, 28)
        start_date = pd.Timestamp(f'{start_year}-{start_month:02d}-{start_day:02d}')
        
        df = generate_mc_hedging_trajectory(
            S0=S0_varied,
            mu=mu_varied,
            sigma=sigma_varied,
            r=r,
            T=T,
            steps=episode_length,
            vol_window=vol_window,
            rng=rng,
            start_date=start_date,
            simulate_option_expiry=True  # Critical: TTM decreases to expiry
        )
        
        trajectories.append(df)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_trajectories} trajectories")
    
    if verbose:
        print(f"  ✓ Generated {n_trajectories} trajectories ({episode_length} steps each)")
        if use_mixed_drift:
            print(f"  Drift distribution: Bullish={drift_counts['bullish']}, Neutral={drift_counts['neutral']}, Bearish={drift_counts['bearish']}")
    
    # Plot trajectories if path provided
    if plot_path is not None:
        try:
            from importlib import import_module
            mc_module = import_module('00_montecarlo_simulations')
            plot_func = getattr(mc_module, 'plot_monte_carlo_trajectories')
            
            # Convert trajectories to paths array (n_trajectories x steps+1)
            paths = np.array([df['underlying_price'].values for df in trajectories])
            
            # Use first trajectory's timestamps for x-axis
            start_date = trajectories[0]['timestamp'].iloc[0]
            
            if verbose:
                print(f"  Saving trajectory plot to: {plot_path}")
            
            # Plot and save
            plot_func(
                paths=paths,
                start_date=start_date,
                freq='B',
                filename=plot_path,
                figsize=(14, 8)
            )
            
            if verbose:
                print(f"  ✓ Plot saved to {plot_path}")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Warning: Could not save plot: {e}")
    
    return trajectories


def generate_train_val_data(
    train_trajectories=None,
    seed=None,
    verbose=True,
    plot_path=None
):
    """
    Generate training dataset using Monte Carlo simulation.
    Each trajectory represents one option hedging episode (30 days by default).
    
    Args:
        train_trajectories: Number of training trajectories (uses config if None)
        seed: Random seed
        verbose: Whether to print progress
        plot_path: Optional path to save trajectory plot
    
    Returns:
        dict: {'train': list of DataFrames}
    """
    if train_trajectories is None:
        train_trajectories = CONFIG.get("mc_train_trajectories", 50)
    if seed is None:
        seed = CONFIG.get("mc_seed", 42)
    
    episode_length = CONFIG.get("mc_episode_length", 30)
    
    if verbose:
        print("\n" + "="*60)
        print("GENERATING MONTE CARLO DATA FOR TRAINING")
        print("="*60)
        print(f"  Episode paradigm: {episode_length}-day option hedging windows")
        print(f"  Each episode simulates: Option sold at t=0, expires at t={episode_length}")
    
    # Generate training data
    train_data = generate_mc_training_data(
        n_trajectories=train_trajectories,
        seed=seed,
        verbose=verbose,
        plot_path=plot_path
    )
    
    if verbose:
        total_train_steps = sum(len(df) for df in train_data)
        print(f"\n  Training: {train_trajectories} episodes ({total_train_steps:,} total steps)")
        print(f"  Each episode: {episode_length} trading days")
        print("="*60 + "\n")
    
    return {
        'train': train_data
    }


if __name__ == "__main__":
    # Test the generator
    print("Testing Monte Carlo Data Generator...")
    print("Generating 30-day option hedging episodes...")
    
    # Generate sample data
    data = generate_train_val_data(
        train_trajectories=5,
        verbose=True
    )
    
    print("\nSample training trajectory (30-day option hedging episode):")
    sample_df = data['train'][0]
    print(sample_df.head(10))
    print(f"\nShape: {sample_df.shape}")
    print(f"\nTime-to-Maturity progression:")
    print(f"  Start TTM: {sample_df['time_to_maturity'].iloc[0]:.4f} years ({sample_df['time_to_maturity'].iloc[0]*252:.1f} days)")
    print(f"  End TTM:   {sample_df['time_to_maturity'].iloc[-1]:.4f} years ({sample_df['time_to_maturity'].iloc[-1]*252:.1f} days)")
    print(f"\nStrike price (fixed throughout episode): ${sample_df['strike'].iloc[0]:.2f}")
    print(f"Initial stock price: ${sample_df['underlying_price'].iloc[0]:.2f}")
    print(f"Final stock price:   ${sample_df['underlying_price'].iloc[-1]:.2f}")
