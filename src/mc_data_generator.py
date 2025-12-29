"""
Monte Carlo Data Generator for Hedging RL Training
Generates synthetic option hedging data using Monte Carlo simulations
for training, while real S&P 500 data is used for testing.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import CONFIG


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
    T=1.0,
    steps=252,
    vol_window=20,
    rng=None,
    start_date=None
):
    """
    Generate a single Monte Carlo trajectory with option hedging data.
    
    Args:
        S0: Initial stock price
        mu: Annual drift
        sigma: Annual volatility for GBM
        r: Risk-free rate
        T: Time horizon (years)
        steps: Number of time steps (trading days)
        vol_window: Window for realized volatility
        rng: Random number generator
        start_date: Start date for timestamps
    
    Returns:
        pd.DataFrame: DataFrame with hedging data
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if start_date is None:
        start_date = pd.Timestamp('2020-01-01')
    
    # Generate stock price path
    S = generate_gbm_path(S0, mu, sigma, T, steps, rng)
    
    # Generate timestamps (business days)
    timestamps = pd.date_range(start=start_date, periods=steps+1, freq='B')
    
    # Calculate realized volatility
    realized_vol = calculate_rolling_volatility(S, window=vol_window, annualization=252)
    
    # Generate strike prices (ATM with small noise)
    noise_moneyness = rng.normal(0, 0.005, steps+1)
    K = S * (1 + noise_moneyness)
    
    # Time to maturity (rolling ~30 day contracts)
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
    steps_per_year=None,
    vol_window=None,
    seed=None,
    verbose=True
):
    """
    Generate Monte Carlo trajectories for training.
    
    Args:
        n_trajectories: Number of 1-year trajectories to generate
        S0: Initial price (uses config default if None)
        mu: Annual drift (uses config default if None)
        sigma: Annual volatility (uses config default if None)
        r: Risk-free rate (uses config default if None)
        steps_per_year: Trading days per year (uses config default if None)
        vol_window: Volatility window (uses config default if None)
        seed: Random seed (uses config default if None)
        verbose: Whether to print progress
    
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
    if steps_per_year is None:
        steps_per_year = CONFIG.get("mc_steps_per_year", 252)
    if vol_window is None:
        vol_window = CONFIG.get("vol_window", 20)
    if seed is None:
        seed = CONFIG.get("mc_seed", 42)
    
    rng = np.random.default_rng(seed)
    trajectories = []
    
    if verbose:
        print(f"\nGenerating {n_trajectories} Monte Carlo trajectories...")
        print(f"  Initial Price: ${S0:.2f}")
        print(f"  Drift (μ): {mu*100:.1f}%")
        print(f"  Volatility (σ): {sigma*100:.1f}%")
        print(f"  Steps/Year: {steps_per_year}")
    
    for i in range(n_trajectories):
        # Vary initial price slightly for each trajectory
        S0_varied = S0 * rng.uniform(0.9, 1.1)
        
        # Vary volatility slightly
        sigma_varied = sigma * rng.uniform(0.8, 1.2)
        
        # Generate trajectory starting from a random year
        start_year = 2010 + i % 10  # Cycle through years 2010-2019
        start_date = pd.Timestamp(f'{start_year}-01-01')
        
        df = generate_mc_hedging_trajectory(
            S0=S0_varied,
            mu=mu,
            sigma=sigma_varied,
            r=r,
            T=1.0,  # 1 year
            steps=steps_per_year,
            vol_window=vol_window,
            rng=rng,
            start_date=start_date
        )
        
        trajectories.append(df)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_trajectories} trajectories")
    
    if verbose:
        print(f"  ✓ Generated {n_trajectories} trajectories ({steps_per_year} steps each)")
    
    return trajectories


def generate_train_val_data(
    train_trajectories=None,
    seed=None,
    verbose=True
):
    """
    Generate training dataset using Monte Carlo simulation.
    
    Args:
        train_trajectories: Number of training trajectories (uses config if None)
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        dict: {'train': list of DataFrames}
    """
    if train_trajectories is None:
        train_trajectories = CONFIG.get("mc_train_trajectories", 50)
    if seed is None:
        seed = CONFIG.get("mc_seed", 42)
    
    if verbose:
        print("\n" + "="*60)
        print("GENERATING MONTE CARLO DATA FOR TRAINING")
        print("="*60)
    
    # Generate training data
    train_data = generate_mc_training_data(
        n_trajectories=train_trajectories,
        seed=seed,
        verbose=verbose
    )
    
    if verbose:
        total_train_steps = sum(len(df) for df in train_data)
        print(f"\n  Training: {train_trajectories} trajectories ({total_train_steps:,} total steps)")
        print("="*60 + "\n")
    
    return {
        'train': train_data
    }


if __name__ == "__main__":
    # Test the generator
    print("Testing Monte Carlo Data Generator...")
    
    # Generate sample data
    data = generate_train_val_data(
        train_trajectories=5,
        verbose=True
    )
    
    print("\nSample training trajectory:")
    print(data['train'][0].head(10))
    print(f"\nShape: {data['train'][0].shape}")
