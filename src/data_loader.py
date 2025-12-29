"""
Data Loader for Hedging RL
Handles loading, splitting, and preparing data for training, validation, and testing

New Architecture (v2):
- Training/Validation: Monte Carlo simulated trajectories (configurable count)
- Testing: Real S&P 500 daily data (2004-2025)
"""

import os
import numpy as np
import pandas as pd
from config import CONFIG, get_data_config
from hedging_env import HedgingEnv
from mc_data_generator import generate_train_val_data


def load_hedging_data(data_path=None):
    """
    Load hedging data from CSV or parquet file
    
    Args:
        data_path: Path to data file. If None, uses config default.
    
    Returns:
        pd.DataFrame: Loaded data with timestamp column parsed
    """
    if data_path is None:
        data_path = CONFIG.get("data_path", "./data/historical_hedging_data.csv")
    
    # Determine file type
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    # Ensure datetime column exists
    datetime_col = CONFIG.get("datetime_column", "timestamp")
    if datetime_col not in df.columns and 'timestamp' in df.columns:
        df[datetime_col] = df['timestamp']
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Sort by timestamp
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    
    print(f"Loaded data: {len(df)} rows")
    print(f"Date range: {df[datetime_col].min()} to {df[datetime_col].max()}")
    
    return df


def split_data_by_years(df, train_years, validation_year, test_year, datetime_col=None):
    """
    Split data by years for training, validation, and testing
    
    Args:
        df: DataFrame with hedging data
        train_years: List of years for training (e.g., [2005, 2006, 2007, 2008, 2009, 2010])
        validation_year: Year for validation (e.g., 2011)
        test_year: Year for testing (e.g., 2012)
        datetime_col: Name of datetime column
    
    Returns:
        dict: Dictionary with 'train', 'validation', 'test' DataFrames
    """
    if datetime_col is None:
        datetime_col = CONFIG.get("datetime_column", "timestamp")
    
    # Ensure datetime column
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract year
    df['_year'] = df[datetime_col].dt.year
    
    # Split data
    train_df = df[df['_year'].isin(train_years)].copy()
    val_df = df[df['_year'] == validation_year].copy()
    test_df = df[df['_year'] == test_year].copy()
    
    # Remove helper column
    for d in [train_df, val_df, test_df]:
        if '_year' in d.columns:
            d.drop('_year', axis=1, inplace=True)
    
    df.drop('_year', axis=1, inplace=True)
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\nData splits:")
    print(f"  Training ({train_years}): {len(train_df)} rows")
    print(f"  Validation ({validation_year}): {len(val_df)} rows")
    print(f"  Test ({test_year}): {len(test_df)} rows")
    
    if len(train_df) == 0:
        raise ValueError(f"No training data found for years {train_years}")
    if len(val_df) == 0:
        print(f"Warning: No validation data found for year {validation_year}")
    if len(test_df) == 0:
        print(f"Warning: No test data found for year {test_year}")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }


def prepare_env_data(df, datetime_col=None):
    """
    Prepare data arrays for HedgingEnv
    
    Args:
        df: DataFrame with hedging data
        datetime_col: Name of datetime column
    
    Returns:
        dict: Dictionary with arrays for environment initialization
    """
    if datetime_col is None:
        datetime_col = CONFIG.get("datetime_column", "timestamp")
    
    data_config = get_data_config()
    
    # Map columns - support both historical_hedging_data.csv and MC-generated data
    price_col = data_config.get("price_column", "option_price")
    stock_col = data_config.get("stock_price_column", "underlying_price")
    moneyness_col = data_config.get("moneyness_column", "moneyness")
    ttm_col = data_config.get("ttm_column", "time_to_maturity")
    
    # Check if columns exist, fallback to alternative names (for MC data)
    if price_col not in df.columns and 'option_price' in df.columns:
        price_col = 'option_price'
    if stock_col not in df.columns and 'underlying_price' in df.columns:
        stock_col = 'underlying_price'
    if ttm_col not in df.columns and 'time_to_maturity' in df.columns:
        ttm_col = 'time_to_maturity'
    if datetime_col not in df.columns and 'timestamp' in df.columns:
        datetime_col = 'timestamp'
    
    # Extract arrays
    env_data = {
        'option_prices': df[price_col].values,
        'stock_prices': df[stock_col].values,
        'moneyness': df[moneyness_col].values,
        'ttm': df[ttm_col].values,
        'timestamps': df[datetime_col].values
    }
    
    return env_data


def create_environment(df, normalization_stats=None, normalize=None, verbose=None):
    """
    Create HedgingEnv from DataFrame
    
    Args:
        df: DataFrame with hedging data
        normalization_stats: Pre-computed normalization statistics (for test/val consistency)
        normalize: Whether to normalize data
        verbose: Whether to print verbose output
    
    Returns:
        HedgingEnv: Initialized environment
    """
    if normalize is None:
        normalize = CONFIG.get("normalize_data", True)
    if verbose is None:
        verbose = CONFIG.get("verbose_evaluation", True)
    
    env_data = prepare_env_data(df)
    
    env = HedgingEnv(
        option_prices=env_data['option_prices'],
        stock_prices=env_data['stock_prices'],
        moneyness=env_data['moneyness'],
        ttm=env_data['ttm'],
        timestamps=env_data['timestamps'],
        normalize=normalize,
        normalization_stats=normalization_stats,
        verbose=verbose
    )
    
    return env


def create_environments_for_training(
    data_path=None,
    train_years=None,
    validation_year=None,
    test_year=None,
    normalize=None,
    verbose=True
):
    """
    Create train and test environments.
    
    NEW BEHAVIOR (when use_montecarlo_training=True in config):
    - Training: Use Monte Carlo simulated trajectories (single pass, no validation)
    - Testing: Use real S&P 500 daily data
    
    LEGACY BEHAVIOR (when use_montecarlo_training=False):
    - All splits from historical CSV file
    
    Args:
        data_path: Path to data file (only used for legacy mode or testing)
        train_years: List of training years (legacy mode only)
        validation_year: Validation year (legacy mode only, ignored if provided)
        test_year: Test year (ignored in MC mode - uses full real data for test)
        normalize: Whether to normalize data
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary with 'train_envs', 'test_env', and metadata
    """
    # Get config values
    use_mc = CONFIG.get("use_montecarlo_training", True)
    
    if normalize is None:
        normalize = CONFIG.get("normalize_data", True)
    
    if use_mc:
        return _create_mc_environments(normalize=normalize, verbose=verbose)
    else:
        return _create_legacy_environments(
            data_path=data_path,
            train_years=train_years,
            validation_year=validation_year,
            test_year=test_year,
            normalize=normalize,
            verbose=verbose
        )


def _create_mc_environments(normalize=True, verbose=True):
    """
    Create environments using Monte Carlo data for training
    and real S&P 500 data for testing.
    
    Returns:
        dict with train_envs, test_env
    """
    if verbose:
        print("\n" + "="*60)
        print("CREATING ENVIRONMENTS (Monte Carlo Mode)")
        print("="*60)
    
    # 1. Generate Monte Carlo trajectories for training
    mc_data = generate_train_val_data(verbose=verbose)
    
    # 2. Create environments from MC trajectories
    # First trajectory for normalization stats
    first_env = create_environment(
        mc_data['train'][0], normalize=normalize, verbose=False
    )
    norm_stats = first_env.get_normalization_stats() if normalize else None
    
    # Create all training environments
    train_envs = []
    for i, df in enumerate(mc_data['train']):
        env = create_environment(
            df, 
            normalization_stats=norm_stats if i > 0 else None,
            normalize=normalize, 
            verbose=False
        )
        # Update norm_stats from first environment
        if i == 0 and normalize:
            norm_stats = env.get_normalization_stats()
        train_envs.append(env)
    
    if verbose:
        print(f"\n  ✓ Created {len(train_envs)} training environments")
    
    # 3. Load real S&P 500 data for testing
    test_env = _create_test_env_from_real_data(norm_stats, normalize, verbose)
    
    return {
        'train_envs': train_envs,
        'test_env': test_env,
        'normalization_stats': norm_stats,
        'mode': 'montecarlo'
    }


def _create_test_env_from_real_data(norm_stats=None, normalize=True, verbose=True):
    """
    Create test environment from real S&P 500 daily data.
    Uses generate_contract.py to convert raw prices to hedging data.
    """
    from generate_contract import generate_historical_hedging_data
    
    sp500_path = CONFIG.get("sp500_data_path", "./data/sp500_data.csv")
    test_start = CONFIG.get("test_start_year", 2004)
    test_end = CONFIG.get("test_end_year", 2025)
    
    # Handle relative paths - check both from current dir and from parent
    if not os.path.isabs(sp500_path):
        if not os.path.exists(sp500_path):
            # Try from parent directory (when running from src/)
            alt_path = os.path.join(os.path.dirname(__file__), '..', sp500_path.lstrip('./'))
            if os.path.exists(alt_path):
                sp500_path = alt_path
    
    if verbose:
        print(f"\n  Loading real S&P 500 data for testing...")
        print(f"    Path: {sp500_path}")
        print(f"    Years: {test_start} - {test_end}")
    
    # Check if raw data exists
    if not os.path.exists(sp500_path):
        print(f"  ⚠ Warning: S&P 500 data not found at {sp500_path}")
        print("    Run download_data.py or provide S&P 500 CSV file")
        return None
    
    # Generate hedging data from S&P 500 prices
    hedging_df = generate_historical_hedging_data(
        csv_path=sp500_path,
        r=CONFIG.get("risk_free_rate", 0.02),
        vol_window=CONFIG.get("vol_window", 20)
    )
    
    # Filter by test years - use 'timestamp' which is the column from generate_contract
    datetime_col = 'timestamp'  # generate_contract.py uses this
    hedging_df['_year'] = hedging_df[datetime_col].dt.year
    test_df = hedging_df[
        (hedging_df['_year'] >= test_start) & 
        (hedging_df['_year'] <= test_end)
    ].copy()
    test_df.drop('_year', axis=1, inplace=True)
    test_df = test_df.reset_index(drop=True)
    
    if len(test_df) == 0:
        print(f"  ⚠ Warning: No test data found for years {test_start}-{test_end}")
        return None
    
    if verbose:
        print(f"    ✓ Test data: {len(test_df)} daily observations")
        print(f"    Date range: {test_df[datetime_col].min().date()} to {test_df[datetime_col].max().date()}")
    
    # Create environment
    test_env = create_environment(
        test_df,
        normalization_stats=norm_stats,
        normalize=normalize,
        verbose=False
    )
    
    return test_env


def _create_legacy_environments(
    data_path=None,
    train_years=None,
    validation_year=None,
    test_year=None,
    normalize=True,
    verbose=True
):
    """
    Legacy mode: Create environments from a single historical CSV file.
    """
    # Get default years from config if not provided
    if train_years is None:
        train_years = CONFIG.get("train_years", [2005, 2006, 2007, 2008, 2009, 2010])
    if validation_year is None:
        validation_year = CONFIG.get("validation_year", 2011)
    if test_year is None:
        test_year = CONFIG.get("test_year", 2012)
    
    # Load and split data
    df = load_hedging_data(data_path)
    data_splits = split_data_by_years(df, train_years, validation_year, test_year)
    
    # Create training environment first to get normalization stats
    if verbose:
        print("\nCreating training environment (legacy mode)...")
    train_env = create_environment(
        data_splits['train'],
        normalize=normalize,
        verbose=verbose
    )
    
    # Get normalization stats from training data
    norm_stats = train_env.get_normalization_stats()
    
    # Create test environment with same normalization
    test_env = None
    if len(data_splits['test']) > 0:
        if verbose:
            print("\nCreating test environment...")
        test_env = create_environment(
            data_splits['test'],
            normalization_stats=norm_stats,
            normalize=normalize,
            verbose=verbose
        )
    
    return {
        'train_envs': [train_env],  # Wrap in list for compatibility
        'test_env': test_env,
        'data_splits': data_splits,
        'normalization_stats': norm_stats,
        'mode': 'legacy'
    }


def get_year_ranges_from_data(df, datetime_col=None):
    """
    Get available year ranges from data
    
    Args:
        df: DataFrame with datetime column
        datetime_col: Name of datetime column
    
    Returns:
        list: Sorted list of unique years
    """
    if datetime_col is None:
        datetime_col = CONFIG.get("datetime_column", "timestamp")
    
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    years = sorted(df[datetime_col].dt.year.unique().tolist())
    return years


def create_train_environments_list(
    data_path=None,
    train_years=None,
    normalize=None,
    verbose=False
):
    """
    Create a list of environments, one per training year
    Useful for episodic training where each episode is one year
    
    Args:
        data_path: Path to data file
        train_years: List of training years
        normalize: Whether to normalize
        verbose: Whether to print verbose output
    
    Returns:
        list: List of (year, environment) tuples
    """
    if train_years is None:
        train_years = CONFIG.get("train_years", [2005, 2006, 2007, 2008, 2009, 2010])
    if normalize is None:
        normalize = CONFIG.get("normalize_data", True)
    
    # Load all data
    df = load_hedging_data(data_path)
    datetime_col = CONFIG.get("datetime_column", "timestamp")
    
    # First pass: compute normalization stats from all training data
    if normalize:
        all_train_df = df[df[datetime_col].dt.year.isin(train_years)].copy()
        temp_env = create_environment(all_train_df, normalize=True, verbose=False)
        norm_stats = temp_env.get_normalization_stats()
    else:
        norm_stats = None
    
    # Create environment for each year
    envs = []
    for year in train_years:
        year_df = df[df[datetime_col].dt.year == year].copy().reset_index(drop=True)
        if len(year_df) > 0:
            env = create_environment(
                year_df,
                normalization_stats=norm_stats,
                normalize=normalize,
                verbose=verbose
            )
            envs.append((year, env))
            print(f"Created environment for year {year}: {len(year_df)} steps")
    
    return envs, norm_stats


if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    print("=" * 60)
    
    use_mc = CONFIG.get("use_montecarlo_training", True)
    
    if use_mc:
        print("\nMode: MONTE CARLO TRAINING")
        print("="*60)
        
        # Test MC environment creation
        envs = create_environments_for_training(verbose=True)
        
        print(f"\n" + "="*60)
        print("ENVIRONMENTS CREATED SUCCESSFULLY")
        print("="*60)
        print(f"Mode: {envs['mode']}")
        print(f"Training environments: {len(envs['train_envs'])}")
        print(f"Validation environments: {len(envs['val_envs'])}")
        print(f"Test environment: {'Yes' if envs['test_env'] else 'No'}")
        
        # Sample environment info
        if envs['train_envs']:
            sample_env = envs['train_envs'][0]
            print(f"\nSample training env:")
            print(f"  Observation space: {sample_env.observation_space.shape}")
            print(f"  Action space: {sample_env.action_space.shape}")
            print(f"  Episode length: {len(sample_env.option_prices)} steps")
        
        if envs['test_env']:
            print(f"\nTest env:")
            print(f"  Episode length: {len(envs['test_env'].option_prices)} steps")
    else:
        print("\nMode: LEGACY (Historical Data)")
        
        # Test with default path
        data_path = "./data/historical_hedging_data.csv"
        
        if os.path.exists(data_path):
            df = load_hedging_data(data_path)
            years = get_year_ranges_from_data(df)
            print(f"\nAvailable years: {years}")
            
            # Test split
            train_years = [2005, 2006, 2007, 2008, 2009, 2010]
            val_year = 2011
            test_year = 2012
            
            # Check if years exist
            available_train = [y for y in train_years if y in years]
            print(f"\nAvailable training years: {available_train}")
            
            if available_train:
                envs = create_environments_for_training(
                    data_path=data_path,
                    train_years=available_train,
                    validation_year=val_year,
                    test_year=test_year,
                    verbose=True
                )
                
                print(f"\nEnvironments created successfully!")
                print(f"Training env state space: {envs['train_envs'][0].observation_space.shape}")
                print(f"Training env action space: {envs['train_envs'][0].action_space.shape}")
        else:
            print(f"Data file not found: {data_path}")
