import numpy as np
import pandas as pd
from config import CONFIG

def calculate_realized_volatility(prices, window, min_periods=None, annualization_factor=None):
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    

    assert (prices > 0).all(), "All prices must be positive"
    assert len(prices) >= 2, "Need at least 2 price observations"
    assert window >= 2, "Window must be at least 2"

    if min_periods is None:
        min_periods = 2
    
    if annualization_factor is None:
        annualization_factor = CONFIG["annualization_factor"]

    log_returns = np.log(prices / prices.shift(1))
    
    realized_vol = log_returns.rolling(
        window=window, 
        min_periods=min_periods
    ).std() * np.sqrt(annualization_factor)
    
    return realized_vol

def fill_initial_volatility(realized_vol, initial_vol_value=None):
    if initial_vol_value is None:
        initial_vol_value = CONFIG["initial_vol"]
    
    filled_vol = realized_vol.copy()
    
    if pd.isna(filled_vol.iloc[0]) and CONFIG.get("use_random_initial_vol", True):
        # Use seeded RNG for reproducibility
        seed = CONFIG.get("seed", 101)
        rng = np.random.default_rng(seed)
        
        vol_min = CONFIG.get("initial_vol_min", 0.05)
        vol_max = CONFIG.get("initial_vol_max", 0.40)
        random_initial_vol = rng.uniform(vol_min, vol_max)
        filled_vol.iloc[0] = random_initial_vol
        
        # Removed verbose print to reduce log spam
    
    filled_vol = filled_vol.fillna(initial_vol_value)
    
    return filled_vol

def calculate_volatility_normalization_stats(df, vol_window=None, initial_vol=None):
    if vol_window is None:
        vol_window = CONFIG["vol_window"]
    if initial_vol is None:
        initial_vol = CONFIG["initial_vol"]
    
    from config import get_data_config
    data_config = get_data_config()
    stock_price_col = data_config["stock_price_column"]
    
    train_vol = calculate_realized_volatility(
        prices=df[stock_price_col],
        window=vol_window,
        min_periods=1,
        annualization_factor=CONFIG["annualization_factor"]
    )
    
    train_vol_filled = fill_initial_volatility(train_vol, initial_vol)
    
    vol_stats = {
        "vol_mean": train_vol_filled.mean(),
        "vol_std": train_vol_filled.std()
    }
    
    if CONFIG.get("verbose_evaluation", False):
        print(f"Volatility normalization stats:")
        print(f"  Mean: {vol_stats['vol_mean']:.4f}")
        print(f"  Std: {vol_stats['vol_std']:.4f}")
        print(f"  Window: {vol_window}")
        print(f"  Initial vol: {initial_vol}")
    
    return vol_stats