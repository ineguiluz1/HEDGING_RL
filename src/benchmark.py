import numpy as np
import pandas as pd
from scipy.stats import norm
from config import CONFIG, get_data_config, get_environment_config


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes delta for option hedging
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
    
    Returns:
        float: Delta value
    """
    # Handle edge cases more gracefully
    if T <= 1/365.25:  # Less than 1 day
        return 1.0 if S > K else 0.0
    if sigma <= 0 or np.isnan(sigma) or np.isnan(S) or np.isnan(K):
        return 1.0 if S > K else 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type.lower() == "call":
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    except:
        return 1.0 if S > K else 0.0


def calculate_delta_from_moneyness(moneyness, T, sigma, r=0.02):
    """
    Calculate delta when we have moneyness instead of S and K separately
    
    Args:
        moneyness: S/K ratio
        T: Time to maturity (in years)
        sigma: Volatility
        r: Risk-free rate
    
    Returns:
        float: Delta value
    """
    if T <= 1/365.25:
        return 1.0 if moneyness > 1 else 0.0
    if sigma <= 0 or np.isnan(sigma):
        return 1.0 if moneyness > 1 else 0.0
    try:
        d1 = (np.log(moneyness) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
    except:
        return 1.0 if moneyness > 1 else 0.0



def delta_hedging(
    df,
    risk_free_rate=None,
    dt=None,
    transaction_cost=None,
    window=None,
    start_date=None,
    end_date=None,
    notional=None
):
    # contract multiplyer for options SPX, SPXW is 100
    M=100

    # Use config defaults if parameters not provided
    env_config = get_environment_config()
    data_config = get_data_config()
    
    if risk_free_rate is None:
        risk_free_rate = env_config["risk_free_rate"]
    if dt is None:
        dt = 1 / CONFIG["annualization_factor"]
    if transaction_cost is None:
        transaction_cost = env_config["transaction_cost"]
    if window is None:
        window = CONFIG.get("benchmark_vol_window", CONFIG["vol_window"])
    if notional is None:
        notional = env_config["notional"]
    
    datetime_col = data_config["datetime_column"]
    price_col = data_config["stock_price_column"]
    option_price_col = data_config["price_column"]
    
    df = df.sort_values(by=datetime_col).copy()

    # Volatility calculation (same as before)
    if start_date is not None:
        vol_base = df[df[datetime_col].dt.year < start_date].copy()
    else:
        vol_base = df.copy()

    vol_base = vol_base.sort_values(by=datetime_col)
    vol_base[price_col] = vol_base[price_col].replace(0, np.nan).ffill()

    if len(vol_base) < window:
        raise ValueError(f"Not enough data to compute realized volatility before {start_date}. "
                        f"Need {window} observations, got {len(vol_base)}")

    from volatility_utils import calculate_realized_volatility, fill_initial_volatility
    initial_vol_series = calculate_realized_volatility(
        prices=vol_base[price_col], 
        window=window,
        annualization_factor=CONFIG["annualization_factor"]
    )
    
    initial_vol_filled = fill_initial_volatility(
        initial_vol_series, 
        initial_vol_value=CONFIG["initial_vol"]
    )
    
    if initial_vol_filled.isna().all():
        raise ValueError("Realized volatility calculation returned all NaN — check input data.")
    
    initial_vol = initial_vol_filled.dropna().iloc[-1]
    
    if CONFIG.get("verbose_evaluation", True):
        print(f"Initial volatility estimate: {initial_vol:.4f}")

    # Filter data for analysis period
    if start_date is not None:
        df = df[df[datetime_col].dt.year >= start_date]
    if end_date is not None:
        df = df[df[datetime_col].dt.year <= end_date]

    df.reset_index(drop=True, inplace=True)
    df[price_col] = df[price_col].replace(0, np.nan).ffill()
    print("\n[DEBUG] First few rows of input data for benchmark:")
    print(df[[datetime_col, price_col, option_price_col, "strike", "option_type"]].head())
    print("\n[DEBUG] Price/Option summary stats:")
    print(df[[price_col, option_price_col]].describe())

    if df[price_col].isna().any():
        raise ValueError(f"Zero or NaN found in {price_col}. Please clean the data.")

    # Calculate realized volatility for the analysis period
    df_realized_vol = calculate_realized_volatility(
        prices=df[price_col], 
        window=window,
        annualization_factor=CONFIG["annualization_factor"]
    )
    df_realized_vol_filled = fill_initial_volatility(
        df_realized_vol, 
        initial_vol_value=initial_vol
    )
    df["Realized Volatility"] = df_realized_vol_filled

    # Initialize tracking columns
    df["Delta"] = 0.0
    df["Stock Position"] = 0.0
    df["Transaction Cost"] = 0.0
    df["PnL"] = 0.0
    df["Reward"] = 0.0
    
    df["Option_Component"] = 0.0
    df["Hedge_Component"] = 0.0
    df["Transaction_Component"] = 0.0
    df["Cash_Account"] = 0.0

    cash_account = 0.0
    
    if CONFIG.get("verbose_evaluation", True):
        print(f"Starting delta hedging simulation...")
        print(f"Data points: {len(df)}")
        print(f"Risk-free rate: {risk_free_rate:.4f}")
        print(f"Transaction cost: {transaction_cost:.4f}")
        print(f"Notional: {notional}")

    skipped_steps = 0
    total_transaction_costs = 0
    
    for t in range(1, len(df) - 1):
        S = df[price_col].iloc[t]         
        K = df["strike"].iloc[t]              
        sigma = df["Realized Volatility"].iloc[t]  
        option_type = df["option_type"].iloc[t]    
        
        if "maturity" in df.columns:
            maturity_days = df["maturity"].iloc[t]
            T = maturity_days / 365.25 
        else:
            T = (len(df) - t) * dt
        
        T = max(T, 1/365.25)
        
        if (
            np.isnan(S) or np.isnan(K) or T <= 0
            or np.isnan(df[option_price_col].iloc[t]) 
            or np.isnan(df[option_price_col].iloc[t + 1])
        ):
            skipped_steps += 1
            continue

        # Calculate Black-Scholes delta
        delta = black_scholes_delta(S, K, T, risk_free_rate, sigma, option_type)
        df.at[t, "Delta"] = delta
        if t < 5:  # only print first 5 steps
            print(f"[t={t}] Date={df[datetime_col].iloc[t]} | "
                  f"S={S:.2f}, K={K:.2f}, opt_price={df[option_price_col].iloc[t]:.4f}, "
                  f"vol={sigma:.4f}, delta={delta:.4f}")
        
        # Clamp delta to reasonable range to avoid extreme positions
        delta = np.clip(delta, -2.0, 2.0)

        # Calculate required position (shares of stock)
        current_position_shares = delta * notional / S  
        previous_position_shares = df["Stock Position"].iloc[t - 1] / df[price_col].iloc[t - 1] if t > 1 and df[price_col].iloc[t - 1] > 0 else 0
        
        # Position adjustment in shares
        hedge_adjustment_shares = current_position_shares - previous_position_shares
        
        # Calculate transaction costs
        trade_cost = abs(hedge_adjustment_shares) * S * transaction_cost
        total_transaction_costs += trade_cost

        # Update tracking variables
        df.at[t, "Transaction Cost"] = trade_cost
        df.at[t, "Stock Position"] = current_position_shares * S  # Dollar value of position
        df.at[t, "Cash_Account"] = cash_account
        
        # Calculate time difference for interest
        if t > 1:
            prev_timestamp = df[datetime_col].iloc[t - 1]
            current_timestamp = df[datetime_col].iloc[t]
            
            # Handle different timestamp types
            if hasattr(current_timestamp, 'total_seconds'):
                time_diff = current_timestamp - prev_timestamp
                time_diff_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
            else:
                current_ts = pd.Timestamp(current_timestamp)
                prev_ts = pd.Timestamp(prev_timestamp)
                time_diff = current_ts - prev_ts
                time_diff_years = time_diff.total_seconds() / (365.25 * 24 * 3600)
                
            interest_factor = (1 + risk_free_rate) ** time_diff_years
        else:
            interest_factor = 1.0
        

        cash_account = cash_account * interest_factor

        cash_account = cash_account - hedge_adjustment_shares * S - trade_cost


        option_component = -1 * (df[option_price_col].iloc[t + 1] - df[option_price_col].iloc[t]) / notional

        previous_position_value = df["Stock Position"].iloc[t - 1] if t > 1 else 0
        hedge_component = (previous_position_value / notional) * (df[price_col].iloc[t + 1] / S - 1)
  
        transaction_component = trade_cost / notional
   
        step_pnl = option_component + hedge_component - transaction_component
        # print(f"    -> step_pnl={step_pnl:.6f} "
        #           f"(option={option_component:.6f}, hedge={hedge_component:.6f}, "
        #           f"tx={transaction_component:.6f})")
        # Store components for debugging
        df.at[t + 1, "Option_Component"] = option_component
        df.at[t + 1, "Hedge_Component"] = hedge_component
        df.at[t + 1, "Transaction_Component"] = transaction_component
        
        # Risk-adjusted reward (same as DRL environment)
        xi = CONFIG.get("risk_aversion", 0.01)
        baseline_reward = 0.1 
        
        reward = baseline_reward + step_pnl - xi * abs(step_pnl)
        if reward < -10: 
            reward = -10 + 0.1 * (reward + 10)
        
        # Store both step P&L and reward
        df.at[t + 1, "PnL"] = step_pnl
        df.at[t + 1, "Reward"] = reward

    if CONFIG.get("verbose_evaluation", True):
        if skipped_steps > 0:
            print(f"Warning: Skipped {skipped_steps} steps due to missing data")
        print(f"Total transaction costs: {total_transaction_costs:.2f}")
        print(f"Average transaction cost per step: {total_transaction_costs/(len(df)-skipped_steps-1):.4f}")
        

        final_pnl = df["PnL"].sum()
        print(f"Final cumulative P&L: {final_pnl:.4f}")
        print(f"Final cumulative P&L ($): ${final_pnl * notional:,.2f}")
        total_option = df["Option_Component"].sum()
        total_hedge = df["Hedge_Component"].sum()
        total_transaction = df["Transaction_Component"].sum()
        
        print(f"P&L Breakdown:")
        print(f"  Option component: {total_option:.4f}")
        print(f"  Hedge component: {total_hedge:.4f}")
        print(f"  Transaction component: {total_transaction:.4f}")
        print(f"  Total: {total_option + total_hedge - total_transaction:.4f}")

    return df


def run_benchmark(
    df,
    start_year=None,
    end_year=None,
    window=None,
    notional=None,
    risk_free_rate=None,
    transaction_cost=None
):

    env_config = get_environment_config()
    data_config = get_data_config()
    
    if start_year is None:
        start_year = CONFIG["test_year"]
    if end_year is None:
        end_year = CONFIG["test_year"]
    if window is None:
        window = CONFIG.get("benchmark_vol_window", CONFIG["vol_window"])
    if notional is None:
        notional = env_config["notional"]
    if risk_free_rate is None:
        risk_free_rate = env_config["risk_free_rate"]
    if transaction_cost is None:
        transaction_cost = env_config["transaction_cost"]
    
    if CONFIG.get("verbose_evaluation", True):
        print(f"\n{'='*50}")
        print(f"RUNNING DELTA HEDGING BENCHMARK")
        print(f"{'='*50}")
        print(f"Configuration:")
        print(f"  Period: {start_year}-{end_year}")
        print(f"  Volatility window: {window} days")
        print(f"  Notional: ${notional:,}")
        print(f"  Risk-free rate: {risk_free_rate:.2%}")
        print(f"  Transaction cost: {transaction_cost:.3%}")
        print(f"{'='*50}")
    
    benchmark_df = delta_hedging(
        df,
        risk_free_rate=risk_free_rate,
        transaction_cost=transaction_cost,
        window=window,
        start_date=start_year,
        end_date=end_year,
        notional=notional
    )

    # P&L for comparison
    benchmark_df["Returns"] = benchmark_df["PnL"].fillna(0)
    benchmark_df["Cumulative PnL"] = benchmark_df["PnL"].cumsum()
    
    if "Reward" in benchmark_df.columns:
        benchmark_df["Reward Returns"] = benchmark_df["Reward"].fillna(0) 
        benchmark_df["Reward Cumulative PnL"] = benchmark_df["Reward"].cumsum()
    
    if CONFIG.get("verbose_evaluation", True):
        final_step_pnl = benchmark_df["Cumulative PnL"].iloc[-1]
        
        print(f"\nBenchmark Results Summary:")
        print(f"  Final Step P&L (Raw): {final_step_pnl:.4f}")
        print(f"  Final Step P&L (Raw) $: ${final_step_pnl * notional:,.2f}")
        
        if "Reward Cumulative PnL" in benchmark_df.columns:
            final_reward = benchmark_df["Reward Cumulative PnL"].iloc[-1]
            print(f"  Final Reward P&L: {final_reward:.4f} (for reference)")
            
        print(f"  Number of trading periods: {len(benchmark_df)}")
        
        # Additional validation
        sharpe_approx = benchmark_df["Returns"].mean() / benchmark_df["Returns"].std() * np.sqrt(252) if benchmark_df["Returns"].std() > 0 else 0
        print(f"  Approximate Sharpe Ratio: {sharpe_approx:.3f}")

    return benchmark_df


def delta_hedging_simple(
    df,
    risk_free_rate=None,
    transaction_cost=None,
    notional=None,
    verbose=True
):
    """
    Simplified delta hedging for historical_hedging_data.csv format
    Uses moneyness and realized_volatility directly from data
    
    Args:
        df: DataFrame with columns: timestamp, underlying_price, option_price, 
            moneyness, time_to_maturity, realized_volatility
        risk_free_rate: Risk-free rate
        transaction_cost: Transaction cost fraction
        notional: Notional amount
        verbose: Whether to print progress
    
    Returns:
        DataFrame with delta hedging results
    """
    env_config = get_environment_config()
    
    if risk_free_rate is None:
        risk_free_rate = env_config["risk_free_rate"]
    if transaction_cost is None:
        transaction_cost = env_config["transaction_cost"]
    if notional is None:
        notional = env_config["notional"]
    
    df = df.copy().reset_index(drop=True)
    
    # Ensure required columns exist
    required_cols = ['timestamp', 'underlying_price', 'option_price', 'moneyness', 'time_to_maturity']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Use realized_volatility from data if available, otherwise calculate
    if 'realized_volatility' in df.columns:
        df['sigma'] = df['realized_volatility']
    else:
        from volatility_utils import calculate_realized_volatility, fill_initial_volatility
        vol_series = calculate_realized_volatility(
            prices=df['underlying_price'],
            window=CONFIG.get("benchmark_vol_window", 20),
            annualization_factor=CONFIG["annualization_factor"]
        )
        df['sigma'] = fill_initial_volatility(vol_series, CONFIG["initial_vol"])
    
    # Initialize result columns
    df["Delta"] = 0.0
    df["Stock_Position"] = 0.0
    df["Transaction_Cost"] = 0.0
    df["PnL"] = 0.0
    df["Reward"] = 0.0
    df["Option_Component"] = 0.0
    df["Hedge_Component"] = 0.0
    df["Transaction_Component"] = 0.0
    
    total_transaction_costs = 0
    previous_position_shares = 0
    
    if verbose:
        print(f"\nRunning Delta Hedging (simple format)...")
        print(f"Data points: {len(df)}")
    
    for t in range(1, len(df) - 1):
        S = df['underlying_price'].iloc[t]
        moneyness = df['moneyness'].iloc[t]
        T = df['time_to_maturity'].iloc[t]  # Already in years
        sigma = df['sigma'].iloc[t]
        
        # Skip if bad data
        if np.isnan(S) or np.isnan(moneyness) or T <= 0 or np.isnan(sigma):
            continue
        
        # Calculate delta from moneyness
        delta = calculate_delta_from_moneyness(moneyness, T, sigma, risk_free_rate)
        delta = np.clip(delta, 0, 1)  # Ensure valid range
        
        df.at[t, "Delta"] = delta
        
        # Calculate position in shares
        current_position_shares = delta * notional / S
        
        # Position adjustment
        hedge_adjustment_shares = current_position_shares - previous_position_shares
        
        # Transaction costs
        trade_cost = abs(hedge_adjustment_shares) * S * transaction_cost
        total_transaction_costs += trade_cost
        
        df.at[t, "Transaction_Cost"] = trade_cost
        df.at[t, "Stock_Position"] = current_position_shares * S
        
        # Calculate P&L components (for next timestep)
        if t + 1 < len(df):
            O_now = df['option_price'].iloc[t]
            O_next = df['option_price'].iloc[t + 1]
            S_now = df['underlying_price'].iloc[t]
            S_next = df['underlying_price'].iloc[t + 1]
            
            # Option component: -1 * (O_t+1 - O_t) / notional
            option_component = -1 * (O_next - O_now) / notional
            
            # Hedge component
            prev_position_value = df["Stock_Position"].iloc[t - 1] if t > 1 else 0
            hedge_component = (prev_position_value / notional) * (S_next / S_now - 1) if S_now > 0 else 0
            
            # Transaction component
            transaction_component = trade_cost / notional
            
            # Step P&L
            step_pnl = option_component + hedge_component - transaction_component
            
            # Risk-adjusted reward
            xi = CONFIG.get("risk_aversion", 0.01)
            reward = step_pnl - xi * abs(step_pnl)
            
            df.at[t + 1, "Option_Component"] = option_component
            df.at[t + 1, "Hedge_Component"] = hedge_component
            df.at[t + 1, "Transaction_Component"] = transaction_component
            df.at[t + 1, "PnL"] = step_pnl
            df.at[t + 1, "Reward"] = reward
        
        previous_position_shares = current_position_shares
    
    # Add cumulative columns
    df["Cumulative PnL"] = df["PnL"].cumsum()
    df["Cumulative Reward"] = df["Reward"].cumsum()
    
    if verbose:
        final_pnl = df["Cumulative PnL"].iloc[-1]
        final_reward = df["Cumulative Reward"].iloc[-1]
        sharpe = df["PnL"].mean() / (df["PnL"].std() + 1e-8) * np.sqrt(252)
        
        print(f"Total transaction costs: {total_transaction_costs:.2f}")
        print(f"Final Cumulative P&L: {final_pnl:.4f}")
        print(f"Final Cumulative Reward: {final_reward:.4f}")
        print(f"Sharpe Ratio: {sharpe:.4f}")
    
    return df


def run_benchmark_simple(
    df,
    year=None,
    notional=None,
    risk_free_rate=None,
    transaction_cost=None,
    verbose=True
):
    """
    Run delta hedging benchmark for a specific year using simple format
    
    Args:
        df: Full DataFrame with all years
        year: Year to run benchmark on (defaults to test_year from config)
        notional: Notional amount
        risk_free_rate: Risk-free rate
        transaction_cost: Transaction cost
        verbose: Whether to print results
    
    Returns:
        DataFrame with benchmark results
    """
    if year is None:
        year = CONFIG.get("test_year", 2012)
    
    # Filter to year
    datetime_col = 'timestamp'
    if datetime_col not in df.columns:
        datetime_col = CONFIG.get("datetime_column", "timestamp")
    
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    year_df = df[df[datetime_col].dt.year == year].copy()
    
    if len(year_df) == 0:
        raise ValueError(f"No data found for year {year}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"DELTA HEDGING BENCHMARK - Year {year}")
        print(f"{'='*60}")
    
    result_df = delta_hedging_simple(
        year_df,
        risk_free_rate=risk_free_rate,
        transaction_cost=transaction_cost,
        notional=notional,
        verbose=verbose
    )
    
    return result_df
