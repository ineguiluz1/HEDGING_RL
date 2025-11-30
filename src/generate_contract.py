import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import datetime
import random

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S: spot price
    K: strike
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility
    """
    if T == 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def generate_option_contracts(prices, r = 0.02, vol_window = 30, min_days = 10, max_days = 60):
    
    # 1. Seleccionar fecha inicial aleatoria
    start_idx = random.randint(0, len(prices) - max_days - 1)
    start_date = prices.index[start_idx]
    
    # 2. Spot price
    S0 = prices.iloc[start_idx]
    
    # 3. Duración aleatoria
    T_days = random.randint(min_days, max_days)
    T = T_days / 252  # en años
    
    # 4. Volatilidad histórica local
    window_start = max(0, start_idx - vol_window)
    hist_returns = np.log(prices.iloc[window_start:start_idx].pct_change().dropna())
    sigma = hist_returns.std() * np.sqrt(252)
    
    # Si hay muy pocos datos al inicio
    if np.isnan(sigma) or sigma == 0:
        sigma = 0.15  # fallback
    
    # 5. Strike aleatorio alrededor de S0
    # Moneyness entre 0.9 y 1.1
    K = S0 * np.random.uniform(0.9, 1.1)
    
    # 6. Tipo de opción
    option_type = random.choice(["call", "put"])
    
    # 7. Precio BS
    option_price = black_scholes(S0, K, T, r, sigma, option_type)
    
    contract = {
        "start_date": start_date,
        "S0": S0,
        "strike": K,
        "maturity_days": T_days,
        "volatility": sigma,
        "r": r,
        "type": option_type,
        "BS_price": option_price
    }
    
    return contract

def contracts_to_csv(contracts, filename):
    df = pd.DataFrame(contracts)
    df.to_csv(filename, index=False)
    print(f"Contracts saved to {filename}")
    
    return df

contracts = []

