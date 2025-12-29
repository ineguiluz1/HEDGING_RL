import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- 1. Modelo de Pricing (Black-Scholes Vectorizado) ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calcula el precio teórico de la opción basándose en el precio REAL del S&P500.
    """
    # Protección contra NaNs o ceros en volatilidad o tiempos muy pequeños
    sigma = np.where(np.isnan(sigma) | (sigma <= 1e-6), 0.15, sigma)
    T = np.maximum(T, 1e-6) # Evitar división por cero en T
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# --- 2. Utilidades ---
def calculate_realized_volatility(log_returns, window=50, annualization_factor=None):
    """
    Calcula la volatilidad histórica real del S&P 500.
    Metodología Sección 4.2.3[cite: 384, 386].
    """
    if annualization_factor is None:
        annualization_factor = np.sqrt(252) 
        
    vol = log_returns.rolling(window=window).std() * annualization_factor
    return vol

def apply_price_capping(prices, threshold=0.20):
    """
    Limita movimientos extremos (Sección 4.3)[cite: 397].
    """
    capped_prices = prices.copy()
    # Usamos un bucle porque la restricción es secuencial (path-dependent)
    for i in range(1, len(capped_prices)):
        prev = capped_prices[i-1]
        curr = capped_prices[i]
        upper = prev * (1 + threshold)
        lower = prev * (1 - threshold)
        
        if curr > upper:
            capped_prices[i] = upper
        elif curr < lower:
            capped_prices[i] = lower
            
    return capped_prices

def load_sp500_data(csv_path):
    """
    Carga y limpia el CSV del S&P 500 con el formato específico de 3 filas de header.
    """
    try:
        # Saltamos las primeras 3 filas que contienen metadatos extraños
        # Asignamos nombres manualmente según el formato que mostraste
        df = pd.read_csv(
            csv_path,
            skiprows=3,
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
            parse_dates=['Date'],
            index_col='Date'
        )
        
        # Ordenar por fecha
        df.sort_index(inplace=True)
        
        # Filtrar solo el rango relevante mencionado en el paper (2004-2024) 
        # Si no tienes datos hasta 2024, tomará lo que haya disponible.
        df = df.loc['2004-01-01':'2025-12-01']
        
        if df.empty:
            raise ValueError("El CSV no contiene datos en el rango 2004-2024 o está vacío.")

        return df['Close']
    
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return None

# --- 3. Generación basada en Histórico ---
def generate_historical_hedging_data(
    csv_path,
    r=0.02,
    vol_window=20, # Ventana de volatilidad (aprox 1 mes en datos diarios)
):
    """
    Genera el dataset usando el S&P 500 histórico como referencia.
    DATOS DIARIOS SIN INTERPOLACIÓN para movimientos más realistas.
    """
    
    print(f"Cargando datos históricos desde {csv_path}...")
    S_series = load_sp500_data(csv_path)
    
    if S_series is None:
        return None

    # Usar datos diarios sin interpolación
    print(f"Usando {len(S_series)} datos diarios (sin interpolación).")
    annualization_factor = np.sqrt(252)  # Factor para volatilidad anualizada

    # Convertir a numpy
    S = S_series.values
    dates = S_series.index
    n_steps = len(S)
    
    print(f"Generando contratos para {n_steps} pasos de tiempo...")

    # --- Construcción de Variables ---
    
    # 1. Volatilidad Realizada (sigma)
    log_returns = pd.Series(np.log(S / np.roll(S, 1)))
    log_returns.iloc[0] = 0
    
    realized_vol = calculate_realized_volatility(
        log_returns, 
        window=vol_window, 
        annualization_factor=annualization_factor
    ).values
    
    # Rellenar NaNs iniciales (Paper sugiere valores aleatorios [0.02, 0.10]) [cite: 390]
    mask_nan = np.isnan(realized_vol)
    realized_vol[mask_nan] = np.random.uniform(0.02, 0.10, size=mask_nan.sum())

    # 2. Strike (K) -> Dinámico para ser siempre ATM (Moneyness ~ 1) [cite: 395]
    # Simulamos pequeñas variaciones de mercado donde no encontramos el strike perfecto
    noise_moneyness = np.random.normal(0, 0.005, n_steps) # Ruido pequeño
    K = S * (1 + noise_moneyness)

    # 3. Time to Maturity (tau) -> Dinámico para ser siempre ~30 días [cite: 395]
    target_maturity_years = 30 / 365.0
    noise_maturity = np.random.normal(0, 0.001, n_steps)
    tau = np.clip(target_maturity_years + noise_maturity, 0.01, 0.2)

    # 4. Precio de la Opción (C) - Teórico basado en S real
    C = black_scholes(S, K, tau, r, realized_vol, option_type="call")
    
    # --- Estructurar DataFrame ---
    df = pd.DataFrame({
        'timestamp': dates,
        'underlying_price': S,
        'strike': K,
        'time_to_maturity': tau,
        'option_price': C,
        'risk_free_rate': r,
        'realized_volatility': realized_vol,
        'moneyness': S / K
    })
    
    # Limpieza final (Price Capping) [cite: 397]
    df['option_price'] = apply_price_capping(df['option_price'].values)
    
    return df

# --- Ejecución ---
if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(ROOT, "data", "sp500_data.csv")
    output_csv = os.path.join(ROOT, "data", "historical_hedging_data.csv")
    
    if os.path.exists(input_csv):
        # Generamos los datos diarios (sin interpolación)
        df_hist = generate_historical_hedging_data(
            input_csv, 
            vol_window=20,
        )
        
        if df_hist is not None:
            df_hist.to_csv(output_csv, index=False)
            print(f"\n¡Éxito! Dataset generado con {len(df_hist)} filas (datos diarios).")
            print(f"Guardado en: {output_csv}")
            print("Primeras filas:")
            print(df_hist[['timestamp', 'underlying_price', 'option_price', 'realized_volatility']].head())
    else:
        print(f"No se encontró el archivo: {input_csv}")