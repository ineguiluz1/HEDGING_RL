import pandas as pd
import numpy as np
import os
import typing


def monte_carlo_paths(S0, mu, sigma, T=5, steps=252, simulations=1000, rng=None):
    """Generate full Monte Carlo price paths using geometric Brownian motion.

    Parameters
    ----------
    S0 : float
        Initial price
    mu : float
        Annual drift (mean return)
    sigma : float
        Annual volatility
    T : int or float
        Time horizon in YEARS (default 5 years)
    steps : int
        Total number of time steps over the horizon T
        For daily trading days, use steps = 252 * T
    simulations : int
        Number of simulation paths to generate
    rng : np.random.Generator, optional
        Random number generator (default: creates a new one)

    Returns
    -------
    np.ndarray
        Array of shape (simulations, steps+1) including the initial price S0.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / steps
    paths = np.zeros((simulations, steps + 1), dtype=float)
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        Z = rng.standard_normal(simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths


def plot_monte_carlo_trajectories(paths: typing.Union[np.ndarray, list],
                                   time_grid: typing.Optional[np.ndarray] = None,
                                   start_date: typing.Optional[pd.Timestamp] = None,
                                   freq: str = "B",
                                   n_plot: typing.Optional[int] = None,
                                   filename: typing.Optional[str] = None,
                                   figsize: typing.Tuple[int, int] = (14, 8),
                                   alpha: typing.Optional[float] = None,
                                   colorful: bool = True):
    """Plot Monte Carlo trajectories with an optional datetime x-axis.

    Parameters
    - paths: array-like with shape (simulations, steps+1) or (steps+1,) for a single path
    - time_grid: optional array of time points (length steps+1). If None and
      `start_date` is provided, a pandas date_range will be constructed using
      `freq` and the number of steps in `paths`.
    - start_date: optional pandas Timestamp or datetime-like to anchor the x-axis
      (useful when paths correspond to trading days starting on a known date).
    - freq: frequency string passed to `pd.date_range` (default 'B' = business day)
    - n_plot: DEPRECATED - ignored, all trajectories are plotted
    - filename: if provided, saves the figure to this path; otherwise shows it
    - figsize: figure size tuple
    - alpha: line transparency for plotted trajectories (auto-adjusted based on number of sims if None)
    - colorful: if True, each trajectory has a different color from a colormap
    """
    import matplotlib as mpl

    # If saving to file in a headless environment, switch to Agg backend
    if filename:
        mpl.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import cm

    paths = np.asarray(paths)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)

    sims, steps_plus_one = paths.shape

    # Build time_grid: prefer explicit start_date -> date_range, else use provided time_grid
    if start_date is not None:
        # start_date may be a pandas Timestamp or datetime; include it as t=0
        time_grid = pd.date_range(start=pd.to_datetime(start_date), periods=steps_plus_one, freq=freq)
    elif time_grid is None:
        time_grid = np.arange(steps_plus_one)
    else:
        time_grid = np.asarray(time_grid)

    # Plot ALL trajectories (not just a sample)
    n_trajectories = sims
    
    # Auto-adjust alpha based on number of trajectories if not provided
    if alpha is None:
        if n_trajectories <= 100:
            alpha = 0.7
        elif n_trajectories <= 500:
            alpha = 0.4
        elif n_trajectories <= 1000:
            alpha = 0.25
        else:
            alpha = 0.15
    
    # Create figure with dark background for better contrast
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Plot individual trajectories with different colors
    if colorful:
        # Use a vibrant colormap for diverse colors
        colors = cm.rainbow(np.linspace(0, 1, n_trajectories))
        for i in range(n_trajectories):
            ax.plot(time_grid, paths[i], color=colors[i], alpha=alpha, linewidth=1.0)
    else:
        for i in range(n_trajectories):
            ax.plot(time_grid, paths[i], color="steelblue", alpha=alpha, linewidth=0.8)

    # plot only the median line (removed percentiles)
    median = np.median(paths, axis=0)
    ax.plot(time_grid, median, color="darkred", lw=3, label="Mediana", zorder=100)

    # Detect datetime-like x-axis and format automatically based on duration
    is_datetime = False
    try:
        # robust check for datetime dtype (works with DatetimeIndex, numpy datetime64, etc.)
        is_datetime = pd.api.types.is_datetime64_any_dtype(time_grid) or isinstance(time_grid, pd.DatetimeIndex)
    except Exception:
        # fallback: check numpy dtype
        try:
            tg = np.asarray(time_grid)
            is_datetime = np.issubdtype(tg.dtype, np.datetime64)
        except Exception:
            is_datetime = False

    if is_datetime:
        # Calculate duration to choose appropriate formatting
        duration = time_grid[-1] - time_grid[0]
        duration_days = duration.days if hasattr(duration, 'days') else (time_grid[-1] - time_grid[0]) / np.timedelta64(1, 'D')
        
        # Automatic formatting based on duration
        if duration_days < 90:  # Less than 3 months
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
        elif duration_days < 365:  # Less than 1 year
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        elif duration_days < 1095:  # Less than 3 years
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
        elif duration_days < 3650:  # Less than 10 years
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        else:  # 10 years or more
            # For very long durations, show years with 2-year intervals
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_minor_locator(mdates.YearLocator())
        
        fig.autofmt_xdate(rotation=45)

    ax.set_xlabel("Fecha" if is_datetime else "Paso", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precio", fontsize=12, fontweight='bold')
    ax.set_title("Trayectorias Monte Carlo - Todas las Simulaciones", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    fig.tight_layout()
    if filename:
        # ensure directory exists
        d = os.path.dirname(filename)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def get_parameters_from_csv(path):
    """
    Load CSV with extra header rows (skip first 3) and columns: 
    Price,Close,High,Low,Open,Volume
    Calculate annual mean return and volatility.
    """
    # Saltamos las filas no útiles
    df = pd.read_csv(path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    
    # Convertimos Date a datetime y ordenamos por fecha
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Calculamos retornos diarios basados en Close
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()

    # Parámetros anuales
    mu = df["Returns"].mean() * 252
    sigma = df["Returns"].std() * np.sqrt(252)
    S0 = df["Close"].iloc[-1]

    return S0, mu, sigma, df


def generate_and_plot_trajectories(years: int, simulations: int = 1000, 
                                   filename: typing.Optional[str] = None,
                                   csv_filename: typing.Optional[str] = None,
                                   alpha: typing.Optional[float] = None):
    """
    Genera y plotea TODAS las trayectorias Monte Carlo para un número dado de años.
    
    Parameters
    ----------
    years : int
        Número de años para las trayectorias
    simulations : int
        Número de simulaciones a generar (default: 1000)
        NOTA: TODAS las trayectorias simuladas serán ploteadas
    filename : str, optional
        Ruta donde guardar el gráfico PNG. Si es None, muestra el gráfico.
    csv_filename : str, optional
        Ruta donde guardar las trayectorias como CSV. Si es None, no se guarda CSV.
    alpha : float, optional
        Transparencia de las líneas (auto-ajustada si es None)
    
    Returns
    -------
    np.ndarray
        Array de trayectorias de forma (simulations, steps+1)
    """
    # Cargar datos históricos
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root_path, "data", "sp500_data.csv")
    S0, mu, sigma, df = get_parameters_from_csv(path)
    
    # Generar trayectorias (252 días de trading por año)
    steps = 252 * years
    paths = monte_carlo_paths(S0, mu, sigma, T=years, steps=steps, simulations=simulations)
    
    # Plotear TODAS las trayectorias
    if filename is not None:
        plot_monte_carlo_trajectories(
            paths, 
            filename=filename, 
            start_date=df["Date"].iloc[-1], 
            freq="B",
            alpha=alpha
        )
    else:
        plot_monte_carlo_trajectories(
            paths, 
            start_date=df["Date"].iloc[-1], 
            freq="B",
            alpha=alpha
        )
    
    # Guardar trayectorias como CSV si se solicita
    if csv_filename is not None:
        save_paths_to_csv(
            paths, 
            csv_filename, 
            start_date=df["Date"].iloc[-1], 
            freq="B",
            years=years
        )
    
    # Estadísticas
    final_prices = paths[:, -1]
    print(f"\n{'='*60}")
    print(f"SIMULACIÓN MONTE CARLO - {years} año(s)")
    print(f"{'='*60}")
    print(f"Precio inicial (S0):      {S0:.2f}")
    print(f"Retorno medio anual (μ):  {mu:.4f} ({mu*100:.2f}%)")
    print(f"Volatilidad anual (σ):    {sigma:.4f} ({sigma*100:.2f}%)")
    print(f"Número de simulaciones:   {simulations}")
    print(f"\nPRECIOS FINALES (después de {years} año(s)):")
    print(f"  Media:                  {np.mean(final_prices):.2f}")
    print(f"  Mediana:                {np.median(final_prices):.2f}")
    print(f"  Mínimo:                 {np.min(final_prices):.2f}")
    print(f"  Máximo:                 {np.max(final_prices):.2f}")
    print(f"{'='*60}\n")
    
    return paths


def save_paths_to_csv(paths: np.ndarray, filename: str, start_date: typing.Optional[pd.Timestamp] = None, 
                      freq: str = "B", years: typing.Optional[int] = None) -> None:
    """
    Guarda las trayectorias Monte Carlo en un archivo CSV.
    
    Parameters
    ----------
    paths : np.ndarray
        Array de trayectorias de forma (simulations, steps+1)
    filename : str
        Nombre del archivo CSV a guardar
    start_date : pd.Timestamp, optional
        Fecha inicial para generar el índice de fechas
    freq : str, default "B"
        Frecuencia para el índice de fechas ('B' = business day)
    years : int, optional
        Número de años (para información en el nombre)
    
    Returns
    -------
    None
    """
    # Crear DataFrame con las trayectorias
    # Cada columna representa una simulación
    df = pd.DataFrame(paths.T)
    
    # Renombrar columnas como Simulación_1, Simulación_2, etc.
    df.columns = [f"Simulacion_{i+1}" for i in range(paths.shape[0])]
    
    # Crear índice de fechas si se proporciona fecha inicial
    if start_date is not None:
        periods = paths.shape[1]
        date_index = pd.date_range(start=start_date, periods=periods, freq=freq)
        df.index = date_index
        df.index.name = "Fecha"
    else:
        # Índice numérico si no hay fecha
        df.index = range(paths.shape[1])
        df.index.name = "Paso"
    
    # Guardar a CSV
    df.to_csv(filename)
    print(f"Trayectorias guardadas en: {filename}")
    print(f"Formato: {paths.shape[0]} simulaciones × {paths.shape[1]} pasos temporales")


# ====================================================================
# CONFIGURACIÓN: Cambia estos parámetros según necesites
# ====================================================================
YEARS = 5              # Número de años para la simulación
SIMULATIONS = 1000     # Número de trayectorias a simular (TODAS se plotean)

# Genera trayectorias y plotea
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_file = os.path.join(root_path, "data", "montecarlo_trajectories.png")
csv_output = os.path.join(root_path, "data", "montecarlo_paths.csv")

paths = generate_and_plot_trajectories(
    years=YEARS,
    csv_filename=csv_output,
    simulations=SIMULATIONS,
    filename=output_file
)

print(f"Gráfico guardado en: {output_file}")
print(f"CSV guardado en: {csv_output}")

# ====================================================================
# EJEMPLO: Para probar con diferentes años y opciones de guardado
# ====================================================================
# paths_1y = generate_and_plot_trajectories(
#     years=1, 
#     simulations=500, 
#     filename="data/mc_1year.png",
#     csv_filename="data/mc_1year.csv"
# )
# 
# paths_10y = generate_and_plot_trajectories(
#     years=10, 
#     simulations=1000, 
#     filename="data/mc_10years.png",
#     csv_filename="data/mc_10years.csv"
# )
# 
# # Solo generar CSV sin gráfico
# paths_csv_only = generate_and_plot_trajectories(
#     years=3, 
#     simulations=2000, 
#     csv_filename="data/mc_3years_only.csv"
# )
