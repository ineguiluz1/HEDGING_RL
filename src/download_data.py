import yfinance as yf

ticker = "^GSPC"  # También podrías usar "SPY" (ETF)

# Descarga datos históricos diarios de los últimos 20 años
data = yf.download(ticker, start="2003-11-17", end="2023-11-17", progress=True)

# Guarda los datos en un archivo CSV
data.to_csv('data/sp500_data.csv')