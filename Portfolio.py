import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# ----------------------- Introduce stocks -------------------------------------------


#tick1 = ['KO', 'GOOGL', 'NVDA','PM', 'AVGO', 'PLTR']
tick1 = ['AAPL', 'MSFT', 'AGG', 'GLD', 'AMD', 'NVDA', 'TGT']
tick2 = ['LMT', 'HWM', 'GD', 'GE', 'RTX']
tick3 = ['AIG']
tick4 = ['AMT']
tick5 = ['NFLX', 'WMT', 'TSLA']
tick6 = ['LLY', 'JNJ']
market_ticker = ['^GSPC']


tickers = tick1
data= pd.DataFrame()

start_date = '2020-01-01'
end_date = '2023-01-01'


# ------------------ Read the data from CSV files for each ticker (not market ticker) -----------------------------------
for ticker in tickers:
    filename = f"data/{ticker}-2000-2025.csv"
    
    try:
        # Read CSV with custom handling for multi-index
        df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)

        
        # Check if the 'Close' column exists for the ticker
        if ('Close', ticker) in df.columns:
            # Filter the data to the desired date range
            df_filtered = df[('Close', ticker)].loc[start_date:end_date]
            data[ticker] = df[('Close', ticker)]
        else:
            print(f"'Close' column not found for {ticker} in {filename}")
    
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# ------------------ Now the market returns -----------------------------------
market_filename = 'data/^GSPC_2000_2025.csv' 
market_df = pd.read_csv(market_filename, parse_dates=['Date'], index_col='Date')
market_data = market_df['^GSPC'].loc[start_date:end_date]
data['^GSPC'] = market_data


returns = data.pct_change().dropna()
market_returns = returns['^GSPC']


# ------------------ Create an Excel file to store the weights -----------------------------------
excel_df = pd.DataFrame({
    'Tickers': tickers,
    'MV Weights': [0] * len(tickers),
    'MC Weights': [0] * len(tickers),
    'CAPM Weights': [0] * len(tickers),
    'Average Weights': [0] * len(tickers)
})

excel_df.to_excel('portfolio.xlsx', index = False)


# Stock returns
stock_returns = returns[tickers]
mean_returns = stock_returns.mean()
cov_matrix = stock_returns.cov()

risk = yf.download('^TNX', period='1d', interval='1d')
risk_free_rate = risk['Close'].iloc[-1] / 100



# ------------------------ Plot the daily returns, histograms -------------------------------------------
"""ret = returns['GOOGL']

plt.figure()
ret.plot(figsize=(6, 3)) # Plot the daily returns for apple stocks

plt.figure()
ret.hist(bins=100, figsize=(5, 3)) # Plot an histogram of daily returns
# Fit a normal distribution
mu, std = norm.fit(ret)

# Plot the Gaussian curve
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.plot(x, p, 'r', linewidth=2)
plt.title(f"Histogram of GOOGL Returns\nFit results: μ = {mu:.4f},  σ = {std:.4f}")
plt.xlabel("Daily Return")
plt.ylabel("Density")
plt.grid(True)
plt.show()
"""