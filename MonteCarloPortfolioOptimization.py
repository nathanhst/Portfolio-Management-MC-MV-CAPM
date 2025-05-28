from Portfolio import stock_returns,tickers,mean_returns,cov_matrix,risk_free_rate,excel_df
from MeanVariancePortfolioOptimization import optimal_weights,opt_return,opt_volatility
from CAPMPortfolioOptimization import optimal_weightsCAPM, returnCAPM, volCAPM, sharpeCAPM
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import pandas as pd


returns = stock_returns


# ---------------------------- Portfolio optimization -----------------------------------------------

num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # rows for return, volatility, Sharpe
weights_record = []

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
    weights /= np.sum(weights)
    weights_record.append(weights)

    portfolio_return = np.dot(weights, mean_returns) * 252  # annualized (there are 252 trading days in a year)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return)/ portfolio_std_dev

    results[0,i] = portfolio_return
    results[1,i] = portfolio_std_dev
    results[2,i] = sharpe_ratio

# ----------- For comparison with MV model --------------

weight_MV,returns_MV,vol_MV = optimal_weights,opt_return,opt_volatility

# ----------- Efficiency frontier --------------
num_bins = 30  # Number of bins to smooth the frontier
return_bins = np.linspace(np.min(results[0,:]), np.max(results[0,:]), num_bins)
min_volatilities = []

# Find the minimum volatility for each return bin
for r in return_bins:
    # Select portfolios with the return closest to the current return level
    return_mask = np.isclose(results[0,:], r, atol=0.005)  # Allow a tolerance
    if np.any(return_mask):  # Check if there are portfolios with that return
        min_volatility = np.min(results[1, return_mask])
        min_volatilities.append(min_volatility)

min_volatilities = np.array(min_volatilities)
smooth_volatilities = np.interp(return_bins, return_bins[:len(min_volatilities)], min_volatilities)

# -------------------- Sharpe ratio ----------------------
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]
excel_df['MC Weights'] = optimal_weights

top_weights = np.mean(np.vstack([optimal_weights, weight_MV]),axis = 0)
optimal_portfolio_return = np.dot(top_weights,mean_returns) * 252
optimal_portfolio_std_dev = np.sqrt(np.dot(top_weights.T, np.dot(cov_matrix * 252, top_weights)))
optimal_sharpe_ratio = (optimal_portfolio_return)/ optimal_portfolio_std_dev
excel_df['Average Weights'] = top_weights

excel_df.to_excel('portfolio.xlsx', index = False)

# -------------------- Plotting time ----------------------
plt.figure(figsize=(8,8))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], marker='*', color='g', s=200, label='Max Sharpe Ratio Monte Carlo')
plt.scatter(vol_MV, returns_MV, marker='*', color='r', s=200, label='Max Sharpe Ratio Mean Variance')
#plt.scatter(volCAPM, returnCAPM, marker='*', color='b', s=200, label='Max Sharpe Ratio CAPM')
plt.scatter(optimal_portfolio_std_dev,optimal_portfolio_return,marker='o', color='orange', s=200, label='Max Sharpe Ratio Averaged')
plt.plot(smooth_volatilities, return_bins, color='black', label='Efficient Frontier', linewidth=2)

plt.title('Efficient Frontier',fontsize = 12)
plt.xlabel('Volatility',fontsize = 12)
plt.ylabel('Return',fontsize = 12)
plt.legend(fontsize = 12)
plt.savefig("EfficientFrontier.svg", format = 'svg')

print("Sharpe Ratio of the optimal portfolio:", results[2, max_sharpe_idx])
print("Return of the optimal portfolio:", optimal_portfolio_return)
print("Volatility of the optimal portfolio:", optimal_portfolio_std_dev)



# ------------------------------------ Plot expected returns ------------------------------------------------

expected_returns = results[1, max_sharpe_idx]
expected_volatility = results[0, max_sharpe_idx]
simulated_returns = np.random.normal(loc=expected_volatility, scale=expected_returns, size=10000)
plt.figure(figsize=(10,6))
plt.hist(simulated_returns, bins=100, density=True, alpha=0.6, color='skyblue')
plt.axvline(expected_returns, color='red', linestyle='dashed', linewidth=2, label='Expected Return')

# Plot 1-sigma interval (68% confidence)
plt.axvline(expected_returns - expected_volatility, color='green', linestyle='dotted', linewidth=2, label='-1σ')
plt.axvline(expected_returns + expected_volatility, color='green', linestyle='dotted', linewidth=2, label='+1σ')

# Labels
plt.title('Distribution of Simulated Portfolio Returns')
plt.xlabel('Return')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.savefig('ExpectedReturnsMonteCarlo.svg',format = 'svg')

