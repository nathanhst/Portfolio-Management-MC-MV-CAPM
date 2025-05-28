from Portfolio import stock_returns,tickers,mean_returns,cov_matrix,risk_free_rate,excel_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

returns = stock_returns
# mean returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualized mean returns
cov_matrix = returns.cov() * 252      # Annualized covariance matrix

# Portfolio performance 
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (returns - risk_free_rate) / std
    return returns, std, sharpe

# maximize sharpe (minimize negative sharpe)
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# constraints for weights
def constraint_sum(weights):
    return np.sum(weights) - 1

# Run optimization
num_assets = len(mean_returns)
initial_guess = num_assets * [1. / num_assets]
bounds = tuple((0, 1) for asset in range(num_assets))
constraints = ({'type': 'eq', 'fun': constraint_sum})

optimized = minimize(
    negative_sharpe,
    initial_guess,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Results
optimal_weights = optimized.x
excel_df['MV Weights'] = optimal_weights
excel_df.to_excel('portfolio.xlsx', index = False)
opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix,risk_free_rate)

print("Optimal weights:", optimal_weights)
print("Expected annual return:", opt_return)
print("Expected annual volatility:", opt_volatility)
print("Sharpe Ratio:", opt_sharpe)