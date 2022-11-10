#Dieses Programm soll dazu dienen ein Portfolio mithilfe der Effizienzkurve der modernen Portfoliotheorie zu optimieren

#Import Python libraries
from pandas_datareader import data as web
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
plt.style.use('fivethirtyeight')

#Tickers & Stock symbols 
assets = ["SKB.DE","1810.HK","BABA","CRM","SNH.DE","MSFT","V","WM","AOS"]

#Gewichtungen der Einzeltitel
weights = np.array([0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])

#Startdatum für Portfolio
stockStartDate = '2013-01-01'

#Enddatum ffür Portfolio (heute)
today = datetime.today().strftime('%Y-%m-%d')
#print(today)

#Dataframe = Close Date der Aktien
df = pd.DataFrame()
for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start = stockStartDate, end = today)['Adj Close']
    print(df)

#Visualisation
title = 'Portfolio Adj. Close Price History'
my_stocks = df

#Plotting
for c in my_stocks.columns.values:
   plt.plot(my_stocks[c], label = c)

plt.title(title)
plt.xlabel('Date', fontsize = 10)
plt.ylabel('Adj. Price $', fontsize = 10)
plt.legend(my_stocks.columns.values, loc = 'upper left')
plt.show()

# Daily returns
returns = df.pct_change()
print(returns)

#create & show annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
print(cov_matrix_annual)

#Portfoliovarianz berechnen
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
print(port_variance)

#Portfoliovola berechnen = STD
port_vola = np.sqrt(port_variance)
print(port_vola)

#Return p.a berechnen
port_SimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
print(port_SimpleAnnualReturn)

#show: expected annual return, vola (risiko -> STD), variance
percent_var = str(round(port_variance, 2) * 100)+ '%'
percent_vol = str(round(port_vola, 2) * 100)+ '%'
percent_ret = str(round(port_SimpleAnnualReturn, 2) * 100)+ '%'

print('Expected annual return: '+ percent_ret)
print('Annual vola / risk: '+ percent_vol)
print('Annual variance: '+ percent_var)

#Efficient frontier import
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

#Portfolio Optimization
#calculation: expected returns + annualised sample covariance matrix (asset returns)
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

#optimize for max sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#discrete AA for each stock
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices=get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = 100000)
allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))