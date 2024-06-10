import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from math import exp, log, sqrt
from contract import *

def simulate_prices(drift=0.15,volatility=0.15, initial_price=100,T=25, n=250, distribution=np.random.normal, m=0, sd=1):
    #np.random.seed(seed)
    ## Trading days in 1 year: 260
    R=log(1+drift)/260
    v=volatility/sqrt(260)

    assets = np.empty((n,T))
    assets[:,0] = initial_price
    for t in range(1,T):
        draws = np.exp(R+distribution(m,sd,n)*v)
        
        assets.T[t] = assets.T[t-1]*draws
    
    return assets

def LSM_american(prices, option, discount_rate = 0.06):
    n = len(prices)
    T = option.T
    cash_flow = np.zeros((n,T))
    discount_rate = log(1+discount_rate)/260
    #260*dr = log(1+r)
    #exp(260*dr) = 1+r

    for path in range(n):
        cash_flow[path,-1] = option.get_payoff(prices[path,-1])
    for t in range(T-2, 0, -1):
        ## Get the in-the-money paths
        X = [] 
        Y = []
        payoffs = {}
        pos_cash_flows = []
        for path in range(n):
            payoff=option.get_payoff(prices[path,t])
            if payoff > 0:
                pos_cash_flows.append(path)
                X.append(prices[path,t])
                payoffs[path]=payoff
                cf = np.argmax(cash_flow[path])
                if cf == 0:
                    Y.append(0)
                else:
                    Y.append(cash_flow[path,cf]*exp(-discount_rate*(cf-t)))

        X = [np.array([1,x,x**2]) for x in X]
      
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.size <=1:
            ## If there is only one path, we can't calculate the betas, read paper to see if LS address this...
            break
        else:
            betas = np.matmul(np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,Y))
    
        p=0
        for path in pos_cash_flows:
            continuation = np.dot(X[p], betas)
            p+=1

            if payoffs[path] > continuation:
                cash_flow[path,t] = payoffs[path]
                cash_flow[path,t+1:] = 0

    present_values = np.zeros(n)
    for path in range(n):
        cf = np.argmax(cash_flow[path])
        present_values[path] = cash_flow[path,cf]*exp(-discount_rate*(cf))
    
    return np.mean(present_values), cash_flow
    #return present_values

def LSM_european(prices, option, discount_rate = 0.06):
    n = len(prices)
    cash_flow = np.zeros(n)
    #discount_rate = log(1+discount_rate)/260

    for path in range(n):
        cash_flow[path] = option.get_payoff(prices[path,-1])*exp(-discount_rate*(option.T-1))

    
    return np.mean(cash_flow)    

def bsm_call(option,rfr=0.06,price=100,vol=0.15/sqrt(260),div=0):
    T=option.T
    strike=option.strike
    N=stats.norm.cdf
    d1=(np.log(price/strike)+(rfr-div+vol**2/2)*T)/(vol*np.sqrt(T))
    d2=d1-vol*np.sqrt(T)
    call = price*np.exp(-div*T)*N(d1)-strike*np.exp(-rfr*T)*N(d2)
    delta=N(d1)
    return call,delta                






