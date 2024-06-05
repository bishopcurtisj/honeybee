import jax.numpy as jnp
import jax
import jax.lax as lax
import numpy as np
import scipy.stats as stats
from math import exp, log, sqrt
from contract import *
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

class JAX_LSM:
    def __init__(self, devices=len(jax.local_devices())):
        self.devices=devices
        self.device_mesh = mesh_utils.create_device_mesh((devices,))
        self.sharding = PositionalSharding(self.device_mesh)
        self.jit_dot = jax.jit(self.cont)

    def simulate_prices(self, drift=0.15,volatility=0.15, initial_price=100,T=25, n=250, distribution=np.random.normal, m=0, sd=1):

        if n%self.devices != 0:
            n= n + self.devices - n%self.devices

        ## Trading days in 1 year: 260
        R=log(1+drift)/260
        v=volatility/sqrt(260)

        assets = np.empty((n,T))
        assets[:,0] = initial_price
        for t in range(1,T):
            draws = np.exp(R+distribution(m,sd,n)*v)
            
            assets.T[t] = assets.T[t-1]*draws
        
        return jnp.asarray(assets)
   
    def cont(self, X,betas):
        return jnp.dot(X,betas)
    


    def LSM_american(self, prices, option, discount_rate = 0.06):

        n = len(prices)
        T = len(prices[0])
        cash_flow = np.zeros((n,T))
        discount_rate = log(1+discount_rate)/260


        for path in range(n):
            cash_flow[path,-1] = option.get_payoff(prices[path,-1])
        for t in range(T-2, 0, -1):
            ## Get the in-the-money paths
            X = [] 
            Y = []
            for path in range(n):
                if option.get_payoff(prices[path,t]) > 0:
                    X.append(prices[path,t])
                    cf = np.argmax(cash_flow[path,:])
                    if cf == 0:
                        Y.append(option.get_payoff(prices[path,t]))
                    else:
                        Y.append(cash_flow[path,cf]/((1+discount_rate)**(cf-t)))

            X = [jnp.array([1,x,x**2]) for x in X]
    
            X = jnp.asarray(X)
            Y = jnp.asarray(Y)
            if Y.size <=1:
                ## If there is only one path, we can't calculate the betas, read paper to see if LS address this...
                break
            else:
                betas = jnp.matmul(jnp.linalg.inv(jnp.dot(X.T,X)), jnp.dot(X.T,Y))
            
            p=0
            for path in range(n):
                if option.get_payoff(prices[path,t]) > 0:
                    exercise= option.get_payoff(prices[path,t])
                    continuation = jnp.dot(X[p], betas)
                    p+=1

                    if exercise > continuation:
                        cash_flow[path,t] = exercise
                        cash_flow[path,t+1:] = 0

        present_values = np.zeros(n)
        for path in range(n):
            cf = np.argmax(cash_flow[path])
            present_values[path] = cash_flow[path,cf]/((1+discount_rate)**(cf))
        
        return np.mean(present_values), cash_flow


    def jax_lsm(self, prices, option, discount_rate = 0.06):
        prices = jax.device_put(prices, self.sharding.reshape(self.devices, 1))
        n = len(prices)
        T = len(prices[0])
        cash_flow = np.zeros((n,T))
        discount_rate = log(1+discount_rate)/260


        for path in range(n):
            cash_flow[path,-1] = option.get_payoff(prices[path,-1])
        for t in range(T-2, 0, -1):
            ## Get the in-the-money paths
            X = [] 
            Y = []
            for path in range(n):
                if option.get_payoff(prices[path,t]) > 0:
                    X.append(prices[path,t])
                    cf = np.argmax(cash_flow[path,:])
                    if cf == 0:
                        Y.append(option.get_payoff(prices[path,t]))
                    else:
                        Y.append(cash_flow[path,cf]/((1+discount_rate)**(cf-t)))

            X = [jnp.array([1,x,x**2]) for x in X]
    
            X = jnp.asarray(X)
            Y = jnp.asarray(Y)
            if Y.size <=1:
                ## If there is only one path, we can't calculate the betas, read paper to see if LS address this...
                break
            else:
                betas = jnp.dot(jnp.linalg.inv(jnp.dot(X.T,X)), jnp.dot(X.T,Y))
            
            p=0
            for path in range(n):
                if option.get_payoff(prices[path,t]) > 0:
                    exercise= option.get_payoff(prices[path,t])
                    continuation = self.jit_dot(X[p],betas)
                    p+=1

                    if exercise > continuation:
                        cash_flow[path,t] = exercise
                        cash_flow[path,t+1:] = 0

        present_values = np.zeros(n)
        for path in range(n):
            cf = np.argmax(cash_flow[path])
            present_values[path] = cash_flow[path,cf]/((1+discount_rate)**(cf))
        
        return np.mean(present_values),cash_flow



    


