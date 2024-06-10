import jax.numpy as jnp
import numpy as np
import scipy.stats as stats
from math import exp, log, sqrt
from contract import *

def euro_binomial_option(option,initial_price, r, vol, div=0.0):
    h=option.h
    u=exp((r-div)*h+vol*sqrt(h))
    d=exp((r-div)*h-vol*sqrt(h))

    Su=initial_price*u
    Sd=initial_price*d
    Cu=option.get_payoff(Su)
    Cd=option.get_payoff(Sd)
    delta_hedge = exp(-div*h)*(Cu-Cd)/(initial_price*(u-d))
    B = exp(-r*h)*(u*Cd-d*Cu)/(u-d)
    option_price = delta_hedge*initial_price+B

    return option_price



def binomial_option(option,initial_price, r, vol, div=0.0):
    h=option.h
    u=exp((r-div)*h+vol*sqrt(h))
    d=exp((r-div)*h-vol*sqrt(h))

    Su=initial_price*u
    Sd=initial_price*d
    Cu=bin_op(option,Su, r, vol, 1,div)
    Cd = bin_op(option,Sd, r, vol,1 ,div)
    
    delta_hedge = exp(-div*h)*(Cu-Cd)/(initial_price*(u-d))
    B = exp(-r*h)*(u*Cd-d*Cu)/(u-d)
    option_price = delta_hedge*initial_price+B 

    return option_price

def bin_op(option,initial_price, r, vol,t, div=0.0):
    if option.T == t:
        return option.get_payoff(initial_price)
    else:
        h=option.h
        u=exp((r-div)*h+vol*sqrt(h))
        d=exp((r-div)*h-vol*sqrt(h))
        Su=initial_price*u
        Sd=initial_price*d
        Cu=bin_op(option,Su, r, vol,t+1,div)
        Cd = bin_op(option,Sd, r, vol,t+1 ,div)

        delta_hedge = exp(-div*h)*(Cu-Cd)/(initial_price*(u-d))
        B = exp(-r*h)*(u*Cd-d*Cu)/(u-d)
        option_price = delta_hedge*initial_price+B 

        return max(option.get_payoff(initial_price),option_price)


