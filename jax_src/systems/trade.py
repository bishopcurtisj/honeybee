
import jax.numpy as jnp

from entities.agent import *
from constants import *
from systems.calculations import update_demands
from entities.market import Market 


# def calculate_market_price(agents: jnp.ndarray, market: Market, min_price: float = 0, max_price: float = 100) -> float:
"""
Calculate the market price of a good given a set of agents and their demands

Args:
    agents (jnp.ndarray): Array of agents
        Before being passed to this function the agents array will be filtered so that only relevant rows/columns are included
        The agents array will be of shape (n, m) with columns: [informed, signal, demand, demand_function, demand_function_params...]

Returns:
    float: Market price of a good
"""
"""
total_demand = jnp.sum(agents[:,1])
if market.price == 0:
    p = (min_price + max_price)/2
else:
    p = market.price

while not jnp.allclose(total_demand, market.supply):
    
    if total_demand < market.supply:
        max_price = p
    else:
        min_price = p

    p = (min_price + max_price)/2
    agents = update_demands(p, agents)
    total_demand = jnp.sum(agents[:,2])
"""
