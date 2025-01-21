
import jax.numpy as jnp

from entities.agent import *
from constants import *
from calculations import update_demands


def calculate_market_price(agents: jnp.ndarray, supply: float, min_price: float = 0, max_price: float = 100) -> float:
    """
    Calculate the market price of a good given a set of agents and their demands
    This function assumes linear demand functions as seen in Routledge (1999)

    Args:
        agents (jnp.ndarray): Array of agents
            Before being passed to this function the agents array will be filtered so that only relevant rows/columns are included
            The agents array will be of shape (n, m) with columns: [informed, signal, demand, demand_function, demand_function_params...]

    Returns:
        float: Market price of a good
    """
    total_demand = jnp.sum(agents[:,1])
    p = (min_price + max_price)/2

    while jnp.allclose(total_demand, supply):
        
        if total_demand < supply:
            max_price = p
        else:
            min_price = p

        p = (min_price + max_price)/2
        agents = update_demands(p, agents)
        total_demand = jnp.sum(agents[:,2])

