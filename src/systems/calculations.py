import numpy as jnp

from ACE_Experiment.globals import globals, config
from entities.agent import *
from entities.market import Market
from systems.trade import *




def calculate_spread(agents: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the bid-ask spread of a set of agents
    agents should have the following columns:
    [informed, signal, bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
    """
    

    if globals.informed:
        return _informed_spread(agents)
    else:
        return _uninformed_spread(agents)

def _informed_spread(agents: jnp.ndarray) -> jnp.ndarray:

    min_price = 0
    max_price = config.max_price
    price = (max_price + min_price) / 2

    for i in range(len(agents)):
        df = DEMAND_REGISTRY[int(agents[i,6])]()
        demand = df(price, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

        while jnp.allclose(demand, 0, atol=1e-2) == False:
            if demand > 0:
                min_price = price
            else:
                max_price = price
            price = (max_price + min_price) / 2
            demand = df(price, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

        bid = price - agents[i,7]
        ask = price + agents[i,7]
        agents[i,2] = bid
        agents[i,3] = ask
        agents[i,4] = df(bid, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])
        agents[i,5] = df(ask, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

    return agents

def _uninformed_spread(agents: jnp.ndarray) -> jnp.ndarray:

    min_price = 0
    max_price = config.max_price
    price = (max_price + min_price) / 2

    for i in range(len(agents)):
        df = DEMAND_REGISTRY[int(agents[i,4])]()
        demand = df(price, agents[i,6:])

        while demand != 0:
            price = price + 0.01
            demand = df(price, agents[i,6:])
        bid = price - agents[i,5]
        ask = price + agents[i,5]
        agents[i,0] = bid
        agents[i,1] = ask
        agents[i,2] = df(bid, agents[i,6:])
        agents[i,3] = df(ask, agents[i,6:])
            
    return agents


def calculate_fitness(agents: jnp.ndarray, repetitions: int, trades: jnp.ndarray, risk_aversion: jnp.ndarray, market: Market) -> jnp.ndarray:
    """
    Calculate the fitness of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function, informed, signal, demand, demand_function, demand_function_params...]"""
    returns = calculate_returns(agents, market, repetitions, trades)
    utilities = calculate_utility(agents, returns, risk_aversion)
    
    for i in range(len(agents)):
        objective_function = OBJECTIVE_REGISTRY[int(agents[i,1])]()
        agents[i, 0] = objective_function(utilities[i], risk_aversion[i])

    return agents
 


def calculate_returns(agents: jnp.ndarray, market: Market, repetition: int, trades: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the returns of a set of agents
    agents should have the following columns:
    [informed, ...]
    trades should have the following columns:
    [quantity, total spendings]
    """

    if globals.informed:
        returns = trades[:, 0] * market.dividends[repetition] - trades[:,1] - market.cost_of_info * agents[:,0]
    else:
        returns = trades[:, 0] * market.dividends[repetition] - trades[:,1]
    return returns

def calculate_utility(agents: jnp.ndarray, returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the utility of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function]
    returns should have the following columns:
    [returns]
    """
    utilities = jnp.zeros((len(agents), len(returns[0]))) 

    for i in range(len(agents)):
        utility_function = UTILITY_REGISTRY[int(agents[i,2])]()
        utilities[i] = utility_function(returns[i], risk_aversion[i])

    return utilities


