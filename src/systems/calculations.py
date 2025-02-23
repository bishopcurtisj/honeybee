import numpy as jnp

from entities.agent import *
from entities.market import Market

GAMMA_CONSTANTS = [1,1]


def calculate_spread(agents: jnp.ndarray, informed: bool = True) -> jnp.ndarray:
    """
    Calculate the bid-ask spread of a set of agents
    agents should have the following columns:
    [informed, signal, bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
    """
    price = 100 ## Dummy needs to be replaced
    if informed:
        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i,6])]()
            demand = df(price, agents[i,8:], agents[i,1], GAMMA_CONSTANTS[int(agents[i,0])])

            while demand != 0:
                price = price + 0.01 # needs to be changed
                demand = df(price, agents[i,8:], agents[i,1], GAMMA_CONSTANTS[int(agents[i,0])])

            bid = price - agents[i,7]
            ask = price + agents[i,7]
            agents[i,2] = bid
            agents[i,3] = ask
            agents[i,4] = df(bid, agents[i,8:], agents[i,1], GAMMA_CONSTANTS[int(agents[i,0])])
            agents[i,5] = df(ask, agents[i,8:], agents[i,1], GAMMA_CONSTANTS[int(agents[i,0])])
    else:
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

    

## For now this is specific to Routledge 2001

def calculate_fitness(agents: jnp.ndarray, repetitions: int, risk_aversion: jnp.ndarray, market: Market, informed: bool = True) -> jnp.ndarray:
    """
    Calculate the fitness of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function, informed, signal, demand, demand_function, demand_function_params...]"""
    
    returns = jnp.ndarray((len(agents), repetitions))
    temp = update_demands(0, agents[:, 3:], informed)

    ## TODO: Consider equilibrium by finding maximum demand 
    if market.supply[0] == 0:    
        market.demand_at_p0 = jnp.sum(temp[:,5])
    else:
        market.demand_at_p0 = jnp.sum(temp[temp[:,5]>=0,5])

    for i in range(repetitions):
        returns[:,i] = calculate_returns(agents[:, 3:], market, i, informed)
    
## Need to add utility function to agents array getting passed to this function
    returns = calculate_utility(agents, returns, risk_aversion)

    for i in range(len(agents)):
        objective_function = OBJECTIVE_REGISTRY[int(agents[i,1])]()
        agents[i, 0] = objective_function(returns[i], risk_aversion[i])
    

    return agents
 


def calculate_returns(agents: jnp.ndarray, market: Market, repetition: int, informed: bool = True) -> jnp.ndarray:
    """
    Calculate the returns of a set of agents
    agents should have the following columns:
    [informed, signal, demand, demand_function, demand_function_params...]
    """
    
    market.price = calculate_market_price(agents, market.supply[repetition], market.demand_at_p0,  informed=informed)
    returns = agents[:,2] * (market.dividends[repetition] - market.price) - market.cost_of_info * agents[:,0]
    return returns

    # return = demand*(dividend - price) - c*informed

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


