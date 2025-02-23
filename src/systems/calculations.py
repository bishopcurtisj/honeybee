import numpy as jnp

from entities.agent import *
from entities.market import Market

GAMMA_CONSTANTS = [1,1]


## TODO: Agent demand functions don't necessarily intersect, need to redesign.
def calculate_market_price(agents: jnp.ndarray, supply: float, maximum_supply: float, min_price: float = 0, max_price: float = 100_000, informed: bool = True) -> float:
    """
    Calculate the market price of a good given a set of agents and their demands

    Args:
        agents (jnp.ndarray): Array of agents
            Before being passed to this function the agents array will be filtered so that only relevant rows/columns are included
            The agents array will be of shape (n, m) with columns: [informed, signal, demand, demand_function, demand_function_params...]
        supply (float): Total supply of a good
        start_price (float, optional): Initial price of a good. Defaults to 0.

        Overwrite these with values relevant to the market
          min_price (float, optional): Minimum price of a good. Defaults to 0. 
          max_price (float, optional): Maximum price of a good. Defaults to 100.
            

    Returns:
        float: Market price of a good
    """

    if maximum_supply < supply:
        supply = maximum_supply*0.9


    p = (min_price + max_price)/2
    
    agents = update_demands(p, agents, informed)


    if supply == 0:
        total_demand = jnp.sum(agents[:,2])
    else:
        total_demand = jnp.sum(agents[agents[:,2]>=0, 2])
    

    iterations = 0
    while not jnp.allclose(total_demand, supply, rtol=0.001):
        
        if total_demand < supply:
            max_price = p
        elif total_demand > supply:
            min_price = p
        else:
            return p

        p = (min_price + max_price)/2
        agents = update_demands(p, agents, informed)
        if supply == 0:
            total_demand = jnp.sum(agents[:,2])
        else:
            total_demand = jnp.sum(agents[agents[:,2]>=0, 2])
        iterations += 1
        if iterations > 10000:
            raise ValueError(f"Price calculation did not converge, current price: {p}, total demand: {total_demand}, supply: {supply}")
    return p

## Pretty confident for loop is the only way, but will reevaluate. Demand Functions should always be pretty simple, so optimization is probably less relevant
def update_demands(price: float, agents: jnp.ndarray, informed: bool) -> jnp.ndarray:
    """
    Update the demand of a set of agents given a price
    agents should have the following columns:
    [informed, signal, demand, demand_function, demand_function_params...]
    """

    if informed:
        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i,3])]()
            agents[i,2] = df(price, agents[i,4:], agents[i,1], GAMMA_CONSTANTS[int(agents[i,0])])
    else:
        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i,1])]()

            agents[i,0] = df(price, agents[i,2:])
            
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


