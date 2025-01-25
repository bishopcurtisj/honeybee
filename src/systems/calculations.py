import jax.numpy as jnp

from entities.agent import *
from entities.market import Market
from constants import *




def calculate_market_price(agents: jnp.ndarray, supply: float, start_price: float = 0, min_price: float = 0, max_price: float = 100, informed: bool = True) -> float:
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
    
    total_demand = jnp.sum(agents[:,1])
    if start_price == 0:
        p = (min_price + max_price)/2
    else:
        p = start_price

    while not jnp.allclose(total_demand, supply):
        
        if total_demand < supply:
            max_price = p
        else:
            min_price = p

        p = (min_price + max_price)/2
        agents = update_demands(p, agents, informed)
        total_demand = jnp.sum(agents[:,2])



## For now this is specific to Routledge 2001
## Takes [fitness, objective_function, informed, signal, demand, demand_function, demand_function_params...] as input

def calculate_fitness(agents: jnp.ndarray, repetitions: int, risk_aversion: jnp.ndarray, market: Market, informed: bool = True) -> jnp.ndarray:
    
    returns = jnp.ndarray((len(agents), repetitions))

    for i in range(repetitions):
        returns[:,i] = calculate_returns(agents[:, 2:], market, i, informed)
    
    returns = calculate_utility(agents, returns, risk_aversion)

    for i in range(len(agents)):
        agents[i, 0] = OBJECTIVE_REGISTRY[agents[i,1]](returns[i], risk_aversion[i])
    

    return agents
 
## Parallelized 1-period simultaions
def calculate_fitness_accel(agents: jnp.ndarray, repetitions: int) -> jnp.ndarray:
    pass


## Pretty confident for loop is the only way, but will reevaluate. Demand Functions should always be pretty simple, so optimization is probably less relevant
## Takes [informed, signal, demand, demand_function, demand_function_params...] as input
def update_demands(price: float, agents: jnp.ndarray, informed: bool, components: dict = None, market: Market = None) -> jnp.ndarray:
    
    """ traders = jnp.where(agents[:,0] == 0)
    if 'informed' not in components.keys():
        columns = [components['demand']['col_idx'], components['demand_function']['col_idx']].extend(components['demand_function']['parameter_idxs'])
        informed = False
    else:
        informed = True
        columns = [components['informed']['col_idx'], components['signal']['col_idx'], components['demand']['col_idx'], components['demand_function']['col_idx']].extend(components['demand_function']['parameter_idxs'])
    subset = traders[:, columns]
    subset = update_demands(market.price, subset,informed)
    #traders[:, columns] = subset
    agents[traders][:, columns] = subset """


    if informed:
        for i in range(len(agents)):
            agents[i,2] = DEMAND_REGISTRY[agents[i,3]](price, agents[i,4:], agents[i,1], GAMMA_CONSTANTS[agents[i,0]])
        else:
            for i in range(len(agents)):
                agents[i,0] = DEMAND_REGISTRY[agents[i,1]](price, agents[i,2:])
            
    return agents


def calculate_utility(agents: jnp.ndarray, returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
    utilities = jnp.empty(len(agents), len(returns)) 

    for i in range(len(agents)):
        utilities[i] = UTILITY_REGISTRY[agents[i,2]](returns[i], risk_aversion[i])


def calculate_returns(agents: jnp.ndarray, market: Market, repetition: int, informed: bool = True) -> jnp.ndarray:
    
    market.price = calculate_market_price(agents, market.supply, market.price, informed)
    return agents[:,2] * (market.dividends[repetition] - market.price) - market.cost_of_info * agents[:,0]

    # return = demand*(dividend - price) - c*informed