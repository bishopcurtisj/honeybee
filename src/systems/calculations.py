import jax.numpy as jnp

from entities.agent import *
from entities.market import Market
from constants import *


## For now this is specific to Routledge 2001
## Takes [fitness, objective_function, informed, signal, demand, demand_function, demand_function_params...] as input
def calculate_fitness(agents: jnp.ndarray, repetitions: int, risk_aversion: jnp.ndarray) -> jnp.ndarray:
    
    returns = jnp.ndarray((len(agents), repetitions))
    for i in range(repetitions):
        returns[:,i] = calculate_returns(agents[:, 2:])
    
    returns = calculate_utility(agents, returns, risk_aversion)

    for i in range(len(agents)):
        agents[i, 0] = OBJECTIVE_REGISTRY[agents[i,1]](returns[i], risk_aversion[i])

    return agents

## Parallelized 1-period simultaions
def calculate_fitness_accel(agents: jnp.ndarray, repetitions: int) -> jnp.ndarray:
    pass


## Pretty confident for loop is the only way, but will reevaluate. Demand Functions should always be pretty simple, so optimization is probably less relevant
## Takes [informed, signal, demand, demand_function, demand_function_params...] as input
def update_demands(price: float, agents: jnp.ndarray, informed: bool, components: dict, market: Market) -> jnp.ndarray:
    
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


def calculate_returns(agents: jnp.ndarray, market: Market) -> jnp.ndarray:
    
    return agents[:,2] * (market.dividend - market.price) - market.cost_of_info * agents[:,0]

    # return = demand*(dividend - price) - c*informed