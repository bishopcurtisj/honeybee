import jax.numpy as jnp

from entities.agent import *
from constants import *


def calculate_fitness(agents: jnp.ndarray) -> jnp.ndarray:
    pass


## Pretty confident for loop is the only way, but will reevaluate. Demand Functions should always be pretty simple, so optimization is probably less relevant
def update_demands(price: float, agents: jnp.ndarray):
    
    for i in range(len(agents)):
        agents[i,2] = DEMAND_REGISTRY[agents[i,3]](price, agents[i,4:], agents[i,1], GAMMA_CONSTANTS[agents[i,0]])
        
    return agents