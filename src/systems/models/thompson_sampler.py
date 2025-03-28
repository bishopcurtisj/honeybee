import numpy as jnp

from ACE_Experiment.globals import globals


def thompson_sampler(agents: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Agents should have the columns:
    ['fitness', 'informed', 'demand_fx_params...']
    Params should have the prior distribution and likelihood function for each demand function parameter
    """

    if globals.informed:

        return informed_thompson(agents, params)
    else:
        return uninformed_thompson(agents, params)
            

def informed_thompson(agents: jnp.ndarray, parameters: jnp.ndarray) -> jnp.ndarray:
    ...

def uninformed_thompson(agents: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    ...

def update_priors(agents: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    ...