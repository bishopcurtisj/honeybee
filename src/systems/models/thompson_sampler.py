import numpy as jnp


def thompson_sampler(agents: jnp.ndarray, params: jnp.ndarray, informed: bool) -> jnp.ndarray:
    """
    Agents should have the columns:
    ['fitness', 'informed', 'demand_fx_params...']
    Params should have the prior distribution and likelihood function for each demand function parameter
    """

    if informed:

        return informed_thompson(agents, params)
    else:
        return uninformed_thompson(agents, params)
            

def informed_thompson(agents: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    demand_fx_params = agents[:, 2:]
    informed_prior = params[:, 0]
    
    params = params[:, 2:] 
    ## Sample from prior on both informed and uninformed to determine whether agent will be informed in following period

def uninformed_thompson(agents: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    ...