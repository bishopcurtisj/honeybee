import numpy as jnp

from ACE_Experiment.globals import globals



def thompson_sampler(agents: jnp.ndarray) -> jnp.ndarray:
    """
    Agents should have the columns:
    ['fitness', 'informed', 'demand_fx_params...']
    Params should have the prior distribution and likelihood function for each demand function parameter
    """
    if globals.informed:
        return informed_thompson(agents)
    else:
        return uninformed_thompson(agents)
            

def informed_thompson(agents: jnp.ndarray) -> jnp.ndarray:
    ...

def uninformed_thompson(agents: jnp.ndarray) -> jnp.ndarray:
    ...

def update_priors(agents: jnp.ndarray) -> jnp.ndarray:
    """
    params: [mean, std, tau]
    """
    params = agents[:, globals.components.demand_fx_params]
    weighted_sum = jnp.sum(globals.trades[:, 0]*globals.trades[:, 1])
    total_quantity_traded = jnp.sum(globals.trades[0])

    sigma_n_sq = 1 / (1 / params[:, 1]**2 + total_quantity_traded / params[:, 3]**2)
    mu_n = sigma_n_sq * (params[:, 0] / params[:, 1]**2 + weighted_sum / params[:, 3]**2)

    params = jnp.array(mu_n, jnp.sqrt(sigma_n_sq), params[:, 2])

    agents[:, globals.components.demand_fx_params] = params

    return agents

