import numpy as jnp

class ThompsonSampler:

    def __call__(self, agents: jnp.ndarray, params: jnp.ndarray, informed: bool) -> jnp.ndarray:
        """
        Agents should have the columns:
        ['fitness', 'informed', 'demand_fx_params...']
        Params should have the prior distribution and likelihood function for each demand function parameter
        """

        if informed:
            demand_fx_params = agents[:, 2:]
            informed_prior = params[:, 0]
            informed_likelihood = params[:, 1]
            params = params[:, 2:] 
            ## Sample from prior on both informed and uninformed to determine whether agent will be informed in following period

        else:
            demand_fx_params = agents[:, 1:]
        
        return agents