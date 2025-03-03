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
            
        
        return agents