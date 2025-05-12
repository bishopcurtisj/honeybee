import numpy as jnp
from jax import vmap

from globals import config, globals
from systems.models.information_policy import INFORMATION_POLICY_REGISTRY
from systems.models.model import Model


class Bayesian(Model):

    def __init__(self):

        self.informed = globals.components.informed
        self.mu_prior = globals.components.mu_prior
        self.sigma_prior = globals.components.sigma_prior
        self.tau = globals.components.tau

    def __call__(self, agents: jnp.ndarray) -> jnp.ndarray:

        if globals.informed:
            return self._informed(agents)
        else:
            return self._uninformed(agents)

    def _uninformed(self, agents: jnp.ndarray) -> jnp.ndarray:
        return self.update_priors(agents, globals.trades)

    def _informed(self, agents: jnp.ndarray) -> jnp.ndarray:

        informed_agents = jnp.where(globals.agents[:, self.informed] == 1)[0]
        uninformed_agents = jnp.where(globals.agents[:, self.informed] == 0)[0]

        samples = jnp.array(
            [
                globals.trades[
                    jnp.random.choice(
                        range(len(globals.trades)),
                        size=(len(globals.trades) * config.uninformed_base_ratio),
                        replace=False,
                    )
                ]
                for _ in range(len(uninformed_agents))
            ]
        )
        agents[informed_agents] = self.update_priors(
            agents[informed_agents],
            globals.trades,
        )
        agents[uninformed_agents] = self.update_priors(
            agents[uninformed_agents],
            samples,
        )

        for key, information_policy in INFORMATION_POLICY_REGISTRY.items():
            agents[
                jnp.where(
                    globals.agents[:, globals.components.information_policy] == key
                )[0]
            ] = information_policy(
                agents[
                    jnp.where(
                        globals.agents[:, globals.components.information_policy] == key
                    )[0]
                ]
            )

        return agents

    def update_priors(self, agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray:
        """
        Updates Bayesian agent priors using their personal trade histories.

        agents: array of shape (num_agents, ...)
        trades: array of shape (num_agents, num_trades, 2) where [:, :, 0] = quantity, [:, :, 1] = price
        """

        mu_prior = agents[:, self.mu_prior]
        sigma_prior = agents[:, self.sigma_prior]
        tau = agents[:, self.tau]

        def update_agent(mu, sigma, tau, agent_trades):
            quantities = agent_trades[:, 0]
            prices = agent_trades[:, 1]

            total_quantity = jnp.sum(quantities)
            weighted_sum = jnp.sum(quantities * prices)

            sigma_n_sq = 1 / (1 / sigma**2 + total_quantity / tau**2)
            mu_n = sigma_n_sq * (mu / sigma**2 + weighted_sum / tau**2)
            return jnp.array([mu_n, jnp.sqrt(sigma_n_sq), tau])

        updated_params = vmap(update_agent)(mu_prior, sigma_prior, tau, trades)

        agents = agents.at[:, globals.components.demand_fx_params].set(updated_params)
        return agents
