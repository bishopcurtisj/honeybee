import numpy as jnp
from jax import vmap

from src.globals import config, globals
from systems.models.information_policy import INFORMATION_POLICY_REGISTRY
from systems.models.model import Model


class Bayesian(Model):

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray:

        if globals.informed:
            return Bayesian._informed(agents)
        else:
            return Bayesian._uninformed(agents)

    def _uninformed(agents: jnp.ndarray) -> jnp.ndarray:
        return Bayesian.update_priors(agents, globals.trades)

    def _informed(agents: jnp.ndarray) -> jnp.ndarray:

        informed_agents = jnp.where(
            globals.agents[:, globals.components.informed] == 1
        )[0]
        uninformed_agents = jnp.where(
            globals.agents[:, globals.components.informed] == 0
        )[0]

        samples = jnp.array(
            [
                jnp.random.choice(
                    globals.trades,
                    size=(len(globals.trades) * config.uninformed_base_ratio),
                    replace=False,
                )
                for _ in len(uninformed_agents)
            ]
        )
        agents[informed_agents] = Bayesian.update_priors(
            agents[informed_agents],
            globals.trades,
        )
        agents[uninformed_agents] = Bayesian.update_priors(
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

    def update_priors(agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray:
        """
        Updates Bayesian agent priors using their personal trade histories.

        agents: array of shape (num_agents, ...)
        trades: array of shape (num_agents, num_trades, 2) where [:, :, 0] = quantity, [:, :, 1] = price
        """
        params = agents[
            :, globals.components.demand_fx_params
        ]  # shape: (num_agents, 3)
        mu_prior = params[:, 0]
        sigma_prior = params[:, 1]
        tau = params[:, 2]

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
