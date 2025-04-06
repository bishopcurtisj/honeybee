import numpy as jnp

from ACE_Experiment.globals import config, globals
from systems.models.model import INFORMATION_POLICY_REGISTRY, Model


class Bayesian(Model):

    def update_priors(agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray:
        """
        params: [mean, std, tau]
        """
        params = agents[:, globals.components.demand_fx_params]
        weighted_sum = jnp.sum(trades[:, 0] * trades[:, 1])
        total_quantity_traded = jnp.sum(trades[0])

        sigma_n_sq = 1 / (
            1 / params[:, 1] ** 2 + total_quantity_traded / params[:, 3] ** 2
        )
        mu_n = sigma_n_sq * (
            params[:, 0] / params[:, 1] ** 2 + weighted_sum / params[:, 3] ** 2
        )

        params = jnp.array(mu_n, jnp.sqrt(sigma_n_sq), params[:, 2])

        agents[:, globals.components.demand_fx_params] = params

        return agents

    def __call__(agents: jnp.ndarray) -> jnp.ndarray:

        if globals.informed:
            return Bayesian._informed(agents)
        else:
            return Bayesian._uninformed(agents)

    def _uninformed(agents: jnp.ndarray) -> jnp.ndarray:
        return Bayesian.update_priors(agents, globals.trades)

    def _informed(agents: jnp.ndarray) -> jnp.ndarray:
        sample = jnp.random.choice(
            globals.trades, size=(len(globals.trades) * config.uninformed_base_ratio)
        )
        agents[jnp.where(globals.agents[:, globals.components.informed] == 1)[0]] = (
            Bayesian.update_priors(
                agents[
                    jnp.where(globals.agents[:, globals.components.informed] == 1)[0]
                ],
                globals.trades,
            )
        )
        agents[jnp.where(globals.agents[:, globals.components.informed] == 0)[0]] = (
            Bayesian.update_priors(
                agents[
                    jnp.where(globals.agents[:, globals.components.informed] == 0)[0]
                ],
                sample,
            )
        )

        return agents
