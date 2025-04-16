from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp
from jax.scipy.stats import norm

from globals import config, globals
from systems.agent_functions.demand import DEMAND_REGISTRY
from systems.learning import model_controller
from systems.models.neural_network import NeuralNetwork


class Spread(ABC):
    name: str

    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class LinearDemandSpread(Spread):
    name = "LinearDemandSpread"

    def __init__(self):
        if globals.informed:
            self.informed = globals.components.informed
        self.signal = globals.components.signal
        self.bid = globals.components.bid
        self.ask = globals.components.ask
        self.bid_quantity = globals.components.bid_quantity
        self.ask_quantity = globals.components.ask_quantity
        self.demand_function = globals.components.demand_function
        self.confidence = globals.components.confidence
        self.beta0 = globals.components.beta0
        self.beta1 = globals.components.beta1
        self.demand_functions = jnp.unique(
            globals.agents[
                jnp.where(globals.agents[:, globals.components.spread_function] == 1)[0]
            ][:, self.demand_function]
        )

    def __call__(self, agents: jnp.ndarray) -> float:
        """
        Calculate the bid-ask spread of a set of agents

        """

        if globals.informed:
            return self._informed_spread(agents)
        else:
            return self._uninformed_spread(agents)

    def _informed_spread(self, agents: jnp.ndarray) -> jnp.ndarray:

        min_price = 0
        max_price = config.max_price
        price = (max_price + min_price) / 2

        for i in range(len(agents)):

            demand = DEMAND_REGISTRY[agents[i, self.demand_function]](
                price,
                agents[i, [self.beta0, self.beta1]],
                agents[i, self.signal],
                config.GAMMA_CONSTANTS[int(agents[i, self.informed])],
            )

            while jnp.allclose(demand, 0, atol=1e-2) == False:
                if demand > 0:
                    min_price = price
                else:
                    max_price = price
                price = (max_price + min_price) / 2
                demand = DEMAND_REGISTRY[agents[i, self.demand_function]](
                    price,
                    agents[i, [self.beta0, self.beta1]],
                    agents[i, self.signal],
                    config.GAMMA_CONSTANTS[int(agents[i, self.informed])],
                )

            bid = price - agents[i, self.confidence]
            ask = price + agents[i, self.confidence]
            agents[i, self.bid] = bid
            agents[i, self.ask] = ask
            agents[i, self.bid_quantity] = DEMAND_REGISTRY[
                agents[i, self.demand_function]
            ](
                bid,
                agents[i, [self.beta0, self.beta1]],
                agents[i, self.signal],
                config.GAMMA_CONSTANTS[int(agents[i, self.informed])],
            )
            agents[i, self.ask_quantity] = DEMAND_REGISTRY[
                agents[i, self.demand_function]
            ](
                ask,
                agents[i, [self.beta0, self.beta1]],
                agents[i, self.signal],
                config.GAMMA_CONSTANTS[int(agents[i, self.informed])],
            )

        return agents

    def _uninformed_spread(agents: jnp.ndarray) -> jnp.ndarray:

        min_price = 0
        max_price = config.max_price
        price = (max_price + min_price) / 2

        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i, 4])]()
            demand = df(price, agents[i, 6:])

            while demand != 0:
                price = price + 0.01
                demand = df(price, agents[i, 6:])
            bid = price - agents[i, 5]
            ask = price + agents[i, 5]
            agents[i, 0] = bid
            agents[i, 1] = ask
            agents[i, 2] = df(bid, agents[i, 6:])
            agents[i, 3] = df(ask, agents[i, 6:])

        return agents


class BayesianSpread(Spread):
    name = "BayesianSpread"

    def __init__(self):

        self.bid = globals.components.bid
        self.ask = globals.components.ask
        self.bid_quantity = globals.components.bid_quantity
        self.ask_quantity = globals.components.ask_quantity
        self.demand_function = globals.components.demand_function
        self.confidence = globals.components.confidence
        self.mu_prior = globals.components.mu_prior
        self.sigma_prior = globals.components.sigma_prior
        self.tau = globals.components.tau
        self.demand_functions = jnp.unique(
            globals.agents[
                jnp.where(globals.agents[:, globals.components.spread_function] == 2)[0]
            ][:, self.demand_function]
        )

    def __call__(self, agents: jnp.ndarray) -> float:
        """
        Calculate the bid-ask spread of a set of agents
        agents should have the following columns:
        [informed, signal, bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
        """

        p = agents[:, self.confidence] / 2
        agents[:, self.bid] = agents[:, self.mu_prior] + agents[
            :, self.sigma_prior
        ] * norm.ppf(p)
        agents[:, self.ask] = agents[:, self.mu_prior] + agents[
            :, self.sigma_prior
        ] * norm.ppf(1 - p)
        for i in self.demand_functions:
            same_demand = jnp.where(agents[:, self.demand_function] == i)[0]
            agents[same_demand, self.bid_quantity] = DEMAND_REGISTRY[i](
                price=agents[same_demand][:, self.bid],
                bid=True,
                params=agents[same_demand][
                    :, [self.mu_prior, self.sigma_prior, self.tau]
                ],
            )
            agents[same_demand, self.ask_quantity] = DEMAND_REGISTRY[i](
                price=agents[same_demand][:, self.ask],
                bid=False,
                params=agents[same_demand][
                    :, [self.mu_prior, self.sigma_prior, self.tau]
                ],
            )
        return agents


class NNSpread(Spread):
    """
    This a placeholder implementation
    """

    def __init__(self):
        self.agent_id = globals.components.agent_id
        self.neural_network: NeuralNetwork = model_controller.model_registry[3]
        self.risk_aversion = globals.components.risk_aversion
        self.bid = globals.components.bid
        self.ask = globals.components.ask
        self.bid_quantity = globals.components.bid_quantity
        self.ask_quantity = globals.components.ask_quantity
        self.demand_function = globals.components.demand_function
        self.confidence = globals.components.confidence

        self.demand_functions = jnp.unique(
            globals.agents[
                jnp.where(globals.agents[:, globals.components.spread_function] == 3)[0]
            ][:, self.demand_function]
        )

    def __call__(self, agents: jnp.ndarray) -> jnp.ndarray:

        agents[:, self.bid] = (1 - agents[:, self.confidence]) * globals.market[
            config.benchmark_price
        ][globals.repetition]
        agents[:, self.ask] = (1 + agents[:, self.confidence]) * globals.market[
            config.benchmark_price
        ][globals.repetition]

        agents[:, self.ask_quantity] = DEMAND_REGISTRY[3](agents, agents[:, self.ask])
        agents[:, self.bid_quantity] = DEMAND_REGISTRY[3](agents, agents[:, self.bid])

        return agents


def register_spread_function(spread_functions: Union[List[Spread], Spread]):
    """
    Registers custom spread function's,
    """
    if type(spread_functions) == Spread or isinstance(spread_functions, Spread):
        spread_functions = [spread_functions]
    for spread_function in spread_functions:
        try:
            assert issubclass(spread_function, Spread) or isinstance(
                spread_function, Spread
            )
        except AssertionError:
            raise ValueError(
                f"Custom spread function {spread_function.name} must be a subclass of Objective"
            )
        SPREAD_REGISTRY[len(SPREAD_REGISTRY)] = spread_function


def spread_factory():

    linear_spread = LinearDemandSpread()
    bayesian_spread = BayesianSpread()
    nn_spread = NNSpread()

    SPREAD_REGISTRY.update({1: linear_spread, 2: bayesian_spread, 3: nn_spread})


SPREAD_REGISTRY = {}
