from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp
from jax.scipy.stats import norm

from ACE_Experiment.globals import config, globals
from systems.agent_functions.demand import DEMAND_REGISTRY


class Spread(ABC):
    name: str

    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class LinearDemandSpread(Spread):
    name = "LinearDemandSpread"

    @staticmethod
    def __call__(agents: jnp.ndarray) -> float:
        """
        Calculate the bid-ask spread of a set of agents
        agents should have the following columns:
        [informed, signal, bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
        """

        if globals.informed:
            return LinearDemandSpread._informed_spread(agents)
        else:
            return LinearDemandSpread._uninformed_spread(agents)

    def _informed_spread(agents: jnp.ndarray) -> jnp.ndarray:

        min_price = 0
        max_price = config.max_price
        price = (max_price + min_price) / 2

        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i, 6])]()
            demand = df(
                price,
                agents[i, 8:],
                agents[i, 1],
                config.GAMMA_CONSTANTS[int(agents[i, 0])],
            )

            while jnp.allclose(demand, 0, atol=1e-2) == False:
                if demand > 0:
                    min_price = price
                else:
                    max_price = price
                price = (max_price + min_price) / 2
                demand = df(
                    price,
                    agents[i, 8:],
                    agents[i, 1],
                    config.GAMMA_CONSTANTS[int(agents[i, 0])],
                )

            bid = price - agents[i, 7]
            ask = price + agents[i, 7]
            agents[i, 2] = bid
            agents[i, 3] = ask
            agents[i, 4] = df(
                bid,
                agents[i, 8:],
                agents[i, 1],
                config.GAMMA_CONSTANTS[int(agents[i, 0])],
            )
            agents[i, 5] = df(
                ask,
                agents[i, 8:],
                agents[i, 1],
                config.GAMMA_CONSTANTS[int(agents[i, 0])],
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

    @staticmethod
    def __call__(agents: jnp.ndarray) -> float:
        """
        Calculate the bid-ask spread of a set of agents
        agents should have the following columns:
        [informed, signal, bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
        """
        p = agents[:, 7] / 2
        agents[:, 2] = agents[:, 8] + agents[:, 9] * norm.ppf(p)
        agents[:, 3] = agents[:, 8] + agents[:, 9] * norm.ppf(1 - p)
        for i in DEMAND_REGISTRY.keys():
            same_demand = jnp.where(agents[6] == i)[0]
            agents[same_demand[:, None]][4] = DEMAND_REGISTRY[i](
                agents[same_demand][:, 2], True, agents[same_demand][:, 8:]
            )
            agents[same_demand[:, None]][5] = DEMAND_REGISTRY[i](
                agents[same_demand][:, 3], False, agents[same_demand][:, 8:]
            )
        return agents


def register_spread_function(spread_functions: Union[List[Spread], Spread]):
    """
    Registers custom spread function's,
    """
    if type(spread_functions) == Spread:
        spread_functions = [spread_functions]
    for spread_function in spread_functions:
        try:
            assert issubclass(spread_function, Spread)
        except AssertionError:
            raise ValueError(
                f"Custom spread function {spread_function.name} must be a subclass of Objective"
            )
        SPREAD_REGISTRY[len(SPREAD_REGISTRY)] = spread_function


SPREAD_REGISTRY = {1: LinearDemandSpread, 2: BayesianSpread}
