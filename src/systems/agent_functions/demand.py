from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp
from jax.random import geometric

from systems.learning import model_controller
from systems.models.neural_network import NeuralNetwork


## Demand function for agents to determine how much of a good to buy
class Demand(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class GS_linear(Demand):
    name = "GS_linear"

    ## If uninformed signal == price
    @staticmethod
    def __call__(
        price: float, coeffs: jnp.ndarray, signal, scaling_factor: float = None
    ) -> float:
        return scaling_factor * (coeffs[0] + coeffs[1] * signal - price)


class BayesianDemand(Demand):
    """
    Returns the draw from the geometric / negative binomial (r=1) that indicates how many units of the
    asset an agent will demand. The probability used in the distribution is the p-value of the price given
    the agents prior price distribution.
    args:
       price: the price of the asset
       bid: whether the quantity is being bought or sold
       params: price distribution [mean, std]
    """

    name = "BayesianDemand"

    @staticmethod
    def __call__(price: jnp.ndarray, bid: bool, params: jnp.ndarray, key=None) -> float:
        prob = (price - params[:, 0]) / params[:, 1]

        if bid:
            return geometric(key, 1 - prob)
        else:
            return geometric(key, prob)


class NeuralNetworkDemand(Demand):

    def __init__(self):
        self.agent_id = globals.components.agent_id
        self.neural_network: NeuralNetwork = model_controller.model_registry[
            "neural_network"
        ]

    def __call__(self, agents: jnp.ndarray, prices: float, *args, **kwds):

        demands = jnp.empty(len(agents))

        for i, agent in enumerate(agents):
            model_info = self.neural_network.models[agent[self.agent_id]]
            model = self.neural_network._load_model(model_info["model_ref"])
            demands[i] = model.predict(prices[i])

        return demands


def register_demand_function(demand_functions: Union[List[Demand], Demand]):
    if type(demand_functions) == Demand:
        demand_functions = [demand_functions]
    for demand_function in demand_functions:
        try:
            assert issubclass(demand_function, Demand)
        except AssertionError:
            raise ValueError(
                f"Custom demand function {demand_function.name} must be a subclass of Demand"
            )
        DEMAND_REGISTRY[len(DEMAND_REGISTRY)] = demand_function


def demand_factory():
    neural_network_demand = NeuralNetworkDemand()
    DEMAND_REGISTRY[3] = neural_network_demand


DEMAND_REGISTRY = {1: GS_linear, 2: BayesianDemand}
