from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp
from jax.random import geometric


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


DEMAND_REGISTRY = {1: GS_linear, 2: BayesianDemand}
