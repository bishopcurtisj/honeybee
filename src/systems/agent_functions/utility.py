from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp


## Utility function for agents to determine how much they value gains relative to risk
class Utility(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class Const_abs_risk_aversion(Utility):
    name = "Const_abs_risk_aversion"

    @staticmethod
    def __call__(returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
        return -jnp.exp(-risk_aversion * returns)


def calculate_utility(
    agents: jnp.ndarray, returns: jnp.ndarray, risk_aversion: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the utility of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function, risk_aversion]
    returns should have the following columns:
    [returns]
    """
    utilities = jnp.zeros((len(agents), len(returns[0])))

    for i in UTILITY_REGISTRY.keys():
        same_util = jnp.where(agents[:, 2] == i)[0]
        utilities[same_util[:, None]] = UTILITY_REGISTRY[i](returns, risk_aversion)

    return utilities


def register_utility_function(utility_functions: Union[List[Utility], Utility]):
    if type(utility_functions) == Utility:
        utility_functions = [utility_functions]
    for utility_function in utility_functions:
        try:
            assert issubclass(utility_function, Utility)
        except AssertionError:
            raise ValueError(
                f"Custom utility function {utility_function.name} must be a subclass of Utility"
            )
        UTILITY_REGISTRY[len(UTILITY_REGISTRY)] = utility_function


UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
