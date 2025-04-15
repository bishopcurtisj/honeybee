from abc import ABC, abstractmethod
from typing import List, Union

import numpy as jnp

from globals import config, globals
from systems.agent_functions.utility import calculate_utility


## Objective function for learning algorithm to maximize
class Objective(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class Mean_variance(Objective):
    name = "Mean_variance"

    @staticmethod
    def __call__(utilities: jnp.ndarray, risk_aversion: float) -> float:
        return jnp.mean(utilities) - jnp.var(utilities) * risk_aversion / 2


def register_objective_function(objective_functions: Union[List[Objective], Objective]):
    """
    Registers custom objective function's,
    """
    if type(objective_functions) == Objective:
        objective_functions = [objective_functions]
    for objective_function in objective_functions:
        try:
            assert issubclass(objective_function, Objective)
        except AssertionError:
            raise ValueError(
                f"Custom objective function {objective_function.name} must be a subclass of Objective"
            )
        OBJECTIVE_REGISTRY[len(OBJECTIVE_REGISTRY)] = objective_function


def calculate_fitness(
    agents: jnp.ndarray, trades: jnp.ndarray, risk_aversion: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the fitness of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function, informed, signal, demand, demand_function, demand_function_params...]
    """
    returns = calculate_returns(agents, config.repetitions, trades)
    utilities = calculate_utility(agents, returns, risk_aversion)

    for i in OBJECTIVE_REGISTRY.keys():
        same_objective = jnp.where(agents[:, 1] == i)
        agents[same_objective[:, None], 0] = OBJECTIVE_REGISTRY[i](
            utilities, risk_aversion
        )

    return agents


# Need to decide the best way to calculate returns, stay with the dividend approach or change to average/last price transacted.
def calculate_returns(agents: jnp.ndarray, trades: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the returns of a set of agents
    agents should have the following columns:
    [informed, ...]
    trades should have the following shape
    (Agent, repetition, [quantity, total spendings])
    """
    returns = (
        trades[:, :, 0] * globals.market[config.benchmark_price]
        - trades[:, :, 1]
        - globals.market.cost_of_info * agents[:, 0]
    )
    return returns


OBJECTIVE_REGISTRY = {1: Mean_variance}
