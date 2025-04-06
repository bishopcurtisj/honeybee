from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import numpy as jnp
from jax.random import geometric
from jax.scipy.stats import norm

from ACE_Experiment.globals import config, globals
from entities.market import Market


class AgentInfo:
    """Dataclass that stores the column associated with each agent attribute"""

    def __init__(self, columns):
        self._columns = columns  # Store original column names
        demand_params = []
        learning_params = []
        for index, name in enumerate(columns):
            if "dfx_" in name:
                demand_params.append(index)
            elif "la_" in name:
                learning_params.append(index)
            else:
                setattr(self, name, index)
        self.demand_fx_params = demand_params
        self.learning_params = learning_params

    def __getitem__(self, key):
        """Allow dict-like access."""
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int) and 0 <= key < len(self._columns):
            return self._columns[key]
        raise KeyError(f"Invalid key: {key}")

    def add(self, name, index):
        """Add a new column."""
        if name in self._columns:
            raise ValueError(f"Column {name} already exists.")

        setattr(self, name, index)
        self._columns.append(name)

    def __repr__(self):
        """Readable representation."""
        return f"ColumnIndexer({self._columns})"

    def keys(self):
        """Return column names."""
        return self._columns

    def values(self):
        """Return column indices."""
        return list(range(len(self._columns)))

    def items(self):
        """Return (column_name, index) pairs."""
        return zip(self._columns, self.values())

    def __iter__(self):
        """Iterate over column names."""
        return iter(self._columns)


## Utility function for agents to determine how much they value gains relative to risk
class Utility(ABC):
    label: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class Const_abs_risk_aversion(Utility):
    label = "Const_abs_risk_aversion"

    @staticmethod
    def __call__(returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
        return -jnp.exp(-risk_aversion * returns)


## Demand function for agents to determine how much of a good to buy
class Demand(ABC):
    label: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class GS_linear(Demand):
    label = "GS_linear"

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

    label = "BayesianDemand"

    @staticmethod
    def __call__(price: jnp.ndarray, bid: bool, params: jnp.ndarray, key=None) -> float:
        prob = (price - params[:, 0]) / params[:, 1]

        if bid:
            return geometric(key, 1 - prob)
        else:
            return geometric(key, prob)


## Objective function for learning algorithm to maximize
class Objective(ABC):
    label: str

    @staticmethod
    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class Mean_variance(Objective):
    label = "Mean_variance"

    @staticmethod
    def __call__(utilities: jnp.ndarray, risk_aversion: float) -> float:
        return jnp.mean(utilities) - jnp.var(utilities) * risk_aversion / 2


class Spread(ABC):
    label: str

    @abstractmethod
    def __call__(*args, **kwargs) -> float: ...


class LinearDemandSpread(Spread):
    label = "LinearDemandSpread"

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
    label = "BayesianSpread"

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


def register_utility_function(utility_functions: Union[List[Utility], Utility]):
    if type(utility_functions) == Utility:
        utility_functions = [utility_functions]
    for utility_function in utility_functions:
        try:
            assert issubclass(utility_function, Utility)
        except AssertionError:
            raise ValueError(
                f"Custom utility function {utility_function.label} must be a subclass of Utility"
            )
        UTILITY_REGISTRY[len(UTILITY_REGISTRY)] = utility_function


def register_demand_function(demand_functions: Union[List[Demand], Demand]):
    if type(demand_functions) == Demand:
        demand_functions = [demand_functions]
    for demand_function in demand_functions:
        try:
            assert issubclass(demand_function, Demand)
        except AssertionError:
            raise ValueError(
                f"Custom demand function {demand_function.label} must be a subclass of Demand"
            )
        DEMAND_REGISTRY[len(DEMAND_REGISTRY)] = demand_function


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
                f"Custom objective function {objective_function.label} must be a subclass of Objective"
            )
        OBJECTIVE_REGISTRY[len(OBJECTIVE_REGISTRY)] = objective_function


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
                f"Custom spread function {spread_function.label} must be a subclass of Objective"
            )
        SPREAD_REGISTRY[len(SPREAD_REGISTRY)] = spread_function


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
        trades[:, :, 0] * globals.market.mean_price
        - trades[:, :, 1]
        - globals.market.cost_of_info * agents[:, 0]
    )
    return returns


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


UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
DEMAND_REGISTRY = {1: GS_linear, 2: BayesianDemand}
OBJECTIVE_REGISTRY = {1: Mean_variance}
SPREAD_REGISTRY = {1: LinearDemandSpread, 2: BayesianSpread}
