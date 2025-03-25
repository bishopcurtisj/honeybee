from dataclasses import dataclass
import numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Union


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
    @abstractmethod
    def __call__(*args, **kwargs) -> float:
        ...

class Const_abs_risk_aversion(Utility):

    label = "Const_abs_risk_aversion"
    @staticmethod
    def __call__(risk_aversion: float, returns: float) -> float:
        return -jnp.exp(-risk_aversion * returns)


## Demand function for agents to determine how much of a good to buy
class Demand(ABC):

    label: str
    @abstractmethod
    def __call__(*args, **kwargs) -> float:
        ...

class GS_linear(Demand):
    label = "GS_linear"
    ## If uninformed signal == price
    @staticmethod
    def __call__(price: float, coeffs: jnp.ndarray, signal, scaling_factor: float = None) -> float:
        return scaling_factor * (coeffs[0] + coeffs[1] * signal - price)


## Objective function for learning algorithm to maximize
class Objective(ABC):
    label: str
    @abstractmethod
    def __call__(*args, **kwargs) -> float:
        ...

class Mean_variance(Objective):
    label = "Mean_variance"

    @staticmethod
    def __call__(returns: float, risk_aversion: float) -> float:
        return jnp.mean(returns) - jnp.var(returns)*risk_aversion/2
    
class Spread(ABC):
    label: str
    @abstractmethod
    def __call__(*args, **kwargs) -> float:
        ...

class LinearDemandSpread(Spread):
    label = "LinearDemandSpread"
    @staticmethod
    def __call__() -> float:
        return 0.01


UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
DEMAND_REGISTRY = {1: GS_linear}
OBJECTIVE_REGISTRY = {1: Mean_variance}

def register_utility_function(id: int, utility_functions: Union[List[Utility], Utility]):
    if type(utility_functions) == Utility:
        utility_functions = [utility_functions]
    for utility_function in utility_functions:
        try:
            assert issubclass(utility_function, Utility)
        except AssertionError:
            raise ValueError(f"Custom utility function {utility_function.label} must be a subclass of Utility")
        UTILITY_REGISTRY[id] = utility_function
    
def register_demand_function(id: int, demand_functions: Union[List[Demand], Demand]):
    if type(demand_functions) == Demand:
        demand_functions = [demand_functions]
    for demand_function in demand_functions:
        try:
            assert issubclass(demand_function, Demand)
        except AssertionError:
            raise ValueError(f"Custom demand function {demand_function.label} must be a subclass of Demand")
        DEMAND_REGISTRY[id] = demand_function

def register_objective_function(id: int, objective_functions: Union[List[Objective], Objective]):
    if type(objective_functions) == Objective:
        objective_functions = [objective_functions]
    for objective_function in objective_functions:  
        try:
            assert issubclass(objective_function, Objective)
        except AssertionError:
            raise ValueError(f"Custom objective function {objective_function.label} must be a subclass of Objective")
        OBJECTIVE_REGISTRY[id] = objective_function
