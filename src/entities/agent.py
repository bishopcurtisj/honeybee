from dataclasses import dataclass
import numpy as jnp
from jax.scipy.stats import norm
from jax.random import geometric
from abc import ABC, abstractmethod
from typing import List, Union
from ACE_Experiment.globals import globals, config
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
    def __call__(price: Union[float, jnp.ndarray], bid: bool, params: jnp.ndarray, key = None) -> float:
        if type(price) == float:
            prob = (price - params[0]) / params[1]
            if bid:
                return geometric(key,1 - prob)
            else:
                return geometric(key, prob)
        ## Implement logic for when multiple agents are passed



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
            df = DEMAND_REGISTRY[int(agents[i,6])]()
            demand = df(price, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

            while jnp.allclose(demand, 0, atol=1e-2) == False:
                if demand > 0:
                    min_price = price
                else:
                    max_price = price
                price = (max_price + min_price) / 2
                demand = df(price, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

            bid = price - agents[i,7]
            ask = price + agents[i,7]
            agents[i,2] = bid
            agents[i,3] = ask
            agents[i,4] = df(bid, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])
            agents[i,5] = df(ask, agents[i,8:], agents[i,1], config.GAMMA_CONSTANTS[int(agents[i,0])])

        return agents

    def _uninformed_spread(agents: jnp.ndarray) -> jnp.ndarray:

        min_price = 0
        max_price = config.max_price
        price = (max_price + min_price) / 2

        for i in range(len(agents)):
            df = DEMAND_REGISTRY[int(agents[i,4])]()
            demand = df(price, agents[i,6:])

            while demand != 0:
                price = price + 0.01
                demand = df(price, agents[i,6:])
            bid = price - agents[i,5]
            ask = price + agents[i,5]
            agents[i,0] = bid
            agents[i,1] = ask
            agents[i,2] = df(bid, agents[i,6:])
            agents[i,3] = df(ask, agents[i,6:])
                
        return agents
    
class BayesianSpread(Spread):
    label = "BayesianSpread"
    @staticmethod
    def __call__(agents: jnp.ndarray) -> float:
        """
        Calculate the bid-ask spread of a set of agents
        agents should have the following columns:
        [bid, ask, bid_quantity, ask_quantity, demand_function, confidence, demand_function_params...]
        """
        p = agents[:,5] / 2
        agents[:, 0] = agents[:, 6] + agents[:, 7] * norm.ppf(p)
        agents[:, 1] = agents[:, 6] + agents[:, 7] * norm.ppf(1-p) 
        ## Need to refactor to accurately use differnt demand functions
        for i in DEMAND_REGISTRY.keys():
            same_demand = jnp.where(agents[4]==i)
            agents[same_demand][2] = DEMAND_REGISTRY[i](agents[same_demand][0], True, agents[same_demand][6:])
            agents[same_demand][3] = DEMAND_REGISTRY[i](agents[same_demand][0], False, agents[same_demand][6:])
        return agents
        


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



def calculate_fitness(agents: jnp.ndarray, repetitions: int, trades: jnp.ndarray, risk_aversion: jnp.ndarray, market: Market) -> jnp.ndarray:
    """
    Calculate the fitness of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function, informed, signal, demand, demand_function, demand_function_params...]"""
    returns = calculate_returns(agents, market, repetitions, trades)
    utilities = calculate_utility(agents, returns, risk_aversion)
    
    for i in range(len(agents)):
        objective_function = OBJECTIVE_REGISTRY[int(agents[i,1])]()
        agents[i, 0] = objective_function(utilities[i], risk_aversion[i])

    return agents

def calculate_returns(agents: jnp.ndarray, market: Market, repetition: int, trades: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the returns of a set of agents
    agents should have the following columns:
    [informed, ...]
    trades should have the following columns:
    [quantity, total spendings]
    """

    if globals.informed:
        returns = trades[:, 0] * market.dividends[repetition] - trades[:,1] - market.cost_of_info * agents[:,0]
    else:
        returns = trades[:, 0] * market.dividends[repetition] - trades[:,1]
    return returns

def calculate_utility(agents: jnp.ndarray, returns: jnp.ndarray, risk_aversion: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the utility of a set of agents
    agents should have the following columns:
    [fitness, objective_function, utility_function]
    returns should have the following columns:
    [returns]
    """
    utilities = jnp.zeros((len(agents), len(returns[0]))) 

    for i in range(len(agents)):
        utility_function = UTILITY_REGISTRY[int(agents[i,2])]()
        utilities[i] = utility_function(returns[i], risk_aversion[i])

    return utilities



UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
DEMAND_REGISTRY = {1: GS_linear, 2: BayesianDemand}
OBJECTIVE_REGISTRY = {1: Mean_variance}