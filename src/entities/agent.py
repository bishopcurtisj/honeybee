from dataclasses import dataclass
import jax.numpy as jnp
from typing import Protocol

from constants import *


## Utility function for agents to determine how much they value gains relative to risk
class Utility(Protocol):
    def __call__(self) -> float:
        ...

@dataclass(frozen=True, kw_only=True, slots=True)
class Const_abs_risk_aversion:

    def __call__(self, risk_aversion: float, returns: float) -> float:
        return -jnp.exp(-risk_aversion * returns)


## Demand function for agents to determine how much of a good to buy
class Demand(Protocol):
    def __call__(self) -> float:
        ...

@dataclass(frozen=True, kw_only=True, slots=True)
class GS_linear:
    ## If uninformed signal == price
    def __call__(self, price: float, coeffs: jnp.ndarray, signal, scaling_factor: float = None) -> float:
        return scaling_factor * (coeffs[0] + coeffs[1] * signal - price)


## Objective function for learning algorithm to maximize
class Objective(Protocol):
    def __call__(self) -> float:
        ...

@dataclass(frozen=True, kw_only=True, slots=True)
class Mean_variance:

    def __call__(self, returns: float, risk_aversion: float) -> float:
        return jnp.mean(returns) - jnp.var(returns)*risk_aversion/2


UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
DEMAND_REGISTRY = {1: GS_linear}
OBJECTIVE_REGISTRY = {1: Mean_variance}

