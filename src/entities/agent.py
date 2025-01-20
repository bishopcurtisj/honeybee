from dataclasses import dataclass
import jax.numpy as jnp
from typing import Protocol


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
    def __call__(self, price: float, informed: int, coeffs: jnp.ndarray) -> float:
        ...

## Objective function for learning algorithm to maximize
class Objective(Protocol):
    def __call__(self) -> float:
        ...

@dataclass(frozen=True, kw_only=True, slots=True)
class Expected_return:

    def __call__(self, returns: float) -> float:
        ...


UTILITY_REGISTRY = {1: Const_abs_risk_aversion}
DEMAND_REGISTRY = {1: GS_linear}
OBJECTIVE_REGISTRY = {1: Expected_return}

