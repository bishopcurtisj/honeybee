from abc import ABC, abstractmethod

import numpy as jnp
from entities.agent import AgentInfo

class Model(ABC):
    """
    Abstract class so that custom learning functions can be implemented
    """
    label: str
    args: dict
    @abstractmethod
    def __init__(self, agents: jnp.ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, agents: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        pass