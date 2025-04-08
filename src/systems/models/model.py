from abc import ABC, abstractmethod

import numpy as jnp


class Model(ABC):
    """
    Abstract class so that custom learning functions can be implemented
    """

    label: str

    @abstractmethod
    def __call__(self, agents: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        pass
