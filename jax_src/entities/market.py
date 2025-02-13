from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp



class Market(ABC):
    dividends: np.ndarray 
    price: float = 0
    last_period_price: float = 0
    supply: np.ndarray = jnp.zeros(100)

    @abstractmethod
    def new_period(self):
        """Prepare the market for a new period/generation"""
        pass




class RoutledgeMarket(Market):

    cost_of_info: float = 1
    supply: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    y: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    z: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    beta0: float = 8  # arbitary values
    beta1: float = 3  # arbitary values
    dividends: jnp.ndarray = beta0 + beta1 * y + z

    def new_period(self):
        self.last_period_price = self.price
        self.y = jnp.array(np.random.normal(0, 1, 100))
        self.z = jnp.array(np.random.normal(0, 1, 100))
        self.dividends = self.beta0 + self.beta1 * self.y + self.z
        self.supply = np.random.normal(0, 1, 100)


    