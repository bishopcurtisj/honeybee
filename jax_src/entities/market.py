from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np



class Market(ABC):
    dividends: np.ndarray 
    price: float = 0
    last_period_price: float = 0
    supply: np.ndarray = np.zeros(100)
    demand_at_p0: float = np.inf

    @abstractmethod
    def new_period(self):
        """Prepare the market for a new period/generation"""
        pass




class RoutledgeMarket(Market):

    cost_of_info: float = 1
    supply: np.ndarray = np.abs(np.random.normal(0, 1, 100))*100
    signal: np.ndarray = np.random.normal(0, 1, 100)
    noise: np.ndarray = np.random.normal(0, 1, 100)
    beta0: float = 8  # arbitary values
    beta1: float = 3  # arbitary values
    dividends: np.ndarray = beta0 + beta1 * signal + noise
    demand_at_p0: float = np.inf

    def new_period(self):
        self.last_period_price = self.price
        self.y = np.random.normal(0, 1, 100)
        self.z = np.random.normal(0, 1, 100)
        self.dividends = self.beta0 + self.beta1 * self.signal + self.noise
        self.supply = np.abs(np.random.normal(0, 1, 100))


    