from dataclasses import dataclass
from abc import ABC, abstractmethod
from systems.trade import OrderBook

import numpy as np



class Market(ABC):
    dividends: np.ndarray 
    price: float = 0
    last_period_price: float = 0
    repetitions: int = 100
    generations: int = 20
    supply: np.ndarray = np.zeros(repetitions)
    demand_at_p0: float = np.inf
    

    @abstractmethod
    def new_period(self):
        """Prepare the market for a new period/generation"""
        pass


class GSOrderMarket(Market):
    order_book: OrderBook = OrderBook()
    repetitions: int = 1_000
    generations: int = 5_000
    cost_of_info: float = 1
    supply: np.ndarray = np.abs(np.random.normal(1000, 10, repetitions))
    signal: np.ndarray = np.random.normal(0, 0.0004, repetitions)
    noise: np.ndarray = np.random.normal(0, 0.0004, repetitions)
    beta0: float = 0.1  # arbitary values
    beta1: float = 1.0  # arbitary values
    dividends: np.ndarray = beta0 + beta1 * signal + noise
    demand_at_p0: float = np.inf

    def new_period(self):
        self.last_period_price = self.price
        self.signal = np.random.normal(0, 1, self.repetitions)
        self.noise = np.random.normal(0, 1, self.repetitions)
        self.dividends = self.beta0 + self.beta1 * self.signal + self.noise
        self.supply = np.abs(np.random.normal(1000, 10, self.repetitions))


class RoutledgeMarket(Market):

    repetitions: int = 1_000
    generations: int = 5_000
    cost_of_info: float = 1
    supply: np.ndarray = np.abs(np.random.normal(1000, 10, repetitions))
    signal: np.ndarray = np.random.normal(0, 0.0004, repetitions)
    noise: np.ndarray = np.random.normal(0, 0.0004, repetitions)
    beta0: float = 0.1  # arbitary values
    beta1: float = 1.0  # arbitary values
    dividends: np.ndarray = beta0 + beta1 * signal + noise
    demand_at_p0: float = np.inf

    def new_period(self):
        self.last_period_price = self.price
        self.signal = np.random.normal(0, 1, self.repetitions)
        self.noise = np.random.normal(0, 1, self.repetitions)
        self.dividends = self.beta0 + self.beta1 * self.signal + self.noise
        self.supply = np.abs(np.random.normal(1000, 10, self.repetitions))


    