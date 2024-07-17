from dataclasses import dataclass
import numpy as np
#import warnings
from typing import Callable, Protocol, Union, NamedTuple
from .system import *
from .constants import *
from .forecasters import *
## import Prezo 

#placeholder
class Contract(NamedTuple):
    expiry: int

class Portfolio(NamedTuple):
    cash: float
    assets: dict ## {symbol: {quantity: int, info: Asset, purchase_price: float}}
    contracts: list ## List of contracts ordered by expiry

    def add_cash(self, amount: float):
        self.cash += amount
    
    def remove_cash(self, amount: float):
        self.cash -= amount

    def add_asset(self, symbol: str, quantity: int, info: Asset, purchase_price: float):
        if symbol in self.assets:
            old_quantity = self.assets[symbol]['quantity']
            self.assets[symbol]['quantity'] += quantity
            self.assets[symbol]['purchase_price'] = (purchase_price*quantity + self.assets[symbol]['purchase_price']*old_quantity)/(quantity + old_quantity)
        else:
            self.assets[symbol] = {'quantity': quantity, 'info': info, 'purchase_price': purchase_price}
        
        self.remove_cash(quantity*purchase_price)

    def remove_asset(self, symbol: str, quantity: int):
        if symbol not in self.assets:
            raise ValueError(f'Asset {symbol} not found in the portfolio')
        if self.assets[symbol]['quantity'] < quantity:
            raise ValueError(f'Not enough quantity of asset {symbol} to sell')
        
        if quantity == self.assets[symbol]['quantity']:
            self.assets.pop(symbol)
        else:
            self.assets[symbol]['quantity'] -= quantity
            self.add_cash(quantity*self.assets[symbol]['purchase_price'])
    
    def add_contract(self, contract: Contract):
        for i in range(len(self.contracts)):
            if self.contracts[i].expiry < self.contracts[i].expiry:
                self.contracts.insert(i, contract)
                return
    
    def remove_contract(self, contract: Contract):
        self.contracts.remove(contract)


class Agent:
    def __init__(self, portfolio: Portfolio, option_pricer: Callable, Forecaster: Forecasters):
        self.portfolio = portfolio
        self.forecaster = create_forecaster(Forecaster)
        self.option_pricer = option_pricer

    



