
from collections import deque
from sortedcontainers import SortedDict
import numpy as jnp

from entities.agent import *


class Order:
    __slots__ = ("trader_id", "price", "quantity")

    def __init__(self, trader_id: str, price: float, quantity: float):
        # self.order_id = uuid.uuid4()  # Unique order ID
        self.trader_id = trader_id
        self.price = price
        self.quantity = quantity
        # self.timestamp = timestamp  # Order arrival time

    def __repr__(self):
        return (
            f"Order(id={self.order_id}, trader={self.trader_id}, "
            f"price={self.price}, qty={self.quantity})"
        )

class OrderBook:
    """Order book for a single asset.
    Orders are stored in two dictionaries, one for buy orders and one for sell orders where prices are the keys and the valuers are deques.
    
    """

    def __init__(self):
        self.buy_orders = SortedDict(lambda x: -x)  # Max-Heap behavior for buy orders
        self.sell_orders = SortedDict()  # Min-Heap behavior for sell orders
        self.agent_trades = {}

    def reset(self):
        self.buy_orders = SortedDict(lambda x: -x)  # Max-Heap behavior for buy orders
        self.sell_orders = SortedDict()  # Min-Heap behavior for sell orders
        self.agent_trades = {}

    def add_order(self, trader_id: str, price: float, quantity: float):
        order = Order(trader_id, price, quantity)

        order = self.match_orders(order)
        if order.quantity == 0:
            return

        if order.quantity < 0:
            # sell order
            if order.price not in self.sell_orders:
                self.sell_orders[order.price] = deque([order])
            else:
                self.sell_orders[order.price].append(order)
        else:
            # buy order
            if order.price not in self.buy_orders:
                self.buy_orders[order.price] = deque([order])
            else:
                self.buy_orders[order.price].append(order)
    

    
    def match_orders(self, order: Order) -> Order:

        best_bid = -next(iter(self.buy_orders))
        best_ask = next(iter(self.sell_orders))

        if order.quantity < 0:
            # sell order
            while order.quantity < 0 and order.price <= best_bid:
                if order.quantity >= self.buy_orders[best_bid][0].quantity:
                    order.quantity += self.buy_orders[best_bid][0].quantity
                    self.agent_trades[self.buy_orders[best_bid][0].trader_id].append((self.buy_orders[best_bid][0].quantity, best_bid))
                    self.agent_trades[order.trader_id].append((-self.buy_orders[best_bid][0].quantity, best_bid))
                    self.buy_orders[best_bid].popleft()
                else:
                    self.buy_orders[best_bid][0].quantity += order.quantity
                    self.agent_trades[self.buy_orders[best_bid][0].trader_id].append((order.quantity, best_bid))
                    self.agent_trades[order.trader_id].append((-order.quantity, best_bid))
                    order.quantity = 0
                self.last_price = best_bid
                
        else:
            # buy order
            while order.quantity > 0 and order.price >= best_ask:
                if order.quantity >= self.sell_orders[best_ask][0].quantity:
                    order.quantity -= self.sell_orders[best_ask][0].quantity
                    self.agent_trades[self.sell_orders[best_ask][0].trader_id].append((-self.sell_orders[best_ask][0].quantity, best_ask))
                    self.agent_trades[order.trader_id].append((self.sell_orders[best_ask][0].quantity, best_ask))
                    self.sell_orders[best_ask].popleft()
                else:
                    self.sell_orders[best_ask][0].quantity -= order.quantity
                    self.agent_trades[self.sell_orders[best_ask][0].trader_id].append((-order.quantity, best_ask))
                    self.agent_trades[order.trader_id].append((order.quantity, best_ask))
                    order.quantity = 0
                self.last_price = best_ask

        return order
    
    def get_agent_trades(self):

        for key in self.agent_trades.keys():
            self.agent_trades[key] = jnp.array(self.agent_trades[key])

        return self.agent_trades
    
    ## Vectorize this later
    def get_trades(self):
        """
        Get all trades in the order book 
        Trades has the form:
        [[quantity, price], [quantity, price], ...]  
        """
        trades = []
        for agent_trades in self.agent_trades.values():
            for trade in agent_trades:
                trades.append(jnp.array(trade))

        return trades