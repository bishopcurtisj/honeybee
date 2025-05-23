from collections import deque
from itertools import chain

from sortedcontainers import SortedDict

import jax.numpy as jnp
from globals import globals


class Order:
    __slots__ = ("trader_id", "price", "quantity")

    def __init__(self, trader_id: str, price: float, quantity: float):
        self.trader_id = trader_id
        self.price = price
        self.quantity = quantity

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
        if trader_id not in self.agent_trades.keys():
            self.agent_trades[trader_id] = []

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

        # need to fix
        try:
            best_bid = next(iter(self.buy_orders))
        except StopIteration:
            best_bid = -1
        try:
            best_ask = next(iter(self.sell_orders))
        except StopIteration:
            best_ask = jnp.inf

        if order.quantity < 0:
            # sell order
            while order.quantity < 0 and order.price <= best_bid:
                if order.quantity >= self.buy_orders[best_bid][0].quantity:
                    order.quantity += self.buy_orders[best_bid][0].quantity
                    self.agent_trades[self.buy_orders[best_bid][0].trader_id].append(
                        (self.buy_orders[best_bid][0].quantity, best_bid)
                    )
                    self.agent_trades[order.trader_id].append(
                        (-self.buy_orders[best_bid][0].quantity, best_bid)
                    )
                    self.buy_orders[best_bid].popleft()
                    if len(self.buy_orders[best_bid]) == 0:
                        del self.buy_orders[best_bid]
                        try:
                            best_bid = next(iter(self.buy_orders))
                        except StopIteration:
                            best_bid = -1
                else:
                    self.buy_orders[best_bid][0].quantity += order.quantity
                    self.agent_trades[self.buy_orders[best_bid][0].trader_id].append(
                        (order.quantity, best_bid)
                    )
                    self.agent_trades[order.trader_id].append(
                        (-order.quantity, best_bid)
                    )
                    order.quantity = 0
                self.last_price = best_bid

        else:
            # buy order
            while order.quantity > 0 and order.price >= best_ask:
                if order.quantity >= self.sell_orders[best_ask][0].quantity:
                    order.quantity += self.sell_orders[best_ask][0].quantity
                    self.agent_trades[self.sell_orders[best_ask][0].trader_id].append(
                        (-self.sell_orders[best_ask][0].quantity, best_ask)
                    )
                    self.agent_trades[order.trader_id].append(
                        (self.sell_orders[best_ask][0].quantity, best_ask)
                    )
                    self.sell_orders[best_ask].popleft()
                    if len(self.sell_orders[best_ask]) == 0:
                        del self.sell_orders[best_ask]
                        try:
                            best_ask = next(iter(self.sell_orders))
                        except StopIteration:
                            best_ask = jnp.inf
                else:
                    self.sell_orders[best_ask][0].quantity -= order.quantity
                    self.agent_trades[self.sell_orders[best_ask][0].trader_id].append(
                        (-order.quantity, best_ask)
                    )
                    self.agent_trades[order.trader_id].append(
                        (order.quantity, best_ask)
                    )
                    order.quantity = 0
                self.last_price = best_ask

        return order

    def get_agent_trades(self):

        for key in self.agent_trades.keys():
            self.agent_trades[key] = jnp.array(self.agent_trades[key])

        return self.agent_trades

    def get_trades(self):
        """
        Get all trades in the order book
        Trades has the form:
        [[quantity, price], [quantity, price], ...]
        """
        trades = list(chain.from_iterable(self.agent_trades.values()))
        trades = jnp.array(trades)
        if len(trades) == 0:
            print(
                f"No trades occured in repetition: {globals.repetition} of generation: {globals.generation}"
            )
            return jnp.array([0, 0, 0])

        mean_price = jnp.mean(trades[:, 1])
        returns = (trades[:, 1] - mean_price) * (
            -1 * trades[:, 0] / jnp.abs(trades[:, 0])
        )
        trades = jnp.hstack((trades, returns[:, jnp.newaxis]))

        return trades
