import numpy as jnp

from entities.agent import UTILITY_REGISTRY


def calculate_trade_utility(trades: jnp.ndarray, utility_function: int) -> jnp.ndarray:
    """
    Calculate the utility of each trade
    Currently uses the average transacted price to calculate return from trade.
    """
    average_price = jnp.mean(trades[:, 1])
    returns = trades[:, 0] * (trades[:, 1] - average_price)
    utility_function = UTILITY_REGISTRY[utility_function]()
    return utility_function(returns)
