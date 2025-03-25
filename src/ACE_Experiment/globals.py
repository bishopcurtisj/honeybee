import numpy as jnp

from entities.agent import AgentInfo
from entities.market import Market


class Globals:

    agents: jnp.ndarray
    components: AgentInfo
    market: Market
    trades: jnp.ndarray
    informed: bool

globals = Globals()