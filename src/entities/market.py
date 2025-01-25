from typing import NamedTuple
import numpy as np
import jax.numpy as jnp

class Market(NamedTuple):
    price: float = 0
    supply: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    cost_of_info: float = 0
    beta0: float = 8  # arbitray values
    beta1: float = 3  # arbitray values
    y: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    z: jnp.ndarray = jnp.array(np.random.normal(0, 1, 100))
    dividends: jnp.ndarray = beta0 + beta1 * y + z