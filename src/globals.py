import json
from typing import List

import numpy as jnp

from entities.agent import AgentInfo


class Globals:

    agents: jnp.ndarray
    components: AgentInfo
    market: object
    trades: jnp.ndarray  # [quantity, price, difference from benchmark (mean/last)]
    informed: bool
    generation: int
    repetition: int

    def __getitem__(self, key):
        """Allow dict-like access."""
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int) and 0 <= key < len(self._columns):
            return self._columns[key]
        elif isinstance(key, list):
            attributes = []
            for k in key:
                attributes.append(self.__getitem__(k))
                return tuple(attributes)
        raise KeyError(f"Invalid key: {key}")


class Config:
    uninformed_base_ratio: float = 0
    mutation_rate: float = 0
    crossover_rate: float = 0
    generations: int = 2
    repetitions: int = 100
    GAMMA_CONSTANTS: List = [0, 0]
    max_price: float = 1000
    memory_optimization: bool = True
    save_models: bool = True
    bootstraps: int = 10
    benchmark_price: str = "mean_price"  # Should map to a field of Market

    def from_json(self, json_path: str):
        with open(json_path) as f:
            config_dict = json.load(f)
        self.__dict__.update(config_dict)


globals = Globals()
config = Config()
