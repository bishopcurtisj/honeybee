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


class Config:
    uninformed_base_ratio: float
    mutation_rate: float
    crossover_rate: float
    generations: int
    repetitions: int
    GAMMA_CONSTANTS: List
    max_price: float
    memory_optimization: bool = True
    save_models: bool = True
    bootstraps: int
    benchmark_price: str  # Should map to a field of Market

    def from_json(self, json_path: str):
        with open(json_path) as f:
            config_dict = json.load(f)
        self.__dict__.update(config_dict)


globals = Globals()
config = Config()
