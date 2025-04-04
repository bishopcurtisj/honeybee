import numpy as jnp
import json
from typing import List


class Globals:

    agents: jnp.ndarray
    components: object
    market: object
    trades: jnp.ndarray #[quantity, price]
    informed: bool

class Config:
    uninformed_base_ratio: float
    mutation_rate: float
    crossover_rate: float
    generations: int
    repetitions: int
    GAMMA_CONSTANTS: List
    max_price: float

    def from_json(self, json_path: str):
        with open(json_path) as f:
            config = json.load(f)
        self.uninformed_base_ratio = config['uninformed_base_ratio']
        self.mutation_rate = config['mutation_rate']
        self.crossover_rate = config['crossover_rate']
        self.generations = config['generations']
        self.repetitions = config['repetitions']
        self.GAMMA_CONSTANTS = config['GAMMA_CONSTANTS']
        self.max_price = config['max_price']



globals = Globals()
config = Config()