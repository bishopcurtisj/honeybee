import sys
import os

import jax.numpy as jnp

from components.initializer import initialize


class Experiment:
    def __init__(self, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json'):
        self.agents, self.components = initialize(agents_file_path, components_file_path)

    def run(self, generations: int = 20, repititions: int = 100):
        pass

    def save(self):
        pass

## Handles the selection of subset to be sent to system
    def parser(self):
        pass