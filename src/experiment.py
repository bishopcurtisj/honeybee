import sys
import os

import jax.numpy as jnp

from components.initializer import initialize


class Experiment:
    def __init__(self, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json'):
        self.agents, self.components = initialize(agents_file_path, components_file_path)

    def run(self):
        pass