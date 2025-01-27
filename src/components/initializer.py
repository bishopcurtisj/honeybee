## Allows for the construction of new agents with custom parameters
import json
import os

import jax.numpy as jnp

def component_parser(components_dictionary: dict):
    ...

def initialize(agents_file_path: str, components_file_path: str) -> tuple[jnp.ndarray, dict[str, dict]]:

    components = json.loads(components_file_path)
    agents = jnp.loadtxt(agents_file_path, delimiter=',')

    return agents, components