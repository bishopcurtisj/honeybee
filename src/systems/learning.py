from dataclasses import dataclass
import jax.numpy as jnp
from typing import Protocol

class LearningFunction(Protocol):
    def __call__(self) -> jnp.ndarray:
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class GeneticAlgorithm:

    def __call__(self) -> jnp.ndarray:
        pass

LEARNING_REGISTRY = {1: GeneticAlgorithm}