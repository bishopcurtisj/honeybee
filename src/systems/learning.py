from typing import List, Union

import jax.numpy as jnp
from globals import globals
from systems.models.bayesian import Bayesian
from systems.models.genetic_algorithm import GeneticAlgorithm
from systems.models.model import Model
from systems.models.neural_network import NeuralNetwork


class ModelController:
    def init_models(self):
        neural_network = NeuralNetwork(
            globals.agents[
                jnp.where(
                    globals.agents[:, globals.components.learning_algorithm] == 3
                )[0]
            ]
        )
        genetic_algorithm = GeneticAlgorithm(
            globals.agents[
                jnp.where(
                    globals.agents[:, globals.components.learning_algorithm] == 1
                )[0]
            ]
        )
        bayesian = Bayesian()
        self.model_registry = {
            1: {"func": genetic_algorithm, "name": "genetic_algorithm"},
            2: {"func": bayesian, "name": "Bayesian"},
            3: {"func": neural_network, "name": "neural_network"},
        }

    def save_models(self):
        for model in self.model_registry.values():
            model["func"].save()

    def register_models(self, models: Union[List[Model], Model]):

        if type(models) == Model:
            models = [models]
        for model in models:
            try:
                assert issubclass(model, Model)
            except AssertionError:
                raise ValueError(
                    f"Custom learning function {model.label} must be a subclass of Model"
                )
            self.model_registry[model.label] = {
                "func": model(
                    globals.agents[
                        jnp.where(
                            globals.agents[:, globals.components.learning_algorithm]
                            == model["id"]
                        )[0]
                    ],
                ),
                "id": len(self.model_registry),
            }

    def learn(self) -> jnp.ndarray:

        for i in globals.learning_algorithm:
            model = self.model_registry[i]
            model_agents = jnp.where(
                globals.agents[:, globals.components.learning_algorithm] == i
            )[0]
            globals.agents[model_agents] = model["func"](globals.agents[model_agents])


model_controller = ModelController()
