from typing import List, Union

import numpy as jnp

from ACE_Experiment.globals import globals
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
        self.model_registry = {
            "genetic_algorithm": {"func": genetic_algorithm, "id": 1},
            "thompson_sampler": {"func": Bayesian, "id": 2},
            "neural_network": {"func": neural_network, "id": 3},
        }

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
                    globals.components,
                ),
                "id": len(self.model_registry),
                "args": model.args,
            }

    def learn(self) -> jnp.ndarray:

        for model in self.model_registry.values():
            model_agents = jnp.where(
                globals.agents[:, globals.components.learning_algorithm] == model["id"]
            )[0]
            globals.agents[model_agents[:, None]] = model["func"](
                globals.agents[model_agents]
            )


model_controller = ModelController()
