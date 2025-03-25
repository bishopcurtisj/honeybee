from typing import List, Union

import numpy as jnp
import numpy.random as random

from ACE_Experiment.globals import globals
from entities.agent import AgentInfo
from systems.models.model import Model
from systems.models.genetic_algorithm import GeneticAlgorithm
from systems.models.neural_network import NeuralNetwork
from systems.models.thompson_sampler import thompson_sampler




class ModelController:
    
    def init_models(self):
        neural_network = NeuralNetwork(globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == 3)[0]], globals.components)
        genetic_algorithm = GeneticAlgorithm(globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == 1)[0]], globals.components)
        self.model_registry = {'genetic_algorithm': {'func': genetic_algorithm, 'id': 1}, 'thompson_sampler': {'func': thompson_sampler, 'id': 2}, 'neural_network': {'func': neural_network, 'id': 3}}
    


    def register_models(self, models: Union[List[Model], Model]):
    
        if type(models) == Model:
            models = [models]
        for model in models:
            try:
                assert issubclass(model, Model)
            except AssertionError:
                raise ValueError(f"Custom learning function {model.label} must be a subclass of Model")
            self.model_registry[model.label] = {'func': model(globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == model['id'])[0]], globals.components), 'id': len(self.model_registry), 'args': model.args}
            

    def learn(self) -> jnp.ndarray:
        
        for model in self.model_registry.values():
            globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == model['id'])[0]] = model['func'](globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == model['id'])[0]], globals.agents[jnp.where(globals.agents[:,globals.components.learning_algorithm] == model['id'])[0]][globals.components.learning_params])


model_controller = ModelController()