from typing import List, Union

import numpy as jnp
import numpy.random as random

from entities.agent import AgentInfo
from systems.models.model import Model
from systems.models.genetic_algorithm import GeneticAlgorithm
from systems.models.neural_network import NeuralNetwork
from systems.models.thompson_sampler import thompson_sampler




class ModelController:
    
    def init_models(self, agents: jnp.ndarray, components: AgentInfo):
        neural_network = NeuralNetwork(agents[jnp.where(agents[:,components.learning_algorithm] == 3)[0]], components)
        genetic_algorithm = GeneticAlgorithm(agents[jnp.where(agents[:,components.learning_algorithm] == 1)[0]], components)
        self.model_registry = {'genetic_algorithm': {'func': genetic_algorithm, 'id': 1}, 'thompson_sampler': {'func': thompson_sampler, 'id': 2}, 'neural_network': {'func': neural_network, 'id': 3}}
    


    def register_models(self, agents: jnp.ndarray, components: AgentInfo, models: Union[List[Model], Model]):
    
        if type(models) == Model:
            models = [models]
        for model in models:
            try:
                assert issubclass(model, Model)
            except AssertionError:
                raise ValueError(f"Custom learning function {model.label} must be a subclass of Model")
            self.model_registry[model.label] = {'func': model(agents[jnp.where(agents[:,components.learning_algorithm] == model['id'])[0]], components), 'id': len(self.model_registry), 'args': model.args}
            

    def learn(self, agents: jnp.ndarray, components: AgentInfo, trades: jnp.ndarray, informed: bool = True) -> jnp.ndarray:
        
        for model in self.model_registry.values():
            agents[jnp.where(agents[:,components.learning_algorithm] == model['id'])[0]] = model['func'](agents[jnp.where(agents[:,components.learning_algorithm] == model['id'])[0]], agents[jnp.where(agents[:,components.learning_algorithm] == model['id'])[0]][components.learning_params], trades, informed)

        return agents

model_controller = ModelController()