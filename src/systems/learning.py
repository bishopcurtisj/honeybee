import numpy as jnp
import numpy.random as random
import pickle
import numpy.random as random

from entities.agent import AgentInfo
from systems.models.genetic_algorithm import GeneticAlgorithm
from systems.models.neural_network import NeuralNetwork
from systems.models.thompson_sampler import ThompsonSampler

class ModelController:
    
    def init_models(self, agents: jnp.ndarray, components: AgentInfo):
        self.ga_learners = jnp.where(agents[:,components.learning_algorithm] == 1)[0]
        self.ts_learners = jnp.where(agents[:,components.learning_algorithm] == 2)[0]
        self.nn_learners = jnp.where(agents[:,components.learning_algorithm] == 3)[0]
        self.genetic_algorithm = GeneticAlgorithm()
        self.thompson_sampler = ThompsonSampler()
        self.neural_network = NeuralNetwork(agents[self.nn_learners], components)
        self.models = [self.genetic_algorithm, self.thompson_sampler, self.neural_network]

    


    def learn(self, agents: jnp.ndarray, components: AgentInfo, informed: bool = True) -> jnp.ndarray:
        agents[self.ga_learners] = self.genetic_algorithm(agents[self.ga_learners], agents[self.ga_learners][components.learning_params], len(agents), informed)
        agents[self.ts_learners] = self.thompson_sampler(agents[self.ts_learners], agents[self.ts_learners][components.learning_params], informed)
        agents = self.neural_network(agents, self.nn_learners)

        return agents

model_controller = ModelController()