import sys
import os

import jax.numpy as jnp

from components.initializer import initialize
from entities.market import Market
from systems.calculations import *
from systems.learning import *


class Experiment:
    def __init__(self, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json'):
        self.agents, self.components = initialize(agents_file_path, components_file_path)
        self.controller = Controller(self)
        self.market = Market()

    def run(self, generations: int = 20, repetitions: int = 100):
        pass

    def save(self):
        pass

## Handles the selection of subset to be sent to system
    

class Controller:

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def learn(self):
        pass

    def trade(self):
        pass

    def calculate_fitness(self):
        traders = jnp.where(self.experiment.agents[:,0] == 0)
        

    def update_agent_demand(self):
        traders = jnp.where(self.experiment.agents[:,0] == 0)
        if 'informed' not in self.experiment.components.keys():
            columns = [2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = [self.experiment.components['informed']['col_idx'] ,self.experiment.components['signal']['col_idx'] ,2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
        subset = traders[:, columns]
        subset = update_demands(self.experiment.market.price, subset,informed)
        #traders[:, columns] = subset
        self.experiment.agents[traders][:, columns] = subset

    def calculate_utility(self):
        pass


    