

import jax.numpy as jnp

from components.initializer import initialize
from entities.market import Market
from systems.calculations import *
from systems.learning import *
from systems.trade import *


class Experiment:
    def __init__(self, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json', market: Market = None):
        self.agents, self.components = initialize(agents_file_path, components_file_path)
        if market is None: 
            self.market = Market()
        else:
            self.market = market
        if 'informed' in self.components.keys():
            self.informed = self.components['informed']['col_idx']
            self.agents[:, self.components['signal']['col_idx']] = jnp.where(self.agents[:, self.informed] == 0, self.market.price, self.market.y[0])

        else:
            self.informed = -1
        self.controller = Controller(self)
        


    def run(self, generations: int = 20, repetitions: int = 100):
        
        ## TODO: reset market each generation
        for i in range(generations):
            self.controller.update_agent_demand()
            self.controller.calculate_agent_fitness(repetitions)
            self.controller.learn()
            self.controller.trade()
        
        self.save()
        return self.agents


    ## Need to test this to verify how write works with jnp arrays
    def save(self):
        with open('./assets/results.csv', 'w') as f:
            f.write(self.agents)

## Handles the selection of subset to be sent to system
    

class Controller:

    def __init__(self, experiment: Experiment):
        self.experiment = experiment


    def learn(self):
        traders = jnp.where(self.experiment.agents[:,0] == 0)
        if self.experiment.informed == -1:
            columns = [4].extend(self.experiment.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = [4, self.experiment.informed].extend(self.experiment.components['demand_function']['parameter_idxs'])

        params = traders[:, self.experiment.components['learning_algorithm']['parameter_idxs']]
        subset = traders[:, columns]
        subset = GeneticAlgorithm(subset, params, len(subset), informed)

        self.experiment.agents[traders][:, columns] = subset
        self.experiment.market.last_period_price = self.experiment.market.price
        self.experiment.agents = jnp.where(self.experiment.agents[:, self.experiment.informed]== 0, self.experiment.market.price, self.experiment.agents[:,self.experiment.components['signal']['col_idx']])


    def trade(self):
        pass

    def calculate_agent_fitness(self, repetitions: int):
        traders = jnp.where(self.experiment.agents[:,0] == 0)
        if self.experiment.informed == -1:
            columns = [4, 5, 2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = [4, 5, self.experiment.informed ,self.experiment.components['signal']['col_idx'] ,2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
        subset = traders[:, columns]
        subset = calculate_fitness(subset, repetitions, traders[:, 8], self.experiment.market, informed)
        #traders[:, columns] = subset
        self.experiment.agents[traders][:, columns] = subset
        

    def update_agent_demand(self):
        traders = jnp.where(self.experiment.agents[:,0] == 0)
        if self.experiment.informed == -1:
            columns = [2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = [self.experiment.informed ,self.experiment.components['signal']['col_idx'] ,2, 3].extend(self.experiment.components['demand_function']['parameter_idxs'])
        subset = traders[:, columns]
        subset = update_demands(self.experiment.market.price, subset,informed)
        #traders[:, columns] = subset
        self.experiment.agents[traders][:, columns] = subset

    def calculate_utility(self):
        pass


    