import json

import jax.numpy as jnp
import numpy as np

from entities.market import Market
from systems.calculations import *
from systems.learning import *
from systems.trade import *




class Experiment:
    def __init__(self, market: Market, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json'):
        
        with open(components_file_path) as f:
            self.components = json.load(f)
        self.agents = jnp.array(np.loadtxt(agents_file_path, delimiter=',', skiprows=1))
        self.market = market
        
        if 'informed' in self.components.keys():
            self.informed = self.components['informed']['col_idx']
            self.agents=self.agents.at[:, self.components['signal']['col_idx']].set(jnp.where(self.agents[:, self.informed] == 0, self.market.price, self.market.y[0]))

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
        learning_function = LEARNING_REGISTRY[self.experiment.components['learning_algorithm']['name']]
        subset = learning_function(subset, params, len(subset), informed)

        self.experiment.agents[traders][:, columns] = subset
        self.experiment.market.new_period()
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
            columns = jnp.array([2, 3].extend(self.experiment.components['demand_function']['parameter_idxs']))
            informed = False
        else:
            informed = True
            columns = jnp.array([self.experiment.informed ,self.experiment.components['signal']['col_idx'] ,2, 3].extend(self.experiment.components['demand_function']['parameter_idxs']))
        subset = traders[:, columns]
        subset = update_demands(self.experiment.market.price, subset,informed)
        #traders[:, columns] = subset
        self.experiment.agents[traders][:, columns] = subset

    def update_agent_demand(self):
        # Extract valid trader indices
        traders = jnp.where(self.experiment.agents[:, 0] == 0)[0]  # Extract the first element of the tuple
        
        if self.experiment.informed == -1:
            # Correctly concatenate lists before converting to jnp.array
            columns = jnp.array([2, 3] + self.experiment.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = jnp.array(
                [self.experiment.informed, self.experiment.components['signal']['col_idx'], 2, 3] + 
                self.experiment.components['demand_function']['parameter_idxs']
            )

        # Correct the way traders and columns are used
        subset = self.experiment.agents[traders][:, columns]  # Corrected indexing
        
        # Update demand function
        subset = update_demands(self.experiment.market.price, subset, informed)
        
        # Store updated values back in agents array
        self.experiment.agents = self.experiment.agents.at[traders, columns].set(subset)

    def calculate_utility(self):
        pass


    