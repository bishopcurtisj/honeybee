import json

import numpy as jnp
import numpy as np

from entities.agent import AgentInfo
from entities.market import Market
from systems.calculations import *
from systems.learning import *
from systems.trade import *


class Experiment:
    def __init__(self, market: Market, agents_file_path: str = './assets/agents.csv', components_file_path: str='./assets/components.json'):
        
        
        with open(agents_file_path) as f:
            self.headers = f.readline().strip().split(',')
        self.agents = np.loadtxt(agents_file_path, delimiter=',', skiprows=1)
        self.market = market
        self.components = AgentInfo(self.headers)
        
        if 'informed' in self.components.keys():
            self.agents[:, self.components.signal] = jnp.where(self.agents[:, self.components.informed] == 0, self.market.price, self.market.signal[0])
        else:
            self.components.add('informed', None)
        
        


    def run(self, generations: int = 20, repetitions: int = 100):
        
        for _ in range(generations):
            self.get_agent_spread()
            self.trade()
            self.calculate_agent_fitness(repetitions)
            self.learn()
            self.trade()
        
        self.save()
        return self.agents


    ## Need to test this to verify how write works with jnp arrays
    def save(self):

        np.savetxt("results.csv", self.agents, delimiter=",", fmt="%.2f", header=",".join(self.headers), comments='')


    def learn(self):
        traders = jnp.where(self.agents[:,self.components.agent_type] == 0)[0]
        if self.components.informed == None:
            columns = jnp.array([self.components.fitness]+self.components.demand_fx_params)
            informed = False
        else:
            informed = True
            columns = jnp.array([self.components.fitness, self.components.informed] + self.components.demand_fx_params)

        params = self.agents[traders][:, jnp.array(self.components.learning_params)]
        subset = self.agents[traders][:, columns]   

        ## Update to allow for different learning algorithms
        learning_function = LEARNING_REGISTRY[self.agents[0, self.components.learning_algorithm]]()

        subset = learning_function(subset, params, len(subset), informed)

        self.agents[traders[:, None], columns] = subset
        self.market.new_period()
        ## Update signal        
        
        if self.components.informed != None:
            self.agents[:,self.components.signal] = jnp.where(self.agents[:, self.components.informed]== 0, self.market.price, self.market.signal[0])

    def trade(self):
        self.order_book = OrderBook()
        traders = jnp.where(self.agents[:,self.components.agent_type] == 0)[0]

    def calculate_agent_fitness(self, repetitions: int):
        traders = jnp.where(self.agents[:,self.components.agent_type] == 0)[0]
        if self.components.informed == None:
            columns = np.array([self.components.fitness, self.components.objective_function, self.components.utility_function, self.components.demand, self.components.demand_function] + self.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = np.array([self.components.fitness, self.components.objective_function, self.components.utility_function, self.components.informed ,self.components.signal ,self.components.demand, self.components.demand_function] + self.components.demand_fx_params)
        subset = self.agents[traders][:, columns]  # Corrected indexing
        
        subset = calculate_fitness(subset, repetitions, self.agents[traders][:, self.components.risk_aversion], self.market, informed)
        self.agents[traders[:, None], columns] = subset

    def get_agent_spread(self):
        """
        Calculate the bid-ask spread of the agents
        """
        traders = jnp.where(self.agents[:, self.components.agent_type] == 0)[0]  # Extract the first element of the tuple
        
        if self.components.informed == None:
            # Correctly concatenate lists before converting to jnp.array
            columns = jnp.array([self.components.bid, self.components.ask, self.components.bid_quantity, self.components.ask_quantity, self.components.demand_function, self.components.confidence] + self.components.demand_fx_params)
            informed = False
        else:
            informed = True
            columns = jnp.array(
                [self.components.informed, self.components.signal, self.components.bid, self.components.ask, self.components.bid_quantity, self.components.ask_quantity, self.components.demand_function, self.components.confidence] + 
                self.components.demand_fx_params
            )

        # Correct the way traders and columns are used
        subset = self.agents[traders][:, columns]  # Corrected indexing
        
        # Update demand function
        subset = update_demands(self.market.price, subset, informed)
        
        # Store updated values back in agents array
        self.agents[traders[:, None], columns] = subset

    