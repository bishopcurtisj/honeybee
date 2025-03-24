import json

import numpy as jnp
import numpy as np

from entities.agent import AgentInfo
from entities.market import Market
from systems.calculations import *
from systems.learning import model_controller



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
        

        model_controller.init_models(self.agents, self.components)
        


    def run(self, generations: int = 20, repetitions: int = 100):
        
        for _ in range(generations):
            
            trades = self.trade(repetitions)
            self.calculate_agent_fitness(repetitions, trades)
            self.learn()
            self.trade()
        
        self.save()
        return self.agents

    def save(self):

        np.savetxt("results.csv", self.agents, delimiter=",", fmt="%.2f", header=",".join(self.headers), comments='')


    def learn(self):
        if self.components.informed == None:
            columns = jnp.array([self.components.fitness]+self.components.demand_fx_params)
            informed = False
        else:
            informed = True
            columns = jnp.array([self.components.fitness, self.components.informed] + self.components.demand_fx_params)

        params = self.agents[:, jnp.array(self.components.learning_params)]
        subset = self.agents[:, columns]   

        subset = model_controller.learn(subset, params, informed)

        self.agents[:, columns] = subset
        self.market.new_period()

        ## Update signal        
        if self.components.informed != None:
            self.agents[:,self.components.signal] = jnp.where(self.agents[:, self.components.informed]== 0, self.market.price, self.market.signal[0])


## Refactor to vectorize
    def trade(self, repetitions: int):
        traders = jnp.where(self.agents[:,self.components.agent_type] == 0)[0]
        self.get_agent_spread()
        agent_ids = traders[:, self.components.id]
        total_agent_trades = jnp.zeros((len(self.agents), repetitions, 2))
        self.trades = []

        for repetition in range(repetitions):
            trade_order = jnp.random.permutation(agent_ids)
            for agent_id in trade_order:
                self.market.order_book.add_order(agent_id, self.agents[agent_id, self.components.bid], self.agents[agent_id, self.components.bid_quantity])
                self.market.order_book.add_order(agent_id, self.agents[agent_id, self.components.ask], self.agents[agent_id, self.components.ask_quantity])
            agent_trades = self.market.order_book.get_trades()
            self.trades += self.market.order_book.get_trades()

            for agent_id in agent_ids:
                if agent_id in agent_trades.keys():
                    total_agent_trades[agent_id, repetition] = np.array(agent_trades[agent_id][:,0].sum(), np.sum(agent_trades[agent_id][:,1]*agent_trades[agent_id][:,0]))
            self.market.order_book.reset()

        self.trades = jnp.array(self.trades)
        return total_agent_trades

    def calculate_agent_fitness(self, repetitions: int, trades: jnp.ndarray):
        traders = jnp.where(self.agents[:,self.components.agent_type] == 0)[0]
        if self.components.informed == None:
            columns = np.array([self.components.fitness, self.components.objective_function, self.components.utility_function, self.components.demand, self.components.demand_function] + self.components['demand_function']['parameter_idxs'])
            informed = False
        else:
            informed = True
            columns = np.array([self.components.fitness, self.components.objective_function, self.components.utility_function, self.components.informed ,self.components.signal ,self.components.demand, self.components.demand_function] + self.components.demand_fx_params)
        subset = self.agents[traders][:, columns]  # Corrected indexing
        
        subset = calculate_fitness(subset, repetitions, trades, self.agents[traders][:, self.components.risk_aversion], self.market, informed)
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
        subset = self.update_demands(self.market.price, subset, informed)
        
        # Store updated values back in agents array
        self.agents[traders[:, None], columns] = subset

    def update_demands():
        pass
    