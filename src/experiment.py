import json
from typing import List, Union

import numpy as jnp
import numpy as np

from entities.agent import AgentInfo
from entities.market import Market
from src.globals import config, globals
from systems.agent_functions.demand import Demand, demand_factory
from systems.agent_functions.demand import register_demand_function as register_demand
from systems.agent_functions.objective import Objective, calculate_fitness
from systems.agent_functions.objective import (
    register_objective_function as register_objective,
)
from systems.agent_functions.spread import SPREAD_REGISTRY, Spread
from systems.agent_functions.spread import register_spread_function as register_spread
from systems.agent_functions.spread import spread_factory
from systems.agent_functions.utility import Utility
from systems.agent_functions.utility import (
    register_utility_function as register_utility,
)
from systems.learning import model_controller
from systems.models.information_policy import (
    InformationDecisionPolicy,
    info_factory,
    register_info_policy,
)
from systems.models.loss import AgentLoss, register_loss
from systems.models.model import Model


class Experiment:
    def __init__(
        self,
        market: Market,
        agents_file_path: str = "./assets/agents.csv",
        config_file_path: str = "./assets/config.json",
    ):

        with open(agents_file_path) as f:
            self.headers = f.readline().strip().split(",")
        globals.agents = np.loadtxt(agents_file_path, delimiter=",", skiprows=1)
        globals.market = market
        globals.components = AgentInfo(self.headers)

        if "informed" in globals.components.keys():
            globals.agents[:, globals.components.signal] = jnp.where(
                globals.agents[:, globals.components.informed] == 0,
                globals.market.price,
                globals.market.signal[0],
            )
            globals.informed = True
            globals.market.cost_of_info = 0.0
        else:
            globals.components.add("informed", None)
            globals.informed = False

        try:
            config.from_json(config_file_path)
        except FileNotFoundError:
            print("Config file not found. Using default values")

        model_controller.init_models(globals.agents, globals.components)
        demand_factory()
        spread_factory()
        info_factory()

    def run(self, generations: int = 20, repetitions: int = 100):

        config.generations = generations
        config.repetitions = repetitions
        globals.generation = 1

        for _ in range(generations):
            trades = self.trade()
            self.calculate_agent_fitness(trades)
            self.learn()
            globals.generation += 1

        self.save()
        return globals.agents

    def save(self):

        np.savetxt(
            "results.csv",
            globals.agents,
            delimiter=",",
            fmt="%.2f",
            header=",".join(self.headers),
            comments="",
        )

    def learn(self):
        if globals.informed == False:
            columns = jnp.array(
                [globals.components.fitness] + globals.components.demand_fx_params
            )

        else:
            columns = jnp.array(
                [globals.components.fitness, globals.components.informed]
                + globals.components.demand_fx_params
            )

        params = globals.agents[:, jnp.array(globals.components.learning_params)]
        subset = globals.agents[:, columns]

        subset = model_controller.learn(subset, params)

        globals.agents[:, columns] = subset
        globals.market.new_period()

        ## Update signal
        if globals.components.informed != None:
            globals.agents[:, globals.components.signal] = jnp.where(
                globals.agents[:, globals.components.informed] == 0,
                globals.market.price,
                globals.market.signal[0],
            )

    ## Refactor to vectorize
    def trade(self):
        traders = jnp.where(globals.agents[:, globals.components.agent_type] == 0)[0]
        self.get_agent_spread()
        agent_ids = traders[:, globals.components.id]
        total_agent_trades = jnp.zeros((len(globals.agents), config.repetitions, 2))
        trades = jnp.empty((0, 3))

        for repetition in range(config.repetitions):
            trade_order = jnp.random.permutation(agent_ids)
            for agent_id in trade_order:
                globals.market.order_book.add_order(
                    agent_id,
                    globals.agents[agent_id, globals.components.bid],
                    globals.agents[agent_id, globals.components.bid_quantity],
                )
                globals.market.order_book.add_order(
                    agent_id,
                    globals.agents[agent_id, globals.components.ask],
                    globals.agents[agent_id, globals.components.ask_quantity],
                )
            agent_trades = globals.market.order_book.get_agent_trades()
            trades = jnp.vstack((trades, globals.market.order_book.get_trades()))

            for agent_id in agent_ids:
                if agent_id in agent_trades.keys():
                    total_agent_trades[agent_id, repetition] = jnp.array(
                        agent_trades[agent_id][:, 0].sum(),
                        np.sum(
                            agent_trades[agent_id][:, 1] * agent_trades[agent_id][:, 0]
                        ),
                    )
            globals.market.order_book.reset()

        globals.trades = trades
        return total_agent_trades

    def calculate_agent_fitness(self, trades: jnp.ndarray):
        traders = jnp.where(globals.agents[:, globals.components.agent_type] == 0)[0]
        if globals.informed == False:
            columns = np.array(
                [
                    globals.components.fitness,
                    globals.components.objective_function,
                    globals.components.utility_function,
                    globals.components.demand,
                    globals.components.demand_function,
                ]
                + globals.components["demand_function"]["parameter_idxs"]
            )
        else:
            columns = np.array(
                [
                    globals.components.fitness,
                    globals.components.objective_function,
                    globals.components.utility_function,
                    globals.components.informed,
                    globals.components.signal,
                    globals.components.demand,
                    globals.components.demand_function,
                ]
                + globals.components.demand_fx_params
            )
        subset = globals.agents[traders][:, columns]  # Corrected indexing

        subset = calculate_fitness(
            subset,
            trades,
            globals.agents[traders][:, globals.components.risk_aversion],
        )
        globals.agents[traders[:, None], columns] = subset

    def get_agent_spread(self):
        """
        Calculate the bid-ask spread of the agents
        """
        traders = jnp.where(globals.agents[:, globals.components.agent_type] == 0)[
            0
        ]  # Extract the first element of the tuple
        if globals.components.informed == None:
            # Correctly concatenate lists before converting to jnp.array
            columns = jnp.array(
                [
                    globals.components.bid,
                    globals.components.ask,
                    globals.components.bid_quantity,
                    globals.components.ask_quantity,
                    globals.components.demand_function,
                    globals.components.confidence,
                ]
                + globals.components.demand_fx_params
            )
        else:
            columns = jnp.array(
                [
                    globals.components.informed,
                    globals.components.signal,
                    globals.components.bid,
                    globals.components.ask,
                    globals.components.bid_quantity,
                    globals.components.ask_quantity,
                    globals.components.demand_function,
                    globals.components.confidence,
                ]
                + globals.components.demand_fx_params
            )
        # Correct the way traders and columns are used
        subset = globals.agents[traders][:, columns]  # Corrected indexing

        for i in SPREAD_REGISTRY.keys():
            same_spread = jnp.where(
                globals.agents[:, globals.components.spread_function] == i
            )
            subset[same_spread[:, None], columns] = SPREAD_REGISTRY[i](
                subset[same_spread][:, columns]
            )

        # Store updated values back in agents array
        globals.agents[traders[:, None], columns] = subset

    def register_demand_function(self, demand_functions: Union[List[Demand], Demand]):
        register_demand(demand_functions)

    def register_objective_function(
        self,
        objective_functions: Union[List[Objective], Objective],
    ):
        register_objective(objective_functions)

    def register_spread_function(self, spread_functions: Union[List[Spread], Spread]):
        register_spread(spread_functions)

    def register_utility_function(
        self, utility_functions: Union[List[Utility], Utility]
    ):
        register_utility(utility_functions)

    def register_loss_function(self, losses: Union[List[AgentLoss], AgentLoss]):
        register_loss(losses)

    def register_information_policy(
        self,
        info_policies: Union[
            List[InformationDecisionPolicy], InformationDecisionPolicy
        ],
    ):
        register_info_policy(info_policies)

    def register_learning_algorithms(self, models: Union[List[Model], Model]):
        model_controller.register_models(models)
