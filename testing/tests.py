import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as jnp
import unittest

from ACE_Experiment.experiment import *
from entities.agent import AgentInfo
from entities.market import RoutledgeMarket


class TestExperiment(unittest.TestCase):

    def test_init(self):
        market = RoutledgeMarket()
        experiment = Experiment(market, 'src/testing/data/agents.csv', 'src/testing/data/components.json')
        self.assertEqual(experiment.market, market)
        self.assertIsInstance(experiment.agents, jnp.ndarray)
        self.assertIsInstance(experiment.components, AgentInfo)

    # def test_dummy_run(self):
    #     market = RoutledgeMarket()
    #     experiment = Experiment(market, 'src/testing/data/dummy_agents.csv', 'src/testing/data/components.json')
    #     results = experiment.run(2, 2)
    #     self.assertIsInstance(results, jnp.ndarray)


    def test_routledge(self):

        market = RoutledgeMarket()
        experiment = Experiment(market, 'src/testing/data/agents.csv', 'src/testing/data/components.json')
        results = experiment.run(repetitions=1000, generations=5_000)
        self.assertIsInstance(results, jnp.ndarray)

if __name__ == '__main__':
    unittest.main() 