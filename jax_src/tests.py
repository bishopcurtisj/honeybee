from ACE_Experiment.experiment import *
import numpy as np
import jax.numpy as jnp
from entities.market import RoutledgeMarket
import json
import unittest

class TestExperiment(unittest.TestCase):

    def test_init(self):
        market = RoutledgeMarket()
        experiment = Experiment(market, 'src/assets/testing/agents.csv', 'src/assets/testing/components.json')
        self.assertEqual(experiment.market, market)
        self.assertIsInstance(experiment.agents, jnp.ndarray)
        self.assertIsInstance(experiment.components, dict)
        

    def test_run(self):
        market = RoutledgeMarket()
        experiment = Experiment(market, 'src/assets/testing/agents.csv', 'src/assets/testing/components.json')
        results = experiment.run(2, 2)
        self.assertIsInstance(results, jnp.ndarray)

if __name__ == '__main__':
    unittest.main()