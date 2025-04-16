import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import unittest

import numpy as jnp

from entities.agent import AgentInfo
from entities.market import GSOrderMarket
from experiment import *
from globals import config, globals


class TestExperiment(unittest.TestCase):

    def setUp(self):
        config.from_json("testing/test_config.json")

    def test_save_models(self):
        config.save_models = True
        market = GSOrderMarket()
        experiment = Experiment(
            market=market,
            agents_file_path="testing/test_bayesians.csv",
        )
        results = experiment.run(2, 100)

        self.assertIsInstance(results, jnp.ndarray)

    # def test_dont_save_models(self):
    #     config.save_models = False
    #     market = GSOrderMarket()
    #     experiment = Experiment(
    #         market=market,
    #         agents_file_path="testing/test_bayesians.csv",
    #     )
    #     results = experiment.run(2, 100)

    #     self.assertIsInstance(results, jnp.ndarray)

    # def test_performance_optimized(self):
    #     config.memory_optimization = False
    #     market = GSOrderMarket()
    #     experiment = Experiment(
    #         market=market,
    #         agents_file_path="testing/test_bayesians.csv",
    #     )
    #     results = experiment.run(2, 100)

    #     self.assertIsInstance(results, jnp.ndarray)


if __name__ == "__main__":
    unittest.main()
