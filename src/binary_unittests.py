import unittest
from binary import *
from LSM import *
from contract import *
import numpy as np
from timeit import timeit
import jax
import jax.numpy as jnp
import jax_lsm
import time
from math import exp, log, sqrt


class binary_unit_tests(unittest.TestCase):

    def test_euro_binomial_option(self):
        option = Contract(strike=40.0, T=1, payoff=call_payoff)
        price = euro_binomial_option(option,41.0,r=0.08,vol=0.30)
        print(price)
        assert np.allclose(price, 7.839,atol=0.005)

    def test_binomial_option(self):
        option = Contract(strike=40.0, T=3, payoff=put_payoff, h=0.333)
        price = binomial_option(option,41.0,r=0.08,vol=0.30)
        print(price)
        assert np.allclose(price, 3.293,atol=0.005)

    def test_LSM_binomial_option(self):
        pass

        
                            


    

    def runTest(self):
        pass

if __name__ == '__main__':
    unittest.main()

