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
        price = np.round(price,3)
        assert np.allclose(price, 7.839,atol=0.001), f'Looking for 7.839, got {price}'

    def test_binomial_option(self):
        option = Contract(strike=40.0, T=3, payoff=put_payoff, h=0.333)
        price = binomial_option(option,41.0,r=0.08,vol=0.30)
        price = np.round(price,3)
        assert np.allclose(price, 3.293,atol=0.001), f'Looking for 3.293, got {price}'

    def test_LSM_binomial_option(self):
        option = Contract(strike=40.0, T=3, payoff=put_payoff, h=0.333)
        bin_price = binomial_option(option,41.0,r=0.08,vol=0.30)
        prices = simulate_prices(drift=0.08,volatility=0.30, initial_price=41.0,T=3, n=1_000_000, h=0.333)
        lsm_price = LSM_american(prices, option, discount_rate = 0.08)[0]
        assert np.allclose(bin_price, lsm_price, rtol=0.1), f'Binomial Price: {bin_price}, LSM Price: {lsm_price}'
        

        
                            


    

    def runTest(self):
        pass

if __name__ == '__main__':
    unittest.main()

