import unittest
from LSM import *
from contract import *
import numpy as np
from timeit import timeit
import jax
import jax.numpy as jnp
import jax_lsm
import time
from math import exp, log, sqrt


class lsm_unit_tests(unittest.TestCase):
    
    def test_simulate_prices(self):
        prices = simulate_prices(drift=0.15,volatility=0.15, initial_price=100,T=25, n=250)
        self.assertEqual(prices.shape, (250,25))
    
    def test_paths_unique(self):
        prices = simulate_prices(drift=0.15,volatility=0.15, initial_price=100,T=25, n=250,m=0,sd=1)
        try:
            assert np.allclose(len(np.unique(prices[:,3])), 250, rtol=0.1) 
        except AssertionError:
            print(np.unique(prices[:,3]))

    def test_LSM_american(self):
        prices = np.array([[1.00,1.09,1.08,1.34],
                          [1.00,1.16,1.26,1.54],
                          [1.00,1.22,1.07,1.03],
                          [1.00,0.93,0.97,0.92],
                          [1.00,1.11,1.56,1.52],
                          [1.00,0.76,0.77,0.90],
                          [1.00,0.92,0.84,1.01],
                          [1.00,0.88,1.22,1.34]])
        option = Contract(strike=1.1, T=4, payoff=put_payoff)
        price, cash_flow = LSM_american(prices, option, discount_rate = exp(0.06*260)-1)
        self.assertEqual(cash_flow.shape, (8,4))
        print(cash_flow)
        print(price)
        assert np.allclose(price, 0.1144,atol=0.015)

    def test_LSM_european(self):
        prices = np.array([[1.00,1.09,1.08,1.34],
                          [1.00,1.16,1.26,1.54],
                          [1.00,1.22,1.07,1.03],
                          [1.00,0.93,0.97,0.92],
                          [1.00,1.11,1.56,1.52],
                          [1.00,0.76,0.77,0.90],
                          [1.00,0.92,0.84,1.01],
                          [1.00,0.88,1.22,1.34]])
        option = Contract(strike=1.1, T=4, payoff=put_payoff)
        price = LSM_european(prices, option, discount_rate = 0.06)
        print(price)
        assert np.allclose(price, 0.0564,atol=0.005)
    
    def test_jax_lsm_american(self):
        prices = np.array([[1.00,1.09,1.08,1.34],
                          [1.00,1.16,1.26,1.54],
                          [1.00,1.22,1.07,1.03],
                          [1.00,0.93,0.97,0.92],
                          [1.00,1.11,1.56,1.52],
                          [1.00,0.76,0.77,0.90],
                          [1.00,0.92,0.84,1.01],
                          [1.00,0.88,1.22,1.34]])
        option = Contract(strike=1.1, T=4, payoff=put_payoff)
        jlsm = jax_lsm.JAX_LSM()
        price, cash_flow = jlsm.LSM_american(prices, option, discount_rate = 0.06)
        self.assertEqual(cash_flow.shape, (8,4))
        #print(cash_flow)
        #print(price)
        assert np.allclose(price, 0.1144,atol=0.015)

    """ def test_jax_lsm(self):
        prices = np.array([[1.00,1.09,1.08,1.34],
                          [1.00,1.16,1.26,1.54],
                          [1.00,1.22,1.07,1.03],
                          [1.00,0.93,0.97,0.92],
                          [1.00,1.11,1.56,1.52],
                          [1.00,0.76,0.77,0.90],
                          [1.00,0.92,0.84,1.01],
                          [1.00,0.88,1.22,1.34]])
        option = Contract(strike=1.1, T=4, payoff=put_payoff)
        jlsm = jax_lsm.JAX_LSM()

        price, cash_flow = jlsm.jax_lsm(prices, option, discount_rate = 0.06)
        self.assertEqual(cash_flow.shape, (8,4))
        #print(cash_flow)
        #print(price)
        assert np.allclose(price, 0.1144,atol=0.015)
        

    def test_simulate_prices_time(self):
        start = time.time()
        simulate_prices(n=1_000_000)
        end = time.time()
        print(f'simulate_prices runtime: {end-start}')

    def test_LSM_american_time(self):
        prices = simulate_prices(n=100_000)
        option = Contract(strike=105, T=25, payoff=call_payoff)
        start = time.time()
        LSM_american(prices, option, discount_rate = 0.06)
        end = time.time()
        print(f'LSM_american runtime: {end-start}') """

    """ def test_jax_lsm_american_time(self):
        prices = simulate_prices(n=100_000)
        option = Contract(strike=105, T=25, payoff=call_payoff)
        jlsm = jax_lsm.JAX_LSM()
        start = time.time()
        jlsm.LSM_american(prices, option, discount_rate = 0.06)
        end = time.time()
        print(f'jax_lsm_american runtime: {end-start}') """

    """ def test_jax_lsm_time(self):
        prices = simulate_prices(n=1_000)
        option = Contract(strike=105, T=25, payoff=call_payoff)
        jlsm = jax_lsm.JAX_LSM()
        start = time.time()
        jlsm.jax_lsm(prices, option, discount_rate = 0.06)
        end = time.time()
        print(f'jax_lsm runtime: {end-start}') """

    def runTest(self):
        pass

if __name__ == '__main__':
    unittest.main()