import numpy.random as npr
import numpy as np
import pandas as pd


POP_SIZE = 10
probability_of_informed = 0.5
beta0_mean = 8
beta0_std = 2
beta1_mean = 3
beta1_std = 1


agent_type = np.ones(POP_SIZE, dtype=int)
idx = np.arange(1,POP_SIZE+1)
demand = np.zeros(POP_SIZE)
demand_function = np.ones(POP_SIZE, dtype=int)
fitness = np.zeros(POP_SIZE)
objective_function = np.ones(POP_SIZE, dtype=int)
utility_function = np.ones(POP_SIZE, dtype=int)
learning_algorithm = np.ones(POP_SIZE, dtype=int)
risk_aversion = npr.uniform(0,1,POP_SIZE)
informed = npr.binomial(1,probability_of_informed,POP_SIZE)
signal = np.zeros(POP_SIZE, dtype=int)
beta0 = npr.normal(beta0_mean, beta0_std, POP_SIZE)
beta1 = npr.normal(beta1_mean, beta1_std, POP_SIZE)
crossover_rate = npr.uniform(0,1,POP_SIZE)
mutation_rate = npr.uniform(0,1,POP_SIZE)

agents = pd.DataFrame({'agent_type': agent_type, 'id': idx, 'demand': demand, 'demand_function': demand_function, 'fitness': fitness, 'objective_function': objective_function, 'utility_function': utility_function, 'learning_algorithm': learning_algorithm, 'risk_aversion': risk_aversion, 'informed': informed, 'signal': signal, 'beta0': beta0, 'beta1': beta1, 'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate})
agents.to_csv('agents.csv', index=False)