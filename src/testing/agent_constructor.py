import numpy.random as npr
import numpy as np
import pandas as pd

POP_SIZE = 1000


def test_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 8
    beta0_std = 5
    beta1_mean = 3
    beta1_std = 4


    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1,POP_SIZE+1)
    demand = np.zeros(POP_SIZE)
    demand_function = np.ones(POP_SIZE, dtype=int)
    fitness = np.zeros(POP_SIZE)
    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)
    risk_aversion = npr.uniform(1,3,POP_SIZE)
    informed = npr.binomial(1,probability_of_informed,POP_SIZE)
    signal = np.zeros(POP_SIZE, dtype=int)
    beta0 = npr.normal(beta0_mean, beta0_std, POP_SIZE)
    beta1 = npr.normal(beta1_mean, beta1_std, POP_SIZE)
    crossover_rate = npr.uniform(0,1,POP_SIZE)
    mutation_rate = npr.uniform(0,1,POP_SIZE)

    agents = pd.DataFrame({'agent_type': agent_type, 'id': idx, 'demand': demand, 'demand_function': demand_function, 'fitness': fitness, 'objective_function': objective_function, 'utility_function': utility_function, 'learning_algorithm': learning_algorithm, 'risk_aversion': risk_aversion, 'informed': informed, 'signal': signal, 'dfx_beta0': beta0, 'dfx_beta1': beta1, 'la_crossover_rate': crossover_rate, 'la_mutation_rate': mutation_rate})
    agents.to_csv('agents.csv', index=False)

def dummy_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 8
    beta0_std = 2
    beta1_mean = 3
    beta1_std = 1


    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1,POP_SIZE+1)
    demand = np.array([2]*POP_SIZE)
    demand_function = np.ones(POP_SIZE, dtype=int)
    fitness = np.array([4]*POP_SIZE)
    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)
    risk_aversion = np.array([8]*POP_SIZE)
    informed = npr.binomial(1,probability_of_informed,POP_SIZE)
    signal = np.array([10]*POP_SIZE, dtype=int)
    beta0 = npr.normal(beta0_mean, beta0_std, POP_SIZE)
    beta1 = npr.normal(beta1_mean, beta1_std, POP_SIZE)
    crossover_rate = npr.uniform(0,1,POP_SIZE)
    mutation_rate = npr.uniform(0,1,POP_SIZE)

    agents = pd.DataFrame({'agent_type': agent_type, 'id': idx, 'demand': demand, 'demand_function': demand_function, 'fitness': fitness, 'objective_function': objective_function, 'utility_function': utility_function, 'learning_algorithm': learning_algorithm, 'risk_aversion': risk_aversion, 'informed': informed, 'signal': signal, 'dfx_beta0': beta0, 'dfx_beta1': beta1, 'la_crossover_rate': crossover_rate, 'la_mutation_rate': mutation_rate})
    agents.to_csv('dummy_agents.csv', index=False)

def routledge_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 0.1
    beta1_mean = 1.0


    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1,POP_SIZE+1)
    demand = np.zeros(POP_SIZE)
    demand_function = np.ones(POP_SIZE, dtype=int)
    fitness = np.zeros(POP_SIZE)
    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)
    risk_aversion = np.array([2.0]*POP_SIZE)
    informed = npr.binomial(1,probability_of_informed,POP_SIZE)
    signal = np.zeros(POP_SIZE, dtype=int)
    beta0 = npr.uniform(beta0_mean-0.1, beta0_mean+0.1, POP_SIZE)
    beta1 = npr.uniform(beta1_mean-0.1, beta1_mean-0.1, POP_SIZE)
    crossover_rate = np.array([0.7]*POP_SIZE)
    mutation_rate = np.array([0.0001]*POP_SIZE)

    agents = pd.DataFrame({'agent_type': agent_type, 'id': idx, 'demand': demand, 'demand_function': demand_function, 'fitness': fitness, 'objective_function': objective_function, 'utility_function': utility_function, 'learning_algorithm': learning_algorithm, 'risk_aversion': risk_aversion, 'informed': informed, 'signal': signal, 'dfx_beta0': beta0, 'dfx_beta1': beta1, 'la_crossover_rate': crossover_rate, 'la_mutation_rate': mutation_rate})
    agents.to_csv('agents.csv', index=False)

if __name__ == '__main__':
    # test_agent_constructor()
    routledge_agent_constructor()
    dummy_agent_constructor()