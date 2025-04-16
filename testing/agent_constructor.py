import numpy as np
import numpy.random as npr
import pandas as pd

POP_SIZE = 1000


def test_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 8
    beta0_std = 5
    beta1_mean = 3
    beta1_std = 4

    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1, POP_SIZE + 1)
    demand = np.zeros(POP_SIZE)
    fitness = np.zeros(POP_SIZE)
    bid = np.zeros(POP_SIZE)
    ask = np.zeros(POP_SIZE)
    bid_quantity = np.zeros(POP_SIZE)
    ask_quantity = np.zeros(POP_SIZE)

    demand_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)

    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    risk_aversion = npr.uniform(1, 3, POP_SIZE)
    spread_function = np.empty(POP_SIZE)
    confidence = np.empty(POP_SIZE)
    loss = [1 if x == 3 else 0 for x in learning_algorithm]

    information_policy = np.empty(POP_SIZE)

    for i in range(POP_SIZE):

        if learning_algorithm[i] == 1:
            information_policy[i] = 0
            spread_function[i] = 1
        elif learning_algorithm[i] == 2:
            information_policy = 1
            spread_function[i] = 2
        elif learning_algorithm == 3:
            information_policy = 2
            spread_function[i] = 3
        else:
            information_policy[i] = 0
            spread_function[i] = 0

        if spread_function[i] == 1:
            confidence[i] = np.random.uniform(3, 10)
        elif spread_function[i] == 2:
            confidence[i] = np.random.uniform(0.05, 0.35)

    info_params = 0

    informed = npr.binomial(1, probability_of_informed, POP_SIZE)
    signal = np.zeros(POP_SIZE, dtype=int)
    beta0 = npr.normal(beta0_mean, beta0_std, POP_SIZE)
    beta1 = npr.normal(beta1_mean, beta1_std, POP_SIZE)

    agents = pd.DataFrame(
        {
            "agent_type": agent_type,
            "id": idx,
            "demand": demand,
            "demand_function": demand_function,
            "fitness": fitness,
            "objective_function": objective_function,
            "utility_function": utility_function,
            "learning_algorithm": learning_algorithm,
            "risk_aversion": risk_aversion,
            "informed": informed,
            "signal": signal,
            "bid": bid,
            "ask": ask,
            "bid_quantity": bid_quantity,
            "ask_quantity": ask_quantity,
            "loss": loss,
            "spread_function": spread_function,
            "confidence": confidence,
            "information_policy": information_policy,
            "dfx_beta0": beta0,
            "dfx_beta1": beta1,
        }
    )
    agents.to_csv("agents.csv", index=False)


def dummy_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 8
    beta0_std = 2
    beta1_mean = 3
    beta1_std = 1

    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1, POP_SIZE + 1)
    demand = np.array([2] * POP_SIZE)
    demand_function = np.ones(POP_SIZE, dtype=int)
    fitness = np.array([4] * POP_SIZE)
    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)
    risk_aversion = np.array([8] * POP_SIZE)
    informed = npr.binomial(1, probability_of_informed, POP_SIZE)
    signal = np.array([10] * POP_SIZE, dtype=int)
    beta0 = npr.normal(beta0_mean, beta0_std, POP_SIZE)
    beta1 = npr.normal(beta1_mean, beta1_std, POP_SIZE)
    crossover_rate = npr.uniform(0, 1, POP_SIZE)
    mutation_rate = npr.uniform(0, 1, POP_SIZE)

    agents = pd.DataFrame(
        {
            "agent_type": agent_type,
            "id": idx,
            "demand": demand,
            "demand_function": demand_function,
            "fitness": fitness,
            "objective_function": objective_function,
            "utility_function": utility_function,
            "learning_algorithm": learning_algorithm,
            "risk_aversion": risk_aversion,
            "informed": informed,
            "signal": signal,
            "dfx_beta0": beta0,
            "dfx_beta1": beta1,
            "la_crossover_rate": crossover_rate,
            "la_mutation_rate": mutation_rate,
        }
    )
    agents.to_csv("dummy_agents.csv", index=False)


def routledge_agent_constructor():
    probability_of_informed = 0.5
    beta0_mean = 0.1
    beta1_mean = 1.0

    agent_type = np.zeros(POP_SIZE, dtype=int)
    idx = np.arange(1, POP_SIZE + 1)
    demand = np.zeros(POP_SIZE)
    demand_function = np.ones(POP_SIZE, dtype=int)
    fitness = np.zeros(POP_SIZE)
    objective_function = np.ones(POP_SIZE, dtype=int)
    utility_function = np.ones(POP_SIZE, dtype=int)
    learning_algorithm = np.ones(POP_SIZE, dtype=int)
    risk_aversion = np.array([2.0] * POP_SIZE)
    informed = npr.binomial(1, probability_of_informed, POP_SIZE)
    signal = np.zeros(POP_SIZE, dtype=int)
    beta0 = npr.uniform(beta0_mean - 0.1, beta0_mean + 0.1, POP_SIZE)
    beta1 = npr.uniform(beta1_mean - 0.1, beta1_mean - 0.1, POP_SIZE)
    crossover_rate = np.array([0.7] * POP_SIZE)
    mutation_rate = np.array([0.0001] * POP_SIZE)

    agents = pd.DataFrame(
        {
            "agent_type": agent_type,
            "id": idx,
            "demand": demand,
            "demand_function": demand_function,
            "fitness": fitness,
            "objective_function": objective_function,
            "utility_function": utility_function,
            "learning_algorithm": learning_algorithm,
            "risk_aversion": risk_aversion,
            "informed": informed,
            "signal": signal,
            "dfx_beta0": beta0,
            "dfx_beta1": beta1,
            "la_crossover_rate": crossover_rate,
            "la_mutation_rate": mutation_rate,
        }
    )
    agents.to_csv("agents.csv", index=False)


def test_bayesians():
    POP_SIZE = 4

    fitness = [0] * POP_SIZE
    informed = [0, 1, 0, 1]
    signal = [0] * POP_SIZE
    bid = [0] * POP_SIZE
    ask = [0] * POP_SIZE
    bid_quantity = [0] * POP_SIZE
    ask_quantity = [0] * POP_SIZE
    demand = [0] * POP_SIZE
    demand_function = [2, 2, 2, 2]
    objective_function = [1] * POP_SIZE
    utility_function = [1] * POP_SIZE
    risk_aversion = [2, 2, 2, 2]
    learning_algorithm = [2, 2, 2, 2]
    spread_function = [2, 2, 2, 2]
    confidence = [0.05, 0.1, 0.2, 0.05]
    loss = [0, 0, 1, 1]
    information_policy = [1, 1, 1, 2]
    agent_type = [0] * POP_SIZE
    agent_id = [0, 1, 2, 3]
    mu_prior = [150, 100, 95, 110]
    sigma_prior = [7, 5, 3, 5]
    tau = [1, 1, 1, 1]
    input_shape = [0, 0, 1, 1]
    hidden_layers = [0, 0, 1, 2]
    hidden_nodes = [0, 0, 16, 16]
    epochs = [0, 0, 5, 5]
    optimizer = [0, 0, 1, 1]
    learning_rate = [0, 0, 0.01, 0.001]
    entropy_coeff = [0, 0, 0.05, 0.07]
    update_frequency = [0, 0, 1, 1]
    info_return = [0] * POP_SIZE
    uninf_return = [0] * POP_SIZE
    beta0 = [0] * POP_SIZE
    beta1 = [0] * POP_SIZE

    agents = pd.DataFrame(
        {
            "fitness": fitness,
            "informed": informed,
            "signal": signal,
            "bid": bid,
            "ask": ask,
            "bid_quantity": bid_quantity,
            "ask_quantity": ask_quantity,
            "demand": demand,
            "demand_function": demand_function,
            "objective_function": objective_function,
            "utility_function": utility_function,
            "risk_aversion": risk_aversion,
            "learning_algorithm": learning_algorithm,
            "spread_function": spread_function,
            "confidence": confidence,
            "loss": loss,
            "information_policy": information_policy,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "tau": tau,
            "input_shape": input_shape,
            "hidden_layers": hidden_layers,
            "hidden_nodes": hidden_nodes,
            "epochs": epochs,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "entropy_coeff": entropy_coeff,
            "update_frequency": update_frequency,
            "info_return": info_return,
            "uninf_return": uninf_return,
            "beta0": beta0,
            "beta1": beta1,
        }
    )
    agents.to_csv("test_bayesians.csv", index=False)


def test_constructor():

    POP_SIZE = 4

    fitness = [0] * POP_SIZE
    informed = [0, 1, 0, 1]
    signal = [0] * POP_SIZE
    bid = [0] * POP_SIZE
    ask = [0] * POP_SIZE
    bid_quantity = [0] * POP_SIZE
    ask_quantity = [0] * POP_SIZE
    demand = [0] * POP_SIZE
    demand_function = [2, 2, 3, 3]
    objective_function = [1] * POP_SIZE
    utility_function = [1] * POP_SIZE
    risk_aversion = [2, 2, 2, 2]
    learning_algorithm = [2, 2, 3, 3]
    spread_function = [2, 2, 3, 3]
    confidence = [0.05, 0.1, 0.2, 0.05]
    loss = [0, 0, 1, 1]
    information_policy = [1, 1, 2, 2]
    agent_type = [0] * POP_SIZE
    agent_id = [0, 1, 2, 3]
    mu_prior = [10, 12, 0, 0]
    sigma_prior = [2, 3, 0, 0]
    tau = [1, 1, 0, 0]
    input_shape = [0, 0, 1, 1]
    hidden_layers = [0, 0, 1, 2]
    hidden_nodes = [0, 0, 16, 16]
    epochs = [0, 0, 5, 5]
    optimizer = [0, 0, 1, 1]
    learning_rate = [0, 0, 0.01, 0.001]
    entropy_coeff = [0, 0, 0.05, 0.07]
    update_frequency = [0, 0, 1, 1]
    info_return = [0] * POP_SIZE
    uninf_return = [0] * POP_SIZE
    beta0 = [0] * POP_SIZE
    beta1 = [0] * POP_SIZE

    agents = pd.DataFrame(
        {
            "fitness": fitness,
            "informed": informed,
            "signal": signal,
            "bid": bid,
            "ask": ask,
            "bid_quantity": bid_quantity,
            "ask_quantity": ask_quantity,
            "demand": demand,
            "demand_function": demand_function,
            "objective_function": objective_function,
            "utility_function": utility_function,
            "risk_aversion": risk_aversion,
            "learning_algorithm": learning_algorithm,
            "spread_function": spread_function,
            "confidence": confidence,
            "loss": loss,
            "information_policy": information_policy,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "mu_prior": mu_prior,
            "sigma_prior": sigma_prior,
            "tau": tau,
            "input_shape": input_shape,
            "hidden_layers": hidden_layers,
            "hidden_nodes": hidden_nodes,
            "epochs": epochs,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "entropy_coeff": entropy_coeff,
            "update_frequency": update_frequency,
            "info_return": info_return,
            "uninf_return": uninf_return,
            "beta0": beta0,
            "beta1": beta1,
        }
    )
    agents.to_csv("test_agents.csv", index=False)


if __name__ == "__main__":
    # test_agent_constructor()
    # routledge_agent_constructor()
    # dummy_agent_constructor()
    # test_constructor()
    test_bayesians()
