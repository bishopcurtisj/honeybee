from abc import ABC, ABCMeta, abstractmethod
from functools import singledispatch
from typing import List, Union

import numpy as jnp
import numpy as np
import tensorflow as tf

from src.globals import config, globals


class InformationDecisionPolicy(ABC):
    name: str

    @abstractmethod
    def __call__(*args, **kwargs): ...


class ReinforcementLearning(InformationDecisionPolicy):

    name: str = "ReinforcementLearning"

    def __init__(
        self,
        agent,
        learning_rate=0.01,
        entropy_coeff=0.05,
        update_frequency=1,
        *args,
        **kwargs,
    ):
        self.logits = tf.Variable([0.0, 0.0])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.entropy_coeff = entropy_coeff
        self.update_frequency = update_frequency
        self.chosen_prob = agent[globals.components.informed]
        self.reset_batch()

    def reset_batch(self):
        # Lists to store batched data
        self.batch_chosen_probabilities = []
        self.batch_rewards = []

    def decide(self):
        probabilities = tf.nn.softmax(self.logits).numpy()
        choice = np.random.choice([0, 1], p=probabilities)
        return choice, probabilities[choice], probabilities

    def record_generation_result(self, chosen_probability, reward):
        self.batch_chosen_probabilities.append(chosen_probability)
        self.batch_rewards.append(reward)

    def update_policy(self):
        chosen_probs_tensor = tf.convert_to_tensor(
            self.batch_chosen_probabilities, dtype=tf.float32
        )
        rewards_tensor = tf.convert_to_tensor(self.batch_rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(self.logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))
            log_probs = tf.math.log(chosen_probs_tensor + 1e-10)
            loss = -tf.reduce_mean(
                log_probs * rewards_tensor + self.entropy_coeff * entropy
            )

        grads = tape.gradient(loss, [self.logits])
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(zip(grads, [self.logits]))

        # Clear batch after update
        self.reset_batch()

    def __call__(self, agent):

        reward = agent[globals.components.fitness]
        self.record_generation_result(self.chosen_prob, reward)
        if globals.generation % self.update_frequency == 0:
            self.update_policy()

        choice, self.chosen_prob, probs = self.decide()
        agent[globals.components.informed] = choice
        return agent


class BayesianInfo(InformationDecisionPolicy):
    name: str = "BayesianInfo"

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray:
        """
        Draws from the posterior distributions in order to determine informed status.
        Draws from a binomial where p = E(Informed)/(E(Informed) + E(Uninformed))
        """
        expected_info_return, expected_uninf_return = agents[
            :, globals.components.info_params
        ]

        expected_info_return[
            jnp.where(agents[:, globals.components.informed] == 1)[0]
        ] = (
            (globals.generation - 1)
            * expected_info_return[
                jnp.where(agents[:, globals.components.informed] == 1)[0]
            ]
            + agents[jnp.where(agents[:, globals.components.informed] == 1)[0]][
                globals.components.fitness
            ]
        ) / globals.generation
        expected_uninf_return[
            jnp.where(agents[:, globals.components.informed] == 0)[0]
        ] = (
            (globals.generation - 1)
            * expected_uninf_return[
                jnp.where(agents[:, globals.components.informed] == 0)[0]
            ]
            + agents[jnp.where(agents[:, globals.components.informed] == 0)[0]][
                globals.components.fitness
            ]
        ) / globals.generation

        agents[:, globals.components.info_params[0]] = expected_info_return
        agents[:, globals.components.info_params[1]] = expected_uninf_return
        p = expected_info_return / (expected_info_return + expected_uninf_return)
        agents[:, globals.components.informed] = jnp.random.binomial(1, p, len(agents))
        return agents


class ThompsonSampling(InformationDecisionPolicy):
    name: str = "ThompsonSampling"

    @staticmethod
    def __call__(agents: jnp.ndarray) -> jnp.ndarray:
        """
        Draws from the posterior distributions in order to determine informed status.
        Draws from a binomial where p = E(Informed)/(E(Informed) + E(Uninformed))
        """
        expected_info_return, expected_uninf_return = agents[
            :, globals.components.info_params
        ]

        expected_info_return[
            jnp.where(agents[:, globals.components.informed] == 1)[0]
        ] = (
            (globals.generation - 1)
            * expected_info_return[
                jnp.where(agents[:, globals.components.informed] == 1)[0]
            ]
            + agents[jnp.where(agents[:, globals.components.informed] == 1)[0]][
                globals.components.fitness
            ]
        ) / globals.generation
        expected_uninf_return[
            jnp.where(agents[:, globals.components.informed] == 0)[0]
        ] = (
            (globals.generation - 1)
            * expected_uninf_return[
                jnp.where(agents[:, globals.components.informed] == 0)[0]
            ]
            + agents[jnp.where(agents[:, globals.components.informed] == 0)[0]][
                globals.components.fitness
            ]
        ) / globals.generation

        agents[:, globals.components.info_params[0]] = expected_info_return
        agents[:, globals.components.info_params[1]] = expected_uninf_return
        p = expected_info_return / (expected_info_return + expected_uninf_return)
        agents[:, globals.components.informed] = jnp.random.binomial(1, p, len(agents))
        return agents


class FixedInformation(InformationDecisionPolicy):
    """
    This information policy does not allow agents to change from their initial informed/uninformed state.
    """

    name: str = "FixedInformation"

    @staticmethod
    @singledispatch
    def __call__(agents, *args, **kwargs):
        raise TypeError(f"Unsupported type {type(agents)}")

    @__call__.register
    def _(agents: None = None, *args, **kwargs):
        return FixedInformation

    @__call__.register
    def _(agents: jnp.ndarray, *args, **kwargs):
        return agents


def register_info_policy(
    info_policies: Union[List[InformationDecisionPolicy], InformationDecisionPolicy],
):
    if type(info_policies) == InformationDecisionPolicy:
        info_policies = [info_policies]
    for info_policy in info_policies:
        try:
            assert issubclass(info_policy, InformationDecisionPolicy)
        except AssertionError:
            raise ValueError(
                f"Custom Information policy {info_policy.name} must be a subclass of InformationDecisionPolicy"
            )
        INFORMATION_POLICY_REGISTRY[len(INFORMATION_POLICY_REGISTRY)] = info_policy


INFORMATION_POLICY_REGISTRY = {
    0: FixedInformation,
    1: BayesianInfo,
    2: ReinforcementLearning,
    3: ThompsonSampling,
}
