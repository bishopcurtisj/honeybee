from abc import ABC, ABCMeta, abstractmethod

import numpy as jnp
import numpy as np
import tensorflow as tf

from entities.agent import AgentInfo


class Model(ABC):
    """
    Abstract class so that custom learning functions can be implemented
    """

    label: str
    args: dict

    @abstractmethod
    def __init__(self, agents: jnp.ndarray, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, agents: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        pass


class AgentLoss(tf.keras.losses.Loss, metaclass=ABCMeta):
    """Abstract parent for all trade fitness losses in the framework.

    Subclasses must implement:
        • call(y_true, y_pred) → Tensor
        • get_config()         → dict   (for serialization)
    """

    name: str
    reduction: tf.keras.losses.Reduction

    @abstractmethod
    def call(self, y_true, y_pred):
        """Compute *negative* fitness so that lower is better."""
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(reduction=self.reduction, name=self.name, **kwargs)

    @abstractmethod
    def get_config(self):
        """Return kwargs needed to recreate the loss (Keras will call this)."""
        raise NotImplementedError


class NegCARA(AgentLoss):

    _NAME: str = "neg_cara"
    _REDUCTION: tf.keras.losses.Reduction = (
        tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    def __init__(self, *args, **kwargs):
        super().__init__(reduction=self.reduction, name=self.name, **kwargs)
        self.risk_aversion = float(args[0])

    def call(self, y_true, y_pred):
        ## y_pred = quantity
        ## y_true = return per share
        returns = y_pred * y_true
        ## The negatives cancel out, so it is just exp(-risk aversion * returns)
        utility = tf.math.exp(-self.risk_aversion * returns)
        return tf.reduce_mean(utility, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(risk_aversion=self.risk_aversion)
        return cfg


class InformationDecisionPolicy(ABC): ...


class ReinforcementLearning(InformationDecisionPolicy):
    def __init__(self, learning_rate=0.01, entropy_coeff=0.05):
        self.logits = tf.Variable([0.0, 0.0])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.entropy_coeff = entropy_coeff
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


class ThompsonSampling(InformationDecisionPolicy): ...


class FixedInformation(InformationDecisionPolicy): ...


LOSS_REGISTRY = {1: NegCARA}
INFORMATION_POLICY_REGISTRY = {
    0: FixedInformation,
    1: ReinforcementLearning,
    2: ThompsonSampling,
}
