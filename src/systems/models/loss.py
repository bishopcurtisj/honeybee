from abc import ABC, ABCMeta, abstractmethod
from typing import List, Union

import numpy as jnp
import numpy as np
import tensorflow as tf

from globals import globals


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

    name: str = "neg_cara"
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE

    def __init__(self, *args, **kwargs):
        super().__init__(reduction=self.reduction, name=self.name, **kwargs)
        agent = args[0]
        self.risk_aversion = agent[globals.components.risk_aversion]

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


def register_loss(losses: Union[List[AgentLoss], AgentLoss]):
    if type(losses) == AgentLoss:
        losses = [losses]
    for loss in losses:
        try:
            assert issubclass(loss, AgentLoss)
        except AssertionError:
            raise ValueError(
                f"Custom loss function {loss.name} must be a subclass of AgentLoss"
            )
        LOSS_REGISTRY[len(LOSS_REGISTRY)] = loss


LOSS_REGISTRY = {1: NegCARA}
