import gc
from functools import singledispatch

import numpy as jnp
import tensorflow as tf

from src.globals import config, globals
from systems.models.information_policy import INFORMATION_POLICY_REGISTRY
from systems.models.loss import LOSS_REGISTRY
from systems.models.model import Model


class NeuralNetwork(Model):
    """
    agents: jnp.ndarray
    """

    def __init__(self, agents: jnp.ndarray):
        self.models = {}
        self.agents = agents
        self.agent_id = globals.components.agent_id
        self.informed = globals.components.informed

        for agent in agents:
            self.models[agent[self.agent_id]] = {
                "input_shape": agent[globals.components.input_shape],
                "hidden_layers": agent[globals.components.hidden_layers],
                "hidden_nodes": agent[globals.components.hidden_nodes],
                "epochs": agent[globals.components.epochs],
            }
            self.models[agent[self.agent_id]]["optimizer"] = OPTIMIZER_REGISTRY[
                agent[globals.components.optimizer]
            ]

            learning_rate, entropy_coeff, update_frequency = agent[
                globals.components["learning_rate", "entropy_coeff", "update_frequency"]
            ]  ## Update this for new InformationDecisionPolicies
            self.models[agent[self.agent_id]][
                "info_policy"
            ] = INFORMATION_POLICY_REGISTRY[
                agent[globals.components.information_policy]
            ](
                agent=agent,
                learning_rate=learning_rate,
                entropy_coeff=entropy_coeff,
                update_frequency=update_frequency,
            )
            self.models[agent[self.agent_id]]["loss"] = LOSS_REGISTRY[
                agent[globals.components.loss]
            ](agent)
            model = self._build_model(
                [
                    self.models[agent[self.agent_id]][k]
                    for k in ("input_shape", "hidden_layers", "hidden_nodes")
                ]
            )
            if config.memory_optimization:
                path = f"models/{agent[self.agent_id]}.keras"
                model.save(path)
                self.models[agent[self.agent_id]]["model_ref"] = path
                tf.keras.backend.clear_session()
                del model
                gc.collect()
            else:
                self.models[agent[self.agent_id]]["model_ref"] = model

    def _build_model(self, params) -> tf.keras.Model:
        input_shape, hidden_layers, hidden_nodes = params
        inputs = tf.keras.Input(shape=(input_shape,))
        x = tf.keras.layers.Dense(hidden_nodes, activation="relu")(inputs)

        for _ in range(hidden_layers - 1):
            x = tf.keras.layers.Dense(hidden_nodes, activation="relu")(x)

        output = tf.keras.layers.Dense(1, activation="linear")(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    @singledispatch
    def _load_model(self, model_ref) -> tf.keras.Model:
        raise TypeError(f"Unsupported type {type(model_ref)}")

    @_load_model.register
    def _(self, model_ref: str) -> tf.keras.Model:
        return tf.keras.models.load_model(model_ref)

    @_load_model.register
    def _(self, model_ref: tf.keras.models.Model) -> tf.keras.Model:
        return model_ref

    @singledispatch
    def _save_model(self, model_ref, model: tf.keras.models.Model, agent_id: int):
        raise TypeError(f"Unsupported type {type(model_ref)}")

    @_save_model.register
    def _(self, model_ref: str, model: tf.keras.models.Model):
        model.save(model_ref)
        tf.keras.backend.clear_session()  # 1 flush Keras’ global graph state  :contentReference[oaicite:0]{index=0}
        del model  # 2 drop Python reference  :contentReference[oaicite:1]{index=1}
        gc.collect()  # 3 ask the GC to reclaim immediately

    @_save_model.register
    def _(self, model_ref: tf.keras.models.Model, model: tf.keras.models.Model):
        pass

    def _prepare_training_data(
        self,
    ) -> jnp.ndarray:
        inputs, outputs = globals.trades[:, 1], globals.trades[:, 2]
        return inputs, outputs

    def _prepare_uninformed_training_data(self) -> jnp.ndarray:
        sample = jnp.random.choice(
            globals.trades, size=(len(globals.trades) * config.uninformed_base_ratio)
        )
        inputs, outputs = sample[:, 1], sample[:, 2]
        return inputs, outputs

    def __call__(self, nn_learners: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:

        if globals.informed:
            nn_learners[jnp.where(globals.agents[:, self.informed] == 1)[0]] = (
                self.informed_agents(
                    nn_learners[jnp.where(globals.agents[:, self.informed] == 1)[0]]
                )
            )
            nn_learners[jnp.where(globals.agents[:, self.informed] == 0)[0]] = (
                self.uninformed_agents(
                    nn_learners[jnp.where(globals.agents[:, self.informed] == 0)[0]]
                )
            )

        else:
            nn_learners = self._informed(nn_learners)

        return nn_learners

    def _informed(self, nn_learners: jnp.ndarray) -> jnp.ndarray:
        for agent in nn_learners:
            X_train, y_train = self._prepare_training_data()
            model_info = self.models[agent[self.agent_id]]
            model: tf.keras.Model = self._load_model(model_info["model_ref"])
            model.compile(optimizer=model_info["optimizer"], loss=model_info["loss"])
            model.fit(X_train, y_train, epochs=model_info["epochs"])

            agent = model_info["info_policy"](agent)

    def _uninformed(self, nn_learners: jnp.ndarray) -> jnp.ndarray:
        for agent in nn_learners:
            X_train, y_train = self._prepare_uninformed_training_data()
            model_info = self.models[agent[self.agent_id]]
            model: tf.keras.Model = self._load_model(model_info["model_ref"])
            model.compile(optimizer=model_info["optimizer"], loss=model_info["loss"])
            model.fit(X_train, y_train, epochs=model_info["epochs"])
            self._save_model(model_info["model_ref"])

        return nn_learners


OPTIMIZER_REGISTRY = {1: "adam"}
