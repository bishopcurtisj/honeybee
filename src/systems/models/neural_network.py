import gc
from functools import singledispatch

import numpy as jnp
import numpy as np
import scipy.optimize as opt
import tensorflow as tf

from ACE_Experiment.globals import config, globals
from systems.models.model import INFORMATION_POLICY_REGISTRY, LOSS_REGISTRY, Model


class NeuralNetwork(Model):
    """
    agents: jnp.ndarray
    """

    def __init__(self, agents: jnp.ndarray):
        self.models = {}
        self.agents = agents
        for agent in agents:
            self.models[agent[globals.components.agent_id]] = {}
            self.models[agent[globals.components.agent_id]][
                "input_shape",
                "hidden_layers",
                "hidden_nodes",
                "demand_fx_params",
                "optimizer",
                "epochs",
                "loss",
                "optimization_steps",
                "learning_rate",
            ] = agent[globals.components.learning_params]
            self.models[agent[globals.components.agent_id]]["loss"] = LOSS_REGISTRY[
                self.models[agent[globals.components.agent_id]]["loss"]
            ](agent[globals.components.loss_params])
            model = self._build_model(
                self.models[agent[globals.components.agent_id]][
                    "input_shape", "hidden_layers", "hidden_nodes", "demand_fx_params"
                ].values()
            )
            if config.memory_optimization:
                path = f"models/{agent[globals.components.agent_id]}.keras"
                model.save(path)
                self.models[agent[globals.components.agent_id]]["model_ref"] = path
                tf.keras.backend.clear_session()
                del model
                gc.collect()
            else:
                self.models[agent[globals.components.agent_id]]["model_ref"] = model

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
        util_func: int,
    ) -> jnp.ndarray:
        inputs, outputs = globals.trades[:, 1], globals.trades[:, 2]
        return inputs, outputs

    def _prepare_uninformed_training_data(self) -> jnp.ndarray:
        sample = jnp.random.choice(
            globals.trades, size=(len(globals.trades) * config.uninformed_base_ratio)
        )
        inputs, outputs = sample[:, 1], sample[:, 2]
        return inputs, outputs

    ## refactor to add informed agents
    def __call__(self, nn_learners: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:

        if globals.informed:
            nn_learners[
                jnp.where(globals.agents[:, globals.components.informed] == 1)[0]
            ] = self.informed_agents(
                nn_learners[
                    jnp.where(globals.agents[:, globals.components.informed] == 1)[0]
                ]
            )
            nn_learners[
                jnp.where(globals.agents[:, globals.components.informed] == 0)[0]
            ] = self.uninformed_agents(
                nn_learners[
                    jnp.where(globals.agents[:, globals.components.informed] == 0)[0]
                ]
            )

        else:
            nn_learners = self.informed_agents(nn_learners)

        return nn_learners

    def informed_agents(self, nn_learners: jnp.ndarray) -> jnp.ndarray:
        for i in range(len(nn_learners)):
            agent = nn_learners[i]
            X_train, y_train = self._prepare_training_data(
                agent[globals.components.demand_fx_params[0]]
            )
            model_info = self.models[agent[globals.components.id]]
            model: tf.keras.Model = self._load_model(model_info["model_ref"])
            model.compile(optimizer=model_info["optimizer"], loss=model_info["loss"])
            model.fit(X_train, y_train, epochs=model_info["epochs"])

        return nn_learners

    def uninformed_agents(self, nn_learners: jnp.ndarray) -> jnp.ndarray:
        for i in range(len(nn_learners)):
            agent = nn_learners[i]
            X_train, y_train = self._prepare_uninformed_training_data(
                agent[globals.components.demand_fx_params[0]]
            )
            model_info = self.models[agent[globals.components.id]]
            model: tf.keras.Model = self._load_model(model_info["model_ref"])
            model.compile(optimizer=model_info["optimizer"], loss=model_info["loss"])
            model.fit(X_train, y_train, epochs=model_info["epochs"])
            self._save_model(model_info["model_ref"])

        return nn_learners
