import numpy as jnp
import tensorflow as tf
import pickle

from entities.agent import AgentInfo

class NeuralNetwork:

    def __init__(self, agents: jnp.ndarray, components: AgentInfo):
        self.model_paths = {}
        for agent in agents:
            model = self._build_model(agent[components.learning_params])
            path = pickle.dumps(model)
            self.model_paths[agent[components.agent_id]] = path

        
    ## Right now assumes informed agents exist
    def _build_model(self, params: jnp.ndarray) -> tf.keras.Model:
        input_shape, hidden_layers, hidden_nodes, demand_fx_params = params
        inputs = tf.keras.Input(shape=(input_shape,))
        x = tf.keras.layers.Dense(hidden_nodes, activation='linear')(inputs)

        for _ in range(hidden_layers-1):
            x = tf.keras.layers.Dense(hidden_nodes, activation='linear')(x)

        output_a = tf.keras.layers.Dense(demand_fx_params, activation='linear')(x)
        output_b = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=[output_a, output_b])
        return model
    
    def _load_model(self, agent_id: int) -> tf.keras.Model:
        return pickle.loads(self.model_paths[agent_id])

    def _reward(self) -> jnp.ndarray:
        ## maximize utility / minimize negative utility
        ...

    def __call__(self, agents: jnp.ndarray) -> jnp.ndarray:
        ...