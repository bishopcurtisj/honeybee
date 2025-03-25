import pickle

import numpy as jnp
import tensorflow as tf
import scipy.optimize as opt

from ACE_Experiment.globals import globals, config
from entities.trades import calculate_trade_utility
from systems.models.model import Model


class NeuralNetwork(Model):

    """
    agents: jnp.ndarray
        Agents should have the columns:
        [id, fitness, informed, demand_function_params...] 
    """

    def __init__(self, agents: jnp.ndarray):
        self.models = {}
        self.agents = agents
        for agent in agents:
            self.models[agent[globals.components.agent_id]] = {}
            self.models[agent[globals.components.agent_id]]['input_shape', 'hidden_layers', 'hidden_nodes', 'demand_fx_params', 'optimizer', 'epochs', 'loss', 'optimization_steps', 'learning_rate'] = agent[globals.components.learning_params]
            model = self._build_model(self.models[agent[globals.components.agent_id]]['input_shape', 'hidden_layers', 'hidden_nodes', 'demand_fx_params'].values())
            path = pickle.dumps(model)
            self.models[agent[globals.components.agent_id]]['path'] = path

        
    def _build_model(self, params) -> tf.keras.Model:
        input_shape, hidden_layers, hidden_nodes = params
        inputs = tf.keras.Input(shape=(input_shape,))
        x = tf.keras.layers.Dense(hidden_nodes, activation='linear')(inputs)

        for _ in range(hidden_layers-1):
            x = tf.keras.layers.Dense(hidden_nodes, activation='linear')(x)

        output = tf.keras.layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model
    
    def _load_model(self, agent_id: int) -> dict:
        return pickle.loads(self.model_paths[agent_id]['path'])


    def _prepare_training_data(self, util_func: int) -> jnp.ndarray:
        outputs = calculate_trade_utility(globals.trades, util_func)
        return globals.trades, outputs
    
    def _prepare_uninformed_training_data(self, util_func: int) -> jnp.ndarray:
        inputs = jnp.random.choice(globals.trades, size=(len(globals.trades)*config.uninformed_base_ratio))
        outputs = calculate_trade_utility(inputs, util_func)
        return inputs, outputs

    def __call__(self,  nn_learners: jnp.ndarray, params) -> jnp.ndarray:
        
        if globals.informed:
            nn_learners[jnp.where(globals.agents[:,globals.components.informed] == 1)[0]] = self.informed_agents(nn_learners[jnp.where(globals.agents[:,globals.components.informed] == 1)[0]])
            nn_learners[jnp.where(globals.agents[:,globals.components.informed] == 0)[0]] = self.uninformed_agents(nn_learners[jnp.where(globals.agents[:,globals.components.informed] == 0)[0]])

        else:
            return self.informed_agents(nn_learners)

    def informed_agents(self, nn_learners: jnp.ndarray) -> jnp.ndarray:

        for i in range(len(nn_learners)):
            agent = nn_learners[i]
            X_train, y_train = self._prepare_training_data(agent[globals.components.demand_fx_params[0]])
            model:  tf.keras.Model = self._load_model(agent[0])
            model_info = self.models[agent[0]]
            model.compile(optimizer=model_info['optimizer'], loss=model_info['loss'])
            model.fit(X_train, y_train, epochs=model_info['epochs'])
            self.model_paths[agent] = pickle.dumps(model)
                          

        return nn_learners
    
    def uninformed_agents(self, nn_learners: jnp.ndarray) -> jnp.ndarray:
        for i in range(len(nn_learners)):
            agent = nn_learners[i]
            X_train, y_train = self._prepare_uninformed_training_data(agent[globals.components.demand_fx_params[0]])
            model:  tf.keras.Model = self._load_model(agent[0])
            model_info = self.models[agent[0]]
            model.compile(optimizer=model_info['optimizer'], loss=model_info['loss'])
            model.fit(X_train, y_train, epochs=model_info['epochs'])
            self.model_paths[agent] = pickle.dumps(model)
                          

        return nn_learners