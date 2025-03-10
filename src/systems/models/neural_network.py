import numpy as jnp
import tensorflow as tf
import pickle
import scipy.optimize as opt

from entities.agent import AgentInfo

class NeuralNetwork:

    def __init__(self, agents: jnp.ndarray, components: AgentInfo):
        self.models = {}
        for agent in agents:
            self.models[agent[components.agent_id]] = {}
            self.models[agent[components.agent_id]]['input_shape', 'hidden_layers', 'hidden_nodes', 'demand_fx_params', 'optimizer', 'epochs', 'loss', 'optimization_steps', 'learning_rate'] = agent[components.learning_params]
            model = self._build_model(self.models[agent[components.agent_id]]['input_shape', 'hidden_layers', 'hidden_nodes', 'demand_fx_params'].values())
            path = pickle.dumps(model)
            self.models[agent[components.agent_id]]['path'] = path

        
    ## Right now assumes informed agents exist
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

    def _reward(self, optimal_input, model) -> jnp.ndarray:
        ## maximize utility / minimize negative utility
        return -1 * self.model.predict(optimal_input)

    def _prepare_training_data(self, agents) -> jnp.ndarray:
        inputs = agents[:, 2:]
        outputs = agents[:, 1]
        return inputs, outputs

    def __call__(self, agents: jnp.ndarray, nn_learners: jnp.ndarray) -> jnp.ndarray:
        """
        agents: jnp.ndarray
            Agents should have the columns:
            [id, fitness, informed, demand_function_params...] 
        """
        X_train, y_train = self._prepare_training_data(agents)
        for i in range(len(agents)):
            agent = agents[i]
            if self.models.get(agent[0], False):
                model = self._load_model(agent[0])
                model_info = self.models[agent[0]]
                model.compile(optimizer=model_info['optimizer'], loss=model_info['loss'])
                model.fit(X_train, y_train, epochs=model_info['epochs'])
                self.model_paths[agent] = pickle.dumps(model)
            
                # calculate optimal input: start with current values
                optimal_input = agents[nn_learners][2:]
                optimal_input = opt.minimize(self._reward, optimal_input, args=(model), method='Nelder-Mead')

                optimal_input = optimal_input.numpy()
                optimal_input[0] = jnp.round(optimal_input[0], 0)
                agents[nn_learners][2:] = jnp.array(optimal_input)

        return agents