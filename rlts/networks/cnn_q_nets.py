import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.agents import CategoricalDqnAgent
from tf_agents.environments.py_environment import PyEnvironment


class RainbowQNet(tf.keras.Sequential):
    """
    Model for the Rainbow DQN agent. This model is a wrapper around the CategoricalQNetwork from the tf_agents library.
    The purpose of this wrapper is to allow the model to be saved and loaded as a single object, with the optimizer state
    for resuming training.
    """

    def __init__(self,
                 observation_shape: tuple,
                 n_actions: int,
                 learning_rate: float,
                 fc_layer_params: list = None,
                 conv_layer_params: list = None,
                 name='RainbowQNet'):
        super(RainbowQNet, self).__init__(name=name)

        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.fc_layer_params = fc_layer_params or [512]
        self.conv_layer_params = conv_layer_params or [(64, 8, 4), (64, 4, 2), (64, 3, 2)]

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.network = CategoricalQNetwork(
            tensor_spec.TensorSpec(shape=observation_shape, dtype=tf.float64, name='observation'),
            tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.int64, minimum=0, maximum=n_actions - 1, name='action'),
            conv_layer_params=self.conv_layer_params,
            fc_layer_params=self.fc_layer_params,
        )

        self.add(self.network)

        self(tf.keras.Input(shape=(1,) + tuple(observation_shape), dtype=tf.float64))

    def get_config(self):
        return {
            'observation_shape': self.observation_shape,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate
        }

    @classmethod
    def from_config(cls, config: dict) -> 'RainbowQNet':
        return cls(**config)

    def create_agent(self, env: PyEnvironment, config: dict) -> CategoricalDqnAgent:
        agent = CategoricalDqnAgent(
            env.time_step_spec(),
            env.action_spec(),
            categorical_q_network=self.network,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            target_update_period=config.get('target_update_period', 10000),
            min_q_value=config.get('r_min', 0),
            max_q_value=config.get('r_max', 1),
            gamma=config.get('discount', 0.99),
            train_step_counter=tf.Variable(0)
        )

        agent.initialize()

        return agent

    @staticmethod
    def load(path: str, config: dict = None) -> 'RainbowQNet':
        try:
            return tf.keras.models.load_model(path, custom_objects={'RainbowQNet': RainbowQNet})
        except IOError or ImportError:
            print('Failed to load model from path: {}'.format(path))
        
        try:
            print('Trying to load from weights')
            model = RainbowQNet(**config)
            model.load_weights(path)
            return model
        except ValueError or ImportError:
            print('Failed to load weights from path: {}'.format(path))

        raise IOError('Failed to load model from path: {}'.format(path))    
