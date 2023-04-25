from mlrl.meta.search_tree import ObjectState
from mlrl.utils import one_hot

from typing import List

import tensorflow as tf
import numpy as np

from gym3.env import Env
from procgen.env import ProcgenGym3Env


class ProcgenProcessing:
    __instance = None

    def __init__(self, agent):
        self.categorical_q_net = agent._q_network
        self.support = agent._support

    @staticmethod
    def set_pretrained_agent(agent):
        ProcgenProcessing.__instance = ProcgenProcessing(agent)

    @staticmethod
    def call(observation) -> np.ndarray:
        processor = ProcgenProcessing.__instance
        categorical_q_net = processor.categorical_q_net
        n_actions = categorical_q_net._num_actions
        n_atoms = categorical_q_net._num_atoms

        encs, _ = categorical_q_net._q_network._encoder(observation)

        q_logits = categorical_q_net._q_network._q_value_layer(encs)
        q_logits = tf.reshape(q_logits, [-1, n_actions, n_atoms])
        q_probabilities = tf.nn.softmax(q_logits)
        q_values = tf.reduce_sum(processor.support * q_probabilities, axis=-1)

        return encs.numpy().squeeze(), q_values.numpy().squeeze()


class ProcgenState(ObjectState):
    """
    Class to handle represent a state of the maze environment.
    The state vector is a 2D vector of the robot position in the maze.
    The gym state is a tuple containing the necessary information
    to set the environment to this state.
    """
    __slots__ = ['state_vec', 'state', 'observation', 'q_values']

    COMBO_STRINGS = [
        '+'.join(combo)
        for combo in ProcgenGym3Env.get_combos(None)
    ]

    ACTIONS = list(range(len(COMBO_STRINGS)))

    @staticmethod
    def extract_state(env: Env) -> 'ProcgenState':
        """
        A static method to extract the state of the environment
        for later restoring and representation.
        """
        return ProcgenState(env)

    def __init__(self, env: Env):
        self.state = env.callmethod('get_state')
        self.observation = env.current_time_step().observation
        self.state_vec, self.q_values = ProcgenProcessing.call(self.observation)

    def set_environment_to_state(self, env: Env):
        env.callmethod('set_state', self.state)

    def get_state_vector(self) -> np.array:
        return np.array(self.state_vec, dtype=np.float32)

    def get_maximum_number_of_actions(self):
        return len(ProcgenState.COMBO_STRINGS)

    def get_actions(self) -> List[int]:
        return ProcgenState.ACTIONS

    def get_action_labels(self) -> List[str]:
        return ProcgenState.COMBO_STRINGS

    def get_action_vector(self, action: int) -> np.array:
        n = self.get_maximum_number_of_actions()
        return one_hot(action, n)

    def get_state_string(self) -> str:
        return ''

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.get_state_string()})'