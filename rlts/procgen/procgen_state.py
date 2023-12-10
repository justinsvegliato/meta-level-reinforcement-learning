from rlts.meta.search_tree import ObjectState
from rlts.utils import one_hot

from typing import List, Tuple
from time import time

import tensorflow as tf
import numpy as np

from gym3.env import Env
from gym3.interop import ToGymEnv
from procgen.env import ProcgenGym3Env


class ProcgenProcessing:
    __instance = None

    def __init__(self, agent):
        self.agent = agent
        self.categorical_q_net = agent._q_network
        self.support = agent._support

    @staticmethod
    def set_pretrained_agent(agent):
        ProcgenProcessing.__instance = ProcgenProcessing(agent)

    @staticmethod
    def get_agent():
        return ProcgenProcessing.__instance.agent

    @staticmethod
    def call(observation) -> Tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def reset_actions():
        ProcgenState.COMBO_STRINGS = [
            '+'.join(combo)
            for combo in ProcgenGym3Env.get_combos(None)
        ]
        ProcgenState.ACTIONS = list(range(len(ProcgenState.COMBO_STRINGS)))

    @staticmethod
    def set_actions(actions: List[str]):
        ProcgenState.reset_actions()
        action_idxs = [ProcgenState.COMBO_STRINGS.index(action) for action in actions]
        ProcgenState.COMBO_STRINGS = actions
        ProcgenState.ACTIONS = action_idxs

    def __init__(self, env: Env = None):
        """
        If an environment is provided, the state and object variables are extracted from it.
        Otherwise, set_variables must be called later.
        """
        if env is not None:
            if isinstance(env, ToGymEnv):
                env = env.env
                _, self.observation, *_ = env.observe()
                self.observation = self.observation / 255.  # need to later handle frame stack and grayscale options
            else:
                self.observation = env.current_time_step().observation

            self.state = env.callmethod('get_state')
            self.state_vec, self.q_values = ProcgenProcessing.call(self.observation)

            self._state_str = str(self.state)
            self._hash_value = hash(self._state_str)
        else:
            self.state_vec = None
            self.state = None
            self.observation = None
            self.q_values = None
            self._state_str = None
            self._hash_value = hash(time())

    def set_variables(self, state, state_vec: np.array, observation: np.array, q_values: np.array):
        self.state = state
        self.state_vec = state_vec
        self.observation = observation
        self.q_values = q_values

        self._state_str = str(self.state)
        self._hash_value = hash(self._state_str)

    def get_observation(self):
        return self.observation

    def get_q_values(self):
        return self.q_values

    def set_environment_to_state(self, env: Env):
        if isinstance(env, ToGymEnv):
            env = env.env
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
        return one_hot(ProcgenState.ACTIONS.index(action), n)

    def get_state_string(self) -> str:
        return self._state_str

    def __hash__(self):
        return self._hash_value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({hash(self)})'
