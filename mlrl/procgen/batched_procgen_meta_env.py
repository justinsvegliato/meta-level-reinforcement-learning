from mlrl.meta.meta_env import MetaEnv
from mlrl.meta.search_tree import SearchTreeNode
from mlrl.procgen.procgen_state import ProcgenProcessing, ProcgenState
from mlrl.procgen.procgen_env import make_vectorised_procgen

from typing import List, Tuple, Optional, Dict

from multiprocessing import pool 
from multiprocessing import dummy as mp_threads
import numpy as np
import gym

import tensorflow as tf
from tensorflow.python.util import nest
from tf_agents.environments.gym_wrapper import spec_from_gym_space
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class NodeExpansionRequest:

    def __init__(self,
                 node: SearchTreeNode[ProcgenState],
                 child_node: SearchTreeNode[ProcgenState],
                 action: int):
        self.node = node
        self.action = action
        self.child_node = child_node

    def get_state(self) -> bytes:
        return self.node.state.state[0]


class BatchedProcgenMetaEnv(PyEnvironment):

    def __init__(self,
                 meta_envs: List[MetaEnv],
                 max_expansions: int,
                 object_config: dict,
                 spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None,
                 simplify_box_bounds: bool = True,
                 match_obs_space_dtype: bool = True,
                 discount: types.Float = 0.99,
                 multithreading: bool = True,
                 auto_reset: bool = True):
        super(BatchedProcgenMetaEnv, self).__init__(auto_reset)

        self.meta_envs = meta_envs
        self.n_meta_envs = len(meta_envs)
        self.n_object_envs = max_expansions * len(meta_envs)
        self.object_config = object_config
        self.object_envs = make_vectorised_procgen(object_config,
                                                   n_envs=self.n_object_envs)

        env = meta_envs[0]

        self._observation_spec = spec_from_gym_space(env.observation_space,
                                                     spec_dtype_map,
                                                     simplify_box_bounds,
                                                     'observation')
        self._flat_obs_spec = tf.nest.flatten(self._observation_spec)

        self._action_spec = spec_from_gym_space(env.action_space,
                                                spec_dtype_map,
                                                simplify_box_bounds,
                                                'action')

        self.expansion_requests: List[NodeExpansionRequest] = []
        self.discount = discount
        self.match_obs_space_dtype = match_obs_space_dtype
        self.multithreading = multithreading

        if multithreading:
            self._pool = mp_threads.Pool(self.n_meta_envs)

    @property
    def envs(self):
        return self.meta_envs

    @property
    def batched(self) -> bool:
        return self.n_meta_envs > 1

    @property
    def batch_size(self) -> int:
        return self.n_meta_envs

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _reset(self):

        time_steps = [
            ts.restart(env.reset()) for env in self.meta_envs
        ]

        return nest_utils.stack_nested_arrays(time_steps)

    def patch_node(self, node: SearchTreeNode[ProcgenState]):

        def patched_create_child(_, object_action, new_node_id) -> SearchTreeNode:
            """
            Patched create_child method that creates a new node in the search tree
            without a populated state. The state variables will be set later
            when the expansion request is fulfilled.
            """
            new_node = SearchTreeNode(
                new_node_id, node, ProcgenState(), object_action,
                0, False, node.discount, node.q_function
            )

            self.expansion_requests.append(NodeExpansionRequest(node, new_node, object_action))

            return new_node

        node.create_child = patched_create_child

    def collect_expansion_requests(self, env: MetaEnv, meta_action: int):

        for node in env.tree.node_list:
            self.patch_node(node)

        env.act(meta_action)

    def handle_requests(self):

        states = self.object_envs.env.get_state()
        for i, request in enumerate(self.expansion_requests):
            states[i] = request.get_state()

        object_action = np.array([0] * self.n_object_envs)
        for i, request in enumerate(self.expansion_requests):
            object_action[i] = request.action

        self.object_envs.env.set_state(states)
        ts = self.object_envs.step(object_action)
        new_states = self.object_envs.env.get_state()

        n_requests = len(self.expansion_requests)
        state_vecs, q_values = ProcgenProcessing.call(ts.observation[:n_requests, ...])

        results = zip(self.expansion_requests,
                      state_vecs,
                      q_values,
                      new_states[:n_requests],
                      ts.observation,
                      ts.reward,
                      ts.is_last())

        for req, state_vec, q_value, new_state, obs, reward, done in results:
            req.child_node.reward = reward
            req.child_node.is_terminal_node = done
            req.child_node.state.set_variables([new_state], state_vec, obs, q_value)

    def _step(self, meta_actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.expansion_requests.clear()

        def collect_expansion_requests(i, action):
            self.collect_expansion_requests(i, action)

        if self.multithreading:
            self._pool.starmap(self.collect_expansion_requests,
                               zip(self.meta_envs, meta_actions))
        else:
            for i, action in enumerate(meta_actions):
                self.get_expansion_requests(i, action)

        if self.expansion_requests:
            self.handle_requests()

        if self.multithreading:
            time_steps = self._pool.map(lambda e: self._create_time_step(e), self.meta_envs)
        else:
            time_steps = [
                self._create_time_step(env) for env in self.meta_envs
            ]

        return nest_utils.stack_nested_arrays(time_steps)

    def _create_time_step(self, env: MetaEnv):
        observation, reward, done, _ = env.observe()
        step_type = ts.StepType.LAST if done else ts.StepType.MID

        if self.match_obs_space_dtype:
            observation = self._to_obs_space_dtype(observation)

        return ts.TimeStep(step_type=step_type,
                           reward=reward,
                           discount=self.discount,
                           observation=observation)

    def _to_obs_space_dtype(self, observation):
        """
        Make sure observation matches the specified space.
        Observation spaces in gym didn't have a dtype for a long time. Now that they
        do there is a large number of environments that do not follow the dtype in
        the space definition. Since we use the space definition to create the
        tensorflow graph we need to make sure observations match the expected
        dtypes.
        Args:
            observation: Observation to match the dtype on.
        Returns:
            The observation with a dtype matching the observation spec.
        """
        # Make sure we handle cases where observations are provided as a list.
        flat_obs = nest.flatten_up_to(self._observation_spec, observation)

        matched_observations = []
        for spec, obs in zip(self._flat_obs_spec, flat_obs):
            matched_observations.append(np.asarray(obs, dtype=spec.dtype))

        return tf.nest.pack_sequence_as(self._observation_spec,
                                        matched_observations)

    def render(self, mode='rgb_array'):
        return np.vstack([env.render(mode=mode) for env in self.meta_envs])
