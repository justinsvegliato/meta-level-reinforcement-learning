from rlts.meta.meta_env import MetaEnv
from rlts.meta.search_tree import SearchTree, SearchTreeNode
from rlts.procgen.procgen_state import ProcgenProcessing, ProcgenState
from rlts.procgen.procgen_env import make_vectorised_procgen

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

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


class ProcgenProcessingRequest(ABC):

    def __init__(self,
                 node: SearchTreeNode[ProcgenState],
                 action: int):
        self.node = node
        self.action = action

    def get_state(self) -> bytes:
        return self.node.state.state[0]

    @abstractmethod
    def process_result(self, state_vec, q_value, new_state, obs, reward, done):
        pass


class NodeExpansionRequest(ProcgenProcessingRequest):

    def __init__(self,
                 node: SearchTreeNode[ProcgenState],
                 child_node: SearchTreeNode[ProcgenState],
                 action: int):
        super().__init__(node, action)
        self.child_node = child_node

    def process_result(self, state_vec, q_value, new_state, obs, reward, done):
        self.child_node.reward = reward
        self.child_node.is_terminal_state = done
        self.child_node.state.set_variables([new_state], state_vec, obs, q_value)


class TerminateProcessRequest(ProcgenProcessingRequest):

    def __init__(self, meta_env: MetaEnv):
        super().__init__(
            meta_env.tree.root_node,
            meta_env.get_best_object_action()
        )
        self.meta_env = meta_env

    def process_result(self, state_vec, q_value, new_state, obs, reward, done):
        """
        Recreates the logic in MetaEnv.terminate but with data from batched
        ProcgenProcessing
        """
        new_tree = SearchTree(
            self.meta_env.object_env,
            ProcgenState(),
            self.meta_env.tree.q_function,
            deterministic=self.meta_env.tree.deterministic,
            discount=self.meta_env.tree.discount
        )
        new_tree.root_node.state.set_variables([new_state], state_vec, obs, q_value)
        new_tree.root_node.is_terminal_state = done

        self.meta_env.tree = new_tree
        self.meta_env.last_object_level_reward = reward

        # Notify observers and update variables
        for observer in self.meta_env.object_level_transition_observers:
            observer(obs, self.action, reward, done, {})

        self.meta_env.search_tree_policy = self.meta_env.make_tree_policy(self.meta_env.tree)

        self.meta_env.reset_computation_variables()


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
                 patch_terminates: bool = True,
                 patch_expansions: bool = True,
                 auto_reset: bool = True):
        super(BatchedProcgenMetaEnv, self).__init__(auto_reset)

        self.meta_envs = meta_envs
        self.n_meta_envs = len(meta_envs)
        self.max_expansions = max_expansions
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

        self.expansion_requests: List[ProcgenProcessingRequest] = []
        self.discount = discount
        self.match_obs_space_dtype = match_obs_space_dtype
        self.multithreading = multithreading

        self.patch_terminates = patch_terminates
        self.patch_expansions = patch_expansions

        self.states = None

        if multithreading:
            self._pool = mp_threads.Pool(self.n_meta_envs)

    def remove_env(self, env: MetaEnv):
        if env not in self.meta_envs:
            return

        if self.n_meta_envs == 1:
            print("Cannot remove last environment")
            return

        self.meta_envs.remove(env)
        self.n_meta_envs = len(self.meta_envs)
        self.n_object_envs = max(1, self.max_expansions * len(self.meta_envs))
        self.object_envs = make_vectorised_procgen(self.object_config,
                                                   n_envs=self.n_object_envs)

        if self.multithreading and self.n_meta_envs > 0:
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

    def patch_terminate(self, meta_env: MetaEnv):

        def patched_terminate():
            self.expansion_requests.append(TerminateProcessRequest(meta_env))
    
        meta_env.terminate = patched_terminate

    def collect_expansion_requests(self, meta_env: MetaEnv, meta_action: int):

        if self.patch_expansions:
            for node in meta_env.tree.node_list:
                self.patch_node(node)

        if self.patch_terminates:
            self.patch_terminate(meta_env)

        meta_env.act(meta_action)

    def handle_requests(self):

        self.states = self.states or self.object_envs.env.get_state()
        object_action = np.array([0] * self.n_object_envs)
        for i, request in enumerate(self.expansion_requests):
            self.states[i] = request.get_state()
            object_action[i] = request.action

        self.object_envs.env.set_state(self.states)
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
            req.process_result(state_vec, q_value, new_state, obs, reward, done)

    def _step(self, meta_actions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        self.expansion_requests.clear()

        if self.multithreading:
            self._pool.starmap(self.collect_expansion_requests,
                               zip(self.meta_envs, meta_actions))
        else:
            for meta_env, action in zip(self.meta_envs, meta_actions):
                self.collect_expansion_requests(meta_env, action)

        if self.expansion_requests:
            self.handle_requests()

        return self.create_time_step()

    def create_time_step(self):

        if self.multithreading:
            time_steps = self._pool.map(self._create_time_step, self.meta_envs)
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
