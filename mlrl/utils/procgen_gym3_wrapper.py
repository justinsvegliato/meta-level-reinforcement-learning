from typing import Any, Dict, Optional, Text

import numpy as np

import gym
import gym.spaces
from gym3.interop import _vt2space
import procgen

import tensorflow as tf
from tensorflow.python.util import nest

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.environments.gym_wrapper import spec_from_gym_space


class ProcgenGym3Wrapper(py_environment.PyEnvironment):
    """
    Base wrapper implementing PyEnvironmentBaseWrapper interface for Gym envs.
    Action and observation specs are automatically generated from the action and
    observation spaces. See base class for py_environment.Base details.

    Based on the GymWrapper class and ToGymEnv classes from tf_agents and gym3.
    """

    def __init__(self,
                 gym_env: procgen.ProcgenGym3Env,
                 discount: types.Float = 1.0,
                 action_repeats: types.Int = 0,
                 match_obs_space_dtype: bool = True,
                 simplify_box_bounds: bool = True,
                 render_kwargs: Optional[Dict[str, Any]] = None,
                 spec_dtype_map: Optional[Dict[gym.Space, np.dtype]] = None):
        """
        Initializes the wrapper.

        Args:
            gym_env: The procgen gym3 environment to wrap. Assumed to contain multiple envs.
            discount: Discount to use for the environment.
            action_repeats: Number of times to repeat the action with each step. 
                If less than 2, then the action is only applied once.
            spec_dtype_map: A dictionary mapping gym spaces to dtypes.
        """
        super(ProcgenGym3Wrapper, self).__init__(handle_auto_reset=False)

        self._n_envs = gym_env.num
        self._gym_env = gym_env
        self._discount = np.array([discount] * self._n_envs)
        self._action_repeats = action_repeats if action_repeats > 0 else 1

        self._gym_observation_space = _vt2space(gym_env.ob_space)
        self._gym_action_space = _vt2space(gym_env.ac_space)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

        self._action_is_discrete = isinstance(self._gym_action_space,
                                              gym.spaces.Discrete)
        self._match_obs_space_dtype = match_obs_space_dtype
        self._observation_spec = spec_from_gym_space(self._gym_observation_space,
                                                     spec_dtype_map,
                                                     simplify_box_bounds,
                                                     'observation')
        self._action_spec = spec_from_gym_space(self._gym_action_space,
                                                spec_dtype_map,
                                                simplify_box_bounds,
                                                'action')
        self._flat_obs_spec = tf.nest.flatten(self._observation_spec)
        self._render_kwargs = render_kwargs or {}
        self._info = None
        self._done = [False] * self._n_envs
        self._is_first = [True] * self._n_envs

    @property
    def gym(self) -> gym.Env:
        return self._gym_env

    @property
    def batched(self) -> bool:
        return self._n_envs > 1

    @property
    def batch_size(self) -> int:
        return self._n_envs

    def __getattr__(self, name: Text) -> Any:
        """Forward all other calls to the base environment."""
        gym_env = super(ProcgenGym3Wrapper, self).__getattribute__('_gym_env')
        return getattr(gym_env, name)

    def get_info(self) -> Any:
        """Returns the gym environment info returned on the last step."""
        return self._info

    def _reset(self):
        self._current_time_step = self._create_time_step()
        return self._current_time_step

    @property
    def done(self) -> bool:
        return all(self._done)

    def _step(self, action):

        if type(action) in [int, np.int32, np.int64]:
            action = np.array([action])

        elif isinstance(action, tf.Tensor):
            action = action.numpy()

        if self._action_repeats > 1:
            reward = 0
            for _ in range(self._action_repeats):
                self._gym_env.act(action)
                r, _, done = self._gym_env.observe()
                reward += r
                if any(done):
                    break

            self._current_time_step = self._create_time_step(reward=reward)

        else:
            self._gym_env.act(action)
            self._current_time_step = self._create_time_step()

        return self._current_time_step

    def _create_time_step(self, reward=None):
        r, observation, self._done = self._gym_env.observe()
        if reward is None:
            reward = r

        step_type = [ts.StepType.MID] * self._n_envs

        for i in range(self._n_envs):
            if self._is_first[i]:
                step_type[i] = ts.StepType.FIRST
                self._is_first[i] = False
            elif self._done[i]:
                step_type[i] = ts.StepType.LAST
                self._is_first[i] = True

        return ts.TimeStep(step_type=np.array(step_type),
                           reward=reward,
                           discount=self._discount,
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

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def close(self) -> None:
        return self._gym_env.close()

    def seed(self, seed: types.Seed) -> types.Seed:
        seed_value = self._gym_env.seed(seed)
        if seed_value is None:
            return 0
        return seed_value

    def render(self, mode: Text = 'rgb_array') -> Any:
        return self._gym_env.render(mode, **self._render_kwargs).squeeze()

    def set_state(self, state: Any) -> None:
        return self._gym_env.set_state(state)

    def get_state(self) -> Any:
        return self._gym_env.get_state()
