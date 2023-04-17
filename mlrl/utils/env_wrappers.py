import collections
import copy
import numpy as np

from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories.time_step import TimeStep


class ImagePreprocessWrapper(wrappers.PyEnvironmentBaseWrapper):
    """normalise and grayscale observations of env."""

    def __init__(self, env, grayscale=True, normalise=True):
        """Initializes a wrapper."""
        super(ImagePreprocessWrapper, self).__init__(env)

        self.grayscale = grayscale
        self.normalise = normalise

        # Update the observation spec in the environment.
        observation_spec = env.observation_spec()

        # Update the observation spec.
        self._new_observation_spec = copy.copy(observation_spec)

        # Redefine pixels spec
        frame_shape = observation_spec.shape
        if grayscale:
            shape = frame_shape[:2] + (1,)
        else:
            shape = frame_shape

        self._new_observation_spec = array_spec.ArraySpec(
            shape=shape,
            dtype=np.float64,
            name='grayscale_pixels')

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self._env.current_time_step().observation

    def preprocess(self, img):
        """Preprocesses the image."""
        if self.normalise:
            img = self._env.current_time_step().observation / 255.
        if self.grayscale:
            img = np.mean(img, axis=-1)
        return img

    def _step(self, action):
        """Steps the environment."""
        time_step = self._env.step(action)
        observations = self.preprocess(time_step.observation)
        return TimeStep(
            time_step.step_type, time_step.reward, time_step.discount,
            observations)

    def _reset(self):
        """Starts a new sequence and returns the first `TimeStep`."""
        time_step = self._env.reset()
        observations = self.preprocess(time_step.observation)
        return TimeStep(
            time_step.step_type, time_step.reward, time_step.discount,
            observations)

    def observation_spec(self):
        """Defines the observations provided by the environment."""
        return self._new_observation_spec


class FrameStack(wrappers.PyEnvironmentBaseWrapper):
    """Stack frames."""

    def __init__(self, env, stack_size):
        """Initializes a wrapper."""
        super(FrameStack, self).__init__(env)
        self.stack_size = stack_size
        self._frames = collections.deque(maxlen=stack_size)

        # Update the observation spec in the environment.
        observation_spec = env.observation_spec()

        # Update the observation spec.
        self._new_observation_spec = copy.copy(observation_spec)

        # Redefine pixels spec
        frame_shape = observation_spec.shape
        stacked_frame_shape = frame_shape[:2] + (frame_shape[2] * stack_size,)
        self._new_observation_spec = array_spec.ArraySpec(
            shape=stacked_frame_shape,
            dtype=observation_spec.dtype,
            name='stacked_frames')

        self._colour_frames = frame_shape[2] == 3

    def _step(self, action):
        """Steps the environment."""
        time_step = self._env.step(action)
        frame = time_step.observation

        # frame stacking
        self._frames.append(frame)

        return TimeStep(
            time_step.step_type, time_step.reward, time_step.discount,
            self.get_stacked_frames())

    def get_stacked_frames(self) -> np.ndarray:
        if self._colour_frames:
            return np.concatenate(self._frames, axis=-1)
        else:
            return np.stack(self._frames, axis=-1)


    def _reset(self):
        """Starts a new sequence and returns the first `TimeStep`."""
        time_step = self._env.reset()
        initial_frame = time_step.observation

        # initial frame stacking
        for _ in range(self.stack_size):
            self._frames.append(initial_frame)

        return TimeStep(
            time_step.step_type, time_step.reward, time_step.discount,
            self.get_stacked_frames())

    def observation_spec(self):
        """Defines the observations provided by the environment."""
        return self._new_observation_spec
