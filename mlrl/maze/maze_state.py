from ..meta.search_tree import ObjectState
from ..utils import one_hot

from typing import List

from gym_maze.envs import MazeEnv
import numpy as np


class MazeState(ObjectState):
    """
    Class to handle represent a state of the maze environment.
    The state vector is a 2D vector of the robot position in the maze.
    The gym state is a tuple containing the necessary information
    to set the environment to this state.
    """
    __slots__ = ['state_vec', 'gym_state']

    @staticmethod
    def extract_state(env: MazeEnv) -> 'MazeState':
        """
        A static method to extract the state of the environment
        for later restoring and representation.
        """
        x, y = env.maze_view._MazeView2D__robot.copy()
        # w, h = env.maze_view.maze_size
        # state_vec = np.array([x / w, y / h], dtype=np.float32)
        state_vec = np.array([x, y], dtype=np.float32)

        maze_pos = (x, y)
        if hasattr(env, '_elapsed_steps'):
            gym_state = (
                maze_pos, env.steps_beyond_done, env.done, env._elapsed_steps
            )
        else:
            gym_state = (
                maze_pos, env.steps_beyond_done, env.done
            )

        return MazeState(state_vec, gym_state)

    def __init__(self, state_vec: np.array, gym_state: tuple):
        self.state_vec = state_vec
        self.gym_state = gym_state
        self.maze_pos, *_ = gym_state

    def set_environment_to_state(self, env):
        robot_pos, steps_beyond_done, done, *_ = self.gym_state

        env.maze_view._MazeView2D__draw_robot(transparency=0)
        env.maze_view._MazeView2D__robot = robot_pos.copy()
        env.maze_view._MazeView2D__draw_robot(transparency=255)

        env.state = robot_pos.copy()
        env.steps_beyond_done = steps_beyond_done
        env.done = done

        if hasattr(env, '_elapsed_steps'):
            env._elapsed_steps = self.gym_state[3]

    def get_maze_pos(self) -> tuple:
        return tuple(self.maze_pos)

    def get_state_vector(self) -> np.array:
        return np.array(self.state_vec, dtype=np.float32)

    def get_maximum_number_of_actions(self):
        return len(self.get_actions())

    def get_actions(self) -> list:
        return [0, 1, 2, 3]

    def get_action_labels(self) -> List[str]:
        return MazeEnv.ACTION

    def get_action_vector(self, action: int) -> np.array:
        _, _, done, *_ = self.gym_state

        if not done:
            return one_hot(action, 4)

        return np.zeros((4,))

    def get_state_string(self) -> str:
        x, y = self.maze_pos
        return f'{x}, {y}'

    def __repr__(self) -> str:
        return f'MazeState({self.get_state_string()})'


class RestrictedActionsMazeState(MazeState):

    @staticmethod
    def extract_state(env) -> 'MazeState':
        """
        A static method to extract the state of the environment
        for later restoring and representation.
        """
        state_vec = np.array(env.maze_view._MazeView2D__robot.copy(),
                             dtype=np.int32)

        if hasattr(env, '_elapsed_steps'):
            gym_state = (
                state_vec, env.steps_beyond_done, env.done, env._elapsed_steps
            )
        else:
            gym_state = (
                state_vec, env.steps_beyond_done, env.done
            )

        actions = [
            i for i, a in enumerate(env.ACTION)
            if env.maze_view.maze.is_open(state_vec, a)
        ]

        return RestrictedActionsMazeState(state_vec, gym_state, actions)

    def __init__(self, state_vec: np.array, gym_state: tuple, actions: list):
        super().__init__(state_vec, gym_state)
        self.actions = actions

    def get_actions(self) -> list:
        return self.actions

    def get_action_labels(self) -> List[str]:
        return [MazeEnv.ACTION[a] for a in self.actions]
