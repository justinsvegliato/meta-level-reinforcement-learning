from .search_tree import ObjectState

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

        return MazeState(state_vec, gym_state)

    def __init__(self, state_vec: np.array, gym_state: tuple):
        self.state_vec = state_vec
        self.gym_state = gym_state

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

    def get_state_vector(self) -> np.array:
        return np.array(self.state_vec, dtype=np.float32)

    def __repr__(self) -> str:
        return f'MazeState(pos={tuple(map(int, self.state_vec))})'
