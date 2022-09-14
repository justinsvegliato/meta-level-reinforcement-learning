import numpy as np

from .search_tree import ObjectState


class MazeState(ObjectState):
    """
    Class to handle represent a state of the maze environment.
    The state vector is a 2D vector of the robot position in the maze.
    The gym state is a tuple containing the necessary information
    to set the environment to this state.
    """
    __slots__ = ['state_vec', 'gym_state']

    @staticmethod
    def extract_state(env: 'MazeEnv') -> 'MazeState':
        """
        A static method to extract the state of the environment
        for later restoring and representation.
        """
        state_vec = np.array(env.maze_view._MazeView2D__robot.copy(),
                             dtype=np.int32)
        gym_state = (state_vec, env.steps_beyond_done, env.done, env._elapsed_steps)
        return MazeState(state_vec, gym_state)

    def __init__(self, state_vec: np.array, gym_state: tuple):
        self.state_vec = state_vec
        self.gym_state = gym_state

    def set_environment_to_state(self, env: 'MazeEnv'):
        robot_pos, steps_beyond_done, done, elapsed_steps = self.gym_state

        env.maze_view._MazeView2D__draw_robot(transparency=0)
        env.maze_view._MazeView2D__robot = robot_pos.copy()
        env.maze_view._MazeView2D__draw_robot(transparency=255)

        env.state = robot_pos.copy()
        env.steps_beyond_done = steps_beyond_done
        env.done = done
        env._elapsed_steps = elapsed_steps

    def get_state_vector(self) -> np.array:
        return np.array(self.state_vec, dtype=np.float32)

    def __repr__(self) -> str:
        return str(tuple(map(int, self.state_vec)))
