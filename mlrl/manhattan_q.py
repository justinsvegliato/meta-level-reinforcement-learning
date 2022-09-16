from mlrl.maze_state import MazeState
from mlrl.q_estimation import QFunction

import numpy as np


class ManhattanQHat(QFunction):

    def __init__(self, maze_env, discount: float = 0.99):
        self.maze_env = maze_env
        self.goal_state = maze_env.maze_view.goal
        self.discount = discount

        # see source code of gym_maze.envs.maze_env.MazeEnv for hard-coded reward values
        # https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/envs/maze_env.py
        self.goal_reward = 1
        self.step_reward = -0.1 / (maze_env.maze_size[0] * maze_env.maze_size[1])

    def get_next_position(self, state: MazeState, action: int) -> np.array:
        """
        Uses the environment to simulate taking the action and return the next state position
        """
        state.set_environment_to_state(self.maze_env)
        elapsed_steps = self.maze_env._elapsed_steps
        next_state, *_ = self.maze_env.step(action)
        # ensure that this doesn't count towards time limit in the environment
        self.maze_env._elapsed_steps = elapsed_steps
        return next_state

    def distance_to_goal_after_taking_action(self, state: MazeState, action: int) -> float:
        """
        Returns the Manhattan distance to the goal
        """
        next_state_pos = self.get_next_position(state, action)
        return np.abs(next_state_pos - self.goal_state).sum()

    def compute_q(self, state: MazeState, action: int) -> float:
        """
        Estimates the Q-value of the given state and action using the Manhattan distance to the goal.
        """

        state_pos = state.get_state_vector()
        if np.array_equal(state_pos, self.goal_state):
            return 0

        dist = self.distance_to_goal_after_taking_action(state, action)

        if dist == 0:
            return self.goal_reward

        exp_goal_reward = self.goal_reward * self.discount ** dist
        exp_cost_to_goal = self.step_reward * (1 - self.discount**dist) / (1 - self.discount)
        return exp_goal_reward + exp_cost_to_goal
