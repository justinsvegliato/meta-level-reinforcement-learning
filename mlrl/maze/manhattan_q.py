from functools import lru_cache
from .maze_state import MazeState
from ..meta.search_tree import QFunction

import numpy as np

from gym_maze.envs.maze_env import MazeEnv


class ManhattanQHat(QFunction):

    def __init__(self, maze_env: MazeEnv, discount: float = 0.99):
        self.maze_env = maze_env
        self.goal_state = np.int32(maze_env.maze_view.goal)
        self.discount = discount

        self.goal_reward = maze_env.goal_reward
        self.step_reward = -maze_env.step_cost

    def get_next_position(self, state: MazeState, action: int) -> np.array:
        """
        Uses the environment to simulate taking the action and return the next state position
        """
        action_str = self.maze_env.ACTION[action]
        curr_pos = np.int32(state.get_maze_pos())
        if self.maze_env.maze_view.is_wall(curr_pos, action_str):
            return curr_pos
        return curr_pos + self.maze_env.maze_view.maze.COMPASS[action_str]

    def distance_to_goal_after_taking_action(self, state: MazeState, action: int) -> float:
        """
        Returns the Manhattan distance to the goal
        """
        next_state_pos = self.get_next_position(state, action)
        return np.abs(next_state_pos - self.goal_state).sum()

    @lru_cache(maxsize=100)
    def get_distance_q(self, dist: float) -> float:
        if dist == 0:
            return self.goal_reward

        exp_goal_reward = self.goal_reward * self.discount ** dist
        exp_cost_to_goal = self.step_reward * (1 - self.discount**dist) / (1 - self.discount)
        return exp_goal_reward + exp_cost_to_goal

    def compute_q(self, state: MazeState, action: int) -> float:
        """
        Estimates the Q-value of the given state and action using the Manhattan distance to the goal.
        """
        state_pos = np.int32(state.get_maze_pos())
        if np.array_equal(state_pos, self.goal_state):
            return 0

        dist = self.distance_to_goal_after_taking_action(state, action)
        return self.get_distance_q(dist)
