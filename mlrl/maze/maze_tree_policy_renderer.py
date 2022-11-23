from typing import List, Tuple
from mlrl.utils.draw_arrow import draw_arrow
from mlrl.meta.tree_policy import SearchTreePolicy
from mlrl.meta.search_tree import SearchTreeNode

import pygame
import numpy as np


def cell_to_screen_coords(env, cell_x: int, cell_y: int) -> tuple:
    screen_x = int(cell_x * env.maze_view.CELL_W + env.maze_view.CELL_W * 0.5 + 0.5)
    screen_y = int(cell_y * env.maze_view.CELL_H + env.maze_view.CELL_H * 0.5 + 0.5)
    return screen_x, screen_y


def draw_arrow_on_maze(env, cell_start: tuple, cell_end: tuple):
    start = pygame.Vector2(*cell_to_screen_coords(env, *cell_start))
    end = pygame.Vector2(*cell_to_screen_coords(env, *cell_end))
    w = env.maze_view.CELL_H / 10
    draw_arrow(env.maze_view.maze_layer, start, end, pygame.Color('orange'), w, w*3, w*2)


def render_tree_policy(env, tree_policy: SearchTreePolicy) -> np.array:
    """
    Renders the sequence of actions of a tree policy on the maze.

    Args:
        - env: The Gym-Maze environment to render the policy on (must be a MazeEnv)
        - tree_policy: The tree policy to render

    Returns:
        - A numpy array of the rendered image
    """

    def recursive_get_arrows(node: SearchTreeNode) -> List[Tuple[int, int]]:
        if node.is_terminal_state:
            return []

        start_cell = tuple(node.state.get_state_vector())
        action = tree_policy.get_action(node.state)
        if node.has_action_children(action):
            child, *_ = node.get_children(action)
            end_cell = tuple(child.state.get_state_vector())
            return [(start_cell, end_cell)] + recursive_get_arrows(child)

        node.state.set_environment_to_state(env)
        next_pos, *_ = env.step(action)
        return [(start_cell, tuple(next_pos))]

    arrows = recursive_get_arrows(tree_policy.tree.get_root())

    root_node = tree_policy.tree.get_root()
    root_node.state.set_environment_to_state(env)
    env.render(mode='rgb_array')
    for start_cell, end_cell in arrows:
        draw_arrow_on_maze(env, start_cell, end_cell)

    env.maze_view._MazeView2D__draw_robot()
    env.maze_view.screen.blit(env.maze_view.maze_layer, (0, 0))

    pygame.display.flip()
    return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))
