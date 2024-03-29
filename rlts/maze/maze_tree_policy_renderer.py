from typing import List, Tuple
from rlts.maze.manhattan_q import ManhattanQHat
from rlts.maze.maze_state import MazeState
from rlts.maze.maze_utils import create_restricted_maze_state
from rlts.utils.draw_arrow import draw_arrow
from rlts.meta.tree_policy import SearchTreePolicy
from rlts.meta.search_tree import SearchTreeNode

import pygame
import numpy as np
from gym_maze.envs import MazeEnv


def cell_to_screen_coords(env, cell_x: int, cell_y: int) -> tuple:
    screen_x = int(cell_x * env.maze_view.CELL_W + env.maze_view.CELL_W * 0.5 + 0.5)
    screen_y = int(cell_y * env.maze_view.CELL_H + env.maze_view.CELL_H * 0.5 + 0.5)
    return screen_x, screen_y


def draw_arrow_on_maze(env, cell_start: tuple, cell_end: tuple,
                       alpha: float = 1, colour: str = 'orange'):
    start = pygame.Vector2(*cell_to_screen_coords(env, *cell_start))
    end = pygame.Vector2(*cell_to_screen_coords(env, *cell_end))
    w = env.maze_view.CELL_H / 10

    colour = pygame.Color(colour)
    if alpha < 1:
        colour.a = int(alpha * 255)

    draw_arrow(env.maze_view.maze_layer, start, end, colour, w, w * 3, w * 2)


def render_tree_policy(env: MazeEnv, tree_policy: SearchTreePolicy,
                       include_leaf_actions=False, show_outside_tree=True) -> np.array:
    """
    Renders the sequence of actions of a tree policy on the maze.

    Args:
        - env: The Gym-Maze environment to render the policy on (must be a MazeEnv)
        - tree_policy: The tree policy to render

    Returns:
        - A numpy array of the rendered image
    """
    if not hasattr(env.maze_view, '_MazeView2D__screen_size'):
        return np.zeros((1, 1, 3))

    def recursive_get_arrows(node: SearchTreeNode[MazeState]) -> List[Tuple[int, int]]:
        if node.is_terminal_state:
            return []

        start_cell = node.state.get_maze_pos()
        action_probs = tree_policy.get_action_probabilities(node.state)
        arrows = []
        for action, prob in action_probs.items():
            if prob == 0:
                continue

            if node.has_action_children(action):
                child, *_ = node.get_children(action)
                end_cell = child.state.get_maze_pos()
                arrows.extend([(start_cell, end_cell, prob)] + recursive_get_arrows(child))

            elif include_leaf_actions:
                node.state.set_environment_to_state(env)
                next_pos, *_ = env.step(action)
                arrows.append((start_cell, tuple(next_pos), prob))

        return arrows

    arrows = recursive_get_arrows(tree_policy.tree.get_root())

    root_node = tree_policy.tree.get_root()
    root_node.state.set_environment_to_state(env)
    env.render(mode='rgb_array')

    if show_outside_tree:
        w, h = env.maze_view.maze_size
        tree_states = [start for start, *_ in arrows]
        not_tree_states = [
            (x, y) for x in range(w) for y in range(h)
            if (x, y) not in tree_states
        ]

        q_func = ManhattanQHat(env)
        for start_cell in not_tree_states:
            if start_cell == tuple(env.maze_view.goal):
                continue

            state = create_restricted_maze_state(np.array(start_cell), env)
            action_probs = tree_policy.get_action_probabilities(state)
            for action, prob in action_probs.items():
                if prob == 0:
                    continue
                end_cell = q_func.get_next_position(state, action)
                draw_arrow_on_maze(env, start_cell, end_cell,
                                   alpha=prob / 2, colour='grey')

    for start_cell, end_cell, prob in arrows:
        draw_arrow_on_maze(env, start_cell, end_cell, alpha=prob)

    env.maze_view._MazeView2D__draw_robot()
    env.maze_view.screen.blit(env.maze_view.maze_layer, (0, 0))

    pygame.display.flip()
    return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))
