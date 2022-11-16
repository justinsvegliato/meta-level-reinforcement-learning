import random
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage

import gym
from gym.envs.registration import register

os.environ["SDL_VIDEODRIVER"] = "dummy"

register(
    id='gym-maze-v0',
    entry_point='gym_maze.envs:MazeEnv',
    nondeterministic=False
)


def make_maze_env(**make_kwargs):
    """
    Args:
        seed (int): The seed to use for the maze generation.
        make_kwargs (dict): Additional arguments to pass to gym.make.

    Returns:
        gym.Env: The maze environment.
    """
    return gym.make("gym-maze-v0", **make_kwargs)


def get_maze_state(env) -> tuple:
    """
    Returns a tuple containing the necessary information
    to restore the environment its current state.
    """
    return (np.array(env.state.copy()), env.steps_beyond_done, env.done)


def set_maze_state(env, state: tuple, update_render=True):
    """ Sets the environment to the given state. """
    robot_state, steps_beyond_done, done = state

    if update_render:
        # pylint: disable=protected-access
        env.maze_view._MazeView2D__draw_robot(transparency=0)

    # pylint: disable=protected-access
    env.maze_view._MazeView2D__robot = robot_state

    if update_render:
        # pylint: disable=protected-access
        env.maze_view._MazeView2D__draw_robot(transparency=255)

    env.state = robot_state
    env.steps_beyond_done = steps_beyond_done
    env.done = done


def downscale(img: np.ndarray, new_dim=(32, 32)) -> np.ndarray:
    """
    Downscale the image to the given dimensions.
    Downscaling is done by taking the minimum the pixels in each patch.
    """
    w, h, *_ = img.shape
    new_w, new_h = new_dim
    kernel = (w // new_w, h // new_h, 1)
    return skimage.measure.block_reduce(img, kernel, np.min)


def dark_mode_maze(img: np.ndarray,
                   brighten_darks=30,
                   brighten_mids=50,
                   light_thresh=240,
                   dark_thresh=50) -> np.ndarray:
    """ Transforms the image to a dark mode version. Purely for aesthetic purposes. """
    img_rgb_sum = np.sum(img, axis=-1)
    light_region = (img_rgb_sum > light_thresh * 3)
    light_region = np.repeat(light_region[:, :, np.newaxis], 3, axis=-1)

    dark_region = (img_rgb_sum < dark_thresh * 3)
    dark_region = np.repeat(dark_region[:, :, np.newaxis], 3, axis=-1)

    other_regions = ~(light_region | dark_region)

    img[light_region] = 255 - img[light_region] + brighten_darks
    img[dark_region] = 255 - img[dark_region]
    img[other_regions] = img[other_regions] + brighten_mids

    return np.clip(img, 0, 255)


def render_maze(env,
                dark_mode=False,
                figsize=None,
                title=None,
                do_downscale=True,
                ax=None,
                show=True):
    """ Renders the maze environment with matplotlib. """
    img = env.render(mode='rgb_array')
    if do_downscale:
        img = downscale(img)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)

    ax.set_axis_off()
    ax.imshow(dark_mode_maze(img) if dark_mode else img)
    ax.set_title(title)

    if show:
        plt.show()
