from typing import List, Optional, Tuple, Union
from IPython.display import HTML, clear_output
import base64

import io
import numpy as np
import silence_tensorflow.auto  # noqa
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.environments import TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.policies import TFPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories import trajectory

import chess
import chess.svg

import cairosvg
from PIL import Image
from io import BytesIO
import imageio
import imageio.core.util

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set()


def plot_to_array(fig) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array.
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', facecolor='white', transparent=False)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def render_chess_board(board, shape=(500, 500)):
    try:
        lastmove = board.peek()
    except IndexError:
        lastmove = None

    svg = chess.svg.board(board, lastmove=lastmove)
    return svg_to_array(svg, shape=shape)


def svg_to_array(svg: str, shape=(128, 128)) -> np.ndarray:
    """
    Converts an SVG string to a numpy array.

    Args:
        svg (str): A string containing the SVG to convert.

    Returns:
        np.ndarray: A numpy array containing the rasterised image.
    """
    png = cairosvg.svg2png(bytestring=svg,
                           output_width=shape[0],
                           output_height=shape[1])
    img = Image.open(BytesIO(png))
    return np.array(img)


def embed_mp4(filename: str, clear_before=True) -> HTML:
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    if clear_before:
        clear_output()

    return HTML(tag)


def stack_renders(frames) -> np.ndarray:
    """ Vertically stacks a list of frames into a single image. """
    imgs = np.array(frames)
    b, h, w, c = imgs.shape
    return imgs.reshape((b * h, w, c))


def create_policy_eval_video(policy: TFPolicy,
                             env: TFEnvironment,
                             max_envs_to_show: Optional[int] = None,
                             rewrite_rewards: bool = False,
                             max_steps: int = 60) -> List[np.ndarray]:
    """
    Creates a video of the policy acting in the given environment.
    If the environment is a batched environment, then multiple episodes
    will be shown stacked vertically.

    The option is available to rewrite the rewards of the environment using
    Retroactive Reward Rewriting.

    Args:
        policy (TFPolicy): The policy to evaluate.
        env (TFEnvironment): The environment to evaluate the policy in.
        max_envs_to_show (Optional[int]): The maximum number of environments to show.
        rewrite_rewards (bool): Whether to rewrite the rewards of the trajectories.
        max_steps (int): The maximum number of steps to run the policy for.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing the frames of the video.
    """
    max_envs_to_show = max_envs_to_show or env.batch_size or 1

    if rewrite_rewards:
        from mlrl.meta.retro_rewards_rewriter import RetroactiveRewardsRewriter

        rewritten_trajs: List[Tuple[Optional[trajectory.Trajectory], dict]] = \
            [(None, {'terminal': [False] * env.batch_size})]
        rewards_rewriter = RetroactiveRewardsRewriter(
            env, rewritten_trajs.append, include_info=True, compute_metrics=False
        )

    if not isinstance(env, TFPyEnvironment):
        env = TFPyEnvironment(env)

    # if not isinstance(policy, TFPolicy):
    #     policy = TFPyPolicy(policy)

    env.reset()

    policy_state = policy.get_initial_state(env.batch_size)

    def render_env(gym_env, i: int = 0):

        if not isinstance(policy, RandomTFPolicy) and hasattr(policy, 'distribution'):
            policy_step = policy.distribution(env.current_time_step(), policy_state)
            if isinstance(policy_step.action, tfp.distributions.Categorical):
                probs = tf.nn.softmax(policy_step.action.logits[i]).numpy()
                return gym_env.render(meta_action_probs=probs)
        else:
            policy_step = policy.action(env.current_time_step(), policy_state)
            info = policy_step.info
            if info is not None and 'dist_params' in info and 'logits' in info['dist_params']:
                probs = tf.nn.softmax(info['dist_params']['logits'][i]).numpy()
                return gym_env.render(meta_action_probs=probs)

        return gym_env.render()

    def get_frames():
        if hasattr(env, 'envs'):
            return [
                render_env(e.gym, i) for i, e in enumerate(env.envs)
                if i < max_envs_to_show
            ]
        else:
            return [render_env(env, 0)]

    frames = [get_frames()]
    for _ in range(max_steps):
        time_step = env.current_time_step()
        action_step = policy.action(env.current_time_step(), policy_state)
        policy_state = action_step.state
        next_time_step = env.step(action_step.action)

        if rewrite_rewards:
            action_step_with_previous_state = action_step._replace(state=policy_state)
            traj = trajectory.from_transition(
                time_step, action_step_with_previous_state, next_time_step)
            rewards_rewriter(traj)

        frames.append(get_frames())

    if rewrite_rewards:
        rewards_rewriter.flush_all()
        frames = _create_rewritten_frames(frames, rewritten_trajs)

    return [stack_renders(renders) for renders in frames]


def _create_rewritten_frames(
        frames: List[List[np.ndarray]],
        rewritten_trajs: List[Tuple[trajectory.Trajectory, dict]]
) -> List[np.ndarray]:
    """
    Takes a list of frames and a list of trajectories and returns a list of frames
    where the rewriten rewards are shown alongside the tree used to evaluate the
    policies considered at each step.

    Args:
        frames (List[List[np.ndarray]]): A list of frames. Each frame is a list of
            numpy arrays containing the frame corresponding to each batch in the environment.
        rewritten_trajs (List[Tuple[trajectory.Trajectory, dict]]): A list of trajectories
            and their corresponding info dictionaries.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing the frames of the video.
    """

    from mlrl.utils.plot_search_tree import plot_tree

    new_frames = []
    n_envs = len(frames[0])
    new_return = [0 for _ in range(n_envs)]
    old_return = [0 for _ in range(n_envs)]

    for (traj, info), renders in zip(rewritten_trajs, frames):
        new_frames.append([])
        for i, frame in enumerate(renders):
            if traj is not None:
                reward = traj.reward[i]

                if traj.is_first()[i]:
                    new_return[i] = 0
                    old_return[i] = 0
            else:
                reward = 0

            new_return[i] += reward

            if 'original_reward' in info:
                original_reward = info['original_reward'][i]
                old_return[i] += original_reward

            fig = plt.figure(tight_layout=True, figsize=(20, 6))
            gs = gridspec.GridSpec(1, 4)

            env_ax = fig.add_subplot(gs[:, :3])
            tree_ax = fig.add_subplot(gs[:, 3:])
            env_ax.axis('off')
            tree_ax.axis('off')

            env_ax.set_title('Meta-level Environment')
            env_ax.imshow(frame)

            eval_tree = info.get('eval_tree', [None] * n_envs)[i]
            is_terminal = info.get('terminal', [False] * n_envs)[i]
            if eval_tree is not None and not is_terminal:
                plot_tree(eval_tree, ax=tree_ax, show=False,
                          title='Evaluation Tree')

            plt.suptitle(f'Environment with Reward Rewriting. '
                         f'Rewritten Reward = {reward:.4f}.\n'
                         f'New Return = {new_return[i]:.4f}. '
                         f'Old Return = {old_return[i]:.4f}', fontsize=16)

            plt.tight_layout()
            new_frames[-1].append(plot_to_array(fig))
            plt.close()

    return new_frames


def create_and_save_policy_eval_video(policy: TFPolicy,
                                      env: TFEnvironment,
                                      filename: str = 'video',
                                      max_steps: int = 60,
                                      rewrite_rewards: bool = False,
                                      fps: int = 1) -> str:
    """
    Creates and saves a video of the policy being evaluating in an environment.

    Args:
        policy (TFPolicy): The policy to evaluate.
        env (TFEnvironment): The environment to evaluate the policy in.
        filename (str): The name of the file to save the video to.
        max_steps (int): The maximum number of steps to run the policy for.
        fps (int): The frames per second of the video.

    Returns:
        str: The path to the saved video.
    """
    frames = create_policy_eval_video(
        policy, env,
        rewrite_rewards=rewrite_rewards,
        max_steps=max_steps
    )

    if not filename.endswith('.mp4'):
        filename = filename + '.mp4'

    with imageio.get_writer(filename, fps=fps, macro_block_size=1) as video:
        for frame in frames:
            video.append_data(frame)

    return filename


def create_random_policy_video(env: Union[TFEnvironment, PyEnvironment],
                               filename: str = 'video',
                               max_steps: int = 60,
                               rewrite_rewards: bool = False,
                               fps: int = 1) -> str:
    """
    Creates and saves a video of a random policy being evaluated in an environment.
    Assumes that environment observations are nested and contain search tokens
    and an action mask.

    Args:
        env (TFEnvironment): The environment to evaluate the policy in.
        filename (str): The name of the file to save the video to.
        max_steps (int): The maximum number of steps to run the policy for.
        fps (int): The frames per second of the video.

    Returns:
        str: The path to the saved video.
    """
    from mlrl.meta.meta_env import mask_token_splitter
    if not isinstance(env, TFPyEnvironment):
        env = TFPyEnvironment(env)
    policy = RandomTFPolicy(env.time_step_spec(),
                            env.action_spec(),
                            observation_and_action_constraint_splitter=mask_token_splitter)
    return create_and_save_policy_eval_video(policy, env,
                                             filename=filename, max_steps=max_steps, fps=fps,
                                             rewrite_rewards=rewrite_rewards)
