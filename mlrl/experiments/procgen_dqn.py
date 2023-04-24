import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np
import cv2

from procgen import ProcgenGym3Env
from gym3 import ExtractDictObWrapper

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.typing.types import FloatOrReturningFloat
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.categorical_q_network import CategoricalQNetwork
from tf_agents.agents import CategoricalDqnAgent
from tf_agents.environments.py_environment import PyEnvironment

from mlrl.utils.render_utils import save_video
from mlrl.runners.eval_runner import EvalRunner
from mlrl.runners.dqn_runner import DQNRun
from mlrl.utils.env_wrappers import ImagePreprocessWrapper, FrameStack
from mlrl.utils.procgen_gym3_wrapper import ProcgenGym3Wrapper
from mlrl.procgen import REWARD_BOUNDS as PROCGEN_REWARD_BOUNDS


def create_epsilon_schedule(train_step_counter: tf.Variable, config: dict) -> FloatOrReturningFloat:
    if config.get('epsilon_schedule', False):
        start_eps = config.get('initial_epsilon', 1.0)
        end_eps = config.get('final_epsilon', 0.1)
        decay_steps = config.get('epsilon_decay_steps', 250000)
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            start_eps, decay_steps, end_eps, power=1.0)

        def get_epsilon() -> float:
            return schedule(train_step_counter)

        def report_epsilon() -> dict:
            return {'Epsilon': get_epsilon()}

        reporters = config.get('log_reporters', [])
        reporters.append(report_epsilon)
        config['log_reporters'] = reporters

        return get_epsilon

    return config.get('epsilon', 0.1)


def create_dqn_agent(env, config: dict) -> Tuple[tf.keras.Model, DdqnAgent]:

    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        conv_layer_params=[(64, 8, 4), (64, 4, 2), (64, 3, 2)],
        fc_layer_params=[512]
    )

    # build weights
    print('Building Q-Network weights...')
    ts = env.current_time_step()
    q_net(ts.observation)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.get('learning_rate', 2.5e-4))
    train_step_counter = tf.Variable(0)

    agent = DdqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=config.get('target_update_period', 10000),
        train_step_counter=train_step_counter,
        epsilon_greedy=create_epsilon_schedule(train_step_counter, config)
    )
    agent.initialize()

    q_model = tf.keras.Sequential([q_net])
    q_model(env.current_time_step().observation)

    return q_model, agent


def create_rainbow_agent(env, config: dict) -> Tuple[tf.keras.Model, CategoricalDqnAgent]:

    categorical_q_net = CategoricalQNetwork(
        tensor_spec.from_spec(env.observation_spec()),
        tensor_spec.from_spec(env.action_spec()),
        conv_layer_params=[(64, 8, 4), (64, 4, 2), (64, 3, 2)],
        fc_layer_params=[512]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.get('learning_rate', 2.5e-4))
    train_step_counter = tf.Variable(0)

    # build weights
    print('Building Categorical Q-Network weights...')
    ts = env.current_time_step()
    categorical_q_net(ts.observation)

    env_name = config.get('env', 'bigfish')
    if env_name not in PROCGEN_REWARD_BOUNDS:
        raise ValueError(f'Unknown reward bounds for procgen env: {env_name}')
    r_min, r_max = PROCGEN_REWARD_BOUNDS[env_name]

    agent = CategoricalDqnAgent(
        tensor_spec.from_spec(env.time_step_spec()),
        tensor_spec.from_spec(env.action_spec()),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        target_update_period=config.get('target_update_period', 10000),
        min_q_value=r_min, max_q_value=r_max,
        gamma=config.get('discount', 0.999),
        train_step_counter=train_step_counter,
        epsilon_greedy=create_epsilon_schedule(train_step_counter, config)
    )

    agent.initialize()

    q_model = tf.keras.Sequential([categorical_q_net])
    q_model(env.current_time_step().observation)

    return q_model, agent


def make_procgen(
        procgen_env_name: str,
        config: dict,
        n_envs: Optional[int] = 64) -> PyEnvironment:

    action_repeats = config.get('action_repeats', 4)
    frame_stack = config.get('frame_stack', 4)
    grayscale = config.get('grayscale', True)

    procgen_gym3 = ExtractDictObWrapper(ProcgenGym3Env(
        num=n_envs,
        num_threads=min(n_envs, 32),
        env_name=procgen_env_name,
        use_backgrounds=False,
        restrict_themes=True,
        distribution_mode='easy'), key='rgb')

    wrapped_procgen_gym3 = ProcgenGym3Wrapper(procgen_gym3, action_repeats=action_repeats)
    env = ImagePreprocessWrapper(wrapped_procgen_gym3, grayscale=grayscale)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)

    env.reset()

    return env


def create_policy_eval_video_frames(
        policy, env,
        render_fn=None,
        steps: int = 60) -> List[np.ndarray]:
    """
    Creates video frames of the policy acting in the given environment.
    If the environment is a batched environment, then multiple episodes
    will be shown stacked vertically.

    Args:
        policy (TFPolicy): The policy to evaluate.
        env (TFEnvironment): The environment to evaluate the policy in.
        steps (int): The maximum number of steps to run the policy for.

    Returns:
        List[np.ndarray]: A list of numpy arrays containing the frames of the video.
    """
    render_fn = render_fn or (lambda e: e.render())

    frames = []

    def observe(_):
        frames.append(render_fn(env))

    driver = py_driver.PyDriver(
        env, policy,
        max_steps=steps,
        observers=[observe]
    )

    driver.run(env.current_time_step())

    return frames


def get_grid_dim(n: int) -> Tuple[int, int]:
    """
    Generates the dimensions of a grid closest to
    square with the given number of cells.
    """
    sqrt_n = int(np.ceil(np.sqrt(n)))
    p = 1
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            p = n / i
    q = n / p
    return int(p), int(q)


def render_env(env):
    """
    Renderers one of a batched env. Scales image for video quality.
    """
    img = env.render()
    img = cv2.resize(img, (4 * img.shape[0], 4 * img.shape[1]),
                     interpolation=cv2.INTER_NEAREST)
    return img


def create_video_renderer(procgen_env_name: str, config: dict) -> Callable:
    """
    Returns a callable that produces a video of 12 games
    played simulataneously by a given policy, saved to a given file path
    """
    n_video_envs = config.get('n_video_envs', 12)
    frame_skip = config.get('action_repeats', 4)
    if frame_skip < 2:
        frame_skip = 1

    video_env = make_procgen(procgen_env_name, config, n_envs=n_video_envs)
    p, q = get_grid_dim(n_video_envs)

    def render_fn(vectorised_env, *_):
        _, observation, _ = vectorised_env.observe()

        def render_env(i):
            img = observation[i]
            img = cv2.resize(img, (4 * img.shape[0], 4 * img.shape[1]),
                             interpolation=cv2.INTER_NEAREST)
            return img

        frames = [render_env(i) for i in range(n_video_envs)]

        return np.concatenate([
            np.concatenate(frames[i * p : (i + 1) * p]) for i in range(q)
        ], axis=1)

    fps = 16 // frame_skip
    video_seconds = config.get('video_seconds', 60)

    def create_video(policy, video_file):
        frames = create_policy_eval_video_frames(
            policy, video_env,
            render_fn=render_fn, steps=fps * video_seconds * n_video_envs
        )
        return save_video(frames, video_file, fps=fps)

    return create_video


def parse_args():
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--env', type=str, default='bigfish',
                        help='Procgen environment.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--experience_batch_size', type=int, default=64,
                        help='Train minibatch batch size.')
    parser.add_argument('--train_steps_per_epoch', type=int, default=20000,
                        help='Number of training steps to perform each epoch.')
    parser.add_argument('--n_collect_envs', type=int, default=64,
                        help='Number of collect envs run in parallel.')
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help='Number of steps to evaluate for.')
    parser.add_argument('--n_eval_envs', type=int, default=64,
                        help='Number evaluation environments to run in parallel.')
    parser.add_argument('--video_seconds', type=int, default=60,
                        help='Number of seconds of video to record.')
    parser.add_argument('--n_video_envs', type=int, default=12,
                        help='Number of video environments to record.')
    parser.add_argument('--frame_stack', type=int, default=4,
                        help='Number frames to stack in observation.')
    parser.add_argument('--action_repeats', type=int, default=0,
                        help='Number of times an action is repeated with each step.')
    parser.add_argument('--grayscale', action='store_true', default=False,
                        help='Whether or not to grayscale the observation image.')

    # Object-level environment parameters
    parser.add_argument('--discount', type=float, default=0.999,
                        help='Discount factor.')

    # Agent parameters
    parser.add_argument('--agent', type=str, default='rainbow',
                        help='Agent class to use.')

    # DQN parameters
    parser.add_argument('--target_network_update_period', type=int, default=10000,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for epsilon-greedy exploration.')
    parser.add_argument('--epsilon_schedule', action='store_true', default=False,
                        help='Whether to use an epsilon-schedule for epsilon-greedy exploration.')
    parser.add_argument('--epsilon_decay_steps', type=int, default=250000,
                        help='Number of steps to decay epsilon over.')
    parser.add_argument('--initial_epsilon', type=float, default=1.0,
                        help='Initial epsilon value.')
    parser.add_argument('--final_epsilon', type=float, default=0.1,
                        help='Final epsilon value.')
    parser.add_argument('--initial_collect_steps', type=int, default=500,
                        help='Number of steps to collect before training.')
    parser.add_argument('--replay_buffer_capacity', type=int, default=16384 // 16,
                        help='Number of steps to collect before training.')

    args = vars(parser.parse_args())
    print('Arguments:')
    for k, v in args.items():
        print(f'\t{k}: {v}')
    print()

    return args


def main():
    config = parse_args()

    n_collect_envs = config.get('n_collect_envs', 64)
    procgen_env_name = config.get('env', 'bigfish')
    print('Creating collect envs...')
    collect_env = make_procgen(procgen_env_name, config, n_envs=n_collect_envs)

    print('Collect env time step spec: ', collect_env.time_step_spec())

    agent = config.pop('agent') if 'agent' in config else 'ddqn'
    if agent == 'ddqn':
        print('Creating DDQN agent...')
        q_net, agent = create_dqn_agent(
            collect_env, config
        )
    elif agent == 'rainbow':
        print('Creating Rainbow agent...')
        q_net, agent = create_rainbow_agent(
            collect_env, config
        )
    else:
        raise ValueError(f'Agent {agent} not supported. Must be one of [ddqn, rainbow]')

    n_eval_envs = config.get('n_eval_envs', 64)
    eval_steps = config.get('eval_steps', 1000)
    eval_envs = make_procgen(procgen_env_name, config, n_envs=n_eval_envs)
    eval_runner = EvalRunner(n_eval_envs * eval_steps, eval_envs, agent.policy)

    video_renderer = create_video_renderer(procgen_env_name, config)

    print(f'Creating {procgen_env_name} DQN run...')
    dqn_run = DQNRun(
        agent, collect_env, q_net,
        eval_runner=eval_runner,
        create_video_fn=video_renderer,
        video_freq=1,
        procgen_env_name=procgen_env_name,
        replay_buffer_max_length=config.get('replay_buffer_capacity', 16384),
        **config
    )

    Path('./tmp').mkdir(parents=True, exist_ok=True)
    print(f'Rendering {procgen_env_name} initial video of collect policy...')
    dqn_run.create_video(dqn_run.collect_policy, 'debug_initial_video')

    dqn_run.execute()


if __name__ == '__main__':
    main()
