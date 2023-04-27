from mlrl.experiments.experiment_utils import create_parser, create_meta_env
from mlrl.runners.ppo_runner import PPORunner
from mlrl.meta.meta_env import MetaEnv
from mlrl.procgen import META_ALLOWED_COMBOS
from mlrl.procgen.procgen_state import ProcgenState, ProcgenProcessing
from mlrl.procgen.procgen_env import make_vectorised_procgen
from mlrl.procgen.meta_renderer import render_tree_policy
from mlrl.experiments.procgen_dqn import create_rainbow_agent

import json
from pathlib import Path
from typing import Tuple, List
import re

import gym

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment

import tensorflow as tf

print(f'Using TensorFlow {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[2:], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def patch_action_repeats(gym_env, config: dict):

    original_step = gym_env.step

    def repeated_step(*args, **kwargs):
        reward = 0
        for _ in range(config.get('action_repeats')):
            observation, r, *info = original_step(*args, **kwargs)
            reward += r
        return observation, reward, *info

    gym_env.step = repeated_step


def patch_normalise(gym_env):
    original_step = gym_env.step

    def normalised_step(*args, **kwargs):
        observation, *rest = original_step(*args, **kwargs)
        return observation / 255., *rest

    gym_env.step = normalised_step


def patch_render(gym_env):

    def render(*_, **__):
        return gym_env.env.observe()[1][0]

    gym_env.render = render


def make_gym_procgen(config: dict):
    env_name = config['env']
    gym_env = gym.make(f'procgen:procgen-{env_name}-v0',
                       distribution_mode='easy',
                       use_backgrounds=False,
                       restrict_themes=True)

    patch_render(gym_env)
    patch_normalise(gym_env)

    # procgen envs cannot be reset and print an annoying warning if you try to do so
    gym_env.reset = lambda *_: None

    if config.get('action_repeats', 0) > 1:
        patch_action_repeats(gym_env, config)

    if config.get('grayscale', False):
        raise NotImplementedError('Grayscale not implemented for gym procgen yet')

    if config.get('frame_stack', 0) > 0:
        raise NotImplementedError('Frame stacking not implemented for gym procgen yet')

    return gym_env


def create_procgen_meta_env(object_config: dict,
                            min_computation_steps: int = 0,
                            **meta_config) -> MetaEnv:

    object_env = make_gym_procgen(object_config)

    def q_hat(s, a):
        return s.q_values[a]

    return create_meta_env(
        object_env, ProcgenState.extract_state(object_env),
        q_hat, meta_config,
        tree_policy_renderer=render_tree_policy,
        min_computation_steps=min_computation_steps
    )


def create_batched_procgen_meta_envs(
        n_envs: int,
        object_config: dict,
        min_computation_steps: int = 0,
        env_multithreading=True, **config) -> BatchedPyEnvironment:

    if n_envs == 0:
        raise ValueError('n_envs must be > 0')

    return BatchedPyEnvironment([
        GymWrapper(create_procgen_meta_env(
            object_config,
            min_computation_steps=min_computation_steps,
            **config
        ))
        for _ in range(n_envs)
    ], multithreading=env_multithreading)


def create_runner_envs(
        object_config: dict,
        n_collect_envs=16, n_video_envs=2, n_eval_envs=8,
        min_train_computation_steps=0,
        env_multithreading=True, **config):

    env = create_batched_procgen_meta_envs(
        n_collect_envs, object_config,
        min_computation_steps=min_train_computation_steps,
        env_multithreading=env_multithreading,
        **config)

    eval_env = create_batched_procgen_meta_envs(
        n_eval_envs, object_config,
        env_multithreading=env_multithreading,
        **config)

    video_env = create_batched_procgen_meta_envs(
        n_video_envs, object_config,
        env_multithreading=env_multithreading,
        **config)

    return env, eval_env, video_env


def parse_model_weights_string(path: str) -> Tuple[int, float]:
    pattern = r"sequential_best_(\d+)_(\d+\.\d+).index"

    match = re.match(pattern, path)

    if match:
        epoch = int(match.group(1))
        value = float(match.group(2))
        return epoch, value

    return None


def get_model_at_return_percentile(model_paths: List[Tuple[str, int, float]], percentile: float) -> Tuple[str, int, float]:
    sorted_paths = sorted(model_paths, key=lambda x: x[2])
    index = round(len(sorted_paths) * percentile)
    return sorted_paths[index]


def load_pretrained_q_network(folder: str, run: str, percentile: float = 1.0, verbose: bool = True):
    folder = f'{folder}/categorical_dqn_agent/{run}'

    with open(folder + '/config.json') as f:
        object_config = json.load(f)

    env = make_vectorised_procgen(object_config, n_envs=1)
    q_net, agent = create_rainbow_agent(env, object_config, verbose=verbose)

    model_paths = [
        (str(path).replace('.index', ''), *values)
        for path in Path(f'{folder}/model_weights').glob('*')
        if path.is_file() and str(path).endswith('.index')
        and (values := parse_model_weights_string(str(path.name))) is not None
    ]

    if verbose:
        print('\n'.join(map(str, model_paths)))

    path, epoch, ret_val = get_model_at_return_percentile(model_paths, percentile)
    object_config['pretrained_epoch'] = epoch
    object_config['pretrained_return'] = ret_val
    object_config['pretrained_run'] = run
    object_config['pretrained_path'] = path
    object_config['pretrained_percentile'] = percentile

    if verbose:
        print(f'Loading model from {run} that scored return value {ret_val} at epoch {epoch}')
        print('Object-level config:')
        for k, v in object_config.items():
            print(f'\t - {k}: {v}')

    q_net.load_weights(path)

    env_name = object_config.get('env', 'coinrun')
    if env_name in META_ALLOWED_COMBOS:
        ProcgenState.set_actions(META_ALLOWED_COMBOS[env_name])

    ProcgenProcessing.set_pretrained_agent(agent)

    return object_config


def create_runner(args):

    object_config = load_pretrained_q_network(
        folder=args['pretrained_runs_folder'],
        run=args['pretrained_run'],
        percentile=args.get('pretrained_percentile', 0.75)
    )

    args['object_level_config'] = object_config
    object_env_name = object_config.get('env', 'coinrun')

    env, eval_env, video_env = create_runner_envs(
        object_config=object_config, **args
    )

    ppo_runner = PPORunner(
        env, eval_env=eval_env, video_env=video_env,
        name=f'procgen-{object_env_name}', **args
    )

    return ppo_runner


def main(args):
    ppo_runner = create_runner(args)
    ppo_runner.run()


def parse_args():
    parser = create_parser()

    parser.add_argument('--pretrained_percentile', type=float, default=0.75,
                        help='Percentile in list of pretrained sorted by reward.')
    parser.add_argument('--pretrained_run', type=str, default='run-16823527592836354',
                        help='Name of the DQN run to load the pretrained Q-network from.')
    parser.add_argument('--pretrained_runs_folder', type=str, default='runs',
                        help='Folder containing pretraining runs.')

    args = vars(parser.parse_args())

    print('Arguments:')
    for k, v in args.items():
        print(f'\t{k}: {v}')
    print()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
