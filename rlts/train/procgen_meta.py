from rlts.train.experiment_utils import create_parser, create_meta_env
from rlts.runners.ppo_runner import PPORunner
from rlts.meta.meta_env import MetaEnv, aggregate_object_level_metrics
from rlts.procgen import META_ALLOWED_COMBOS
from rlts.procgen.procgen_state import ProcgenState, ProcgenProcessing
from rlts.procgen.procgen_env import make_vectorised_procgen
from rlts.procgen.batched_procgen_meta_env import BatchedProcgenMetaEnv
from rlts.procgen.meta_renderer import render_tree_policy
from rlts.procgen.time_limit_observer import TimeLimitObserver
from rlts.train.procgen_dqn import create_rainbow_agent
from rlts.utils.system import restrict_gpus

import json
from pathlib import Path
from typing import Tuple, List
import re

import gym
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


def patch_action_repeats(gym_env, config: dict):

    original_step = gym_env.step

    def repeated_step(*args, **kwargs):
        reward = 0
        for _ in range(config.get('action_repeats')):
            observation, r, done, info = original_step(*args, **kwargs)
            reward += r
            if done:
                break
        return observation, reward, done, info

    gym_env.step = repeated_step


def patch_normalise(gym_env):
    original_step = gym_env.step

    def normalised_step(*args, **kwargs):
        observation, *rest = original_step(*args, **kwargs)
        return observation / 255., *rest

    gym_env.step = normalised_step

    # procgen envs cannot be reset and print an annoying warning if you try to do so
    gym_env.reset = lambda *_: gym_env.env.observe()[1][0] / 255.


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

    if config.get('action_repeats', 0) > 1:
        patch_action_repeats(gym_env, config)

    if config.get('grayscale', False):
        raise NotImplementedError('Grayscale not implemented for gym procgen yet')

    if config.get('frame_stack', 0) > 0:
        raise NotImplementedError('Frame stacking not implemented for gym procgen yet')

    return gym_env


def create_procgen_meta_env(object_config: dict,
                            min_computation_steps: int = 0,
                            render_plans: bool = True,
                            **meta_config) -> MetaEnv:

    object_env = make_gym_procgen(object_config)
    meta_config['object_discount'] = object_config.get('discount', 0.99)

    def q_hat(s, a):
        return s.q_values[a]

    return create_meta_env(
        object_env,
        ProcgenState.extract_state(object_env),
        q_hat,
        meta_config,
        tree_policy_renderer=render_tree_policy if render_plans else None,
        min_computation_steps=min_computation_steps
    )


def create_batched_procgen_meta_envs(
        n_envs: int,
        object_config: dict,
        min_computation_steps: int = 0,
        max_object_level_steps: int = 500,
        patch_terminates: bool = True,
        patch_expansions: bool = True,
        env_multithreading=True,
        **config) -> BatchedPyEnvironment:

    if n_envs == 0:
        raise ValueError('n_envs must be > 0')

    meta_envs = [
        create_procgen_meta_env(
            object_config,
            min_computation_steps=min_computation_steps,
            **config
        )
        for _ in range(n_envs)
    ]
    max_expands_per_env = len(ProcgenState.ACTIONS)

    for env in meta_envs:
        time_limit = TimeLimitObserver(env, max_object_level_steps)
        env.object_level_transition_observers.append(time_limit)

    return BatchedProcgenMetaEnv(meta_envs,
                                 max_expands_per_env,
                                 object_config,
                                 patch_terminates=patch_terminates,
                                 patch_expansions=patch_expansions,
                                 multithreading=env_multithreading)


def create_runner_envs(
        object_config: dict,
        n_collect_envs=16,
        n_video_envs=2,
        n_eval_envs=8,
        min_train_computation_steps=0,
        env_multithreading=True, **config):

    env = create_batched_procgen_meta_envs(
        n_collect_envs, object_config,
        min_computation_steps=min_train_computation_steps,
        env_multithreading=env_multithreading,
        **config)

    if n_eval_envs > 0:
        eval_env = create_batched_procgen_meta_envs(
            n_eval_envs, object_config,
            env_multithreading=env_multithreading,
            **config)
    else:
        eval_env = None

    if n_video_envs > 0:
        video_env = create_batched_procgen_meta_envs(
            n_video_envs, object_config,
            env_multithreading=env_multithreading,
            **config)
    else:
        video_env = None

    return env, eval_env, video_env


def parse_model_weights_string(path: str) -> Tuple[int, float]:
    pattern = r"sequential_best_(\d+)_(-?\d+\.\d+).index"

    match = re.match(pattern, path)

    if match:
        epoch = int(match.group(1))
        value = float(match.group(2))
        return epoch, value

    return None


def get_model_at_return_percentile(model_paths: List[Tuple[str, int, float]],
                                   percentile: float) -> Tuple[str, int, float]:
    sorted_paths = sorted(model_paths, key=lambda x: x[2])
    index = round(len(sorted_paths) * percentile)
    return sorted_paths[max(0, min(len(sorted_paths) - 1, index))]


def get_model_at_epoch(model_paths: List[Tuple[str, int, float]],
                       epoch: float) -> Tuple[str, int, float]:
    for path, e, ret_val in model_paths:
        if e == epoch:
            return path, ret_val
    raise ValueError(f'No model found for epoch {epoch}')


def load_pretrained_q_network(folder: str,
                              run: str,
                              percentile: float = 1.0,
                              epoch: int = None,
                              verbose: bool = True):
    """
    Loads the pretrained Q-network from a run and returns the config used to train it.
    The Q-network is passed to the ProcgenProcessing class and stored as a static variable
    so that it can be used to compute Q-values and encode environment states.
    The ProcgenState class is also updated to use the correct action space.
    """
    folder = f'{folder}/categorical_dqn_agent/{run}'

    with open(folder + '/config.json') as f:
        object_config = json.load(f)

    if verbose:
        print('Object-level config:')
        for k, v in object_config.items():
            print(f'\t - {k}: {v}')

    env = make_vectorised_procgen(object_config, n_envs=1)
    if verbose:
        print(f'Created environment {env}')

    q_net, agent = create_rainbow_agent(env, object_config, verbose=verbose)

    model_paths = [
        (str(path).replace('.index', ''), *values)
        for path in Path(f'{folder}/model_weights').glob('*')
        if path.is_file() and str(path).endswith('.index')
        and (values := parse_model_weights_string(str(path.name))) is not None
    ]

    if verbose:
        print('\n'.join(map(str, model_paths)))

    if epoch is not None:
        path, ret_val = get_model_at_epoch(model_paths, epoch)
    else:
        path, epoch, ret_val = get_model_at_return_percentile(model_paths, percentile)

    object_config['pretrained_epoch'] = epoch
    object_config['pretrained_return'] = ret_val
    object_config['pretrained_run'] = run
    object_config['pretrained_path'] = path
    object_config['pretrained_percentile'] = percentile

    if verbose:
        print(f'Loading model from {run} that scored return value {ret_val} at epoch {epoch}')

    q_net.load_weights(path)

    env_name = object_config.get('env', 'coinrun')
    if env_name in META_ALLOWED_COMBOS:
        ProcgenState.set_actions(META_ALLOWED_COMBOS[env_name])

    ProcgenProcessing.set_pretrained_agent(agent)

    return object_config


def get_object_level_metrics(batched_env: BatchedPyEnvironment):
    return aggregate_object_level_metrics([
        meta_env.get_object_level_metrics()
        for meta_env in batched_env.meta_envs
    ])


def reset_object_level_metrics(batched_env: BatchedPyEnvironment):
    for meta_env in batched_env.meta_envs:
        meta_env.reset_metrics()


def end_of_epoch_callback(logs: dict, runner: PPORunner):
    collect_object_level_metrics = get_object_level_metrics(runner.collect_env)
    for metric, value in collect_object_level_metrics.items():
        logs[f'Collect{metric}'] = value
    reset_object_level_metrics(runner.collect_env)

    if runner.eval_env is not None and any('Eval' in k for k in logs):
        eval_object_level_metrics = get_object_level_metrics(runner.eval_env)
        for metric, value in eval_object_level_metrics.items():
            logs[f'Eval{metric}'] = value
        reset_object_level_metrics(runner.eval_env)


def create_runner(args: dict) -> PPORunner:

    object_config = load_pretrained_q_network(
        folder=args.get('pretrained_runs_folder', 'runs'),
        run=args.get('pretrained_run', 'run-16823527592836354'),
        percentile=args.get('pretrained_percentile', 0.75),
        epoch=args.get('pretrained_epoch', None),
    )

    args['object_level_config'] = object_config
    object_env_name = object_config.get('env', 'coinrun')

    env, eval_env, video_env = create_runner_envs(
        object_config=object_config, **args
    )

    run_group = f'rlts-procgen-{object_env_name}'
    ppo_runner = PPORunner(
        env,
        eval_env=eval_env,
        video_env=video_env,
        wandb_group=run_group,
        end_of_epoch_callback=end_of_epoch_callback,
        rewrite_rewards=not args.get('no_rewrite_rewards', False),
        **args
    )

    return ppo_runner


def main(args: dict):
    ppo_runner = create_runner(args)
    ppo_runner.run()


def parse_args() -> dict:
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

    if args.get('gpus'):
        restrict_gpus(args['gpus'])

    main(args)
