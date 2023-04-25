from mlrl.experiments.experiment_utils import parse_args, create_meta_env
from mlrl.runners.ppo_runner import PPORunner
from mlrl.meta.meta_env import MetaEnv
from mlrl.procgen.procgen_state import ProcgenState, ProcgenProcessing
from mlrl.procgen.procgen_env import make_procgen
from mlrl.experiments.procgen_dqn import create_rainbow_agent

import json
from pathlib import Path
from typing import Tuple
import re

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


def create_procgen_meta_env(object_config: dict,
                            min_computation_steps: int = 0,
                            **meta_config) -> MetaEnv:

    object_env = make_procgen(object_config, n_envs=1)

    def q_hat(s, a):
        return s.q_values[a]

    return create_meta_env(
        object_env, ProcgenState.extract_state(object_env),
        q_hat, meta_config,
        # tree_policy_renderer=render_tree_policy,
        min_computation_steps=min_computation_steps
    )


def create_batched_procgen_envs(
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


def create_batched_procgen_meta_envs(
        env_batch_size: int,
        object_config: dict,
        n_video_envs=0, n_eval_envs=0, min_train_computation_steps=0,
        env_multithreading=True, **config):

    env = create_batched_procgen_envs(
        env_batch_size, object_config,
        min_computation_steps=min_train_computation_steps,
        env_multithreading=env_multithreading,
        **config)

    eval_env = create_batched_procgen_envs(
        n_eval_envs, object_config,
        env_multithreading=env_multithreading,
        **config)

    video_env = create_batched_procgen_envs(
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


def load_pretrained_q_network(run: str = None):
    run = run or 'run-16823527592836354'
    folder = f'./sync/dqn/categorical_dqn_agent/{run}'

    with open(folder + '/config.json') as f:
        object_config = json.load(f)

    env = make_procgen(object_config, n_envs=1)
    q_net, agent = create_rainbow_agent(env, object_config)

    model_paths = [
        (str(path).replace('.index', ''), *values)
        for path in Path(f'{folder}/model_weights').glob('*')
        if path.is_file() and str(path).endswith('.index') 
        and (values := parse_model_weights_string(str(path.name))) is not None
    ]
    path, epoch, ret_val = max(model_paths, key=lambda x: x[2])
    print(f'Loading model from {run} that scored return value {ret_val} at epoch {epoch}')
    print('Object-level config:')
    for k, v in object_config.items():
        print(f'\t - {k}: {v}')

    q_net.load_weights(path)

    ProcgenProcessing.set_pretrained_agent(agent)

    return object_config


def main(args):
    object_config = load_pretrained_q_network(run=args.get('pretraining_run', None))
    for k, v in object_config.items():
        args['object_level_' + k] = v
    object_env_name = object_config.get('env', 'coinrun')

    env, eval_env, video_env = create_batched_procgen_meta_envs(object_config=object_config, **args)
    ppo_runner = PPORunner(
        env, eval_env=eval_env, video_env=video_env,
        name=f'procgen-{object_env_name}', **args
    )
    ppo_runner.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
