from rlts.train.experiment_utils import create_parser, create_meta_env
from rlts.runners.ppo_runner import PPORunner
from rlts.meta.search_tree import ObjectState
from rlts.meta.meta_env import MetaEnv
from rlts.maze.maze_state import MazeState, RestrictedActionsMazeState
from rlts.maze.manhattan_q import ManhattanQHat, InadmissableManhattanQHat
from rlts.maze.maze_env import make_maze_env
from rlts.maze.maze_tree_policy_renderer import render_tree_policy

from typing import Type

from tf_agents.train.gym_wrapper import GymWrapper
from tf_agents.train.batched_py_environment import BatchedPyEnvironment


def get_maze_name(config: dict) -> str:
    name = ''

    if config.get('procgen_maze', False):
        name += 'procgen_'

    n = config.get('maze_size', 5)
    agent = config.get('agent', 'ppo')
    name += f'{n}x{n}_maze_{agent}'

    if config.get('rewrite_rewards', False):
        name += '_rrr'
    if config.get('finish_on_terminate', False):
        name += '_fot'

    return name


def create_maze_meta_env(object_state_cls: Type[ObjectState] = None,
                         min_computation_steps: int = 0,
                         enable_render: bool = True,
                         maze_size: int = 5,
                         procgen_maze: bool = True,
                         seed: int = 0,
                         **config) -> MetaEnv:
    object_state_cls = object_state_cls or RestrictedActionsMazeState

    object_env = make_maze_env(
        seed=seed,
        maze_size=(maze_size, maze_size),
        goal_reward=1,
        generate_new_maze_on_reset=procgen_maze,
        enable_render=enable_render
    )

    if config.get('q_hat_inadmissable_action', None) is not None:
        q_hat = InadmissableManhattanQHat(
            bad_action=object_env.ACTION.index(config['q_hat_inadmissable_action']),
            overestimation_factor=config.get('q_hat_overestimation_factor', 2.0),
            maze_env=object_env
        )
    else:
        q_hat = ManhattanQHat(object_env)

    return create_meta_env(
        object_env, object_state_cls.extract_state(object_env),
        q_hat, config,
        tree_policy_renderer=render_tree_policy,
        min_computation_steps=min_computation_steps
    )


def create_batched_maze_envs(
        n_envs: int,
        enable_render=False,
        min_computation_steps: int = 0,
        env_multithreading=True, **config) -> BatchedPyEnvironment:

    if n_envs == 0:
        return None

    if config.get('restricted_maze_states', True):
        state_cls = RestrictedActionsMazeState
    else:
        state_cls = MazeState

    return BatchedPyEnvironment([
        GymWrapper(create_maze_meta_env(
            object_state_cls=state_cls,
            enable_render=enable_render,
            min_computation_steps=min_computation_steps,
            seed=config.get('seed', 0) + i,
            **{k: v for k, v in config.items() if k != 'seed'}
        ))
        for i in range(n_envs)
    ], multithreading=env_multithreading)


def create_batched_maze_meta_envs(
        n_collect_envs=16, n_video_envs=2, n_eval_envs=8,
        min_train_computation_steps=0,
        env_multithreading=True, **config):

    env = create_batched_maze_envs(
        n_collect_envs,
        enable_render=False,
        min_computation_steps=min_train_computation_steps,
        env_multithreading=env_multithreading,
        **config)

    eval_env = create_batched_maze_envs(
        n_eval_envs,
        enable_render=False,
        env_multithreading=env_multithreading,
        **config)

    video_env = create_batched_maze_envs(
        n_video_envs,
        enable_render=True,
        env_multithreading=env_multithreading,
        **config)

    return env, eval_env, video_env


def create_runner(args):
    env, eval_env, video_env = create_batched_maze_meta_envs(**args)
    return PPORunner(
        env, eval_env=eval_env, video_env=video_env,
        name=get_maze_name(args), **args
    )


def main(args):
    ppo_runner = create_runner(args)
    ppo_runner.run()


def parse_args():
    parser = create_parser()

    parser.add_argument('--maze_size', type=int, default=10,
                        help='Size of the maze.')
    parser.add_argument('--procgen_maze', type=bool, default=True,
                        help='Whether to use a procgen maze.')
    parser.add_argument('--restricted_maze_states', type=bool, default=True,
                        help='Whether to restrict movements and node expansions to only free spaces.')
    parser.add_argument('--q_hat_inadmissable_action', type=str, default=None,
                        help='Action to estimate inadmissably.')
    parser.add_argument('--q_hat_inadmissable_overestimation', type=float, default=2.0,
                        help='Distance overestimation to use for inadmissable action.')

    args = vars(parser.parse_args())

    print('Arguments:')
    for k, v in args.items():
        print(f'\t{k}: {v}')
    print()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
