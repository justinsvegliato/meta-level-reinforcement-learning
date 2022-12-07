from mlrl.experiments.experiment_utils import parse_args, create_meta_env
from mlrl.experiments.ppo_runner import PPORunner
from mlrl.meta.search_tree import ObjectState
from mlrl.meta.meta_env import MetaEnv
from mlrl.maze.maze_state import RestrictedActionsMazeState
from mlrl.maze.manhattan_q import ManhattanQHat
from mlrl.maze.maze_env import make_maze_env
from mlrl.maze.maze_tree_policy_renderer import render_tree_policy

from typing import Type

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment


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


def create_maze_meta_env(object_state_cls: Type[ObjectState] = RestrictedActionsMazeState,
                         args: dict = None,
                         min_computation_steps: int = 0,
                         enable_render: bool = True) -> MetaEnv:
    args = args or {}

    maze_n = args.get('maze_size', 5)
    procgen = bool(args.get('procgen_maze', True))

    object_env = make_maze_env(
        seed=args.get('seed', 0),
        maze_size=(maze_n, maze_n),
        goal_reward=1,
        generate_new_maze_on_reset=procgen,
        enable_render=enable_render
    )

    manhattan_q_hat = ManhattanQHat(object_env)

    return create_meta_env(
        object_env, object_state_cls.extract_state(object_env),
        manhattan_q_hat, args,
        tree_policy_renderer=render_tree_policy,
        min_computation_steps=min_computation_steps
    )


def create_batched_maze_meta_envs(
        env_batch_size,
        n_video_envs=0, n_eval_envs=0, min_train_computation_steps=0,
        env_multithreading=True, **config):

    def create_envs(n_envs, enable_render=False, min_computation_steps: int = 0):
        if n_envs == 0:
            return None

        return BatchedPyEnvironment([
            GymWrapper(create_maze_meta_env(
                RestrictedActionsMazeState, config,
                enable_render=enable_render,
                min_computation_steps=min_computation_steps
            ))
            for _ in range(n_envs)
        ], multithreading=env_multithreading)

    env = create_envs(env_batch_size,
                      enable_render=False,
                      min_computation_steps=min_train_computation_steps)
    eval_env = create_envs(n_eval_envs, enable_render=False)
    video_env = create_envs(n_video_envs, enable_render=True)

    return env, eval_env, video_env


def main():
    args = parse_args()
    env, eval_env, video_env = create_batched_maze_meta_envs(**args)
    ppo_runner = PPORunner(
        env, eval_env=eval_env, video_env=video_env,
        name=get_maze_name(args), **args
    )
    ppo_runner.run()


if __name__ == "__main__":
    main()
