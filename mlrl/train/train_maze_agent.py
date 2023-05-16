from mlrl.train.experiment_utils import (
    parse_args, create_agent, create_training_run,
    create_batched_tf_meta_env, create_meta_env
)
from mlrl.meta.search_tree import ObjectState
from mlrl.meta.meta_env import MetaEnv
from mlrl.maze.maze_state import RestrictedActionsMazeState, MazeState
from mlrl.maze.manhattan_q import ManhattanQHat
from mlrl.maze.maze_env import make_maze_env

from typing import Type


def get_maze_name(args) -> str:
    name = ''

    if args.get('maze_procgen', False):
        name += 'procgen_'

    n = args.get('maze_size', 5)
    agent = args['agent']
    name += f'{n}x{n}_maze_{agent}'

    return name


def create_maze_meta_env(object_state_cls: Type[ObjectState],
                         args: dict) -> MetaEnv:
    maze_n = args.get('maze_size', 5)
    procgen = bool(args.get('maze_procgen', False))

    object_env = make_maze_env(
        seed=args.get('seed', 0),
        maze_size=(maze_n, maze_n),
        goal_reward=1,
        render_shape=(64, 64),
        generate_new_maze_on_reset=procgen,
    )

    q_hat = ManhattanQHat(object_env)

    return create_meta_env(
        object_env,
        object_state_cls.extract_state(object_env),
        q_hat,
        args
    )


def main():

    args = parse_args()

    if args.get('restricted_maze_states', True):
        object_state_cls = RestrictedActionsMazeState
    else:
        object_state_cls = MazeState

    tf_env = create_batched_tf_meta_env(
        lambda: create_maze_meta_env(object_state_cls, args),
        args.get('env_batch_size')
    )

    agent, models = create_agent(tf_env, **args)

    run = create_training_run(
        agent, tf_env, models, args, get_maze_name(args)
    )
    run.execute()


if __name__ == "__main__":
    main()
