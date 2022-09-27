from .utils import parse_args, create_dqn_agent, create_training_run
from ..maze.maze_env import make_maze_env
from ..maze.maze_state import MazeState
from ..maze.manhattan_q import ManhattanQHat
from ..meta.search_tree import SearchTree
from ..meta.meta_env import MetaEnv
from ..meta.search_q_model import SearchQModel

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper


def main():

    args = parse_args()

    object_env = make_maze_env(
        seed=0, maze_size=(5, 5), goal_reward=1, render_shape=(64, 64),
        generate_new_maze_on_reset=True
    )

    q_hat = ManhattanQHat(object_env)
    initial_tree = SearchTree(object_env, MazeState.extract_state(object_env), q_hat)
    meta_env = MetaEnv(object_env, initial_tree, max_tree_size=args.get('max_tree_size', 10),
                       object_action_to_string=lambda a: object_env.ACTION[a])
    tf_env = TFPyEnvironment(GymWrapper(meta_env))

    q_net = SearchQModel()
    agent = create_dqn_agent(tf_env, q_net, **args)

    run = create_training_run(agent, tf_env, q_net, args, 'procgen_maze_dqn')

    run.execute()


if __name__ == "__main__":
    main()
