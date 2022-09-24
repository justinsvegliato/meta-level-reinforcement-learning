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

    def make_maze_search_tree(env) -> SearchTree:
        return SearchTree(env, extract_state=MazeState.extract_state)

    meta_env = MetaEnv(object_env, q_hat, make_maze_search_tree, max_tree_size=args.get('max_tree_size', 10))
    tf_env = TFPyEnvironment(GymWrapper(meta_env))
    q_net = SearchQModel()
    agent = create_dqn_agent(tf_env, q_net, args)
    run = create_training_run(agent, tf_env, q_net, args, 'simple_procgen_grid_dqn')
    run.execute()


if __name__ == "__main__":
    main()
