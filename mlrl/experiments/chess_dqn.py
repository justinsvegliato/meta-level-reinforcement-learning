from .utils import parse_args, create_dqn_agent, create_training_run
from ..meta.search_tree import SearchTree
from ..meta.meta_env import MetaEnv
from ..meta.search_q_model import SearchQModel
from ..chess.chess_env import ChessVsRandom
from ..chess.chess_state import ChessState, ChessQFunction

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper

import badgyal


def main():

    args = parse_args()

    object_env = ChessVsRandom()
    object_env.reset()

    chess_network = badgyal.BGNet(cuda=True)
    q_hat = ChessQFunction(chess_network)
    initial_tree = SearchTree(object_env, ChessState.extract_state(object_env), q_hat)
    meta_env = MetaEnv(object_env, initial_tree, max_tree_size=32)
    tf_env = TFPyEnvironment(GymWrapper(meta_env))

    q_net = SearchQModel()
    agent = create_dqn_agent(tf_env, q_net, args)
    run = create_training_run(agent, tf_env, q_net, args, 'single_maze_dqn')
    run.execute()


if __name__ == "__main__":
    main()
