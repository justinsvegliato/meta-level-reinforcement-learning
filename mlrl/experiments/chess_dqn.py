from .utils import create_dqn_agent, create_training_run
from ..meta.search_tree import SearchTree
from ..meta.meta_env import MetaEnv
from ..meta.search_q_model import SearchQModel
from ..chess.chess_env import ChessVsRandom
from ..chess.chess_state import ChessState, ChessQFunction

import argparse

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper

import badgyal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to use.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to run for.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of epochs to run for.')
    parser.add_argument('--num_eval_episodes', type=int, default=3,
                        help='Number of episodes to evaluate for.')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Number of steps to evaluate for.')
    parser.add_argument('--initial_collect_steps', type=int, default=500,
                        help='Number of steps to collect before training.')
    parser.add_argument('--video_steps', type=int, default=300,
                        help='Number of steps to record a video for.')
    parser.add_argument('--video_freq', type=int, default=5,
                        help='Frequency of recording videos.')
    parser.add_argument('--max_tree_size', type=int, default=16,
                        help='Maximum number of nodes in the search tree.')

    args = vars(parser.parse_args())

    print('Arguments:')
    for k, v in args.items():
        print(f'\t{k}: {v}')
    print()

    return args


def main():

    args = parse_args()

    object_env = ChessVsRandom()
    object_env.reset()

    chess_network = badgyal.BGNet(cuda=True)
    q_hat = ChessQFunction(chess_network)
    initial_tree = SearchTree(object_env, ChessState.extract_state(object_env), q_hat)
    meta_env = MetaEnv(object_env, initial_tree, max_tree_size=10)
    tf_env = TFPyEnvironment(GymWrapper(meta_env))

    q_net = SearchQModel(n_object_actions=meta_env.n_object_actions, head_dim=32)
    agent = create_dqn_agent(tf_env, q_net, **args)
    run = create_training_run(agent, tf_env, q_net, args, 'chess_dqn')
    run.execute()


if __name__ == "__main__":
    main()
