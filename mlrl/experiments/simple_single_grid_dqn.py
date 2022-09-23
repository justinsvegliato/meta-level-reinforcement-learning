from ..run import TrainingRun
from ..maze.maze_env import make_maze_env
from ..maze.maze_state import MazeState
from ..maze.manhattan_q import ManhattanQHat
from ..meta.search_tree import SearchTree
from ..meta.meta_env import MetaEnv, mask_invalid_action_constraint_splitter
from ..meta.search_q_model import SearchQModel

import argparse

import tensorflow as tf
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.utils import common
from tf_agents.networks.sequential import Sequential


def create_dqn_agent(tf_env: TFPyEnvironment, q_net: tf.keras.Model, args: dict) -> DqnAgent:
    """ Creates a DQN agent. """

    optimizer = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'])

    train_step_counter = tf.Variable(0)

    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=Sequential([q_net]),
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        observation_and_action_constraint_splitter=mask_invalid_action_constraint_splitter,
        train_step_counter=train_step_counter
    )

    agent.initialize()

    return agent


def main():

    args = parse_args()

    object_env = make_maze_env(
        seed=0, maze_size=(5, 5), goal_reward=1, render_shape=(64, 64)
    )
    q_hat = ManhattanQHat(object_env)

    def make_maze_search_tree(env) -> SearchTree:
        return SearchTree(env, extract_state=MazeState.extract_state)

    meta_env = MetaEnv(object_env, q_hat, make_maze_search_tree, max_tree_size=10)
    tf_env = TFPyEnvironment(GymWrapper(meta_env))
    q_net = SearchQModel()
    agent = create_dqn_agent(tf_env, q_net, args)

    run = TrainingRun(
        agent, tf_env, q_net,
        num_epochs=args['num_epochs'],
        experience_batch_size=args['batch_size'],
        train_steps_per_epoch=args['steps_per_epoch'],
        num_eval_episodes=args['num_eval_episodes'],
        eval_steps=args['eval_steps'],
        name='simple_single_grid_dqn'
    )

    run.execute()


def parse_args():
    parser = argparse.ArgumentParser()

    # default hyperparams chosen from sweep on baseline_ft_ae
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to use.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to run for.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of epochs to run for.')
    parser.add_argument('--num_eval_episodes', type=int, default=1,
                        help='Number of episodes to evaluate for.')
    parser.add_argument('--eval_steps', type=int, default=250,
                        help='Number of steps to evaluate for.')

    return vars(parser.parse_args())


if __name__ == "__main__":
    main()
