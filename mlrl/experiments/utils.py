from ..run import TrainingRun
from ..meta.meta_env import mask_invalid_action_constraint_splitter

import argparse

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.networks.sequential import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment


def create_dqn_agent(tf_env: TFPyEnvironment,
                     q_net: tf.keras.Model,
                     args: dict) -> DqnAgent:
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


def create_training_run(
    agent: DqnAgent,
    tf_env: TFPyEnvironment,
    model: tf.keras.Model,
    args: dict,
    name: str
) -> TrainingRun:
    return TrainingRun(
        agent, tf_env, model,
        num_epochs=args['num_epochs'],
        experience_batch_size=args['batch_size'],
        train_steps_per_epoch=args['steps_per_epoch'],
        num_eval_episodes=args['num_eval_episodes'],
        eval_steps=args['eval_steps'],
        initial_collect_steps=args['initial_collect_steps'],
        video_steps=args['video_steps'],
        name=name
    )


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size to use.')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to run for.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of epochs to run for.')
    parser.add_argument('--num_eval_episodes', type=int, default=3,
                        help='Number of episodes to evaluate for.')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of steps to evaluate for.')
    parser.add_argument('--initial_collect_steps', type=int, default=500,
                        help='Number of steps to collect before training.')
    parser.add_argument('--video_steps', type=int, default=60,
                        help='Number of steps to record a video for.')
    parser.add_argument('--max_tree_size', type=int, default=10,
                        help='Maximum number of nodes in the search tree.')

    return parser


def parse_args(verbose=True):
    parser = create_parser()
    args = vars(parser.parse_args())
    if verbose:
        print('Arguments:')
        for k, v in args.items():
            print(f'\t{k}: {v}')
        print()
    return args
