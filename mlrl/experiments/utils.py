from ..run import TrainingRun
from ..meta.meta_env import mask_invalid_action_constraint_splitter

import argparse

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.networks.sequential import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment


def create_dqn_agent(tf_env: TFPyEnvironment,
                     q_net: tf.keras.Model,
                     learning_rate=1e-3,
                     target_network_update_period=500,
                     meta_discount=0.99,
                     agent='dqn',
                     **_) -> DqnAgent:
    """ Creates a DQN agent. """

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    if agent == 'dqn':
        agent = DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=Sequential([q_net]),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            observation_and_action_constraint_splitter=mask_invalid_action_constraint_splitter,
            target_update_period=target_network_update_period,
            gamma=meta_discount,
            train_step_counter=train_step_counter
        )
    elif agent == 'ddqn':
        agent = DdqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=Sequential([q_net]),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            observation_and_action_constraint_splitter=mask_invalid_action_constraint_splitter,
            target_update_period=target_network_update_period,
            gamma=meta_discount,
            train_step_counter=train_step_counter
        )
    else:
        raise ValueError(f'Unknown agent: {agent}')

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
        video_freq=args['video_freq'],
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
    parser.add_argument('--eval_steps', type=int, default=250,
                        help='Number of steps to evaluate for.')
    parser.add_argument('--initial_collect_steps', type=int, default=500,
                        help='Number of steps to collect before training.')
    parser.add_argument('--video_steps', type=int, default=120,
                        help='Number of steps to record a video for.')
    parser.add_argument('--video_freq', type=int, default=5,
                        help='Number of steps to record a video for.')
    parser.add_argument('--max_tree_size', type=int, default=10,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--target_network_update_period', type=int, default=500,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--object_discount', type=float, default=0.99,
                        help='Discount factor in object-level environment.')
    parser.add_argument('--meta_discount', type=float, default=0.99,
                        help='Discount factor in meta-level environment.')
    parser.add_argument('--epsilon_greedy', type=float, default=0.1,
                        help='Epsilon for epsilon-greedy exploration.')
    parser.add_argument('--agent', type=str, default='ddqn',
                        help='Agent class to use.')

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
