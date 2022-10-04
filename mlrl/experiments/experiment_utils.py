from ..run import TrainingRun
from ..meta.meta_env import mask_token_splitter
from ..meta.search_networks import SearchQNetwork, SearchActorNetwork, SearchValueNetwork
from ..meta.meta_env import MetaEnv
from ..meta.search_tree import QFunction, SearchTree, ObjectState

import argparse
from typing import Union, List

import gym

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.networks.sequential import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.agents.ppo.ppo_agent import PPOAgent


def create_batched_tf_meta_env(make_meta_env, batch_size: int) -> TFPyEnvironment:
    return TFPyEnvironment(BatchedPyEnvironment([
        GymWrapper(make_meta_env()) for _ in range(batch_size)
    ]))


def create_meta_env(object_env: gym.Env,
                    init_state: ObjectState,
                    q_hat: QFunction,
                    args: dict,
                    object_action_to_string=None) -> MetaEnv:
    """
    Creates a meta environment from a given object environment,
    an initial state, and a Q-function.
    """
    initial_tree = SearchTree(object_env, init_state, q_hat)
    meta_env = MetaEnv(object_env,
                       initial_tree,
                       max_tree_size=args.get('max_tree_size', 10),
                       split_mask_and_tokens=args.get('split_mask_and_tokens', True),
                       object_action_to_string=object_action_to_string)

    if args.get('meta_time_limit', None):
        return gym.wrappers.time_limit.TimeLimit(meta_env, args['meta_time_limit'])

    return meta_env


def create_agent(tf_env: TFPyEnvironment,
                 learning_rate=1e-3,
                 target_network_update_period=500,
                 meta_discount=0.99,
                 agent='dqn',
                 **args) -> TFAgent:
    """ Creates a DQN agent. """

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    if agent in ['dqn', 'ddqn']:
        print('Creating DQN/DDQN agent...')

        q_net = SearchQNetwork(head_dim=args.get('transformer_head_dim', 32),
                               n_layers=args.get('transformer_n_layers', 2),
                               n_heads=args.get('transformer_n_heads', 2))

        Agent = DqnAgent if agent == 'dqn' else DdqnAgent

        agent = Agent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=Sequential([q_net]),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            observation_and_action_constraint_splitter=mask_token_splitter,
            target_update_period=target_network_update_period,
            gamma=meta_discount,
            train_step_counter=train_step_counter
        )
        print('Initialising agent...')
        agent.initialize()

        return agent, q_net

    elif agent == 'ppo_agent':
        print('Creating PPO agent...')

        actor_net = SearchActorNetwork(head_dim=args.get('transformer_head_dim', 32),
                                       n_layers=args.get('transformer_n_layers', 2),
                                       n_heads=args.get('transformer_n_heads', 3))

        value_net = SearchValueNetwork(head_dim=args.get('transformer_head_dim', 32),
                                       n_layers=args.get('transformer_n_layers', 2),
                                       n_heads=args.get('transformer_n_heads', 3))

        agent = PPOAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            actor_net=Sequential([actor_net]),
            value_net=Sequential([value_net]),
            optimizer=optimizer,
            train_step_counter=train_step_counter,
            # compute_value_and_advantage_in_train=False,
            # update_normalizers_in_train=False,
            discount_factor=meta_discount
        )

        print('Initialising agent...')
        agent.initialize()

        return agent, [actor_net, value_net]

    else:
        raise ValueError(f'Unknown agent: {agent}')


def create_training_run(
    agent: DqnAgent,
    tf_env: TFPyEnvironment,
    model: Union[tf.keras.Model, List[tf.keras.Model]],
    args: dict,
    name: str
) -> TrainingRun:
    print('Creating training run...')
    return TrainingRun(
        agent, tf_env, model,
        num_epochs=args.get('num_epochs', 10),
        experience_batch_size=args.get('experience_batch_size', 64),
        train_steps_per_epoch=args.get('steps_per_epoch', 1000),
        num_eval_episodes=args.get('num_eval_episodes', 3),
        eval_steps=args.get('eval_steps', 250),
        initial_collect_steps=args.get('initial_collect_steps', 500),
        video_steps=args.get('video_steps', 300),
        video_freq=args.get('video_freq', 5),
        name=name,
        run_args=args
    )


def create_parser():
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--experience_batch_size', type=int, default=64,
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

    # Environment parameters
    parser.add_argument('--max_tree_size', type=int, default=10,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--object_discount', type=float, default=0.99,
                        help='Discount factor in object-level environment.')
    parser.add_argument('--meta_discount', type=float, default=0.99,
                        help='Discount factor in meta-level environment.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--env_batch_size', type=int, default=2,
                        help='Batch size for the environment.')

    # Maze parameters
    parser.add_argument('--maze_size', type=int, default=5,
                        help='Size of the maze.')
    parser.add_argument('--procgen_maze', type=bool, default=False,
                        help='Whether to use a procgen maze.')
    parser.add_argument('--restricted_maze_states', type=bool, default=True,
                        help='Whether to restrict movements and node expansions to only free spaces.')

    # Agent parameters
    parser.add_argument('--agent', type=str, default='ppo_agent',
                        help='Agent class to use.')
    parser.add_argument('--transformer_head_dim', type=int, default=16,
                        help='Head dimension for the agent transformer.')
    parser.add_argument('--transformer_n_layers', type=int, default=2,
                        help='Number of agent transformer layers.')
    parser.add_argument('--transformer_n_heads', type=int, default=3,
                        help='Number of agent transformer heads.')
    parser.add_argument('--target_network_update_period', type=int, default=500,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--epsilon_greedy', type=float, default=0.1,
                        help='Epsilon for epsilon-greedy exploration.')

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