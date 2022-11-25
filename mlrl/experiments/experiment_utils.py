from mlrl.run import TrainingRun
from mlrl.meta.meta_env import mask_token_splitter
from mlrl.networks.search_q_net import SearchQNetwork
from mlrl.networks.search_actor_nets import SearchActorNetwork
from mlrl.networks.search_value_net import SearchValueNetwork
from mlrl.meta.meta_env import MetaEnv
from mlrl.meta.search_tree import QFunction, SearchTree, ObjectState

import argparse
from typing import Union, List

import gym

import silence_tensorflow.auto  # noqa
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
                    config: dict,
                    tree_policy_renderer=None) -> MetaEnv:
    """
    Creates a meta environment from a given object environment,
    an initial state, and a Q-function.
    """
    initial_tree = SearchTree(object_env, init_state, q_hat)
    meta_env = MetaEnv(object_env,
                       initial_tree,
                       max_tree_size=config.get('max_tree_size', 10),
                       split_mask_and_tokens=config.get('split_mask_and_tokens', True),
                       expand_all_actions=config.get('expand_all_actions', False),
                       computational_rewards=config.get('computational_rewards', True),
                       finish_on_terminate=config.get('finish_on_terminate', False),
                       tree_policy_renderer=tree_policy_renderer)

    if config.get('meta_time_limit', None):
        return gym.wrappers.time_limit.TimeLimit(meta_env, config['meta_time_limit'])

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

        q_net = SearchQNetwork(d_model=args.get('transformer_d_model', 32),
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

        actor_net = SearchActorNetwork(d_model=args.get('transformer_d_model', 32),
                                       n_layers=args.get('transformer_n_layers', 2),
                                       n_heads=args.get('transformer_n_heads', 3))

        value_net = SearchValueNetwork(d_model=args.get('transformer_d_model', 32),
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
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--experience_batch_size', type=int, default=512,
                        help='Batch size to use.')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Train batch size to use in PPO')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to run for.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of epochs to run for.')
    parser.add_argument('--num_eval_episodes', type=int, default=3,
                        help='Number of episodes to evaluate for.')
    parser.add_argument('--eval_steps', type=int, default=1600,
                        help='Number of steps to evaluate for.')
    parser.add_argument('--initial_collect_steps', type=int, default=500,
                        help='Number of steps to collect before training.')
    parser.add_argument('--n_video_steps', type=int, default=120,
                        help='Number of steps to record a video for.')
    parser.add_argument('--video_freq', type=int, default=5,
                        help='Frequency of recording videos.')
    parser.add_argument('--n_video_envs', type=int, default=5,
                        help='Number of video environments to record.')

    # Meta-level environment parameters
    parser.add_argument('--expand_all_actions', type=bool, default=True,
                        help='Whether to expand all actions in the meta environment '
                             'with each computational action.')
    parser.add_argument('--max_tree_size', type=int, default=32,
                        help='Maximum number of nodes in the search tree.')
    parser.add_argument('--meta_discount', type=float, default=0.99,
                        help='Discount factor in meta-level environment.')
    parser.add_argument('--meta_time_limit', type=int, default=500,
                        help='Maximum number of steps in meta-level environment.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--env_batch_size', type=int, default=32,
                        help='Batch size for the environment.')
    parser.add_argument('--computational_rewards', type=bool, default=True,
                        help='Whether to use computational rewards.')
    parser.add_argument('--rewrite_rewards', type=bool, default=True,
                        help='Whether to rewrite computational rewards.')
    parser.add_argument('--finish_on_terminate', type=bool, default=True,
                        help='Whether to finish meta-level episode on computational terminate action.')

    # Object-level environment parameters
    parser.add_argument('--object_discount', type=float, default=0.99,
                        help='Discount factor in object-level environment.')

    # Maze parameters
    parser.add_argument('--maze_size', type=int, default=5,
                        help='Size of the maze.')
    parser.add_argument('--procgen_maze', type=bool, default=True,
                        help='Whether to use a procgen maze.')
    parser.add_argument('--restricted_maze_states', type=bool, default=True,
                        help='Whether to restrict movements and node expansions to only free spaces.')

    # Agent parameters
    parser.add_argument('--agent', type=str, default='ppo',
                        help='Agent class to use.')
    parser.add_argument('--transformer_d_model', type=int, default=16,
                        help='Head dimension for the agent transformer.')
    parser.add_argument('--transformer_n_layers', type=int, default=2,
                        help='Number of agent transformer layers.')
    parser.add_argument('--transformer_n_heads', type=int, default=3,
                        help='Number of agent transformer heads.')
    parser.add_argument('--n_lstm_layers', type=int, default=3,
                        help='Number of lstm layers.')

    # DQN parameters
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
