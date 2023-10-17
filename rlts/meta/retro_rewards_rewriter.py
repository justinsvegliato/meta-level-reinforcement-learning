import math
from typing import Any, Callable, List, Optional, Union
from rlts.meta.meta_env import MetaEnv
from rlts.meta.search_tree import SearchTree
from rlts.meta.tree_policy import SearchTreePolicy
# from mlrl.meta.metrics import MeanFinalPolicyValueMetric
from rlts.maze.maze_utils import construct_maze_policy_string

from tf_agents.trajectories import trajectory
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.metrics import py_metrics

import gym_maze
import numpy as np


class TrajectoryRewriterWrapper:
    """
    Wrapper class for a trajectory that allows for rewriting the reward.
    Specifically, this class handles the case where the trajectory is from a
    batched environment, and the reward needs to be rewritten for each
    environment in the batch.
    """

    def __init__(self,
                 env: Union[BatchedPyEnvironment, PyEnvironment, TFEnvironment],
                 traj: trajectory.Trajectory):
        self.env = env
        self.traj = traj
        self.n_envs = env.batch_size or 1
        self.is_handled = {
            i: False for i in range(self.n_envs)
        }
        self.original_rewards = np.array(traj.reward)
        self.prev_policies = {
            i: self.get_env(i).prev_search_policy for i in range(self.n_envs)
        }
        self.policies = {
            i: self.get_env(i).search_tree_policy for i in range(self.n_envs)
        }
        self.eval_trees = {i: None for i in range(self.n_envs)}
        self.prev_policy_values = {i: 0. for i in range(self.n_envs)}
        self.policy_values = {i: 0. for i in range(self.n_envs)}
        self.computational_rewards = {i: 0. for i in range(self.n_envs)}

    def is_terminal(self, i: int = 0) -> bool:
        return self.traj.is_last()[i] or self.traj.action[i] == 0

    def get_env(self, i: int = 0) -> MetaEnv:
        env = self.env.envs[i] if hasattr(self.env, 'envs') else self.env
        if hasattr(env, 'gym'):
            return env.gym
        return env

    def get_policy(self, i: int = 0) -> Optional[SearchTreePolicy]:
        return self.policies[i]

    def get_prev_policy(self, i: int = 0) -> Optional[SearchTreePolicy]:
        return self.prev_policies[i]

    def get_reward(self, i: int = 0) -> float:
        return self.traj.reward[i]

    def replace_reward(self, i: int, new_reward: float):
        mask = np.zeros_like(self.traj.reward)
        mask[i] = 1
        replaced_reward = self.traj.reward * (1 - mask) + new_reward * mask
        self.traj = self.traj._replace(reward=replaced_reward)

    def rewrite_reward(self, i: int, final_tree: Optional[SearchTree], verbose: bool = False):
        """
        Rewrites the reward for the ith environment in the batch.
        Marks the ith environment as handled.

        Args:
            i: The index of the environment in the batch.
            final_tree: The final search tree for the ith environment to use for
                evaluating each action.
        """
        self.is_handled[i] = True
        if self.is_terminal(i):
            return

        prev_policy = self.get_prev_policy(i)
        policy = self.get_policy(i)
        if None in [prev_policy, policy, final_tree]:
            return

        prev_policy_value = prev_policy.evaluate(final_tree)
        policy_value = policy.evaluate(final_tree)
        computational_reward = policy_value - prev_policy_value

        self.prev_policy_values[i] = prev_policy_value
        self.policy_values[i] = policy_value
        self.computational_rewards[i] = computational_reward

        meta_env = self.get_env(i)
        reward = computational_reward - meta_env.cost_of_computation

        original_reward = self.original_rewards[i]
        if verbose and math.fabs(reward - original_reward) > 1e-9:
            print(self._get_policy_update_string(meta_env, prev_policy, policy))
            print(f'Rewriting reward from {original_reward:.4f} to '
                  f'{reward:.4f} = ({policy_value:.4f} - {prev_policy_value:.4f})'
                  f' - {meta_env.cost_of_computation:.4f}\n')

        self.replace_reward(i, reward)
        self.eval_trees[i] = final_tree

    def all_rewrites_handled(self) -> bool:
        return all(self.is_handled.values())

    def is_rewrite_handled(self, i: int = 0) -> bool:
        return self.is_handled[i]

    def get_info(self) -> dict:
        return {
            'original_reward': self.original_rewards,
            'prev_policy': self.prev_policies,
            'policy': self.policies,
            'eval_tree': self.eval_trees,
            'terminal': [self.is_terminal(i) for i in range(self.n_envs)],
            'prev_policy_value': self.prev_policy_values,
            'policy_value': self.policy_values,
            'computational_reward': self.computational_rewards
        }


class RetroactiveRewardsRewriter:
    """
    The class creates a callable object that can be used as a callback for
    the tf_agents driver. The callback is called after each meta-env step
    with a trajectory object. The trajectories are stored in a list
    as they are recieved, and once a computation is terminated (i.e. the
    agent takes an action a terminate action or the episode ends), the callback
    will rewrite each of the stored rewards according to the final search tree.
    The idea is that this improves the reliability of the reward signal, as
    the agent will be able to learn from the final search tree, rather than
    the prior search tree at the time of the action.

    After the rewards are rewritten, the trajectories are flushed from the
    class using a call to a given callable.

    The class is designed to work with a batched environment, and will
    rewrite the rewards for each environment in the batch. It will only
    flush the trajectories once all of the environments have had a chance
    to rewrite their rewards.
    """

    def __init__(self,
                 env: Union[TFEnvironment, PyEnvironment, BatchedPyEnvironment],
                 add_batch: Union[Callable[[trajectory.Trajectory], Any],
                                  Callable[[trajectory.Trajectory, dict], Any]],
                 include_info: bool = False,
                 metric_buffer_size: int = 10000,
                 metric_name: str = 'RewrittenAverageReturn',
                 compute_metrics: bool = True,
                 verbose: bool = False):
        """
        Args:
            env: The environment to use for the callback.
            add_batch: A callable that is called with a trajectory object when
                the rewards have been rewritten.
            include_info: Whether to include the info dict with the trajectory
                when calling add_batch.
            verbose: Whether to print out information about the policy updates
        """
        self.env = env
        self.verbose = verbose
        self.include_info = include_info
        self.compute_metrics = compute_metrics

        if hasattr(env, 'batch_size'):
            self.n_envs = env.batch_size or 1
        else:
            self.n_envs = 1

        self.add_batch: Callable[[trajectory.Trajectory], Any] = add_batch
        self.trajectories: List[TrajectoryRewriterWrapper] = []

        self.return_metric = py_metrics.AverageReturnMetric(buffer_size=metric_buffer_size,
                                                            batch_size=self.n_envs,
                                                            name=metric_name)
        # self.mean_policy_value_metric = MeanFinalPolicyValueMetric()

    def get_env(self, i: int = 0) -> MetaEnv:
        env = self.env.envs[i] if hasattr(self.env, 'envs') else self.env
        if hasattr(env, 'gym'):
            return env.gym
        return env

    def reset(self):
        self.trajectories = []
        for metric in self.get_metrics():
            metric.reset()

    def flush_all(self):
        for i in range(self.n_envs):
            self.rewrite_rewards(i)
        self.flush_handled_trajectories()
        self.reset()

    def flush_handled_trajectories(self):
        """
        Flushes trajectories that have had their rewards rewritten across all
        environments in the batch using the add_batch callable.
        """
        for traj_wrapper in self.trajectories:
            if traj_wrapper.all_rewrites_handled():
                if self.compute_metrics:
                    self.return_metric(traj_wrapper.traj)
                if self.include_info:
                    self.add_batch((traj_wrapper.traj, traj_wrapper.get_info()))
                else:
                    self.add_batch(traj_wrapper.traj)

        self.trajectories = [
            traj_wrapper for traj_wrapper in self.trajectories
            if not traj_wrapper.all_rewrites_handled()
        ]

    def _get_policy_update_string(self, meta_env, prev_policy, policy):
        if isinstance(meta_env.object_env, gym_maze.envs.MazeEnv):
            maze_string_prev = construct_maze_policy_string(meta_env, prev_policy)
            maze_string_now = construct_maze_policy_string(meta_env, policy)
            string = 'Previous Policy\t\t\t\tNew Policy\b=n'
            string += '\n'.join((
                f'{l1}\t\t{l2}' for l1, l2 in zip(maze_string_prev.splitlines(),
                                                  maze_string_now.splitlines())
            ))
            return string
        else:
            return f'{prev_policy} -> {policy}'

    def get_unhandled_trajectories(self, i: int = 0):
        return [t for t in self.trajectories if not t.is_rewrite_handled(i)]

    def rewrite_rewards(self, i: int):
        """
        Rewrites the rewards for the ith environment in the batch using the current
        search tree for evaluating each of the previous policies.

        The rewards are rewritten in the order that the trajectories were recieved,
        and marked as handled. This method does not flush the trajectories.
        """
        unhandled_trajectories = self.get_unhandled_trajectories(i)
        if not unhandled_trajectories:
            return

        # Get final tree to use for reward calculation
        if len(unhandled_trajectories) > 1:
            # -2 because the last trajectory is the terminate action
            # and so the policy has already been reset
            last_policy = unhandled_trajectories[-2].get_policy(i)
            final_tree = last_policy.tree if last_policy is not None else None
            # self.mean_policy_value_metric(final_tree.root_node.get_value())
        else:
            final_tree = None  # implies that terminate was the first and only action

        if self.verbose and len(unhandled_trajectories) > 1:
            print(f'Rewriting rewards with the final tree:\n{final_tree}')

        # Rewrite rewards
        for traj_wrapper in unhandled_trajectories:
            traj_wrapper.rewrite_reward(i, final_tree, self.verbose)

        if self.verbose:
            print()

    def __call__(self, traj: trajectory.Trajectory):
        """
        Takes a trajectory and stores it in a list. Once a sequence of trajectories
        is terminated, the rewards are rewritten and the trajectories are flushed.
        """
        traj_wrapper = TrajectoryRewriterWrapper(self.env, traj)
        self.trajectories.append(traj_wrapper)
        any_rewrites = False
        for i in range(self.n_envs):
            if traj_wrapper.is_terminal(i):
                self.rewrite_rewards(i)
                any_rewrites = True

        if any_rewrites:
            self.flush_handled_trajectories()

    def get_metrics(self):
        # return [self.return_metric, self.mean_policy_value_metric]
        return [self.return_metric]
