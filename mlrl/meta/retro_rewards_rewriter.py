import math
from mlrl.meta.meta_env import MetaEnv
from mlrl.maze.maze_utils import construct_maze_policy_string

from collections import OrderedDict
from functools import lru_cache

from tf_agents.trajectories import trajectory
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.utils.nest_utils import split_nested_tensors
from tf_agents.train.utils import spec_utils

import gym_maze


class RetroactiveRewardsRewriter:

    def __init__(self,
                 env: PyEnvironment,
                 collect_data_spec,
                 add_batch: callable,
                 verbose: bool = False):
        self.env = env
        self.collect_data_spec = collect_data_spec
        self.verbose = verbose

        self.observation_tensor_spec, *_ = spec_utils.get_tensor_specs(env)
        if isinstance(env, BatchedPyEnvironment):
            self.n_envs = env.batch_size
        else:
            self.n_envs = 1

        self.add_batch = add_batch
        self.trajectories_and_policies = [[] for _ in range(self.n_envs)]

    def get_env(self, i: int) -> MetaEnv:
        gym_wrapper_env = self.env if self.n_envs == 1 else self.env.envs[i]
        return gym_wrapper_env.gym

    def index_tensor(self, tensor_or_nest, spec, i):
        if isinstance(tensor_or_nest, OrderedDict):
            return OrderedDict([
                (k, v[i]) for k, v in tensor_or_nest.items()
            ])
        elif isinstance(tensor_or_nest, dict):
            return split_nested_tensors(
                tensor_or_nest, spec, self.n_envs
            )[i]
        elif tensor_or_nest == ():
            return ()
        else:
            return tensor_or_nest[i]

    def get_env_traj(self, traj: trajectory.Trajectory, i: int) -> trajectory.Trajectory:
        """
        Constructs a trajectory object specific to the ith environment in the batch.
        """
        observation = self.index_tensor(traj.observation, self.observation_tensor_spec, i)
        policy_info = self.index_tensor(traj.policy_info, self.collect_data_spec.policy_info, i)

        return trajectory.Trajectory(
            step_type=traj.step_type[i],
            action=traj.action[i],
            observation=observation,
            next_step_type=traj.next_step_type[i],
            policy_info=policy_info,
            reward=traj.reward[i],
            discount=traj.discount[i]
        )

    def is_terminate(self, traj: trajectory.Trajectory) -> bool:
        return traj.is_last() or traj.action.numpy() == 0

    def rewrite_rewards_and_flush(self, i: int):
        meta_env = self.get_env(i)
        final_tree = meta_env.tree

        if self.verbose and len(self.trajectories_and_policies[i]) > 1:
            print(f'Rewriting rewards with the final tree:\n{final_tree}')

        @lru_cache(maxsize=1000)
        def get_policy_value(policy):
            return policy.tree_conditioned_root_value_estimate(final_tree)

        for prev_policy, traj, policy in self.trajectories_and_policies[i]:
            if prev_policy is not None:
                prev_policy_value = get_policy_value(prev_policy)
                policy_value = get_policy_value(policy)

                computational_reward = policy_value - prev_policy_value
                reward = computational_reward - meta_env.cost_of_computation

                if self.verbose and math.fabs(reward - traj.reward) > 1e-9:
                    if isinstance(meta_env.object_env, gym_maze.envs.MazeEnv):
                        string1 = construct_maze_policy_string(meta_env, prev_policy)
                        string2 = construct_maze_policy_string(meta_env, policy)
                        print('Previous Policy\t\t\t\tNew Policy')
                        print('\n'.join((
                            f'{l1}\t\t{l2}' for l1, l2 in zip(string1.splitlines(), string2.splitlines())
                        )))
                    else:
                        print(policy)
                    print(f'Rewriting reward from {traj.reward:.4f} to '
                          f'{reward:.4f} = ({policy_value:.4f} - {prev_policy_value:.4f})'
                          f' - {meta_env.cost_of_computation:.4f}\n')

                traj = traj._replace(reward=reward)

            self.add_batch(traj)

        if self.verbose:
            print()

        self.trajectories_and_policies[i] = []

    def __call__(self, traj):
        for i in range(self.n_envs):
            env_traj = self.get_env_traj(traj, i)
            env = self.get_env(i)
            if self.is_terminate(env_traj):
                self.rewrite_rewards_and_flush(i)
                continue

            item = (env.prev_search_policy, env_traj, env.search_tree_policy)
            self.trajectories_and_policies[i].append(item)
