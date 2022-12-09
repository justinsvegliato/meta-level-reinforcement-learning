from mlrl.meta.meta_env import MetaEnv

from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import policy_step
import tensorflow as tf


def get_a_star_action(meta_env: MetaEnv, action_mask: list) -> int:
    """
    Chooses the node expansion according the the A* search algorithm,
    i.e. choose node n that maximises f(n) = g(n) + h(n), where g(n) is the return
    to reach the node and h(n) is the "heuristic" return from the node.
    In other words, the expected return from the root if passing through the node.
    """
    available_expansions = [
        i for i, can_take in enumerate(action_mask)
        if i > 0 and can_take
    ]

    if not available_expansions:
        return 0

    def eval_node(a: int) -> float:
        node = meta_env.tree.node_list[a - 1]
        return node.get_exp_root_return()

    return max(available_expansions, key=eval_node)


class AStarPolicy(PyPolicy):

    def __init__(self, env: BatchedPyEnvironment):
        super().__init__(env.time_step_spec(), env.action_spec())
        self.env = env
        self.batch_size = env.batch_size

    def _action(self, time_step, policy_state, seed=None):
        obs = time_step.observation

        action = tf.convert_to_tensor([
            get_a_star_action(self.env.envs[i], obs['action_mask'][i])
            for i in range(self.batch_size)
        ], dtype=self.action_spec.dtype)

        return policy_step.PolicyStep(action, policy_state, info=())
