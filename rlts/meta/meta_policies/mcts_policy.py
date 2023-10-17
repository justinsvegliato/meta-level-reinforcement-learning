from rlts.meta.meta_env import MetaEnv

from collections import Counter
import numpy as np

from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import policy_step
import tensorflow as tf


def get_mcts_action(meta_env: MetaEnv, action_mask: list, c: float = 0.5) -> int:
    """
    Chooses the node expansion according the the MCTS algorithm
    """
    available_expansions = [
        i for i, can_take in enumerate(action_mask)
        if i > 0 and can_take
    ]

    if not available_expansions:
        return 0

    state_counter = Counter([node.state for node in meta_env.tree.node_list])
    sa_counter = Counter([
        (node.parent.state, node.action)
        for node in meta_env.tree.node_list
    ])

    def mcts_eval(node, object_action):
        state_count = state_counter[node.state]
        state_act_count = sa_counter[(node.state, object_action)]
        ucb = np.sqrt(np.log(state_count) / (1 + state_act_count))
        return node.get_q_value(object_action) + c * ucb

    def eval_meta_action(meta_action: int) -> float:
        node = meta_env.tree.node_list[meta_action - 1]
        return max(
            mcts_eval(node, object_action)
            for object_action in node.state.get_actions()
        )

    return max(available_expansions, key=eval_meta_action)


class MCTSPolicy(PyPolicy):

    def __init__(self, env: BatchedPyEnvironment, exploration_const: float = 0.5):
        super().__init__(env.time_step_spec(), env.action_spec())
        self.env = env
        self.batch_size = env.batch_size
        self.exploration_const = exploration_const

    def _action(self, time_step, policy_state, seed=None):
        obs = time_step.observation

        action = tf.convert_to_tensor([
            get_mcts_action(self.env.envs[i],
                            obs['action_mask'][i],
                            c=self.exploration_const)
            for i in range(self.batch_size)
        ], dtype=self.action_spec.dtype)

        return policy_step.PolicyStep(action, policy_state, info=())
