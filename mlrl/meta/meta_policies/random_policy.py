from mlrl.meta.meta_env import mask_token_splitter

import tensorflow as tf
import numpy as np
from tf_agents.policies.random_py_policy import RandomPyPolicy


def create_random_search_policy(env):
    return RandomPyPolicy(
        env.time_step_spec(),
        env.action_spec(),
        observation_and_action_constraint_splitter=mask_token_splitter)


def create_random_search_policy_no_terminate(env):
    """
    Creates a policy that uniformly chooses a random node expansion
    to perform, but never terminates the episode.
    """

    def mask_token_splitter_no_terminate(obs):
        tokens, mask = mask_token_splitter(obs)
        not_terminate = np.ones(mask.shape)
        for i in range(mask.shape[0]):
            # if there are no more expansions, don't mask terminate
            if np.sum(mask[i, 1:].numpy()) > 1:
                not_terminate[i, 0] = 0
        mask = mask * tf.convert_to_tensor(not_terminate, dtype=tf.int32)
        return tokens, mask

    return RandomPyPolicy(
        env.time_step_spec(),
        env.action_spec(),
        observation_and_action_constraint_splitter=mask_token_splitter_no_terminate)
