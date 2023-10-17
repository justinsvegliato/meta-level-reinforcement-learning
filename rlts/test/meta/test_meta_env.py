from rlts.train.maze_meta import create_maze_meta_env
from rlts.maze.maze_state import RestrictedActionsMazeState
from rlts.meta.meta_env import MetaEnv

import numpy as np


def test_meta_env_expand_all():
    """ Test that the meta environment expands all actions. """
    config = {'expand_all_actions': True}

    for seed in range(10):
        config['seed'] = seed
        env: MetaEnv = create_maze_meta_env(RestrictedActionsMazeState, **config)
        assert env.expand_all_actions
        env.reset()

        assert len(env.tree.node_list) == 1

        action = 1
        node_to_expand_idx = action - 1  # expand the root node
        node_to_expand = env.tree.node_list[node_to_expand_idx]
        expected_children = len(node_to_expand.state.get_actions())
        expected_tree_size = 1 + expected_children
        env.step(action)

        assert len(env.tree.node_list) == expected_tree_size, f'Tree: {env.tree}'
        assert len(env.tree.get_root().children) == expected_children, f'Tree: {env.tree}'


def test_meta_env_expand_all_correct_obs_and_action():
    """
    Checks that the meta environment that expands all actions has
    the correct observation and action spaces.
    """

    max_tree_size = 10
    config = {'expand_all_actions': True, 'max_tree_size': max_tree_size}
    env = create_maze_meta_env(RestrictedActionsMazeState, **config)
    env.reset()
    obs = env.get_observation()

    assert MetaEnv.ACTION_MASK_KEY in obs
    assert MetaEnv.SEARCH_TOKENS_KEY in obs
    search_tokens = obs[MetaEnv.SEARCH_TOKENS_KEY]
    n_tokens, token_dim = search_tokens.shape
    assert n_tokens == 1 + max_tree_size, f'Expected {1 + max_tree_size} tokens, got {n_tokens}'

    root_node = env.tree.get_root()
    state_dim = root_node.state.get_state_vector_dim()
    action_dim = root_node.state.get_action_vector_dim()
    exp_dim = state_dim + action_dim + 2 * max_tree_size + env.n_meta_feats
    assert token_dim == exp_dim, f'Expected {exp_dim}, got {token_dim}'

    obs_shape = env.observation_space[MetaEnv.SEARCH_TOKENS_KEY].shape
    assert obs_shape == (n_tokens, exp_dim), 'Environment observation space has wrong shape'

    action_mask = obs[MetaEnv.ACTION_MASK_KEY]
    assert action_mask.shape == (n_tokens,)

    assert np.all(action_mask[:2] == 1.), 'Root node and terminate should be legal'
    assert np.all(action_mask[2:] == 0.), 'All other actions should be illegal'


def test_meta_env_expand_one():
    """ Test that the meta environment expands one action when configured to do so. """
    config = {'expand_all_actions': False}

    for seed in range(10):
        config['seed'] = seed
        env = create_maze_meta_env(RestrictedActionsMazeState, **config)
        assert not env.expand_all_actions
        env.reset()

        assert len(env.tree.node_list) == 1

        # expand the root node with first legal action
        action = 1
        expected_tree_size = 2
        env.step(action)

        assert len(env.tree.node_list) == expected_tree_size, f'Tree: {env.tree}'
        assert len(env.tree.get_root().children) == 1, f'Tree: {env.tree}'
