from typing import Callable, Tuple

import numpy as np
import gym

from mlrl.q_estimation import QFunction, SimpleSearchBasedQEstimator
from mlrl.search_tree import SearchTree, SearchTreeNode


class MetaEnv(gym.Env):
    """
    Class that wraps a gym_maze environment and allows for the meta-learning of the search problem
    """

    def __init__(self,
                 env: gym.Env,
                 q_hat: QFunction,
                 make_tree: Callable[[gym.Env], SearchTree],
                 object_state_vec_size: int = 2,
                 max_tree_size: int = 10):
        """
        Args:
            env: The object-level environment to wrap
            q_hat: The Q-hat function to use for the search tree at the leaf nodes
            make_tree: A function that takes an environment and returns a search 
                tree with the root node set to the current state
            object_state_vec_size: The size of the vector representation of the object state
            max_tree_size: The maximum number of nodes in the search tree
        """
        self.env = env
        self.q_hat = q_hat
        self.max_tree_size = max_tree_size
        
        self.n_object_actions = self.env.action_space.n
        self.action_space = gym.spaces.Discrete(1 + max_tree_size * self.n_object_actions)
        
        self.n_meta_data = 4 # number of meta data features: mask, parent node idx, action, reward
        self.observation_space = gym.spaces.Box(
            low=0, high=max_tree_size,
            shape=(max_tree_size, self.n_meta_data + object_state_vec_size),
            dtype=np.float32
        )

        self.make_tree = make_tree
        self.tree = None
        self.n_computations = 0

    def reset(self):
        self.env.reset()
        self.tree = self.get_root_tree()
        self.n_computations = 0

    def get_root_tree(self) -> SearchTree:
        """ Creates a new tree with the root node set to the current state of the environment """
        return self.make_tree(self.env)

    def tokenise_node(self, node: SearchTreeNode) -> np.array:
        meta_features = np.array([1, node.get_parent_id(), node.get_action(), node.reward],
                                 dtype=np.float32)
        return np.concatenate([meta_features, node.state.get_state_vector()])

    def get_observation(self) -> np.array:
        """
        Returns the observation of the meta environment.
        This is a tensor of shape (max_tree_size, n_meta_features + object_state_vec_size).
        The meta-features are: mask, parent node idx, action, reward, where mask is 1 if the row
        corresponds to a node in the tree and 0 otherwise. The parent node idx is the index of the
        parent node in the observation tensor. The action is the action taken to reach the node.
        The reward is the reward received from the underlying environment when the node was reached.
        """
        obs = np.array([
            self.tokenise_node(node)
            for node in self.tree.node_list
        ])
        padding = np.zeros((self.max_tree_size - obs.shape[0], obs.shape[1]))
        return np.concatenate([obs, padding])

    def get_object_action(self) -> int:
        q_est = SimpleSearchBasedQEstimator(self.q_hat, self.tree)
        return max(range(self.env.action_space.n),
                   key=lambda a: q_est.compute_q(self.tree.get_root(), a))
    
    def step(self, computational_action: int) -> Tuple[np.array, float, bool, dict]:
        """
        Performs a step in the meta environment. The action is interpreted as follows:
        - action == 0: terminate search and perform a step in the underlying environment
        - action > 0: expand the search tree by one node with the given action

        For example, if the underlying environment has 4 actions and the tree is currently
        of size 2, then the valid action space of the meta environment contains 1 + 2 * 4 = 9
        computational actions. Taking computational action 6 expands the second node in the tree
        with object-level action 1 (actions are 0-indexed).

        Args:
            computational_action: The action to perform in the meta environment

        Returns:
            observation: The observation of the meta environment. This is a tensor of shape
                (max_tree_size, n_meta_features + object_state_vec_size).
            reward: The reward received from the underlying environment
            done: Whether the underlying environment is done
            info: Additional information
        """
        if computational_action == 0 or self.tree.get_num_nodes() >= self.max_tree_size:
            # Perform a step in the underlying environment
            root_state = self.tree.get_root().get_state()
            root_state.set_environment_to_state(self.env)
            _, reward, done, info = self.env.step(self.get_object_action())
            self.tree = self.get_root_tree()
            return self.get_observation(), reward, done, info

        # perform a computational action on the search tree
        node_idx = (computational_action - 1) // self.n_object_actions
        object_action = (computational_action - 1) % self.n_object_actions

        reward = 0
        self.tree.expand(node_idx, object_action)
        self.n_computations += 1

        return self.get_observation(), reward, False, dict()
