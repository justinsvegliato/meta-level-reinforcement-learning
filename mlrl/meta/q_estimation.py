from functools import lru_cache
from typing import Callable
from mlrl.networks.search_q_net import SearchQNetwork
from .search_tree import SearchTree, SearchTreeNode

from abc import ABC, abstractmethod

import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf


class SearchQEstimator(ABC):

    def __init__(self, discount: float = 0.99):
        self.discount = discount

    @abstractmethod
    def compute_root_q(self, tree: SearchTree, action: int) -> float:
        """
        Computes the Q-value for a given state and action using the search
        tree and the Q-hat function to evaluate the leaf nodes.

        Args:
            search_tree_node: The node in the search tree corresponding to the state
            action: The action to evaluate
        """
        pass

    def compute_root_value(self, search_tree: SearchTree) -> float:
        """
        Computes the value of a given state using the search tree and the Q-hat function
        """
        actions = search_tree.get_root().get_actions()
        return max((self.compute_root_q(search_tree, action) for action in actions), default=0)


class RecursiveDeterministicEstimator(SearchQEstimator):
    """
    Assumes a deterministic environment and only uses child nodes to compute Q-values
    """

    def __init__(self, discount: float = 0.99):
        self.discount = discount

    def compute_value(self, search_node: SearchTreeNode) -> float:
        """
        Computes the value of a given state using the search tree and the Q-hat function
        """
        actions = search_node.get_actions()
        return max((self.compute_q(search_node, action) for action in actions), default=0)

    def compute_q(self, search_tree_node: SearchTreeNode, action: int) -> float:
        """
        Computes the Q-value for a given state and action using the search
        tree and the Q-hat function to evaluate the leaf nodes.

        Args:
            search_tree_node: The node in the search tree corresponding to the state
            action: The action to evaluate
        """
        children = search_tree_node.get_children()
        if action in children and children[action]:
            child_node = children[action][0]
            reward = child_node.get_reward_received()
            return reward + self.discount * self.compute_value(child_node)

        return search_tree_node.get_q_value(action)

    def compute_root_q(self, tree: SearchTree, action: int) -> float:
        """
        Computes the Q-value for a given state and action using the search
        tree and the Q-hat function to evaluate the leaf nodes.

        Args:
            search_tree_node: The node in the search tree corresponding to the state
            action: The action to evaluate
        """
        return self.compute_q(tree.get_root(), action)


class SearchQModelEstimator(SearchQEstimator):
    """
    Uses a Q-model to compute Q-values
    """

    def __init__(self,
                 q_net: SearchQNetwork,
                 tree_tokeniser: Callable[[SearchTree], tf.Tensor]):
        self.q_net = q_net
        self.tree_tokeniser = tree_tokeniser
    
    @lru_cache
    def compute_root_q_distribution(self, search_tree: SearchTree) -> tf.Tensor:
        """
        Computes the Q-value distribution for a given state using the search tree and the Q-hat function
        """
        tokens = self.tree_tokeniser(search_tree)
        q_values = self.q_net(tokens)
        return q_values

    def compute_root_q(self, search_tree: SearchTree, action: int) -> float:
        """
        Computes the Q-value for a given state and action using the search
        tree and the Q-hat function to evaluate the leaf nodes.

        Args:
            search_tree_node: The node in the search tree corresponding to the state
            action: The action to evaluate
        """
        q_values = self.compute_root_q_distribution(search_tree)
        return q_values[action]

    def compute_root_value(self, search_tree: SearchTree) -> float:
        """
        Computes the value of a given state using the search tree and the Q-hat function
        """
        q_values = self.compute_root_q_distribution(search_tree)
        return tf.reduce_max(q_values)
