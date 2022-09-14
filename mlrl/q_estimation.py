from .search_tree import ObjectState, SearchTree, SearchTreeNode

from abc import ABC, abstractmethod


class QFunction(ABC):
    """
    Abstract class for a Q-function
    """

    @abstractmethod
    def compute_q(self, state: ObjectState, action: int) -> float:
        pass

    def __call__(self, state: ObjectState, action: int) -> float:
        return self.compute_q(state, action)


class SimpleSearchBasedQEstimator:
    """
        Assumes a deterministic environment and only uses child nodes to compute Q-values
    """

    def __init__(self, q_hat: QFunction, search_tree: SearchTree,
                 n_actions: int = 4,
                 discount: float = 0.99):
        self.n_actions = n_actions
        self.discount = discount
        self.q_hat = q_hat
        self.search_tree = search_tree

    def compute_q(self, search_tree_node: SearchTreeNode, action: int) -> float:
        """
        Computes the Q-value for a given state and action using the search
        tree and the Q-hat function to evaluate the leaf nodes.

        Args:
            search_tree_node: The node in the search tree corresponding to the state
            action: The action to evaluate
        """
        children = search_tree_node.get_children()
        if action in children:
            child_node = children[action][0]
            reward = child_node.get_reward_received()
            return reward + self.discount * self.compute_value(child_node)

        return self.q_hat(search_tree_node.get_state(), action)

    def compute_value(self, search_tree_node: SearchTreeNode) -> float:
        """ Computes the value of a given state using the search tree and the Q-hat function """
        return max(self.compute_q(search_tree_node, action) for action in range(self.n_actions))
