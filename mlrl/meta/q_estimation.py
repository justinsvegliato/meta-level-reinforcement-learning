from .search_tree import SearchTree, SearchTreeNode


class SimpleSearchBasedQEstimator:
    """
    Assumes a deterministic environment and only uses child nodes to compute Q-values
    """

    def __init__(self, search_tree: SearchTree, discount: float = 0.99):
        self.discount = discount
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
        if action in children and children[action]:
            child_node = children[action][0]
            reward = child_node.get_reward_received()
            return reward + self.discount * self.compute_value(child_node)

        return search_tree_node.get_q_value(action)

    def compute_value(self, search_tree_node: SearchTreeNode) -> float:
        """
        Computes the value of a given state using the search tree and the Q-hat function
        """
        actions = search_tree_node.get_state().get_actions()
        return max(self.compute_q(search_tree_node, action) for action in actions)
