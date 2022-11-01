from typing import Optional
from mlrl.meta.search_tree import SearchTree, ObjectState
from mlrl.meta.q_estimation import SearchQEstimator, RecursiveDeterministicEstimator

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class SearchPolicy(ABC):

    def __init__(self, tree: SearchTree):
        self.tree = tree

    @abstractmethod
    def get_action(self, state: ObjectState) -> int:
        pass


class GreedyPolicy(SearchPolicy):

    def __init__(self, tree: SearchTree, object_discount: float = 0.99):
        super().__init__(tree)
        self.estimator = RecursiveDeterministicEstimator(object_discount)

    def q_aggregation(self, q_val1: float, q_val2) -> float:
        """
        How to aggregate competing Q-values for the same state and action
        from different parts of the tree.

        The default implementation returns the minimum value, 
        i.e. a pessimistic estimate of the Q-value.
        """
        if q_val1 is None:
            return q_val2
        if q_val2 is None:
            return q_val1

        return min(q_val1, q_val2)

    def get_action(self, state: ObjectState) -> int:
        """
        This method returns the action with the highest Q-value for the given state
        using the tree policy to evaluate the Q-values.
        """

        q_values = defaultdict(lambda: None)
        state_action_in_tree = defaultdict(lambda: False)
        for node in self.tree.node_list:
            if node.state == state:
                for action in node.get_children():
                    state_action_in_tree[action] = True
                    q_est = self.estimator.compute_q(node, action)
                    q_values[action] = self.q_aggregation(q_values[action], q_est)

        # If a state-action pair is not in the tree, we return the Q-hat function
        for action in state.get_actions():
            if not state_action_in_tree[action]:
                q_values[action] = self.tree.q_function(state, action)

        return max(q_values, key=q_values.get)
