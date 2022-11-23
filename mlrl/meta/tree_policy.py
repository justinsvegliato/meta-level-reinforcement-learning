from typing import Dict, List, Tuple
from mlrl.meta.search_tree import SearchTree, SearchTreeNode, ObjectState

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class SearchTreePolicy(ABC):

    def __init__(self,
                 tree: SearchTree,
                 optimal_q_estimator: 'SearchOptimalQEstimator',
                 object_discount: float = 0.99):
        # makes a copy of the tree to avoid changing
        # the policy if the original tree is changed
        self.tree = tree.copy()
        self.object_discount = object_discount
        self.estimator = optimal_q_estimator

    @abstractmethod
    def get_action(self, state: ObjectState) -> int:
        pass

    @abstractmethod
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        pass

    def evaluate(
            self, evaluation_tree: SearchTree, verbose=False
    ) -> float:
        """
        Estimate the value of the given state under the current policy,
        given the evaluation tree.
        """
        def recursive_compute_value(node: SearchTreeNode) -> float:

            probabilities = self.get_action_probabilities(node.state)
            value = 0
            if verbose:
                print(', '.join([
                    f'P({node.state.get_action_label(a)} | {node.state}) = {p}'
                    for a, p in probabilities.items()
                ]))

            for action, prob in probabilities.items():
                if prob == 0:
                    continue

                children = node.get_children()
                if action in children and children[action]:
                    q_value = np.mean([
                        child.reward + self.object_discount * recursive_compute_value(child)
                        for child in children[action]
                    ])
                    if verbose:
                        action_label = node.state.get_action_label(action)
                        print(f'Recursive Q-hat({node.state}, {action_label}) = {q_value:.3f} ')

                else:
                    q_value = self.estimator.estimate_optimal_q_value(evaluation_tree, node.state, action)
                    if verbose:
                        action_label = node.state.get_action_label(action)
                        print(f'Leaf evaluation: Q-hat({node.state}, {action_label}) = {q_value:.5f}')

                value += prob * q_value

            if verbose:
                print(f'Value({node.state}) = {value:.5f}')

            return value

        return recursive_compute_value(self.tree.get_root())


class GreedySearchTreePolicy(SearchTreePolicy):

    def __init__(self, tree: SearchTree, object_discount: float = 0.99):
        from mlrl.meta.q_estimation import DeterministicOptimalQEstimator
        estimator = DeterministicOptimalQEstimator(object_discount)
        super().__init__(tree, estimator, object_discount)

    def get_action(self, state: ObjectState) -> int:
        """
        This method returns the action with the highest Q-value for the given
        state using the tree policy to evaluate the Q-values.
        """
        q_values = self.estimator.estimate_optimal_q_values(self.tree, state)
        return max(q_values, key=q_values.get)

    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        action = self.get_action(state)
        probabilities = defaultdict(lambda: 0.0)
        probabilities[action] = 1.0
        return probabilities

    def __repr__(self) -> str:

        def build_trajectory(node: SearchTreeNode) -> List[str]:
            action = self.get_action(node.state)
            if node.has_action_children(action):
                child = node.children[action][0]
                return [node.state.get_action_label(action)] + build_trajectory(child)
            return [node.state.get_action_label(action)]

        traj_string = ''.join(build_trajectory(self.tree.get_root()))
        return f'Greedy Policy Trajectory: {traj_string}'
