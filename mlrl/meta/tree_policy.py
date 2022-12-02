# from functools import lru_cache
from functools import lru_cache
from typing import Dict, List
from mlrl.meta.search_tree import SearchTree, SearchTreeNode, ObjectState

from abc import ABC, abstractmethod

import numpy as np


class SearchTreePolicy(ABC):

    def __init__(self,
                 tree: SearchTree,
                 optimal_q_estimator,
                 object_discount: float = 0.99):
        # makes a copy of the tree to avoid changing
        # the policy if the original tree is changed
        self.tree = tree.copy()
        self.object_discount = object_discount
        from mlrl.meta.q_estimation import SearchOptimalQEstimator
        self.estimator: SearchOptimalQEstimator = optimal_q_estimator

    def get_action(self, state: ObjectState) -> int:
        """
        Get the action to take in the given state.
        """
        action_probs = self.get_action_probabilities(state)
        return np.random.choice(list(action_probs.keys()),
                                p=list(action_probs.values()))

    @abstractmethod
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        pass

    @lru_cache(maxsize=100)
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

    def __init__(self,
                 tree: SearchTree,
                 break_ties_randomly: bool = True,
                 object_discount: float = 0.99):
        self.break_ties_randomly = break_ties_randomly

        from mlrl.meta.q_estimation import DeterministicOptimalQEstimator
        estimator = DeterministicOptimalQEstimator(object_discount)
        super().__init__(tree, estimator, object_discount)

    @lru_cache(maxsize=100)
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        q_values = self.estimator.estimate_optimal_q_values(self.tree, state)
        max_q = max(q_values.values())
        max_actions = [a for a, q in q_values.items() if q == max_q]
        probs = {a: 0. for a in range(len(state.get_actions()))}

        if self.break_ties_randomly:
            for a in max_actions:
                probs[a] = 1 / len(max_actions)
        else:
            probs[max_actions[0]] = 1.

        return probs

    def __repr__(self) -> str:

        def build_trajectory(node: SearchTreeNode) -> List[str]:
            action = self.get_action(node.state)
            if node.has_action_children(action):
                child = node.children[action][0]
                return [node.state.get_action_label(action)] + build_trajectory(child)
            return [node.state.get_action_label(action)]

        traj_string = ''.join(build_trajectory(self.tree.get_root()))
        return f'Greedy Policy Trajectory: {traj_string}'
