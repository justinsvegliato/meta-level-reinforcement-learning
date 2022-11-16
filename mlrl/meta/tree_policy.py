from typing import Dict
from mlrl.meta.search_tree import SearchTree, SearchTreeNode, ObjectState

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class SearchTreePolicy(ABC):

    def __init__(self,
                 tree: SearchTree,
                 estimator: 'SearchQEstimator',
                 object_discount: float = 0.99):
        # makes a copy of the tree to avoid changing
        # the policy if the original tree is changed
        self.tree = tree.copy()
        self.object_discount = object_discount
        self.estimator = estimator

    @abstractmethod
    def get_action(self, state: ObjectState) -> int:
        pass

    @abstractmethod
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        pass

    def q_aggregation(self, q_val1: float, q_val2: float) -> float:
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

    def compute_cycle_value(self, leaf_node: SearchTreeNode) -> float:
        """
        Computes the value of a cycle in the tree.
        Assumes that the cycle 
        """
        cycle_len = 0
        cycle_return = 0
        node = leaf_node

        if leaf_node.duplicate_state_ancestor is None:
            raise ValueError('Leaf node does not have a duplicate state ancestor')
            
        while node != leaf_node.duplicate_state_ancestor:
            cycle_return += node.reward * self.object_discount ** cycle_len
            cycle_len += 1
            node = node.parent

        return cycle_return / (1 - self.object_discount**cycle_len)

    def estimate_optimal_q_values(self, state: ObjectState, verbose=False) -> Dict[ObjectState, float]:

        q_values = defaultdict(lambda: None)
        state_nodes = self.tree.get_state_nodes(state)
        if verbose:
            nodes_str = '\n'.join(state_nodes)
            print(f'Estimating optimal Q-values for state {state} from nodes:\n{nodes_str}')

        for action in state.get_actions():
            children = sum([node.get_children()[action] for node in state_nodes], [])
            if children:
                if verbose:
                    print('Aggregating Q-value estimates from children:', children)

                q_values[action] = 0

                for child in children:
                    if child.duplicate_state_ancestor is None:
                        if verbose:
                            print()
                        child_q_values = self.estimate_optimal_q_values(child.state,
                                                                        verbose=verbose)
                        value = max(child_q_values.values())
                    else:
                        value = self.compute_cycle_value(child)
                        if verbose:
                            print(f'Cycle value of duplicate child {child}:', value)

                    q_values[action] = child.reward + self.object_discount * value

                q_values[action] /= len(children)

            else:
                q_values[action] = self.tree.q_function(state, action)
                
                if verbose:
                    print(f'Computing Q-value from estimator on leaf node:'
                          f'Q({state}, {state.get_action_label(action)}) =', q_values[action])

        if verbose:
            print()
            print('Optimal Q-values:\n', '\n'.join([
                f'Q*({state}, {state.get_action_label(a)}) = {q}'
                for a, q in q_values.items()
            ]))

        return q_values

    def tree_conditioned_root_value_estimate(
            self, evaluation_tree: SearchTree, debug_verbose=False
            ) -> float:
        """
        Estimate the value of the given state under the current policy,
        given the evaluation tree.
        """
        def recursive_compute_value(node: SearchTreeNode) -> float:
            probabilities = self.get_action_probabilities(node.state)
            value = 0
            if debug_verbose:
                print(', '.join([
                    f'P({node.state.get_action_label(a)} | {node.state}) = {p}'
                    for a, p in probabilities.items()
                ]))

            for action, prob in probabilities.items():
                if prob == 0:
                    continue

                children = node.get_children()
                if action not in children:
                    q_value = evaluation_tree.q_function.compute_q(node.state, action)
                    if debug_verbose:
                        action_label = node.state.get_action_label(action)
                        print(f'Leaf evaluation: Q-hat({node.state}, {action_label}) = {q_value:.5f}')

                else:
                    q_value = None
                    for child in children[action]:
                        if child.duplicate_state_ancestor is None:
                            q = child.reward + self.object_discount * recursive_compute_value(child)
                        else:
                            q = child.reward + self.object_discount * self.compute_cycle_value(child)

                        if debug_verbose:
                            action_label = node.state.get_action_label(action)
                            print(f'Recursive Q-hat({node.state}, {action_label}) = {q:.5f}')

                        q_value = self.q_aggregation(q_value, q)

                value += prob * q_value

            if debug_verbose:
                print(f'Value({node.state}) = {value:.5f}')

            return value

        return recursive_compute_value(evaluation_tree.get_root())


class GreedySearchTreePolicy(SearchTreePolicy):

    def __init__(self, tree: SearchTree, object_discount: float = 0.99):
        from mlrl.meta.q_estimation import RecursiveDeterministicEstimator
        estimator = RecursiveDeterministicEstimator(object_discount)
        super().__init__(tree, estimator, object_discount)

    def get_action(self, state: ObjectState) -> int:
        """
        This method returns the action with the highest Q-value for the given
        state using the tree policy to evaluate the Q-values.
        """
        q_values = self.estimate_optimal_q_values(state)
        return max(q_values, key=q_values.get)

    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        action = self.get_action(state)
        probabilities = defaultdict(lambda: 0.0)
        probabilities[action] = 1.0
        return probabilities
