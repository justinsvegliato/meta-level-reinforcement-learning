from typing import Dict, List, Tuple
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

    def compute_cycle_value(self, cycle_traj: List[Tuple[ObjectState, int, float]]) -> float:
        """
        Computes the value of a cycle in the tree.
        """
        cycle_len = len(cycle_traj)
        cycle_return = sum([
            reward * self.object_discount ** t
            for t, (_, _, reward) in enumerate(cycle_traj)
        ])

        return cycle_return / (1 - self.object_discount**cycle_len)

    @staticmethod
    def find_cycle(
            state: ObjectState,
            trajectory: List[Tuple[ObjectState, int, float]]
    ) -> List[Tuple[ObjectState, int, float]]:
        state_idxs = [
            i for i, (s, _, _) in enumerate(trajectory)
            if s == state
        ]
        if state_idxs:
            i = state_idxs[0]
            return trajectory[i:]
        return []

    def estimate_optimal_q_values(
            self, state: ObjectState, verbose=False, trajectory=None
    ) -> Dict[ObjectState, float]:
        trajectory = trajectory or []

        q_values = defaultdict(lambda: None)
        state_nodes = self.tree.get_state_nodes(state)
        if verbose:
            nodes_str = '\n'.join(map(str, state_nodes)) if state_nodes else '[]'
            print(f'Estimating optimal Q-values for state {state} from nodes:\n{nodes_str}')

        for action in state.get_actions():
            children = sum([node.get_children()[action] for node in state_nodes], [])
            if children:
                if verbose:
                    print('Aggregating Q-value estimates from children:', children)

                q_values[action] = 0

                for child in children:
                    cycle_trajectory = self.find_cycle(child.state, trajectory)
                    if cycle_trajectory:
                        value = self.compute_cycle_value(cycle_trajectory)
                        if verbose:
                            print(f'Cycle value of duplicate child {child}:', value)
                    else:
                        if verbose:
                            print()
                        child_q_values = self.estimate_optimal_q_values(
                            child.state,
                            trajectory=trajectory + [(state, action, child.reward)],
                            verbose=verbose
                        )
                        value = max(child_q_values.values())

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
        def recursive_compute_value(
                node: SearchTreeNode,
                trajectory: List[Tuple[ObjectState, int, float]] = None
        ) -> float:
            trajectory = trajectory or []

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
                        cycle_trajectory = self.find_cycle(child.state, trajectory)
                        if cycle_trajectory:
                            child_value = self.compute_cycle_value(cycle_trajectory)
                        else:
                            child_value = recursive_compute_value(
                                child, trajectory=trajectory + [(node.state, action, child.reward)])

                        q = child.reward + self.object_discount * child_value

                        if debug_verbose:
                            action_label = node.state.get_action_label(action)
                            print(f'Recursive Q-hat({node.state}, {action_label}) = {child.reward:.3f} '
                                  f'+ {self.object_discount:.3f} * {child_value:.3f} = {q:.5f}')

                        q_value = self.q_aggregation(q_value, q)

                value += prob * q_value

            if debug_verbose:
                print(f'Value({node.state}) = {value:.5f}')

            return value

        return recursive_compute_value(evaluation_tree.get_root())


class GreedySearchTreePolicy(SearchTreePolicy):

    def __init__(self, tree: SearchTree, object_discount: float = 0.99):
        from mlrl.meta.q_estimation import RecursiveDeterministicOptimalQEstimator
        estimator = RecursiveDeterministicOptimalQEstimator(object_discount)
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

    def __repr__(self) -> str:

        def build_trajectory(node: SearchTreeNode) -> List[str]:
            action = self.get_action(node.state)
            if action in node.children and node.children[action]:
                child = node.children[action][0]
                return [node.state.get_action_label(action)] + build_trajectory(child)
            return [node.state.get_action_label(action)]

        traj_string = ''.join(build_trajectory(self.tree.get_root()))
        return f'Greedy Policy Trajectory: {traj_string}'
