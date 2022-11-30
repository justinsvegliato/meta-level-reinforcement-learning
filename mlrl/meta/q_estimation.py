# from functools import lru_cache
from typing import Callable, Dict, List, Tuple
from mlrl.networks.search_q_net import SearchQNetwork
from .search_tree import ObjectState, SearchTree

from abc import ABC, abstractmethod

import silence_tensorflow.auto  # noqa
import tensorflow as tf


class SearchOptimalQEstimator(ABC):

    def __init__(self, discount: float = 0.99):
        self.discount = discount

    @abstractmethod
    def estimate_optimal_value(
            self, tree: SearchTree, state: ObjectState) -> float:
        pass

    @abstractmethod
    def estimate_optimal_q_value(
            self, tree: SearchTree, state: ObjectState, action: int) -> float:
        pass

    @abstractmethod
    def estimate_optimal_q_values(
            self, tree: SearchTree, state: ObjectState, verbose=False, trajectory=None
    ) -> Dict[int, float]:
        pass


class DeterministicOptimalQEstimator(SearchOptimalQEstimator):
    """
    Assumes a deterministic environment and only uses child nodes to compute Q-values
    """

    def __init__(self, discount: float = 0.99):
        super().__init__(discount)

    def estimate_optimal_value(
            self, tree: SearchTree, state: ObjectState, trajectory=None, verbose=False) -> float:
        q_values = self.estimate_optimal_q_values(tree, state, trajectory=trajectory, verbose=verbose)
        return max(q_values.values())

    def estimate_optimal_q_value(
            self, tree: SearchTree, state: ObjectState, action: int,
            trajectory=None, verbose=False
    ) -> float:
        trajectory = trajectory or []

        state_nodes = tree.get_state_nodes(state)
        children = sum([node.get_children(action) for node in state_nodes], [])
        if children:
            if verbose:
                print('Aggregating Q-value estimates from children:', children)

            q_value = 0

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
                        tree, child.state,
                        trajectory=trajectory + [(state, action, child.reward)],
                        verbose=verbose
                    )
                    value = max(child_q_values.values())

                q_value = child.reward + self.discount * value

            q_value /= len(children)

        else:
            q_value = tree.q_function(state, action)

            if verbose:
                print(f'Computing Q-value from estimator on leaf node:'
                      f'Q({state}, {state.get_action_label(action)}) =', q_value)

        return q_value

    def estimate_optimal_q_values(
            self, tree: SearchTree, state: ObjectState, verbose=False, trajectory=None
    ) -> Dict[int, float]:
        trajectory = trajectory or []

        if verbose:
            state_nodes = tree.get_state_nodes(state)
            nodes_str = '\n'.join(map(str, state_nodes)) if state_nodes else '[]'
            print(f'Estimating optimal Q-values for state {state} from nodes:\n{nodes_str}')

        q_values = {
            action: self.estimate_optimal_q_value(tree, state, action,
                                                  trajectory=trajectory, verbose=verbose)
            for action in state.get_actions()
        }

        if verbose:
            print()
            print('Optimal Q-values:\n', '\n'.join([
                f'Q*({state}, {state.get_action_label(a)}) = {q}'
                for a, q in q_values.items()
            ]))

        return q_values

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

    def compute_cycle_value(self, cycle_traj: List[Tuple[ObjectState, int, float]]) -> float:
        """
        Computes the value of a cycle in the tree.
        """
        cycle_len = len(cycle_traj)
        cycle_return = sum([
            reward * self.discount ** t
            for t, (_, _, reward) in enumerate(cycle_traj)
        ])
        return cycle_return / (1 - self.discount**cycle_len)


class SearchQModelEstimator(SearchOptimalQEstimator):
    """
    Uses a Q-model to compute Q-values
    """

    def __init__(self,
                 q_net: SearchQNetwork,
                 tree_tokeniser: Callable[[SearchTree], tf.Tensor]):
        self.q_net = q_net
        self.tree_tokeniser = tree_tokeniser

    # @lru_cache
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
