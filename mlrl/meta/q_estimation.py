# from functools import lru_cache
from typing import Callable, Dict, List, NoReturn, Tuple
from mlrl.networks.search_q_net import SearchQNetwork
from .search_tree import ObjectState, SearchTree, find_cycle

from abc import ABC, abstractmethod

import silence_tensorflow.auto  # noqa
import tensorflow as tf


class SearchOptimalQEstimator(ABC):

    def __init__(self, discount: float = 0.99):
        self.discount = discount

    @abstractmethod
    def estimate_optimal_q_value(
            self, tree: SearchTree, state: ObjectState, action: int) -> float:
        pass

    @abstractmethod
    def estimate_and_cache_optimal_q_values(self, tree: SearchTree, verbose=False):
        """
        Estimates the optimal Q values for all states in the tree and stores them in the tree.
        """
        pass


class DeterministicOptimalQEstimator(SearchOptimalQEstimator):
    """
    Assumes a deterministic environment and only uses child nodes to compute Q-values
    """

    def __init__(self, discount: float = 0.99):
        super().__init__(discount)

    def estimate_optimal_q_value(
            self, tree: SearchTree, state: ObjectState, action: int,
            trajectory=None, verbose=False
    ) -> float:
        trajectory = trajectory or []

        state_nodes = tree.get_state_nodes(state)
        children = sum([node.get_children(action) for node in state_nodes], [])
        if any(node.is_terminal for node in state_nodes):
            q_value = 0
            if verbose:
                print('Q-value of terminal state:', q_value)

        elif children:
            if verbose:
                print('Aggregating Q-value estimates from children:', children)

            q_value = 0

            for child in children:
                if child.is_terminal:
                    value = 0
                else:
                    cycle_trajectory = find_cycle(child.state, trajectory)
                    if cycle_trajectory:
                        value = self.compute_cycle_value(cycle_trajectory)
                        if verbose:
                            print(f'Cycle value of duplicate child {child}:', value)
                    else:
                        child_q_values = self.estimate_optimal_q_distribution(
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

        for node in state_nodes:
            node.set_q_value(action, q_value)

        return q_value

    def estimate_optimal_q_distribution(
            self, tree: SearchTree, state: ObjectState,
            verbose=False, trajectory=None
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

    def estimate_and_cache_optimal_q_values(self, tree: SearchTree, verbose=False) -> NoReturn:
        self.estimate_optimal_q_distribution(tree, tree.root_node.state, verbose=verbose)

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
