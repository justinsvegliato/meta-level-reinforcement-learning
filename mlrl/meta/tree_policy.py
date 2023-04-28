# from functools import lru_cache
from functools import lru_cache
from typing import Dict, List, Tuple
from mlrl.meta.search_tree import SearchTree, SearchTreeNode, ObjectState, find_cycle

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

    def get_action_index(self, state: ObjectState) -> int:
        """
        Get the action to take in the given state.
        
        The action is returned as an integer indicating the index of
        the gym action in the ObjectState.get_actions representation.

        For example, an environment may have an action space of unhashable
        objects, e.g. uci move objects in gymchess, q-distributions are maintained
        as dictionaries of integers to floats, and the action space is a list of
        uci move objects. In this case, the action returned by this function
        would be the index of the uci move object in the action space.
        """
        action_probs = self.get_action_probabilities(state)
        return np.random.choice(list(action_probs.keys()),
                                p=list(action_probs.values()))

    @abstractmethod
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:
        pass

    def compute_exp_cycle_value(
            self, cycle_traj: List[Tuple[ObjectState, int, float, float]]) -> float:
        """
        Computes the expected value of a cycle in the tree.
        """
        cycle_len = len(cycle_traj)
        cycle_return = sum([
            reward * self.object_discount ** t
            for t, (*_, reward, _) in enumerate(cycle_traj)
        ])

        traj_prob = np.prod([p for *_, p in cycle_traj])

        return cycle_return / (1 - traj_prob * self.object_discount**cycle_len)

    @lru_cache(maxsize=100)
    def evaluate(
            self, evaluation_tree: SearchTree, verbose=False
    ) -> float:
        """
        Estimate the value of the given state under the current policy,
        given the evaluation tree.
        """
        def recursive_compute_value(node: SearchTreeNode, trajectory=None) -> float:
            trajectory = trajectory or []

            cycle = find_cycle(node.state, trajectory)
            if cycle:
                return self.compute_exp_cycle_value(cycle)

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

                children: List[SearchTreeNode] = node.get_children(action)
                if children:
                    q_value = np.mean([
                        child.reward + self.object_discount * recursive_compute_value(
                            child, trajectory + [(node.state, action, child.reward, prob)]
                        )
                        for child in children
                    ])
                    if verbose:
                        action_label = node.state.get_action_label(action)
                        print(f'Recursive Q-hat({node.state}, {action_label}) = {q_value:.3f} ')

                else:
                    q_value = evaluation_tree.q_function(node.state, action)
                    if verbose:
                        action_label = node.state.get_action_label(action)
                        print(f'Leaf evaluation: Q-hat({node.state}, {action_label}) = {q_value:.5f}')

                value += prob * q_value

            if verbose:
                print(f'Value({node.state}) = {value:.5f}')

            return value

        return recursive_compute_value(evaluation_tree.get_root())


class GreedySearchTreePolicy(SearchTreePolicy):

    def __init__(self,
                 tree: SearchTree,
                 break_ties_randomly: bool = True,
                 object_discount: float = 0.99):
        self.break_ties_randomly = break_ties_randomly

        from mlrl.meta.q_estimation import DeterministicOptimalQEstimator
        estimator = DeterministicOptimalQEstimator(object_discount)
        super().__init__(tree, estimator, object_discount)

        self.estimator.estimate_and_cache_optimal_q_values(self.tree)

    @lru_cache(maxsize=100)
    def get_action_probabilities(self, state: ObjectState) -> Dict[int, float]:

        state_nodes = self.tree.get_state_nodes(state)
        if not state_nodes:
            q_values = {
                a_idx: self.tree.q_function(state, action)
                for a_idx, action in enumerate(state.get_actions())
            }
        else:
            node, *_ = state_nodes  # all nodes should have the same q-values
            q_values = {
                a_idx: node.get_q_value(action)
                for a_idx, action in enumerate(node.state.get_actions())
            }

        max_q = max(q_values.values())
        max_actions = [a for a, q in q_values.items() if q == max_q]
        probs = {a: 0. for a in q_values}

        if self.break_ties_randomly:
            for a in max_actions:
                probs[a] = 1 / len(max_actions)
        else:  # choose the first action in the list of max actions
            probs[max_actions[0]] = 1.

        return probs

    def __repr__(self) -> str:

        def build_trajectory(node: SearchTreeNode) -> List[str]:
            action = self.get_action_index(node.state)
            if node.has_action_children(action):
                child = node.children[action][0]
                return [node.state.get_action_label(action)] + build_trajectory(child)
            return [node.state.get_action_label(action)]

        traj_string = ''.join(build_trajectory(self.tree.get_root()))
        return f'Greedy Policy Trajectory: {traj_string}'
