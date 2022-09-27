from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from collections import defaultdict

import gym
import numpy as np


class ObjectState(ABC):
    """ An abstract class for a state of an object-level environment """

    @staticmethod
    @abstractmethod
    def extract_state(env: gym.Env) -> 'ObjectState':
        """
        A static method to extract the current state of the environment
        for later restoring and representation.
        """
        pass

    @abstractmethod
    def set_environment_to_state(self, env: gym.Env):
        """ A method to set the environment to the state of the object. """
        pass

    @abstractmethod
    def get_state_vector(self) -> np.array:
        """ Returns a vector representation of the state. """
        pass

    @abstractmethod
    def get_actions(self) -> list:
        """ Returns a list of actions that can be taken from the current state. """
        pass

    @abstractmethod
    def get_action_vector(self, action) -> np.array:
        """ Returns a vector representation of the action. """
        pass

    def get_action_vector_dim(self) -> int:
        """
        Computes the dimension of the action vectors for the problem domain.
        This should be the same for all actions.
        """
        actions = self.get_actions()
        if actions:
            return self.get_action_vector(actions[0]).shape[0]

        raise Exception(
            'Cannot determine action vector dimension: '
            'No actions available for this state'
        )

    def get_state_vector_dim(self) -> int:
        """
        Computes the dimension of the state vectors for the problem domain.
        This should be the same for all states.
        """
        return self.get_state_vector().shape[0]

    def get_maximum_number_of_actions(self):
        """
        Returns the maximum number of actions that can be taken from any state.
        If the number of actions is not fixed, this should be overriden.
        """
        return len(self.get_actions())

    def get_state_string(self) -> str:
        """ Returns a string representation of the state. """
        return str(self.get_state_vector())


class QFunction(ABC):
    """
    Abstract class for a Q-function
    """

    @abstractmethod
    def compute_q(self, state: ObjectState, action: int) -> float:
        pass

    def __call__(self, state: ObjectState, action: int) -> float:
        return self.compute_q(state, action)


class SearchTreeNode:
    """
    A class to represent a node in the search tree.
    The node contains the state of the environment, the action that led to this state,
    the reward received, and the children of the node, and other important data.
    """

    __slots__ = [
        'parent', 'node_id', 'state', 'action',
        'reward', 'children', 'done', 'q_function'
    ]

    def __init__(self,
                 node_id: int,
                 parent: 'SearchTreeNode',
                 state: ObjectState,
                 action: int,
                 reward: float,
                 done: bool,
                 q_function: QFunction):
        """
        Args:
            node_id: The id of the node. Used to identify the node in the search tree
                and the observation tensor in the meta-environment.
            parent: The parent node of the current node. None if the node is the root.
            state: The state of the environment at the current node.
            action: The action that led to the current node. -1 if the node is the root.
            reward: The reward received at upond reaching the current node.
            done: Whether the current node is a terminal node. If so, the node cannot be expanded.
        """
        self.node_id = node_id
        self.parent = parent
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.children: Dict[int, List['SearchTreeNode']] = defaultdict(list)
        self.q_function = q_function

    def expand_node(self,
                    env: gym.Env,
                    action_idx: int,
                    new_node_id: int) -> 'SearchTreeNode':
        """
        Expands the node by "simulating" taking the given action in the environment
        from the node's state and creating a new child node.
        """
        if self.done:
            raise Exception('Cannot expand a terminal node')

        self.state.set_environment_to_state(env)
        object_action = self.state.get_actions()[action_idx]
        _, reward, done, *_ = env.step(object_action)
        next_state: ObjectState = self.state.extract_state(env)

        child_node = SearchTreeNode(
            new_node_id, self, next_state, object_action,
            reward, done, self.q_function
        )
        self.children[object_action].append(child_node)
        return child_node

    def get_q_value(self, action: int) -> float:
        return self.q_function(self.state, action)

    def get_id(self) -> int:
        return self.node_id

    def get_parent(self) -> 'SearchTreeNode':
        return self.parent

    def is_root(self) -> bool:
        return self.parent is None

    def get_parent_id(self) -> int:
        if not self.is_root():
            return self.parent.node_id
        return -1

    def get_children(self) -> Dict[int, List['SearchTreeNode']]:
        """
            Returns a dictionary of the children of the node.
            The keys are the actions that led to the children, and the values are the children.
            As the environment may be stochastic, there may be multiple children for each action.
        """
        return self.children

    def can_expand(self) -> bool:
        return not self.done

    def get_trajectory(self) -> Tuple[int, float, ObjectState]:
        return (self.action, self.reward, self.state)

    def get_state(self) -> ObjectState:
        return self.state

    def get_action(self) -> int:
        return self.action or -1

    def get_preceding_action(self) -> int:
        return self.action

    def get_reward_received(self) -> float:
        return self.reward

    def __repr__(self, depth=0) -> str:
        children_str = '\n'.join([
            c.__repr__(depth + 1) 
            for a in self.children for c in self.children[a]
        ])

        node_str = f'[{self.state}, {self.can_expand()}]'
        maybe_newline = '\n' if children_str else ''

        if depth == 0:
            return f'{node_str}{maybe_newline}{children_str}'

        transition_str = f'|---[{self.action}, {self.reward}]-->'

        return '\t' * (depth - 1) + f'{transition_str} {node_str}{maybe_newline}{children_str}'


class SearchTree:
    """
    A class to represent the search tree.
    Serves as a wrapper for the root node of the search tree that provides O(1)
    access to a list of all nodes. Thereby, the search tree can be used as a data structure
    for the meta-environment and computational actions  can be performed on the search
    tree without having to traverse the tree by indexing nodes in the list with a given action.
    """

    def __init__(self,
                 env: gym.Env,
                 root_state: ObjectState,
                 q_function: QFunction,
                 deterministic: bool = True):
        self.env = env
        self.deterministic = deterministic
        self.q_function = q_function
        self.root_node: SearchTreeNode = SearchTreeNode(
            0, None, root_state, None, 0, False, q_function
        )
        self.node_list: List[SearchTreeNode] = [self.root_node]

    def is_action_valid(self, node: SearchTreeNode, action: int) -> bool:
        """
        Checks whether the given action permits a valid for the given node.
        This is only relevant for deterministic environments as we do not want to try the
        same action multiple times.
        """
        return self.deterministic and action not in node.children

    def expand(self, node_idx: int, action: int):
        """ Expands the node with the given index by taking the given action. """
        if node_idx >= len(self.node_list):
            raise Exception(
                f"Node index out of bounds: {node_idx=}, {action=}, {len(self.node_list)=}"
            )

        node = self.node_list[node_idx]
        if node.can_expand():
            child_node = node.expand_node(self.env, action, len(self.node_list))
            self.node_list.append(child_node)
            return True
        return False

    def get_nodes(self) -> List[SearchTreeNode]:
        return self.node_list

    def get_num_nodes(self) -> int:
        return len(self.node_list)

    def get_root(self) -> SearchTreeNode:
        return self.root_node

    def __repr__(self) -> str:
        return str(self.root_node)
