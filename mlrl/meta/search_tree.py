from abc import ABC, abstractmethod
import copy
from functools import cached_property
from typing import Generic, List, Optional, Tuple, Dict, TypeVar, Union

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
    def get_state_vector(self) -> np.ndarray:
        """ Returns a vector representation of the state. """
        pass

    @abstractmethod
    def get_actions(self) -> list:
        """ Returns a list of actions that can be taken from the current state. """
        pass

    @abstractmethod
    def get_action_vector(self, action) -> np.ndarray:
        """ Returns a vector representation of the action. """
        pass

    def get_action_labels(self) -> List[str]:
        """ Returns a list of labels for the actions. """
        return [str(a) for a in self.get_actions()]

    def get_action_label(self, action) -> str:
        """ Returns a label for the action. """
        if action not in self.get_actions():
            return ''

        return {
            a: l for a, l in zip(self.get_actions(), self.get_action_labels())
        }[action]

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

    def __eq__(self, other: 'ObjectState') -> bool:
        return np.array_equal(self.get_state_vector(), other.get_state_vector())

    def __hash__(self) -> int:
        return hash(self.get_state_string())


def find_cycle(
        state: ObjectState,
        trajectory: List[Tuple[ObjectState, int, float]]
) -> List[Tuple[ObjectState, int, float]]:
    state_idxs = [
        i for i, (s, *_) in enumerate(trajectory)
        if s == state
    ]
    if state_idxs:
        i = state_idxs[0]
        return trajectory[i:]
    return []


class QFunction(ABC):
    """
    Abstract class for a Q-function
    """

    @abstractmethod
    def compute_q(self, state: ObjectState, action: int) -> float:
        pass

    def __call__(self, state: ObjectState, action: int) -> float:
        return self.compute_q(state, action)


StateType = TypeVar('StateType', bound=ObjectState)


class SearchTreeNode(Generic[StateType]):
    """
    A class to represent a node in the search tree.
    The node contains the state of the environment, the action that led to this state,
    the reward received, and the children of the node, and other important data.
    """

    # __slots__ = [
    #     'parent', 'node_id', 'state', 'action', 'reward',
    #     'children', 'is_terminal_state', 'q_function', 'tried_actions',
    #     '_duplicate_state_ancestor', 'q_values', 'depth', 'path', 'path_actions',
    #     'path_return'
    # ]

    def __init__(self,
                 node_id: int,
                 parent: Optional['SearchTreeNode[StateType]'],
                 state: StateType,
                 action: Optional[int],
                 reward: float,
                 done: bool,
                 discount: float,
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
            q_function: The Q-function used to compute the action Q-values from the node state.
        """
        self.node_id = node_id
        self.parent = parent
        self.state: StateType = state
        self.action = action
        self.reward = reward
        self.discount = discount
        self.is_terminal_state = done
        self.children: Dict[int, List['SearchTreeNode']] = dict()
        self.q_function = q_function
        self.q_values: Dict[int, float] = dict()
        self.tried_actions = []
        self._duplicate_state_ancestor = None

    @cached_property
    def depth(self) -> int:
        """
        Returns the depth of the node in the search tree.
        The root node has depth 0.
        """
        if self.parent is None:
            return 0

        return self.parent.depth + 1

    @cached_property
    def path(self) -> List['SearchTreeNode[StateType]']:
        """
        Returns the path from the root to the current node.
        The root node is the first element of the list.
        """
        if self.parent is None:
            return [self]

        return self.parent.path + [self]

    @cached_property
    def path_actions(self) -> List[int]:
        """
        Returns the actions taken from the root to the current node.
        The root node has action -1.
        """
        if self.depth == 0:
            return []
        return [node.action for node in self.path[1:]]

    @cached_property
    def path_return(self) -> float:
        """
        Returns the return of the path from the root to the current node.
        """
        if self.depth == 0:
            return 0
        return self.parent.path_return + self.reward * self.discount ** (self.depth - 1)

    @property
    def duplicate_state_ancestor(self) -> Optional['SearchTreeNode[StateType]']:
        return self.find_duplicate_state_ancestor()

    def find_duplicate_state_ancestor(self) -> Optional['SearchTreeNode[StateType]']:
        """
        Returns the first ancestor of the given child node that has the same state as the given parent node.
        If no such ancestor exists, returns None.
        """
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
            if current_node.state == self.state:
                return current_node

        return None

    def get_q_value(self, action: int) -> float:
        """
        Returns the Q-value of the given action.
        If the Q-value has not been computed yet, it is computed using Q-hat and stored.
        """
        if action not in self.q_values:
            self.q_values[action] = self.q_function(self.state, action)

        return self.q_values[action]

    def get_value(self) -> float:
        """
        Returns the value of the node, i.e. the maximum Q-value of the node.
        """
        return max(self.get_q_value(a) for a in self.state.get_actions())

    def get_exp_root_return(self) -> float:
        """
        Returns the expected discounted return from the root node passing through the current node.
        """
        return self.path_return + self.get_value() * self.discount ** self.depth

    def set_q_value(self, action: int, q_value: float):
        """
        Sets the Q-value of the given action.
        """
        self.q_values[action] = q_value

    def expand_node(self,
                    env: gym.Env,
                    action_idx: int,
                    new_node_id: int) -> 'SearchTreeNode':
        """
        Expands the node by "simulating" taking the given action in the environment
        from the node's state and creating a new child node.
        """
        if self.is_terminal_state:
            raise Exception('Cannot expand a terminal node')

        self.state.set_environment_to_state(env)
        object_action = self.state.get_actions()[action_idx]
        _, reward, done, *_ = env.step(object_action)
        next_state: StateType = self.state.extract_state(env)

        child_node = SearchTreeNode(
            new_node_id, self, next_state, object_action,
            reward, done, self.discount, self.q_function
        )
        self.tried_actions.append(action_idx)
        return child_node

    def get_path_to_root(self) -> List['SearchTreeNode[StateType]']:
        """
        Returns a list of nodes from the root to the current node.
        The list is ordered from the current node to the root.
        The list does not include the current node.
        """
        path = []
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
            path.append(current_node)
        return path

    def add_child_node(self, node: 'SearchTreeNode[StateType]'):
        if node.action not in self.children:
            self.children[node.action] = [node]
        else:
            self.children[node.action].append(node)

    def has_action_children(self, action: int) -> bool:
        return action in self.children and len(self.children[action]) > 0

    def get_id(self) -> int:
        return self.node_id

    def get_parent(self) -> Optional['SearchTreeNode[StateType]']:
        return self.parent

    def is_root(self) -> bool:
        return self.parent is None

    def get_parent_id(self) -> int:
        if self.parent is not None:
            return self.parent.node_id
        return -1

    def get_children(
        self, action: Optional[int] = None
    ) -> Union[Dict[int, List['SearchTreeNode[StateType]']],
               List['SearchTreeNode[StateType]']]:
        """
            Returns a dictionary of the children of the node for each action
            if no action is given, or a list of the children of the node for the given action.
            The keys are the actions that led to the children, and the
            values are the children. To account for the environment
            potentially being stochastic, there may be multiple
            children for each action.
        """
        if action is not None:
            return self.children[action] if action in self.children else []
        return self.children

    def can_expand(self) -> bool:
        if self.duplicate_state_ancestor is not None:
            return False

        all_tried = all(
            action_idx in self.tried_actions
            for action_idx in range(len(self.state.get_actions()))
        )

        return (not self.is_terminal_state) and (not all_tried)

    def get_trajectory(self) -> Tuple[Optional[int], float, StateType]:
        return (self.action, self.reward, self.state)

    def get_state(self) -> StateType:
        return self.state

    def get_action(self) -> int:
        return self.action if self.action is not None else -1

    def get_reward_received(self) -> float:
        return self.reward

    def __repr__(self, depth=0) -> str:
        children_str = '\n'.join([
            c.__repr__(depth + 1)
            for a in self.children for c in self.children[a]
        ])

        node_str = f'({self.state})' if self.can_expand() else '{' + str(self.state) + '}'
        maybe_newline = '\n' if children_str else ''

        if depth == 0:
            return f'{node_str}{maybe_newline}{children_str}'

        if self.parent is not None:
            action_label = self.parent.state.get_action_label(self.action)
            if self.action in self.parent.q_values:
                q = self.parent.q_values[self.action]
                transition_str = f'|---[{action_label}, {self.reward:.3f}, {q:.3f}]-->'
            else:
                transition_str = f'|---[{action_label}, {self.reward:.3f}]-->'
            return '\t' * (depth - 1) + f'{transition_str} {node_str}{maybe_newline}{children_str}'

        raise AttributeError('Expected node to have a parent.')

    def __hash__(self):
        return hash((self.node_id, self.state, self.reward, self.action))


class SearchTree(Generic[StateType]):
    """
    A class to represent the search tree.
    Serves as a wrapper for the root node of the search tree that provides O(1)
    access to a list of all nodes. Thereby, the search tree can be used as a data structure
    for the meta-environment and computational actions  can be performed on the search
    tree without having to traverse the tree by indexing nodes in the list with a given action.
    """

    def __init__(self,
                 env: gym.Env,
                 root: Union[StateType, SearchTreeNode],
                 q_function: QFunction,
                 max_size: int = 10,
                 discount: float = 0.99,
                 deterministic: bool = True):
        self.env = env
        self.deterministic = deterministic
        self.q_function = q_function
        self.max_size = max_size
        self.discount = discount
        self.root_node: SearchTreeNode[StateType] = root if isinstance(root, SearchTreeNode) else SearchTreeNode(
            0, None, root, None, 0, False, self.discount, q_function
        )
        self.root_node.node_id = 0
        self.node_list: List[SearchTreeNode[StateType]] = []
        self.state_nodes = {}
        self.add_node(self.root_node)

    def copy(self) -> 'SearchTree':
        """
        Makes a copy of the tree structure.
        Preserves the node ids and the parent-child relationships.
        The original and the copy share the same Q-function and the same environment objects.
        """
        root = self.get_root()

        def recursive_copy(node, parent):
            new_node = SearchTreeNode(
                node.node_id, parent, node.state, node.action, node.reward,
                node.is_terminal_state, self.discount, self.q_function
            )
            new_node.tried_actions = copy.deepcopy(node.tried_actions)
            for a in node.children:
                for child in node.children[a]:
                    new_node.add_child_node(recursive_copy(child, new_node))
            return new_node

        new_root = recursive_copy(root, None)
        new_tree = SearchTree(
            self.env, new_root, self.q_function, self.max_size,
            self.discount, self.deterministic)

        def recursive_update_node_list(node):
            if node not in new_tree.node_list:
                new_tree.add_node(node)
            for a in node.children:
                for child in node.children[a]:
                    recursive_update_node_list(child)

        recursive_update_node_list(new_root)
        new_tree.node_list.sort(key=lambda n: n.node_id)

        return new_tree

    def get_subtree(self, node_id: int, action: int) -> 'SearchTree':
        """
        Creates the subtree rooted with the child node corresponding to the given node and action.
        """
        root_node = self.node_list[node_id].children[action][0]
        root_node.parent = None

        sub_tree = SearchTree(self.env,
                              root_node,
                              self.q_function,
                              self.max_size,
                              self.discount,
                              self.deterministic)

        def add_children(node: SearchTreeNode):
            for children in node.get_children().values():
                for child in children:
                    # add to node list and recurse
                    sub_tree.add_node(child)
                    add_children(child)

        add_children(root_node)

        return sub_tree

    def get_root_subtree(self, action: int) -> 'SearchTree':
        """
        Creates the subtree rooted with the child node corresponding to the given action
        taken from the root of the current tree.
        """
        return self.get_subtree(0, action)

    def get_state_nodes(self, state: StateType) -> List[SearchTreeNode[StateType]]:
        """
        Returns all nodes in the tree that correspond to the given state.
        """
        return self.state_nodes.get(state, [])

    def is_action_valid(self, node: SearchTreeNode, action_idx: int) -> bool:
        """
        Checks whether the given action permits a valid expansion for the given node.
        """
        action = node.state.get_actions()[action_idx]
        return action not in node.children and node.can_expand()

    def has_valid_expansions(self, node: SearchTreeNode) -> bool:
        """
        Checks whether the given node has any valid expansions.
        """
        return node.can_expand() and any(
            self.is_action_valid(node, action_idx)
            for action_idx in range(node.state.get_maximum_number_of_actions())
        )

    def expand_all(self, node_idx: int):
        """ Expands all valid actions for the given node. """
        node_idx = int(node_idx)

        if node_idx >= len(self.node_list):
            raise Exception(
                f"Node index out of bounds: {node_idx=}, {len(self.node_list)=}"
            )

        node = self.node_list[node_idx]
        for action_idx in range(len(node.state.get_actions())):
            if len(self.node_list) >= self.max_size:
                break
            if self.is_action_valid(node, action_idx):
                self.expand_action(node_idx, action_idx)

    def expand_action(self, node_idx: int, action_idx: int):
        """
        Expands the node with the given index by taking the given action.

        Args:
            node_idx: The index in the tree node list of the node to expand.
            action_idx: The index in the node's action list of the action to take.
        """

        node_idx = int(node_idx)
        action_idx = int(action_idx)

        if node_idx >= len(self.node_list):
            raise Exception(
                f"Node index out of bounds: {node_idx=}, {action_idx=}, {len(self.node_list)=}"
            )

        node = self.node_list[node_idx]
        if node.can_expand():
            child_node = node.expand_node(self.env, action_idx, len(self.node_list))
            node.add_child_node(child_node)
            self.add_node(child_node)

    def add_node(self, node: SearchTreeNode):
        """ Adds a node to the tree. """
        node.node_id = len(self.node_list)
        self.node_list.append(node)
        if node.duplicate_state_ancestor is None:
            if node.state not in self.state_nodes:
                self.state_nodes[node.state] = [node]
            else:
                self.state_nodes[node.state].append(node)

    def set_max_size(self, max_size: int):
        self.max_size = max_size

    def get_nodes(self) -> List[SearchTreeNode]:
        return self.node_list

    def get_num_nodes(self) -> int:
        return len(self.node_list)

    def get_root(self) -> SearchTreeNode:
        return self.root_node

    def __repr__(self) -> str:
        return str(self.root_node)

    def __hash__(self) -> int:
        return hash(tuple(self.node_list))
