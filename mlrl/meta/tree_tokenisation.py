from functools import cached_property
from typing import List
from mlrl.meta.search_tree import SearchTree, SearchTreeNode
from mlrl.utils import one_hot, compute_positional_encoding

from abc import ABC, abstractmethod

import numpy as np


class TreeTokeniser(ABC):
    """Abstract base class for tree tokenisers."""

    def __init__(self,
                 max_tokens: int,
                 one_hot_node_id: bool = False,
                 pos_enc_dim: int = 16,
                 n_terminate_tokens: int = 1):
        self.max_tokens = max_tokens  # (non-terminate) tokens
        self.one_hot_node_id = one_hot_node_id
        self.pos_enc_dim = pos_enc_dim
        self.n_tokens = self.max_tokens + n_terminate_tokens

    @abstractmethod
    def tokenise(self, tree: SearchTree) -> np.ndarray:
        """Tokenise a tree into a list of tokens."""
        pass

    @abstractmethod
    def get_token_labels(self) -> List[str]:
        """Get the labels for the tokens."""
        pass

    @cached_property
    def tree_token_dim(self) -> int:
        """Get the dimension of the tree token."""
        return len(self.get_token_labels())

    def get_terminate_token(self) -> np.ndarray:
        """Get the token for terminate action."""
        return np.array([1., 1.] + [0.] * (self.tree_token_dim - 3) + [1.])

    def pad(self, tokens: np.ndarray) -> np.ndarray:
        """Pad a list of tokens to the maximum tree size."""
        if tokens.size > 0:
            if len(tokens.shape) < 2:
                raise ValueError(f'Tokens must be a 2D array. {tokens.shape=}')

            padding = np.zeros((self.n_tokens - tokens.shape[0], tokens.shape[1]))
            return np.concatenate([tokens, padding], axis=0)
        else:
            return np.zeros((self.n_tokens, self.tree_token_dim))

    def encode_node_id(self, node_id: int) -> np.ndarray:
        """Encode the node id as a vector."""
        if self.one_hot_node_id:
            return one_hot(node_id, self.max_tokens)
        else:
            return compute_positional_encoding(node_id, self.pos_enc_dim)

    def get_id_dim(self) -> int:
        """Get the dimension of the id vector."""
        if self.one_hot_node_id:
            return self.max_tokens
        else:
            return self.pos_enc_dim

    @abstractmethod
    def can_tokenise(self, tree: SearchTree) -> bool:
        """Check if a tree can be tokenised."""
        pass


class NodeTokeniser(TreeTokeniser):
    """Tokenise a tree into a list of vectors, with a token corresponding to each node."""

    def __init__(self,
                 max_tokens: int,
                 action_vec_dim: int,
                 state_vec_dim: int,
                 include_unexpandable_leaves: bool = False,
                 **kwargs):
        super().__init__(max_tokens, **kwargs)
        self.action_vec_dim = action_vec_dim
        self.state_vec_dim = state_vec_dim
        self.include_unexpandable_leaves = include_unexpandable_leaves

    def node_tokenisation(self, tree: SearchTree, node: SearchTreeNode) -> np.ndarray:
        """
        Generates a token for the given node.
        This will be used when computational actions correspond to expanding a node by
        trying all possible actions, thus it does not include information about
        an action to expand the node with.

        Token contains the following information:
            - Attention mask. Whether the token contains valid information or is padding.
            - Can expand. Whether the node can be expanded. Used to mask out invalid actions.
            - Antecedent action vector. Vector encoding the action that was taken to get to this node.
            - Reward. The reward given for the antecendent action.
            - State vector. The state vector of the node.
            - ID: The id of the node.
            - Parent ID: The id of the parent node.
            - Terminate Action: The action that terminates the search.
                Always zero for tokens generated by this function.

        Args:
            node: The node to tokenise
            action_idx: The index of the action to expand the node with. If this is greater than the
                number of actions the node can be expanded with, the token will be a padding token.

        Returns:
            A 1-dimensional numpy array of length token_dim.
        """
        state = node.get_state()

        id_vec = self.encode_node_id(node.get_id())
        if node.is_root():
            parent_id_vec = np.zeros((self.max_tokens,), dtype=np.float32)
            action_taken_vec = np.zeros((self.action_vec_dim,), dtype=np.float32)
        else:
            parent_id_vec = self.encode_node_id(node.get_parent_id())
            action_taken_vec = state.get_action_vector(node.get_action())

        # meta features contains a mask attention and the reward
        meta_features = np.array([
            1., tree.has_valid_expansions(node), node.reward
        ], dtype=np.float32)

        state_vec = state.get_state_vector()

        return np.concatenate([
            meta_features, id_vec, parent_id_vec,
            action_taken_vec, state_vec,
            np.array([0.])  # not a terminate token
        ])

    def tokenise(self, tree: SearchTree) -> np.ndarray:
        terminate_token = self.get_terminate_token()
        tokens = np.array([terminate_token] + [
            self.node_tokenisation(tree, node)
            for node in tree.node_list
            if self.include_unexpandable_leaves or not node.is_unexpandable_leaf()
        ], dtype=np.float32)
        return self.pad(tokens)

    def get_token_labels(self) -> List[str]:
        meta_features = [
            'obs_mask', 'can_expand', 'reward'
        ]

        id_dim = self.get_id_dim()
        id_vec = [f'id_{i}' for i in range(id_dim)]
        parent_id_vec = [f'parent_id_{i}' for i in range(id_dim)]
        action_taken_vec = [f'action_taken_{i}' for i in range(self.action_vec_dim)]
        state_vec = [f'state_{i}' for i in range(self.state_vec_dim)]
        return meta_features + id_vec + parent_id_vec + \
            action_taken_vec + state_vec + [r'$\perp$']

    def can_tokenise(self, tree: SearchTree) -> bool:
        """ Check if a tree can be tokenised."""
        n = sum([
            1 for node in tree.node_list
            if self.include_unexpandable_leaves or not node.is_unexpandable_leaf()
        ])
        return n <= self.max_tokens

    def get_node_idx(self, tree: SearchTree, computational_action: int) -> int:
        """Get the index of the node that the given computational action expands."""
        if self.include_unexpandable_leaves:
            return computational_action - 1

        # skip unexpandable leaves
        node_idx = 0
        for node in tree.node_list:
            if not node.is_unexpandable_leaf():
                if computational_action - 1 == node_idx:
                    return node.get_id()
                node_idx += 1

        raise ValueError(f'Computational action {computational_action} is not valid.')


class NodeActionTokeniser(NodeTokeniser):
    """
    Tokenise a tree into a list of vectors, with a
    token corresponding to each node-action pair.
    """

    def tokenise(self, tree: SearchTree) -> np.ndarray:
        tree_tokens = []
        # appends each action vector to each node token
        for node in tree.node_list:
            node_token = self.node_tokenisation(tree, node)
            for action in node.state.get_actions():
                action_vec = node.state.get_action_vector(action)
                token = np.concatenate([node_token, action_vec])
                tree_tokens.append(token)

        terminate_token = self.get_terminate_token()
        tokens = np.array([terminate_token] + tree_tokens, dtype=np.float32)
        return self.pad(tokens)

    def get_token_labels(self) -> List[str]:
        labels = super().get_token_labels()
        action_vec = [f'action_{i}' for i in range(self.action_vec_dim)]
        return labels + action_vec

    def can_tokenise(self, tree: SearchTree) -> bool:
        """ Check if a tree can be tokenised."""
        n = sum([
            node.state.get_maximum_number_of_actions()
            for node in tree.node_list
            if self.include_unexpandable_leaves or not node.is_unexpandable_leaf()
        ])
        return n <= self.max_tokens

    def get_node_idx(self, tree: SearchTree, computational_action: int) -> int:
        """Get the index of the node that the given computational action expands."""
        raise NotImplementedError('This tokeniser does not support this method yet.')
