from .q_estimation import QFunction, SimpleSearchBasedQEstimator
from .search_tree import SearchTree, SearchTreeNode

from typing import Callable, Tuple, Dict

import numpy as np
import gym


class MetaEnv(gym.Env):
    """
    Class that wraps a gym_maze environment and allows for the meta-learning of the search problem
    """

    def __init__(self,
                 env: gym.Env,
                 q_hat: QFunction,
                 make_tree: Callable[[gym.Env], SearchTree],
                 cost_of_computation: float = 0.001,
                 computational_rewards: bool = True,
                 object_state_vec_size: int = 2,
                 max_tree_size: int = 10,
                 object_env_discount: float = 0.99):
        """
        Args:
            env: The object-level environment to wrap
            q_hat: The Q-hat function to use for the search tree at the leaf nodes
            make_tree: A function that takes an environment and returns a search
                tree with the root node set to the current state
            object_state_vec_size: The size of the vector representation of the object state
            max_tree_size: The maximum number of nodes in the search tree
        """
        self.object_env = env
        self.q_hat = q_hat
        self.max_tree_size = max_tree_size
        self.object_env_discount = object_env_discount
        self.cost_of_computation = cost_of_computation
        self.computational_rewards = computational_rewards

        self.n_object_actions = self.object_env.action_space.n
        self.action_space = gym.spaces.Discrete(1 + max_tree_size * self.n_object_actions)

        self.n_meta_data = 3  # number of meta data features: parent node idx, action, reward
        tree_token_space = gym.spaces.Box(
            low=0, high=max_tree_size,
            shape=(max_tree_size, self.n_meta_data + object_state_vec_size),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            'search_tree_tokens': tree_token_space,
            'valid_action_mask': gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32)
        })

        self.make_tree = make_tree
        self.tree = None
        self.n_computations = 0

    def reset(self):
        self.object_env.reset()
        self.tree = self.get_root_tree()
        self.n_computations = 0
        return self.get_observation()

    def get_root_tree(self) -> SearchTree:
        """ Creates a new tree with the root node set to the current state of the environment """
        return self.make_tree(self.object_env)

    def tokenise_node(self, node: SearchTreeNode) -> np.array:
        meta_features = np.array([node.get_parent_id(), node.get_action(), node.reward],
                                 dtype=np.float32)
        return np.concatenate([meta_features, node.state.get_state_vector()])

    def get_observation(self) -> Dict[str, np.array]:
        """
        Returns the observation of the meta environment.

        This is a dictionary with the following keys:
        - search_tree_tokens: A tensor of shape (max_tree_size, n_meta_features + object_state_vec_size)
        - valid_action_mask: A tensor of shape (1 + max_tree_size,)

        The search tree tokens are a set of vectors with each token corresponding to a node in the tree,
        and with any remaining rows zeroed out. Each token contains meta-features and object-features.
        The meta-features are: parent node idx, action, reward. The parent node idx is the index of the
        parent node in the observation tensor. The action is the action taken to reach the node.
        The reward is the reward received from the underlying environment when the node was reached.
        The object-features are the vector representation of the object state.

        The valid action mask is a binary vector with a 1 in each position corresponding to a valid
        computational action.
        """
        obs = np.array([
            self.tokenise_node(node)
            for node in self.tree.node_list
        ])
        padding = np.zeros((self.max_tree_size - obs.shape[0], obs.shape[1]))
        search_tokens = np.concatenate([obs, padding], axis=0)

        action_mask = np.zeros((self.action_space.n,), dtype=np.int32)
        action_mask[0] = 1
        for node in self.tree.node_list:
            for action in range(self.n_object_actions):
                if self.tree.is_action_valid(node, action):
                    action_mask[node.get_id() * self.n_object_actions + action + 1] = 1

        return {
            'search_tree_tokens': search_tokens,
            'valid_action_mask': action_mask
        }

    def get_best_object_action(self) -> int:
        q_est = SimpleSearchBasedQEstimator(self.q_hat, self.tree,
                                            self.n_object_actions, self.object_env_discount)
        return max(range(self.object_env.action_space.n),
                   key=lambda a: q_est.compute_q(self.tree.get_root(), a))

    def root_q_distribution(self) -> np.array:
        q_est = SimpleSearchBasedQEstimator(self.q_hat, self.tree,
                                            self.n_object_actions, self.object_env_discount)
        return np.array([q_est.compute_q(self.tree.get_root(), a) for a in range(self.n_object_actions)])

    def get_computational_reward(self, prior_action: int) -> float:
        """
        Difference between the value of best action before and after the computation,
        both considered under the Q-distribution derived from the tree after the computation.
        """
        q_dist = self.root_q_distribution()
        return q_dist.max() - q_dist[prior_action]

    def step(self, computational_action: int) -> Tuple[np.array, float, bool, dict]:
        """
        Performs a step in the meta environment. The action is interpreted as follows:
        - action == 0: terminate search and perform a step in the underlying environment
        - action > 0: expand the search tree by one node with the given action

        For example, if the underlying environment has 4 actions and the tree is currently
        of size 2, then the valid action space of the meta environment contains 1 + 2 * 4 = 9
        computational actions. Taking computational action 6 expands the second node in the tree
        with object-level action 1 (actions are 0-indexed).

        Args:
            computational_action: The action to perform in the meta environment

        Returns:
            observation: The observation of the meta environment. This is a tensor of shape
                (max_tree_size, n_meta_features + object_state_vec_size).
            reward: The reward received from the underlying environment
            done: Whether the underlying environment is done
            info: Additional information
        """
        if computational_action == 0 or self.tree.get_num_nodes() >= self.max_tree_size:
            # Perform a step in the underlying environment
            root_state = self.tree.get_root().get_state()
            root_state.set_environment_to_state(self.object_env)
            _, reward, done, info = self.object_env.step(self.get_best_object_action())
            self.tree = self.get_root_tree()
            return self.get_observation(), reward, done, info

        if self.computational_rewards:
            # Keep track of prior action for later comparison
            prior_action = self.get_best_object_action()

        # perform a computational action on the search tree
        node_idx = (computational_action - 1) // self.n_object_actions
        object_action = (computational_action - 1) % self.n_object_actions
        
        self.tree.expand(node_idx, object_action)
        self.n_computations += 1

        reward = -self.cost_of_computation
        if self.computational_rewards:
            reward += self.get_computational_reward(prior_action)

        return self.get_observation(), reward, False, dict()

    def get_action_strings(self) -> Dict[int, str]:
        n = self.n_object_actions
        return {
            0: 'Terminate Search',
            **{
                i: f'Expand Node {(i - 1) // n} with Action ' + self.object_env.ACTION[(i - 1) % n]
                for i in range(1, self.action_space.n)
            }
        }
