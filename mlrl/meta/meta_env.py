from mlrl.meta.q_estimation import SearchQEstimator, RecursiveDeterministicEstimator
from mlrl.meta.search_tree import SearchTree, SearchTreeNode
from mlrl.utils import one_hot
from mlrl.utils.plot_search_tree import plot_tree
from mlrl.utils.render_utils import plot_to_array

from typing import Callable, Optional, Tuple, Dict, Any, List, Union

import numpy as np
import gym

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set()


def mask_token_splitter(tokens_and_mask):
    tokens = tokens_and_mask[MetaEnv.SEARCH_TOKENS_KEY]
    mask = tokens_and_mask[MetaEnv.ACTION_MASK_KEY]
    return tokens, mask


class MetaEnv(gym.Env):
    """
    Class that wraps a gym_maze environment and allows for the
    meta-learning of the search problem
    """

    SEARCH_TOKENS_KEY = 'search_tree_tokens'
    ACTION_MASK_KEY = 'action_mask'

    def __init__(self,
                 object_env: gym.Env,
                 initial_tree: SearchTree,
                 cost_of_computation: float = 0.001,
                 computational_rewards: bool = True,
                 max_tree_size: int = 10,
                 expand_all_actions: bool = False,
                 search_q_estimator: Optional[SearchQEstimator] = None,
                 object_action_to_string: Callable[[Any], str] = None,
                 object_reward_min: float = 0.0,
                 object_reward_max: float = 1.0,
                 object_env_discount: float = 0.99,
                 one_hot_action_space: bool = False,
                 split_mask_and_tokens: bool = True,
                 dump_debug_images: bool = True):
        """
        Args:
            env: The object-level environment to wrap
            initial_tree: The initial search tree to use, this should be a tree with a root node.
            cost_of_computation: The cost of computing a node in the search tree
            computational_rewards: Whether to give computational reward
            max_tree_size: The maximum number of nodes in the search tree
            expand_all_actions: Whether to have each computational action correspond to
                expanding all actions at a given node, or just one specified action.
                The latter increases the number of actions in the meta-action space and
                the size of the observations by a factor of the number of actions in the
                object-level environment.
        """
        self.object_env = object_env
        self.max_tree_size = max_tree_size
        self.expand_all_actions = expand_all_actions
        self.object_env_discount = object_env_discount
        self.cost_of_computation = cost_of_computation
        self.computational_rewards = computational_rewards
        self.split_mask_and_tokens = split_mask_and_tokens
        self.dump_debug_images = dump_debug_images

        self.tree = initial_tree
        self.tree.set_max_size(max_tree_size)
        self.n_computations = 0

        self.search_q_estimator = search_q_estimator or RecursiveDeterministicEstimator(object_env_discount)

        # Setup gym spaces

        object_state = initial_tree.get_root().get_state()
        self.n_object_actions = object_state.get_maximum_number_of_actions()

        if self.expand_all_actions:
            self.action_space_size = 1 + max_tree_size
        else:
            self.action_space_size = 1 + max_tree_size * self.n_object_actions

        self.one_hot_action_space = one_hot_action_space
        if one_hot_action_space:
            self.action_space = gym.spaces.Box(
                low=0, high=1, shape=(self.action_space_size,),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Discrete(self.action_space_size)

        # meta data features: attn mask, can expand, reward, is terminate
        self.n_meta_feats = 5
        self.state_vec_dim = object_state.get_state_vector_dim()
        self.action_vec_dim = object_state.get_action_vector_dim()

        if self.expand_all_actions:
            # If we are expanding all actions, then the token does not include a
            # representation of the action to be expanded
            self.tree_token_dim = self.n_meta_feats + \
                2 * self.max_tree_size + self.state_vec_dim + self.action_vec_dim
        else:
            # 2 * max_tree_size because we include the id of the corresponding node
            # and the id of the parent node
            # 2 * self.action_vec_dim because we include the vector for the action
            # taken to reach the node and the vector for the action to be expanded
            self.tree_token_dim = self.n_meta_feats + \
                2 * self.max_tree_size + self.state_vec_dim + 2 * self.action_vec_dim

        tree_token_space = gym.spaces.Box(
            low=object_reward_min, high=object_reward_max,
            shape=(self.action_space_size, self.tree_token_dim),
            dtype=np.float32
        )

        self.object_reward_min = object_reward_min
        self.object_reward_max = object_reward_max

        if split_mask_and_tokens:
            self.observation_space = gym.spaces.Dict({
                'search_tree_tokens': tree_token_space,
                'action_mask': gym.spaces.Box(
                    low=0, high=1, shape=(self.action_space_size,), dtype=np.int32)
            })
        else:
            self.observation_space = tree_token_space

        # Variables for rendering
        self.last_meta_action = None
        self.last_meta_reward = 0
        self.last_computational_reward = 0
        self.object_action_to_string = object_action_to_string or (lambda a: str(a))
        self.meta_action_strings = self.get_action_strings()
        self.steps = 0

        self.reset()

    def reset(self):
        self.object_env.reset()
        self.tree = self.get_root_tree()
        self.n_computations = 0
        self.last_meta_action = None
        self.last_meta_reward = 0
        self.last_computational_reward = 0
        self.steps = 0
        return self.get_observation()

    def get_root_tree(self) -> SearchTree:
        """
        Creates a new tree with the root node set to the
        current state of the environment
        """
        old_root_node = self.tree.get_root()
        old_state = old_root_node.get_state()
        new_root_state = old_state.extract_state(self.object_env)
        return SearchTree(
            self.object_env, new_root_state, self.tree.q_function,
            deterministic=self.tree.deterministic,
            max_size=self.max_tree_size
        )

    def node_action_tokenisation(self, node: SearchTreeNode, action_idx: int) -> np.array:
        """
        Generates a token for the given node and action index.
        This will be used when computational actions correspond to explicitly expanding
        a node with a given action, rather than expanding a node by trying all possible actions.

        Token contains the following information:
            - Attention mask. Whether the token contains valid information or is padding.
            - Can expand. Whether the node can be expanded. Used to mask out invalid actions.
            - Antecedent action vector. Vector encoding the action that was taken to get to this node.
            - Reward. The reward given for the antecendent action.
            - Action vector. Action to expand the node with.
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
        actions = node.state.get_actions()

        if action_idx < len(actions):
            action = actions[action_idx]
        else:
            return np.zeros(self.tree_token_dim)

        id_vec = one_hot(node.get_id(), self.max_tree_size)
        if node.is_root():
            parent_id_vec = np.zeros((self.max_tree_size,), dtype=np.float32)
            action_taken_vec = np.zeros((self.action_vec_dim,), dtype=np.float32)
        else:
            parent_id_vec = one_hot(node.get_parent_id(), self.max_tree_size)
            action_taken_vec = state.get_action_vector(node.get_action())

        # meta features contains a mask attention and the reward
        meta_features = np.array([
            1., self.tree.is_action_valid(node, action), node.reward,
        ], dtype=np.float32)

        action_vec = state.get_action_vector(action)
        state_vec = state.get_state_vector()

        return np.concatenate([
            meta_features, id_vec, parent_id_vec,
            action_taken_vec, action_vec, state_vec,
            [0.]  # not a terminate token
        ])

    def node_tokenisation(self, node: SearchTreeNode) -> np.array:
        """
        Generates a token for the given node.
        This will be used when computational actions correspond to expanding a node by
        trying all possible actions, thus it does not include information about
        an action to expand the node with.

        Token contains the following information:
            - Attention mask. Whether the token contains valid information or is padding.
            - Can expand. Whether the node can be expanded. Used to mask out invalid actions.
            - Antecendet action vector. Vector encoding the action that was taken to get to this node.
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

        id_vec = one_hot(node.get_id(), self.max_tree_size)
        if node.is_root():
            parent_id_vec = np.zeros((self.max_tree_size,), dtype=np.float32)
            action_taken_vec = np.zeros((self.action_vec_dim,), dtype=np.float32)
        else:
            parent_id_vec = one_hot(node.get_parent_id(), self.max_tree_size)
            action_taken_vec = state.get_action_vector(node.get_action())

        # meta features contains a mask attention and the reward
        meta_features = np.array([
            1., self.tree.has_valid_expansions(node), node.reward
        ], dtype=np.float32)

        state_vec = state.get_state_vector()

        return np.concatenate([
            meta_features, id_vec, parent_id_vec,
            action_taken_vec, state_vec,
            [0.]  # not a terminate token
        ])

    def get_token_labels(self) -> List[str]:
        meta_features = [
            'obs_mask', 'can_expand', 'reward',
            'value-estimate' if self.expand_all_actions else 'q-estimate'
        ]
        id_vec = [f'id_{i}' for i in range(self.max_tree_size)]
        parent_id_vec = [f'parent_id_{i}' for i in range(self.max_tree_size)]
        action_taken_vec = [f'action_taken_{i}' for i in range(self.action_vec_dim)]
        action_vec = [f'action_{i}' for i in range(self.action_vec_dim)]
        state_vec = [f'state_{i}' for i in range(self.state_vec_dim)]
        if self.expand_all_actions:
            return meta_features + id_vec + parent_id_vec + \
                action_taken_vec + state_vec + [r'$\perp$']
        else:
            return meta_features + id_vec + parent_id_vec + \
                action_taken_vec + action_vec + state_vec + [r'$\perp$']

    def get_observation(self) -> Dict[str, np.array]:
        """
        Returns the observation of the meta environment.

        This is a dictionary with the following keys:
        - search_tree_tokens: A tensor of shape
            (max_tree_size, n_meta_features + object_state_vec_size)
        - valid_action_mask: A tensor of shape (1 + max_tree_size,)

        The search tree tokens are a set of vectors with each token corresponding
        to a node in the tree, and with any remaining rows zeroed out. Each token
        contains meta-features and object-features. The meta-features are: parent
        node idx, action, reward. The parent node idx is the index of the parent
        node in the observation tensor. The action is the action taken to reach the
        node. The reward is the reward received from the underlying environment
        when the node was reached.The object-features are the vector representation
        of the object state.

        The valid action mask is a binary vector with a 1 in each position
        corresponding to a valid computational action.
        """
        if len(self.tree.node_list) > self.max_tree_size:
            raise RuntimeError(
                f'Tree has {len(self.tree.node_list)} nodes, '
                f'but max_tree_size is {self.max_tree_size}'
            )

        terminate_token = [1., 1.] + [0.] * (self.tree_token_dim - 3) + [1.]
        if self.expand_all_actions:
            obs = np.array([terminate_token] + [
                self.node_tokenisation(node)
                for node in self.tree.node_list
            ])
        else:
            obs = np.array([terminate_token] + [
                self.node_action_tokenisation(node, action_idx)
                for node in self.tree.node_list
                for action_idx in range(self.n_object_actions)
            ])

        n_tokens = self.action_space_size
        if obs.size > 0:
            padding = np.zeros((n_tokens - obs.shape[0], obs.shape[1]))
            search_tokens = np.concatenate([obs, padding], axis=0)
        else:
            search_tokens = np.zeros((n_tokens, self.tree_token_dim))

        if self.split_mask_and_tokens:
            action_mask = search_tokens[:, 1].astype(np.int32)
            return {
                'search_tree_tokens': search_tokens,
                'action_mask': action_mask
            }
        else:
            return search_tokens

    def get_best_object_action(self) -> int:
        actions = self.tree.get_root().get_state().get_actions()
        action_idx, _ = max(
            enumerate(actions),
            key=lambda item: self.search_q_estimator.compute_root_q(self.tree, item[1]),
            default=(None, None)
        )
        return action_idx

    def root_q_distribution(self) -> np.array:
        root_state = self.tree.get_root().get_state()
        return np.array([
            self.search_q_estimator.compute_root_q(self.tree, a)
            for a in root_state.get_actions()
        ])

    def get_computational_reward(self, prior_action: int) -> float:
        """
        Difference between the value of best action before and after the computation,
        both considered under the Q-distribution derived from the tree after the computation.
        """
        q_dist = self.root_q_distribution()
        self.last_computational_reward = q_dist.max() - q_dist[prior_action]
        return self.last_computational_reward

    def step(self,
             computational_action: Union[int, list, np.array]
             ) -> Tuple[np.array, float, bool, dict]:
        """
        Performs a step in the meta environment. The action is interpreted as follows:
        - action == 0: terminate search and perform a step in the underlying environment
        - action > 0: expand a node of the search tree.

        The expansion operation has two modes:
        - expand_all_actions: expand all actions from the given node
        - expand_given_action: expand the given action from the given node

        For example of expanding a given action, if the underlying environment
        has 4 actions and the tree is currently of size 2, then the valid action
        space of the meta environment contains 1 + 2 * 4 = 9 computational
        actions. Taking computational action 6 expands the second node in the
        tree with object-level action 1 (actions are 0-indexed).

        In a where the meta environment is configured to expand all actions,
        the same action space would be 1 + 2 = 3 computational actions.

        Args:
            computational_action: The action to perform in the meta environment

        Returns:
            observation: The observation of the meta environment. This is a tensor
                of shape (max_tree_size, n_meta_features + object_state_vec_size).
            reward: The reward received from the underlying environment
            done: Whether the underlying environment is done
            info: Additional information
        """
        try:
            if self.one_hot_action_space and not isinstance(computational_action, int):
                computational_action = np.argmax(computational_action)

            self.last_meta_action = computational_action
            self.steps += 1

            if computational_action == 0 or self.tree.get_num_nodes() >= self.max_tree_size:
                # Perform a step in the underlying environment
                best_action_idx = self.get_best_object_action()
                root_state = self.tree.get_root().get_state()
                action = root_state.get_actions()[best_action_idx]

                self.set_environment_to_root_state()
                _, self.last_meta_reward, done, info = self.object_env.step(action)

                self.tree = self.get_root_tree()

                info.update({
                    'computational_reward': 0,
                    'object_level_reward': self.last_meta_reward
                })

                return self.get_observation(), self.last_meta_reward, done, info

            if self.computational_rewards:
                # Keep track of prior action for later comparison
                prior_action = self.get_best_object_action()

            if self.expand_all_actions:
                # Expand all actions from the given node
                node_idx = (computational_action - 1)
                self.tree.expand_all(node_idx)
            else:
                # perform a computational action on the search tree
                node_idx = (computational_action - 1) // self.n_object_actions
                object_action_idx = (computational_action - 1) % self.n_object_actions
                self.tree.expand_action(node_idx, object_action_idx)

            self.n_computations += 1

            # Compute reward (named "last_meta_reward" for readability in later access)
            self.last_meta_reward = -self.cost_of_computation
            if self.computational_rewards:
                self.last_meta_reward += self.get_computational_reward(prior_action)

            # Set the environment to the state of the root node for inter-step consistency
            self.set_environment_to_root_state()

            info = {
                'computational_reward': self.last_computational_reward,
                'object_level_reward': 0
            }

            return self.get_observation(), self.last_meta_reward, False, info

        except Exception as e:
            self._dump_debug_info(e, computational_action)
            raise e

    def _dump_debug_info(self,
                         e: Exception,
                         computational_action: int = None,
                         debug_dir: str = None,
                         open_debug_server: bool = True):
        """
        Dumps debug information to the provided folder or to `./debug/{timestamp}`.
        Starts a debug server on port 5678 to allow inspection of the program state.

        Args:
            computational_action: The action that was taken in the meta environment
            e: The exception that was raised
            debug_dir: The directory to dump the debug information to
            open_debug_server: Whether to open a debug server
        """
        try:
            import time
            debug_dir = debug_dir or f'./debug/{int(time.time() * 1000)}'

            from pathlib import Path
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

            import traceback
            with open(f'{debug_dir}/exception_log.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())

            info = self.get_info()
            info['exception'] = str(e)
            if computational_action is not None:
                info['computational_action'] = int(computational_action)

            import json
            with open(f'{debug_dir}/env_info.json', 'w') as f:
                json.dump(info, f)

            self.plot_search_tokens(show=False)
            plt.savefig(f'{debug_dir}/crash_search_tokens.png')
            plt.close()

            self.render(save_fig_to=f'{debug_dir}/crash_render.png')

        finally:
            if open_debug_server:
                import debugpy
                # 5678 is the default attach port in the VS Code debug configurations.
                # 0.0.0.0 is used to allow remote debugging through docker.
                debugpy.listen(('0.0.0.0', 5678))
                print("Waiting for debugger attach")
                debugpy.wait_for_client()
                print("Debugger attached")
                debugpy.breakpoint()
                print('Breakpoint reached')

    def get_info(self) -> dict:
        return {
            'n_computations': int(self.n_computations),
            'last_meta_reward': float(self.last_meta_reward),
            'last_computation_reward': float(self.last_computational_reward),
            'last_meta_action': int(self.last_meta_action),
            'tree': str(self.tree),
        }

    def set_environment_to_root_state(self):
        root_state = self.tree.get_root().get_state()
        root_state.set_environment_to_state(self.object_env)

    def get_action_strings(self) -> Dict[int, str]:
        if self.expand_all_actions:
            return {
                0: 'Terminate Search',
                **{
                    i: f'Expand Node {(i - 1)}'
                    for i in range(1, self.action_space_size)
                }
            }
        else:
            n = self.n_object_actions
            return {
                0: 'Terminate Search',
                **{
                    i: f'Expand Node {(i - 1) // n} with Action '
                    + self.object_action_to_string((i - 1) % n)
                    for i in range(1, self.action_space_size)
                }
            }

    def get_render_title(self) -> str:

        if self.last_meta_action is None:
            return 'Initial State'

        action_string = self.meta_action_strings[int(self.last_meta_action)]

        if self.tree.get_num_nodes() != 1:
            computational_reward = self.last_computational_reward
        else:
            computational_reward = 0

        action = self.get_best_object_action()

        return f'Meta-action: [{action_string}] | '\
               f'Meta-Reward: {self.last_meta_reward:.3f} | '\
               f'Best Object-action: {self.object_action_to_string(action)} | '\
               f'Computational-Reward: {computational_reward:.3f} | '\
               f't = {self.steps}'

    def render(self,
               mode: str = 'rgb_array',
               save_fig_to: Optional[str] = None,
               meta_action_probs: np.array = None,
               plt_show: bool = False) -> np.ndarray:
        """
        Renders the meta environment as three plots showing the state of the
        object-level environment, the current search tree (i.e. the meta-level
        state), and the Q-distribution over the object-level actions derived
        from the tree and Q-hat function. Additional information regarding the
        last meta-action, meta-reward, best object-level action, and computational
        reward is also displayed.

        Args:
            mode: The rendering mode. Unused and only included for compatibility
                with gym environments.
            plt_show: Whether to call plt.show() after rendering. This is useful
                for interactive environments.

        Returns:
            A numpy array containing the rendered image.
        """
        fig = plt.figure(tight_layout=True, figsize=(15, 6))
        n = 5
        given_action_probs = meta_action_probs is not None
        total_rows = n + 1 if given_action_probs else n
        gs = gridspec.GridSpec(total_rows, 3 * n)

        object_env_ax = fig.add_subplot(gs[:n, :n * 1])
        tree_ax = fig.add_subplot(gs[:n, n:n * 2])
        root_dist_ax = fig.add_subplot(gs[:n, n * 2:])

        plt.suptitle(self.get_render_title())

        self.set_environment_to_root_state()
        object_env_img = self.object_env.render(mode=mode)

        # Render the underlying environment in left subplot
        object_env_ax.set_axis_off()
        object_env_ax.imshow(object_env_img)
        object_env_ax.set_title('Object-level Environment')

        # Render the search tree in the middle subplot
        plot_tree(self.tree, ax=tree_ax, show=False,
                  object_action_to_string=self.object_action_to_string)

        # Render the Q-distribution in the right subplot
        self.plot_root_q_distribution(root_dist_ax)

        if given_action_probs:
            probs_ax = fig.add_subplot(gs[-1, :])
            self.plot_action_probs(probs_ax, meta_action_probs)

        plt.tight_layout(rect=[0, 0.03, 1, .9])

        meta_env_img = plot_to_array(fig)

        if save_fig_to is not None:
            plt.savefig(save_fig_to, transparent=False)

        if plt_show:
            plt.show()
        else:
            plt.close()

        return meta_env_img

    def plot_action_probs(self, probs_ax, meta_action_probs):
        """
        Plots heatmap of the meta-action probabilities on the given axis.

        Args:
            probs_ax: The axis to plot on.
            meta_action_probs: The meta-action probabilities to plot.
        """
        meta_action_probs = meta_action_probs[: 1 + len(self.tree.node_list)]
        meta_action_probs = np.reshape(meta_action_probs, (1, meta_action_probs.size))

        probs_ax.set_title('Meta-level Action probabilities')
        probs_ax = sns.heatmap(meta_action_probs, annot=True,
                               fmt='.3f', vmin=0, vmax=1, cbar=False)
        label_strings = [r'$\perp$'] + [
            f'Expand Node {i}'
            for i, _ in enumerate(self.tree.node_list)
        ]
        probs_ax.set_xticks(ticks=0.5 + np.arange(len(label_strings)),
                            labels=label_strings, ha='center')
        probs_ax.set_yticks(ticks=[])

    def plot_root_q_distribution(self, ax):
        """
        Plots the Q-distribution over the object-level actions derived from the
        current search tree and Q-hat function.

        Args:
            ax: The matplotlib axis on which to plot the Q-distribution.
        """
        root_state = self.tree.get_root().get_state()
        action_labels = root_state.get_action_labels()
        if action_labels:
            q_dist = self.root_q_distribution()
            sns.barplot(x=list(range(q_dist.shape[0])), y=q_dist, ax=ax)
            ax.set_xticklabels(action_labels)

        ax.set_ylim([self.object_reward_min, self.object_reward_max])
        ax.set_title('Root Q-Distribution')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

    def plot_search_tokens(self,
                           ax: plt.Axes = None,
                           show: bool = True,
                           annot_fmt: str = '.3f'):
        """
        Plots the search tokens for each node in the search tree.
        """
        if ax is None:
            plt.figure(figsize=(25, 10))

        obs = self.get_observation()
        tokens = obs['search_tree_tokens'] if self.split_mask_and_tokens else obs
        ax = sns.heatmap(tokens, annot=True, fmt=annot_fmt, ax=ax)
        for t in ax.texts:
            if t.get_text() and float(t.get_text()) == 0:
                t.set_text('')

        ax.set_xticklabels(self.get_token_labels(), rotation=45)
        ax.set_title(
            'Search Tree Tokens: Each token represents a node and '
            'object-level action, i.e. a potential expansion of the search tree.'
        )

        if show:
            plt.show()
