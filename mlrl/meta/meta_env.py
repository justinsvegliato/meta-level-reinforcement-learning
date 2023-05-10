from mlrl.meta.q_estimation import SearchOptimalQEstimator, DeterministicOptimalQEstimator
from mlrl.meta.search_tree import SearchTree
from mlrl.meta.tree_policy import GreedySearchTreePolicy, SearchTreePolicy
from mlrl.meta.tree_tokenisation import NodeActionTokeniser, NodeTokeniser
from mlrl.utils import clean_for_json
from mlrl.utils.plot_search_tree import plot_tree
from mlrl.utils.render_utils import plot_to_array

from typing import Callable, List, Optional, Tuple, Dict, Union, NoReturn

import numpy as np
import gym

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set()


TransitionObserver = Callable[[np.ndarray, float, bool, dict], NoReturn]
TreePolicyProducer = Callable[[SearchTree], SearchTreePolicy]
TreePolicyRenderer = Callable[[gym.Env, SearchTreePolicy], np.ndarray]


gym.envs.register(
    id='MetaEnv-v0',
    entry_point='mlrl.meta.meta_env:MetaEnv',
)


class ObjectLevelMetrics:

    def __init__(self, episode_complete_callback: Optional[Callable[[dict], None]] = None):
        self.return_val = 0
        self.n_steps = 0
        self.episode_stats = []
        self.episode_complete_callback = episode_complete_callback

    def reset(self):
        self.return_val = 0
        self.n_steps = 0
        self.episode_stats = []

    def get_num_episodes(self):
        return len(self.episode_stats)

    def __call__(self, obs, reward, done, info):
        self.return_val += np.sum(reward)
        if done:
            stats = {
                'return': self.return_val,
                'steps': self.n_steps
            }
            self.episode_stats.append(stats)
            if self.episode_complete_callback is not None:
                self.episode_complete_callback(stats)

            self.return_val = 0
            self.n_steps = 0
        else:
            self.n_steps += np.size(reward)

    def get_live_results(self):
        sum_of_returns = self.return_val + sum([stat['return'] for stat in self.episode_stats])
        total_steps = self.n_steps + sum([stat['steps'] for stat in self.episode_stats])
        n_episodes = len(self.episode_stats) + int(self.n_steps > 0)

        return {
            'ObjectLevelMeanReward': sum_of_returns / max(1, n_episodes),
            'ObjectLevelMeanStepsPerEpisode': total_steps / max(1, n_episodes),
            'ObjectLevelEpisodes': n_episodes,
            'ObjectLevelCurrentEpisodeReturn': self.return_val,
            'ObjectLevelCurrentEpisodeSteps': self.n_steps
        }

    def get_results(self):
        sum_of_returns = sum([stat['return'] for stat in self.episode_stats])
        total_steps = sum([stat['steps'] for stat in self.episode_stats])
        n_episodes = len(self.episode_stats)

        return {
            'ObjectLevelMeanReward': sum_of_returns / max(1, n_episodes),
            'ObjectLevelMeanStepsPerEpisode': total_steps / max(1, n_episodes),
            'ObjectLevelEpisodes': n_episodes,
            'ObjectLevelCurrentEpisodeReturn': self.return_val,
            'ObjectLevelCurrentEpisodeSteps': self.n_steps
        }


def aggregate_object_level_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    sum_reward = 0
    n_steps = 0
    total_episodes = 0
    curr_returns = 0
    curr_n_steps = 0

    for metric in metrics:
        n_episodes = metric['ObjectLevelEpisodes']
        if n_episodes == 0:
            continue
        sum_reward += metric['ObjectLevelMeanReward'] * n_episodes
        n_steps += metric['ObjectLevelMeanStepsPerEpisode'] * n_episodes
        curr_returns += metric['ObjectLevelCurrentEpisodeReturn']
        curr_n_steps += metric['ObjectLevelCurrentEpisodeSteps']
        total_episodes += n_episodes

    n_envs = len(metrics)

    return {
        'ObjectLevelMeanReward': sum_reward / max(1, total_episodes),
        'ObjectLevelMeanStepsPerEpisode': n_steps / max(1, total_episodes),
        'ObjectLevelEpisodes': total_episodes,
        'ObjectLevelCurrentEpisodeReturn': curr_returns / max(1, n_envs),
        'ObjectLevelCurrentEpisodeSteps': curr_n_steps / max(1, n_envs)
    }


def mask_token_splitter(tokens_and_mask):
    tokens = tokens_and_mask[MetaEnv.SEARCH_TOKENS_KEY]
    mask = tokens_and_mask[MetaEnv.ACTION_MASK_KEY]
    return tokens, mask


class MetaEnv(gym.Env):
    """
    Class that wraps a gym environment and allows for the meta-learning of the search problem
    """

    SEARCH_TOKENS_KEY = 'search_tree_tokens'
    ACTION_MASK_KEY = 'action_mask'

    def __init__(self,
                 object_env: gym.Env,
                 initial_tree: SearchTree,
                 cost_of_computation: float = 0.001,
                 computational_rewards: bool = True,
                 max_tree_size: int = 10,
                 expand_all_actions: bool = True,
                 finish_on_terminate: bool = False,
                 keep_subtree_on_terminate: bool = True,
                 root_based_computational_rewards: bool = False,
                 search_optimal_q_estimator: Optional[SearchOptimalQEstimator] = None,
                 make_tree_policy: Optional[TreePolicyProducer] = None,
                 tree_policy_renderer: Optional[TreePolicyRenderer] = None,
                 object_reward_min: float = 0.0,
                 object_reward_max: float = 1.0,
                 object_env_discount: float = 0.99,
                 one_hot_action_space: bool = False,
                 split_mask_and_tokens: bool = True,
                 random_cost_of_computation: bool = True,
                 cost_of_computation_interval: Tuple[float, float] = (0.0, 0.05),
                 min_computation_steps: int = 0,
                 open_debug_server_on_fail: bool = False,
                 object_level_transition_observers: Optional[List[TransitionObserver]] = None,
                 verbose: bool = False,
                 reset_on_crash: bool = True,
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
        # Meta env params
        self.object_env = object_env
        self.max_tree_size = max_tree_size
        self.expand_all_actions = expand_all_actions
        self.keep_subtree_on_terminate = keep_subtree_on_terminate
        self.object_env_discount = object_env_discount

        self.computational_rewards = computational_rewards
        self.root_based_computational_rewards = root_based_computational_rewards
        self.finish_on_terminate = finish_on_terminate
        self.min_computation_steps = min_computation_steps
        self.object_level_metrics = ObjectLevelMetrics()
        self.object_level_transition_observers = [self.object_level_metrics] + (object_level_transition_observers or [])

        self.random_cost_of_computation = random_cost_of_computation
        self.cost_of_computation_interval = cost_of_computation_interval
        self.cost_of_computation = cost_of_computation
        self.cost_of_computation = self.get_next_computation_cost()

        # Functions
        self.tree_policy_renderer = tree_policy_renderer

        self.optimal_q_estimator = search_optimal_q_estimator or \
            DeterministicOptimalQEstimator(object_env_discount)

        self.make_tree_policy = make_tree_policy or \
            (lambda tree: GreedySearchTreePolicy(tree, object_env_discount))
        self.search_tree_policy = self.make_tree_policy(initial_tree)
        self.prev_search_policy = None

        # Utility params
        self.split_mask_and_tokens = split_mask_and_tokens
        self.dump_debug_images = dump_debug_images
        self.open_debug_server_on_fail = open_debug_server_on_fail and not reset_on_crash
        self.reset_on_crash = reset_on_crash
        self.verbose = verbose

        # Meta env state
        self.tree = initial_tree
        self.n_computations = 0
        self.done = False

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

        # Meta data features: attn mask, can expand, reward, is terminate
        self.n_meta_feats = 4
        self.state_vec_dim = object_state.get_state_vector_dim()
        self.action_vec_dim = object_state.get_action_vector_dim()

        if self.expand_all_actions:
            # If we are expanding all actions, then the token does not include a
            # representation of the  to be expanded
            # self.tree_token_dim = self.n_meta_feats + \
            #     2 * self.max_tree_size + self.state_vec_dim + self.action_vec_dim

            self.tree_tokeniser = NodeTokeniser(self.max_tree_size,
                                                self.action_vec_dim,
                                                self.state_vec_dim)
        else:
            # 2 * max_tree_size because we include the id of the corresponding node
            # and the id of the parent node
            # 2 * self.action_vec_dim because we include the vector for the action
            # taken to reach the node and the vector for the action to be expanded
            # self.tree_token_dim = self.n_meta_feats + \
            #     2 * self.max_tree_size + self.state_vec_dim + 2 * self.action_vec_dim

            self.tree_tokeniser = NodeActionTokeniser(self.max_tree_size,
                                                      self.action_vec_dim,
                                                      self.state_vec_dim)

        self.update_tree_meta_vars()
        self.tree_token_dim = self.tree_tokeniser.tree_token_dim

        tree_token_space = gym.spaces.Box(
            low=object_reward_min, high=object_reward_max,
            shape=(self.action_space_size, self.tree_token_dim),
            dtype=np.float64
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
        self.last_object_level_reward = 0
        self.meta_action_strings = self.get_action_strings()
        self.steps = 0

    def get_next_computation_cost(self):
        if self.random_cost_of_computation:
            return np.random.uniform(*self.cost_of_computation_interval)
        else:
            return self.cost_of_computation

    def update_tree_meta_vars(self):
        min_cost, max_cost = self.cost_of_computation_interval

        if max_cost < min_cost:
            raise ValueError(f'{max_cost = } is less than {min_cost = }')

        if np.abs(max_cost - min_cost) > 1e-9:
            normed_cost = (self.cost_of_computation - min_cost) / (max_cost - min_cost)
        else:
            normed_cost = min_cost

        self.tree_tokeniser.set_meta_vars(cost_of_computation=normed_cost)

    def reset(self):
        self.object_env.reset()
        self.tree = self.get_root_tree()
        self.search_tree_policy = self.make_tree_policy(self.tree)
        self.reset_computation_variables()
        return self.get_observation()

    def reset_computation_variables(self):
        self.prev_search_policy = None
        self.n_computations = 0
        self.last_meta_action = None
        self.last_meta_reward = 0
        self.last_computational_reward = 0
        self.steps = 0
        self.cost_of_computation = self.get_next_computation_cost()
        self.update_tree_meta_vars()

    def reset_metrics(self):
        self.object_level_metrics.reset()

    def get_object_level_metrics(self) -> dict:
        return self.object_level_metrics.get_results()

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
            discount=self.tree.discount
        )

    def get_token_labels(self) -> List[str]:
        """ Returns the labels for the tokens in the observation """
        return self.tree_tokeniser.get_token_labels()

    def get_observation(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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

        search_tokens = self.tree_tokeniser.tokenise(self.tree)

        if self.n_computations < self.min_computation_steps:
            search_tokens[0, 1] = 0.  # Set to 0 so terminal cannot be selected

        if self.split_mask_and_tokens:
            action_mask = search_tokens[:, 1].astype(np.int32)
            return {
                'search_tree_tokens': search_tokens,
                'action_mask': action_mask
            }
        else:
            return search_tokens

    def get_best_object_action(self) -> int:
        # """
        # Returns the best action in the object-level environment.
        # The action is returned as an integer indicating the index of
        # the gym action in the ObjectState.get_actions representation.

        # For example, an environment may have an action space of unhashable
        # objects, e.g. uci move objects in gymchess, q-distributions are maintained
        # as dictionaries of integers to floats, and the action space is a list of
        # uci move objects. In this case, the action returned by this function
        # would be the index of the uci move object in the action space.
        # """
        root_state = self.tree.get_root().get_state()
        return self.search_tree_policy.get_action(root_state)

    # def get_best_object_action(self):
    #     action_idx = self.get_best_object_action_index()
    #     root_state = self.tree.get_root().get_state()
    #     return root_state.get_actions()[action_idx]

    def root_q_distribution(self) -> Dict[int, float]:
        self.optimal_q_estimator.estimate_and_cache_optimal_q_values(self.tree)
        root_node = self.tree.get_root()
        return {a: root_node.get_q_value(a) for a in root_node.state.get_actions()}

    def get_computational_reward(self) -> float:
        """
        Difference between the value of best action before and after the
        computation, both considered under the Q-distribution derived from
        the tree after the computation.
        """
        if self.root_based_computational_rewards:
            # q-distribution under the current tree
            q_dist = self.root_q_distribution()
            # best action under the current policy (assumed to be created with the previous tree)
            prior_action = self.get_best_object_action()
            self.last_computational_reward = max(q_dist.values()) - q_dist[prior_action]

        else:
            if self.verbose:
                print('Estimating value of new policy:\n', self.tree)

            updated_policy_value = self.search_tree_policy.evaluate(self.tree, verbose=self.verbose)

            if self.verbose:
                print()
                print('Estimating value of prior policy:\n', self.search_tree_policy.tree)

            prior_policy_value = self.prev_search_policy.evaluate(self.tree, verbose=self.verbose)

            self.last_computational_reward = updated_policy_value - prior_policy_value
            if self.verbose:
                print()
                print(f'Computational Reward = {updated_policy_value:.3f} - '
                      f'{prior_policy_value:.3f} = {self.last_computational_reward:.3f}')

        return self.last_computational_reward

    def step(self,
             computational_action: Union[int, list, np.ndarray]
             ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float, bool, dict]:
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
                computational_action = int(np.argmax(computational_action))

            self.prev_search_policy = self.search_tree_policy
            self.last_computational_reward = 0
            self.last_meta_action = computational_action
            self.steps += 1

            if computational_action == 0 or not self.tree_tokeniser.can_tokenise(self.tree):
                self.done = True
                return self.terminate_step()

            self.done = False

            self.perform_computational_action(computational_action)

            self.search_tree_policy = self.make_tree_policy(self.tree)

            meta_reward = -self.cost_of_computation
            if self.computational_rewards:
                meta_reward += self.get_computational_reward()

            # Set the environment to the state of the root node for inter-step consistency
            self.set_environment_to_root_state()
            self.last_meta_reward = meta_reward

            info = {
                'computational_reward': self.last_computational_reward,
                'object_level_reward': 0
            }

            return self.get_observation(), meta_reward, False, info

        except Exception as e:
            return self._handle_exception(e, computational_action)

    def terminate_step(self):

        # Perform a step in the underlying environment
        action = self.get_best_object_action()

        self.set_environment_to_root_state()
        object_obs, object_r, done, info = self.object_env.step(action)
        self.last_object_level_reward = object_r

        self.object_level_metrics(object_obs, object_r, done, info)
        if self.object_level_transition_observers is not None:
            for observer in self.object_level_transition_observers:
                observer(object_obs, object_r, done, info)

        if not self.finish_on_terminate and self.keep_subtree_on_terminate and self.tree.get_root().has_action_children(action):
            self.tree = self.tree.get_root_subtree(action)
        else:
            self.tree = self.get_root_tree()
            self.last_meta_reward = object_r
            self.prev_search_policy = None
            self.n_computations = 0

        self.search_tree_policy = self.make_tree_policy(self.tree)

        return self.observe_terminate()

    def observe_terminate(self):

        if self.finish_on_terminate:
            return self.get_observation(), 0., self.done, {}

        info = {
            'computational_reward': 0,
            'object_level_reward': self.last_object_level_reward
        }

        return self.get_observation(), self.last_meta_reward, self.done, info

    def perform_computational_action(self, computational_action: int):
        """
        Performs a computational action on the tree.
        """
        self.last_object_level_reward = 0.
        self.n_computations += 1

        if self.expand_all_actions:
            # Expand all actions from the given node
            node_idx = self.tree_tokeniser.get_node_idx(self.tree, computational_action)
            self.tree.expand_all(node_idx)
        else:
            # perform a computational action on the search tree
            node_idx = (computational_action - 1) // self.n_object_actions
            object_action_idx = (computational_action - 1) % self.n_object_actions
            self.tree.expand_action(node_idx, object_action_idx)

    def act(self, computational_action: Union[int, list, np.ndarray]):
        try:
            if self.one_hot_action_space and not isinstance(computational_action, int):
                computational_action = int(np.argmax(computational_action))

            self.prev_search_policy = self.search_tree_policy
            self.last_computational_reward = 0
            self.last_meta_action = computational_action
            self.steps += 1

            if computational_action == 0 or not self.tree_tokeniser.can_tokenise(self.tree):
                self.done = True
                self.terminate()
            else:
                self.done = False
                self.perform_computational_action(computational_action)

        except Exception as e:
            return self._handle_exception(e, computational_action)

    def terminate(self):

        # Perform a step in the underlying environment
        action = self.get_best_object_action()

        self.set_environment_to_root_state()
        object_obs, object_r, done, info = self.object_env.step(action)
        self.last_object_level_reward = object_r

        for observer in self.object_level_transition_observers:
            observer(object_obs, object_r, done, info)

        if not self.finish_on_terminate and self.keep_subtree_on_terminate and self.tree.get_root().has_action_children(action):
            self.tree = self.tree.get_root_subtree(action)
        else:
            self.tree = self.get_root_tree()

        self.search_tree_policy = self.make_tree_policy(self.tree)

        self.reset_computation_variables()

    def observe(self):
        try:
            if self.done:
                return self.observe_terminate()

            self.search_tree_policy = self.make_tree_policy(self.tree)

            meta_reward = -self.cost_of_computation
            if self.computational_rewards:
                meta_reward += self.get_computational_reward()

            # Set the environment to the state of the root node for inter-step consistency
            self.set_environment_to_root_state()
            self.last_meta_reward = meta_reward

            info = {
                'computational_reward': self.last_computational_reward,
                'object_level_reward': 0
            }

            return self.get_observation(), meta_reward, self.done, info

        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, e: Exception, *args):
        debug_dir = self._dump_debug_info(e, *args,
                                          open_debug_server=self.open_debug_server_on_fail)
        if self.reset_on_crash:
            print(f'Resetting environment after crash: {e}')
            print(f'Debug info dumped to {debug_dir}')
            return self.reset(), 0., True, {}
        else:
            raise e

    def _dump_debug_info(self,
                         e: Exception,
                         computational_action: Optional[int] = None,
                         debug_dir: Optional[str] = None,
                         open_debug_server: bool = True) -> str:
        """
        Dumps debug information to the provided folder or to `./outputs/debug/{timestamp}`.
        Starts a debug server on port 5678 to allow inspection of the program state.

        Args:
            computational_action: The action that was taken in the meta environment
            e: The exception that was raised
            debug_dir: The directory to dump the debug information to
            open_debug_server: Whether to open a debug server

        Returns:
            The directory that the debug information was dumped to
        """
        from time import time
        debug_dir = debug_dir or f'./outputs/debug/{time()}'

        from pathlib import Path
        Path(debug_dir).mkdir(parents=True, exist_ok=True)

        try:
            import traceback
            with open(f'{debug_dir}/exception_log.txt', 'a') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
        except Exception as e2:
            print(f'Failed to dump exception log: {e2}')

        try:
            info = self.get_info()
            info['exception'] = str(e)
            if computational_action is not None:
                info['computational_action'] = int(computational_action)

            import json
            with open(f'{debug_dir}/env_info.json', 'w') as f:
                json.dump(info, f, indent=4)

        except Exception as e2:
            print(f'Failed to dump environment info: {e2}')

        try:
            obs = self.get_observation()
            if isinstance(obs, dict):
                obs['token_dim_labels'] = self.tree_tokeniser.get_token_labels()
            else:
                obs = {
                    'observation': obs,
                    'token_dim_labels': self.tree_tokeniser.get_token_labels()
                }
            with open(f'{debug_dir}/observation.json', 'w') as f:
                json.dump(clean_for_json(obs), f, indent=4)

            # self.plot_search_tokens(show=False)
            # plt.savefig(f'{debug_dir}/crash_search_tokens.png')
            # plt.close()

        except Exception as debug_e:
            print(f'Failed to write observation: {debug_e}')
            with open(f'{debug_dir}/observation.json', 'w') as f:
                f.write(f'Failed to write observation: {debug_e}')
                import traceback
                f.write(traceback.format_exc())

        try:
            self.render(save_fig_to=f'{debug_dir}/crash_render.png')
        except Exception as debug_e:
            print(f'Failed to dump render: {debug_e}')

        try:
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
        except Exception as debug_e:
            print(f'Failed to start debug server: {debug_e}')

        return debug_dir

    def get_object_level_seed(self) -> Optional[int]:
        if not hasattr(self.object_env, 'get_seed'):
            return None
        get_seed = getattr(self.object_env, 'get_seed')
        return get_seed()

    def get_info(self) -> dict:
        return {
            'n_computations': int(self.n_computations),
            'last_meta_reward': float(self.last_meta_reward),
            'last_computation_reward': float(self.last_computational_reward),
            'last_meta_action': int(self.last_meta_action),
            'tree': str(self.tree),
            'object_level_seed': self.get_object_level_seed(),
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
            raise NotImplementedError('Not implemented for expand_all_actions=False')
            # action_strings = dict()
            # return {
            #     0: 'Terminate Search',
            #     **{
            #         i: f'Expand Node {i} with Action {node.state.get_action_label(action)}'
            #         for i, node in enumerate(self.tree.get_nodes())
            #         for action in node.state.get_actions()
            #     }
            # }

    def get_render_title(self) -> str:

        if self.last_meta_action is None:
            object_info = self.get_object_level_info_string()
            return f'Initial State\n{object_info}'

        i = int(self.last_meta_action)
        if i in self.meta_action_strings:
            action_string = self.meta_action_strings[i]
        else:
            action_string = 'Invalid'

        if self.tree.get_num_nodes() != 1:
            computational_reward = self.last_computational_reward
        else:
            computational_reward = 0

        action = self.get_best_object_action()
        root_node = self.tree.get_root()
        action_label = root_node.state.get_action_label(action)

        meta_info = f'Meta-action: [{action_string}] | '\
                    f'Meta-Reward: {self.last_meta_reward:.3f} | '\
                    f'Best Object-action: {action_label} | '\
                    f'Comp-Reward: {computational_reward:.3f} | '\
                    f'Comp-Cost: {self.cost_of_computation:.3f} | '\
                    f't = {self.steps}'

        object_info = self.get_object_level_info_string()

        return f'{meta_info}\n{object_info}'

    def get_object_level_info_string(self) -> str:
        return f'Object-level Reward: {self.last_object_level_reward:.2f} | ' + \
            ' | '.join(f'{k}: {v:.2f}' for k, v in self.get_object_level_metrics().items())

    def render(self,
               mode: str = 'rgb_array',
               save_fig_to: Optional[str] = None,
               meta_action_probs: Optional[np.ndarray] = None,
               remove_duplicate_tree_states: bool = True,
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

        if self.tree_policy_renderer is None:
            object_env_img = self.object_env.render(mode=mode)
        else:
            object_env_img = self.tree_policy_renderer(self.object_env,
                                                       self.search_tree_policy)

        # Render the underlying environment in left subplot
        object_env_ax.set_axis_off()
        object_env_ax.imshow(object_env_img)

        seed = self.get_object_level_seed()
        if seed is not None:
            object_env_ax.set_title(f'Object-level Environment [Seed: {seed}]')
        else:
            object_env_ax.set_title('Object-level Environment')

        # Render the search tree in the middle subplot
        plot_tree(self.tree, ax=tree_ax, show=False,
                  remove_duplicate_states=remove_duplicate_tree_states)

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

        self.set_environment_to_root_state()

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

        n_actions = len(self.tree.node_list)

        probs_ax.set_title('Meta-level Action probabilities')
        probs_ax = sns.heatmap(meta_action_probs, annot=n_actions <= 16,
                               fmt='.3f', vmin=0, vmax=1, cbar=False)

        def label(i: int) -> str:
            if n_actions > 16:
                return ''
            if n_actions >= 10:
                return str(i)
            return f'Expand Node {i}'

        label_strings = [r'$\perp$' if n_actions < 16 else ''] + [
            str(i) for i, _ in enumerate(self.tree.node_list)
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
            q_dist = np.array(list(q_dist.values()))

            sns.barplot(x=list(range(q_dist.size)), y=q_dist, ax=ax)
            ax.bar_label(ax.containers[0], labels=action_labels,
                         label_type='center', rotation=90, color='white')
            ax.set_xticklabels([''] * len(action_labels))

            # if any([len(label) > 3 for label in action_labels]):
            #     rotation = 90
            # else:
            #     rotation = 0

            # ax.set_xticklabels(action_labels, rotation=rotation)

            min_val = min(self.object_reward_min, min(q_dist))
            max_val = max(self.object_reward_max, max(q_dist))
            ax.set_ylim([min_val, max_val])

        ax.set_title('Root Q-Distribution')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

    def plot_search_tokens(self,
                           ax: Optional[plt.Axes] = None,
                           show: bool = True,
                           annot_fmt: str = '.3f'):
        """
        Plots the search tokens for each node in the search tree.
        """
        if ax is None:
            plt.figure(figsize=(25, 10))

        obs = self.get_observation()
        tokens = obs['search_tree_tokens'] if self.split_mask_and_tokens else obs
        ax: plt.Axes = sns.heatmap(tokens, annot=True, fmt=annot_fmt, ax=ax)
        for t in ax.texts:
            if t.get_text() and float(t.get_text()) == 0:
                t.set_text('')

        ax.set_xticklabels(self.get_token_labels(), rotation=65)
        ax.set_title(
            'Search Tree Tokens: Each token represents a node and '
            'object-level action, i.e. a potential expansion of the search tree.'
        )

        if show:
            plt.show()
