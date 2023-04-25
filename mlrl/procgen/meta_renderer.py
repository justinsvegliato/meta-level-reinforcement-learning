from typing import List, Tuple

from gym3 import ToGymEnv
import numpy as np

from mlrl.meta.tree_policy import SearchTreePolicy
from mlrl.meta.search_tree import SearchTreeNode
from mlrl.procgen.procgen_state import ProcgenState


def render_tree_policy(env: ToGymEnv, tree_policy: SearchTreePolicy,
                       include_leaf_actions=False,
                       blend_mode: str = 'screen') -> np.array:
    """
    Renders the sequence of actions of a tree policy on the procegen environment.
    Overlays the planned future states on the current state.

    Args:
        - env: The procgen environment to render on
        - tree_policy: The tree policy to render

    Returns:
        - A numpy array of the rendered image
    """

    def recursive_get_transitions(node: SearchTreeNode[ProcgenState]) -> List[Tuple[int, int]]:
        if node.is_terminal_state:
            return []

        start_obs = node.state.observation
        action_probs = tree_policy.get_action_probabilities(node.state)
        transitions = []
        for action, prob in action_probs.items():
            if prob == 0:
                continue

            if node.has_action_children(action):
                child, *_ = node.get_children(action)
                end_obs = child.state.observation
                transitions.extend([(start_obs, end_obs, prob)] + recursive_get_transitions(child))

            elif include_leaf_actions:
                node.state.set_environment_to_state(env)
                next_obs, *_ = env.step(action)
                transitions.append((start_obs, next_obs, prob))

        return transitions

    root = tree_policy.tree.get_root()
    transitions = recursive_get_transitions(root)

    root.state.set_environment_to_state(env)
    img = root.state.observation

    for _, end_obs, prob in transitions:
        frame = 0.25 * prob * end_obs
        if blend_mode == 'add':
            img += frame
        elif blend_mode == 'max':
            img = np.maximum(img, frame)
        elif blend_mode == 'min':
            img = np.minimum(img, frame)
        elif blend_mode == 'screen':
            img = 1 - (1 - img) * (1 - frame)

    return np.clip(img[0], 0, 1)
