from mlrl.utils.hierarchical_pos import hierarchy_pos_large_tree
from mlrl.meta.search_tree import SearchTree, SearchTreeNode
from mlrl.meta.tree_policy import SearchTreePolicy

from typing import Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def construct_tree(tree: nx.DiGraph,
                   policy: SearchTreePolicy,
                   node: SearchTreeNode,
                   remove_duplicate_states=True):

    tree.add_node(
        node.node_id,
        search_tree_node=node,
        state=node.get_state().get_state_string()
    )

    for action, children in node.get_children().items():
        for child in children:
            if remove_duplicate_states and child.duplicate_state_ancestor is not None:
                continue
            construct_tree(tree, policy, child, remove_duplicate_states=remove_duplicate_states)

            if policy is not None:
                base_edge_colour = np.array([255., 164., 36.]) / 255.
                state = node.get_state()
                prob = policy.get_action_probabilities(state)[action]
                colour = base_edge_colour * (0.25 + 0.75 * prob if prob > 0 else 0.)
            else:
                colour = np.array([0., 0., 0.])

            tree.add_edge(node.node_id, child.node_id,
                          parent=node, child=child,
                          colour=colour,
                          action=node.state.get_action_label(action),
                          reward=child.get_reward_received(),
                          id=child.get_id()),


def plot_tree(search_tree: SearchTree,
              policy: Optional[SearchTreePolicy] = None,
              figsize=(20, 20),
              show_reward=False,
              show_id=False, ax=None,
              can_expand_colour='tab:green',
              remove_duplicate_states=True,
              node_label_threshold=20,
              min_state_string_length=10,
              small_node_size=150,
              large_node_size=1200,
              edge_width=1,
              zoom=10,
              show=True,
              title='Search Tree',
              **draw_kwargs):

    nx_tree = nx.DiGraph()
    construct_tree(nx_tree, policy, search_tree.get_root(), remove_duplicate_states=remove_duplicate_states)

    pos = hierarchy_pos_large_tree(nx_tree, search_tree.get_root().node_id, width=250, height=250)

    if len(search_tree.node_list) < node_label_threshold:
        edge_labels = {
            (n1, n2): '{}{}'.format(data['action'], '-' + data['reward'] if show_reward else '')
            for n1, n2, data in nx_tree.edges(data=True)
        }
    else:
        edge_labels = {
            (n1, n2): '' for n1, n2, _ in nx_tree.edges(data=True)
        }

    edge_colours = [
        data['colour'] for *_, data in nx_tree.edges(data=True)
    ]

    node_labels = {
        node: data['state'] if len(data['state']) < min_state_string_length else ''
        for node, data in nx_tree.nodes(data=True)
    }

    if all([len(node_labels[node]) == 0 for node in node_labels]):
        node_size = small_node_size
    else:
        node_size = large_node_size

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    ax.set_title(title)

    if can_expand_colour is not None:

        def get_colour(node: SearchTreeNode):
            if node.can_expand():
                return can_expand_colour
            if node.duplicate_state_ancestor is not None:
                return 'tab:red'
            else:
                return 'tab:blue'

        colour_map = [
            get_colour(search_tree.node_list[node_idx])
            for node_idx in nx_tree.nodes()
        ]
        nx.draw(nx_tree, pos, node_size=node_size, ax=ax,
                node_color=colour_map, width=edge_width,
                edge_color=edge_colours, **draw_kwargs)
    else:
        nx.draw(nx_tree, pos, node_size=node_size, ax=ax,
                width=edge_width, edge_color=edge_colours, **draw_kwargs)

    nx.draw_networkx_edge_labels(nx_tree, pos, edge_labels=edge_labels, ax=ax)
    nx.draw_networkx_labels(nx_tree, pos, labels=node_labels, ax=ax, font_color='white')
    ax.set_axis_off()
    axis = plt.gca()
    axis.set_xlim([axis.get_xlim()[0] - zoom, axis.get_xlim()[1] + zoom])
    axis.set_ylim([axis.get_ylim()[0] - zoom, axis.get_ylim()[1] + zoom])

    if show:
        plt.show()
