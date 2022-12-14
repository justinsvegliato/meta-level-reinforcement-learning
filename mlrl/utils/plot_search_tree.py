from .hierarchical_pos import hierarchy_pos_large_tree
from ..meta.search_tree import SearchTree, SearchTreeNode

import networkx as nx

import matplotlib.pyplot as plt


def construct_tree(tree: nx.DiGraph,
                   node: SearchTreeNode,
                   env):

    tree.add_node(
        node.node_id,
        state=node.get_state().get_state_string()
    )

    for action, children in node.get_children().items():
        for child in children:
            construct_tree(tree, child, env)
            tree.add_edge(node.node_id, child.node_id,
                          action=node.state.get_action_label(action),
                          reward=child.get_reward_received(),
                          id=child.get_id()),


def plot_tree(search_tree: SearchTree, figsize=(20, 20),
              show_reward=False, show_id=False, ax=None,
              can_expand_colour='tab:green',
              show=True, title='Search Tree'):

    nx_tree = nx.DiGraph()
    construct_tree(nx_tree, search_tree.get_root(), search_tree.env)

    pos = hierarchy_pos_large_tree(nx_tree, search_tree.get_root().node_id, width=250, height=250)
    edge_labels = {
        (n1, n2): '{}{}'.format(data['action'], '-' + data['reward'] if show_reward else '')
        for n1, n2, data in nx_tree.edges(data=True)
    }

    node_labels = {
        node: data['state'] if len(data['state']) < 10 else ''
        for node, data in nx_tree.nodes(data=True)
    }

    if all([len(node_labels[node]) == 0 for node in node_labels]):
        node_size = 150
    else:
        node_size = 1200

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
            get_colour(search_tree.node_list[node_idx]) for node_idx in nx_tree.nodes()
        ]
        nx.draw(nx_tree, pos, node_size=node_size, ax=ax, node_color=colour_map)
    else:
        nx.draw(nx_tree, pos, node_size=node_size, ax=ax)

    nx.draw_networkx_edge_labels(nx_tree, pos, edge_labels=edge_labels, ax=ax)
    nx.draw_networkx_labels(nx_tree, pos, labels=node_labels, ax=ax, font_color='white')
    ax.set_axis_off()
    axis = plt.gca()
    zoom = 10
    axis.set_xlim([axis.get_xlim()[0] - zoom, axis.get_xlim()[1] + zoom])
    axis.set_ylim([axis.get_ylim()[0] - zoom, axis.get_ylim()[1] + zoom])

    if show:
        plt.show()
