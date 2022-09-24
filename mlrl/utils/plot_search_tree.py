from .hierarchical_pos import hierarchy_pos_large_tree
from ..meta.search_tree import SearchTree, SearchTreeNode

import networkx as nx

import matplotlib.pyplot as plt


def construct_tree(tree: nx.DiGraph, node: SearchTreeNode, env):
    tree.add_node(hash(node), state=str(node.state))
    for action, children in node.get_children().items():
        for child in children:
            construct_tree(tree, child, env)
            tree.add_edge(hash(node), hash(child),
                          action=env.ACTION[action],
                          reward=child.get_reward_received(),
                          id=child.get_id()),


def plot_tree(search_tree: SearchTree, figsize=(20, 20),
              show_reward=False, show_id=False, ax=None,
              show=True, title='Search Tree'):

    nx_tree = nx.DiGraph()
    construct_tree(nx_tree, search_tree.get_root(), search_tree.env)

    pos = hierarchy_pos_large_tree(nx_tree, hash(search_tree.get_root()), width=250, height=250)
    edge_labels = {
        (n1, n2): '{}{}'.format(data['action'], '-' + data['reward'] if show_reward else '')
        for n1, n2, data in nx_tree.edges(data=True)
    }

    def get_node_label(node):
        vec = tuple(nx_tree.nodes.get(node)['state'].get_state_vector())
        if show_id:
            return f'{vec} - {nx_tree.nodes.get(node)["state"].get_id()}'
        return str(vec)

    node_labels = {node: get_node_label(node) for node in nx_tree.nodes()}

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    ax.set_title(title)

    nx.draw(nx_tree, pos, node_size=1800, ax=ax)

    nx.draw_networkx_edge_labels(nx_tree, pos, edge_labels=edge_labels, ax=ax)
    nx.draw_networkx_labels(nx_tree, pos, labels=node_labels, ax=ax, font_color='white')
    ax.set_axis_off()
    axis = plt.gca()
    zoom = 10
    axis.set_xlim([axis.get_xlim()[0] - zoom, axis.get_xlim()[1] + zoom])
    axis.set_ylim([axis.get_ylim()[0] - zoom, axis.get_ylim()[1] + zoom])

    if show:
        plt.show()
