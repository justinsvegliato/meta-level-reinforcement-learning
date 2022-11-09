from .hierarchical_pos import hierarchy_pos_large_tree
from ..meta.search_tree import SearchTree, SearchTreeNode

import networkx as nx

import matplotlib.pyplot as plt


def construct_tree(tree: nx.DiGraph,
                   node: SearchTreeNode,
                   env,
                   object_action_to_string):

    tree.add_node(
        node.node_id,
        state=node.get_state().get_state_string()
    )

    for action, children in node.get_children().items():
        for child in children:
            construct_tree(tree, child, env, object_action_to_string)
            tree.add_edge(node.node_id, child.node_id,
                          action=object_action_to_string(action),
                          reward=child.get_reward_received(),
                          id=child.get_id()),


def plot_tree(search_tree: SearchTree, figsize=(20, 20),
              show_reward=False, show_id=False, ax=None,
              object_action_to_string=None,
              can_expand_colour='tab:green',
              show=True, title='Search Tree'):

    nx_tree = nx.DiGraph()
    object_action_to_string = object_action_to_string or (lambda x: str(x))
    construct_tree(nx_tree, search_tree.get_root(), search_tree.env, object_action_to_string)

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
        try:
            colour_map = [
                can_expand_colour if search_tree.node_list[node_idx].can_expand() else 'tab:blue'
                for node_idx in nx_tree.nodes()
            ]
            nx.draw(nx_tree, pos, node_size=node_size, ax=ax, node_color=colour_map)
        except Exception:
            import debugpy
            # 5678 is the default attach port in the VS Code debug configurations.
            # 0.0.0.0 is used to allow remote debugging through docker.
            debugpy.listen(('0.0.0.0', 5678))
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            print("Debugger attached")
            debugpy.breakpoint()
            print('Breakpoint reached')
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
