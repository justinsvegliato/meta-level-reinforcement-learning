import networkx as nx
import random


def hierarchy_pos(graph, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike


    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(graph):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(iter(nx.topological_sort(graph)))
            # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(graph.nodes))

    def _hierarchy_pos(graph, root,
                       width=1., vert_gap=0.2, vert_loc=0, x_center=0.5,
                       pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed
        '''

        if pos is None:
            pos = {root: (x_center, vert_loc)}
        else:
            pos[root] = (x_center, vert_loc)
        children = list(graph.neighbors(root))
        if not isinstance(graph, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = x_center - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(graph, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, x_center=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(graph, root, width, vert_gap, vert_loc, xcenter)


def hierarchy_pos_large_tree(graph, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing

       Code written by burubum:
       https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
       '''
    total_key = "total"
    current_key = "current"

    def make_levels(levels, node=root, current_level=0, parent=None):
        """ Compute the number of nodes for each level """
        if current_level not in levels:
            levels[current_level] = {total_key : 0, current_key : 0}
        levels[current_level][total_key] += 1
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, current_level + 1, node)
        return levels

    def make_pos(pos, node=root, current_level=0, parent=None, vert_loc=0):
        dx = 1 / levels[current_level][total_key]
        left = dx / 2
        pos[node] = ((left + dx * levels[current_level][current_key]) * width, vert_loc)
        levels[current_level][current_key] += 1
        neighbors = graph.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, current_level + 1, node, vert_loc - vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {level: {total_key: levels[level], current_key: 0} for level in levels}
    vert_gap = height / (max([level for level in levels]) + 1)
    return make_pos({})
