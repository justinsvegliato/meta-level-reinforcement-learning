import numpy as np


def construct_maze_string(maze_view, actions_dict=None) -> str:
    """
    Construct a string representation of the maze.
    Optionally, include the actions taken at each state.

    Args:
        maze_view (MazeView): The maze view from the gym-maze enviroment.
        actions_dict (dict): A dictionary mapping states (tuples) to actions (strings).
            e.g. {(0, 0): 'E', (0, 1): 'S'}

    Returns:
        str: A string representation of the maze. For example:
            |---|---|---|---|---|
            | X →   →   →   |   |
            |---|   |---| ↓ |   |
            |           |   →   |
            |   |---|   |---| ↓ |
            |           |       |
            |---|---|---|   |---|
            |               |   |
            |   |---|---|---|   |
            |                   |
            |---|---|---|---|---|
            Arrows indicate the actions taken at each state.
            The X indicates the current position in the maze.
    """

    w, h = maze_view.maze_size
    actions_dict = actions_dict or {}

    maze_str = '|---' * w + '|\n'

    for y in range(h):
        maze_str += '|'
        for x in range(w):
            is_robot_pos = (maze_view.robot == np.array([x, y])).all()
            if maze_view.is_wall((x, y), 'E'):
                maze_str += ' X |' if is_robot_pos else '   |'
            else:
                maze_str += ' X ' if is_robot_pos else '   '
                move_left = (x, y) in actions_dict and actions_dict[(x, y)] == 'E'
                move_right = (x + 1, y) in actions_dict and actions_dict[(x + 1, y)] == 'W'
                if move_left and move_right:
                    maze_str += '↔'
                elif move_left:
                    maze_str += '→'
                elif move_right:
                    maze_str += '←'
                else:
                    maze_str += ' '

        maze_str += '\n|'
        for x in range(w):
            if maze_view.is_wall((x, y), 'S'):
                maze_str += '---|'
            else:
                move_down = (x, y) in actions_dict and actions_dict[(x, y)] == 'S'
                move_up = (x, y + 1) in actions_dict and actions_dict[(x, y + 1)] == 'N'
                if move_up and move_down:
                    maze_str += ' ↕ |'
                elif move_down:
                    maze_str += ' ↓ |'
                elif move_up:
                    maze_str += ' ↑ |'
                else:
                    maze_str += '   |'
        maze_str += '\n'
    return maze_str


def construct_maze_policy_string(meta_env, policy) -> str:
    """
    Construct a string representation of the maze with the policy overlaid.

    Args:
        meta_env (MetaEnv): The meta environment.
        policy (SearchTreePolicy): The policy to use to determine the actions
            taken at each state.

    Returns:
        str: A string representation of the maze. For example:
            |---|---|---|---|---|
            | X →   →   →   |   |
            |---|   |---| ↓ |   |
            |           |   →   |
            |   |---|   |---| ↓ |
            |           |       |
            |---|---|---|   |---|
            |               |   |
            |   |---|---|---|   |
            |                   |
            |---|---|---|---|---|
            Arrows indicate the actions taken at each state.
            The X indicates the current position in the maze.
    """
    maze_view = meta_env.object_env.maze_view

    def build_trajectory(node) -> list:
        action = policy.get_action(node.state)
        state = tuple(map(int, node.state.get_state_vector()))
        item = (state, node.state.get_action_label(action))
        if action in node.children and node.children[action]:
            child = node.children[action][0]
            return [item] + build_trajectory(child)
        return [item]

    actions_dict = dict(build_trajectory(policy.tree.root_node))
    meta_env.set_environment_to_root_state()
    return construct_maze_string(maze_view, actions_dict)
