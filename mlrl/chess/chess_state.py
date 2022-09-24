from ..meta.search_tree import ObjectState

import chess
import numpy as np


class ChessState(ObjectState):
    """
    Class to handle represent a state of the chess environment.
    """
    __slots__ = ['board_epd']

    @staticmethod
    def extract_state(env) -> 'ChessState':
        """
        A static method to extract the state of the environment
        for later restoring and representation.
        """

        # Extended Position Description of the board for later restoring
        board_epd = env._board.epd()

        return ChessState(board_epd)

    def __init__(self, board_epd: str):
        self.board_epd = board_epd

    def set_environment_to_state(self, env):
        env._board.set_epd(self.board_epd)

    def get_state_vector(self) -> np.array:
        return np.array([0], dtype=np.float32)

    def get_actions(self) -> list:
        """ Returns a list of actions that can be taken from the current state. """
        board = chess.Board(self.board_epd)
        return list(board.legal_moves)[:self.get_maximum_number_of_actions()]

    def get_action_vector(self, action) -> np.array:
        """ Returns a vector representation of the action. """
        from_x, from_y = action.from_square % 8, action.from_square // 8
        to_x, to_y = action.to_square % 8, action.to_square // 8
        return np.array([from_x / 8, from_y / 8, to_x / 8, to_y / 8], dtype=np.float32)

    def get_maximum_number_of_actions(self):
        """
        Returns the maximum number of actions that can be taken from any state.
        If there are more actions than this, the legal actions will be truncated
        by taking the top actions.
        """
        return 5

    def __repr__(self) -> str:
        return f'ChessState(EPD={self.board_epd})'
