from ..meta.search_tree import ObjectState, QFunction

import badgyal
from badgyal.board2planes import bulk_board2planes
import chess
import chess.svg
import numpy as np
import torch


class ChessBoardEmbedder:
    __instance = None

    def __new__(cls, *args):
        """ Singleton pattern """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args)
        return cls.__instance

    def __init__(self):
        self.network = badgyal.BGNet(cuda=True)
        self.embedding_network = torch.nn.Sequential(*(
            [self.network.net.conv_block, self.network.net.residual_stack]
            + list(self.network.net.value_head.children())[:-3]
        ))

    def embed(self, board: chess.Board) -> np.ndarray:
        input = bulk_board2planes([board])
        input = input.pin_memory().cuda(non_blocking=True)
        with torch.jit.optimized_execution(True):
            with torch.no_grad():
                return self.embedding_network(input).cpu().numpy()[0]


class ChessState(ObjectState):
    """
    Class to handle represent a state of the chess environment.
    """
    __slots__ = ['board_epd', 'state_vec', 'actions']

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

        board_embedder = ChessBoardEmbedder()
        board = chess.Board(self.board_epd)
        self.state_vec = board_embedder.embed(board)

        policy, _ = board_embedder.network.eval(board)
        sorted_policy = sorted(policy.items(), key=lambda x: -x[1])
        n = self.get_maximum_number_of_actions()
        self.actions = [
            chess.Move.from_uci(action_uci)
            for action_uci, _ in sorted_policy[:n]
        ]

    def set_environment_to_state(self, env):
        env._board.set_epd(self.board_epd)

    def get_state_vector(self) -> np.array:
        return self.state_vec

    def get_actions(self) -> list:
        """
        Returns a list of actions that can be taken from the current state.
        """
        return self.actions

    def get_action_vector(self, action) -> np.array:
        """
        Returns a vector representation of the action.
        """
        from_x, from_y = action.from_square % 8, action.from_square // 8
        to_x, to_y = action.to_square % 8, action.to_square // 8
        return np.array([from_x / 8, from_y / 8, to_x / 8, to_y / 8],
                        dtype=np.float32)

    def get_maximum_number_of_actions(self):
        """
        Returns the maximum number of actions that can be taken from any state.
        If there are more actions than this, the legal actions will be truncated
        by taking the top actions.
        """
        return 5

    def __repr__(self) -> str:
        return f'ChessState(EPD={self.board_epd})'


class ChessQFunction(QFunction):

    def __init__(self, network):
        self.network = network

    def compute_q(self, state: ChessState, action: chess.Move) -> float:
        board = chess.Board()
        board.set_epd(state.board_epd)
        policy, _ = self.network.eval(board, softmax_temp=1.61)
        return policy[action.uci()]
