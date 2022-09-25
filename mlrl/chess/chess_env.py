from ..utils.render_utils import render_chess_board

from typing import Callable, Tuple
import random

import chess
import gym_chess
import badgyal


class ChessVsAgent(gym_chess.envs.Chess):

    def __init__(self, agent: Callable[[chess.Board], int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = agent

    def step(self, action: int) -> Tuple[chess.Board, float, bool, dict]:
        _, reward, done, _ = super().step(action)
        if not done:
            observation, _, done, info = super().step(self.agent(self._board))
        return observation, reward, done, info

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return render_chess_board(self._board)

        return super().render(mode)


class ChessVsRandom(ChessVsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(self.get_action, *args, **kwargs)

    def get_action(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


class ChessVsSimpleNetwork(ChessVsAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(self.get_action, *args, **kwargs)
        self.network = badgyal.BGNet(cuda=True)

    def get_action(self, board: chess.Board) -> chess.Move:
        policy, *_ = self.network.eval(board)
        action = random.choices(list(policy), weights=list(policy.values()))[0]
        return chess.Move.from_uci(action)
