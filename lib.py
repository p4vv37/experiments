import random

import numpy as np


def find_four(board, current_player):
    """
    Returns True iff the player has connected  4 (or more)
    This is much faster if written in C or Cython
    """
    for pos, direction in POS_DIR:
        streak = 0
        while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
            if board[pos[0], pos[1]] == current_player:
                streak += 1
                if streak == 4:
                    return True
            else:
                streak = 0
            pos = pos + direction
    return False


POS_DIR = np.array(
    [[[i, 0], [0, 1]] for i in range(6)]
    + [[[0, i], [1, 0]] for i in range(7)]
    + [[[i, 0], [1, 1]] for i in range(1, 3)]
    + [[[0, i], [1, 1]] for i in range(4)]
    + [[[i, 6], [1, -1]] for i in range(1, 3)]
    + [[[0, i], [1, -1]] for i in range(3, 7)]
)


class RandomPlayer(object):

    def __init__(self, name="Human"):
        self.name = name

    # noinspection PyMethodMayBeStatic
    def ask_move(self, game):
        possible_moves = game.possible_moves()
        move = possible_moves[random.randint(0, len(possible_moves) - 1)]
        return move
