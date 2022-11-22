#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # TODO: implement a way to generate all the legal moves from a given game state / board position

        def possible(i, j, value):
            return game_state.board.get(i, j) == SudokuBoard.empty \
                   and not TabooMove(i, j, value) in game_state.taboo_moves

        all_moves = [Move(i, j, value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value)]

        # TODO: Implement an evaluation function that assigns a numerical score to any game state

        # TODO: Implement a variant of the minimax tree search algorithm
        # --> Assign the best score at the moment to self.propose_move(), and update this every time

        while True:
            self.propose_move(random.choice(all_moves))
            time.sleep(0.2)

