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

        def checkEmpty(board, N):
            ''' Function that returns a list of empty cells for the current board '''
            emptyCells = []
            for k in range(N**2):
                i,j = SudokuBoard.f2rc(board, k)
                if board.get(i,j) == SudokuBoard.empty:
                    emptyCells.append([i,j])
            print(emptyCells)
            return emptyCells
        
        def checkColumn(i, j, value):
            ''' Function that returns a boolean value whether or not the input value fits in the column '''
            for col in range(N):
                if game_state.board.get(col, j) == value:
                    return False
            return True
        
        def checkRow(i, j, value):
            ''' Function that returns a boolean value whether or not the input value fits in the row '''
            for row in range(N):
                if game_state.board.get(i, row) == value:
                    return False
            return True
        
        def checkBlock(i, j, value):
            ''' Function that returns a boolean value whether or not the input value fits in the block '''
            x = i - i % game_state.board.m
            y = j - j % game_state.board.n
            for col in range(game_state.board.m):
                for row in range(game_state.board.n):
                    if game_state.board.get(x+col, y+row) == value:
                        return False
            return True
        
        def completeColumn(i, j):
            ''' Function that returns a boolean value whether or not the column will be filled if i, j is filled '''
            for col in range(N):
                if game_state.board.get(col, j) == SudokuBoard.empty \
                        and col != i:
                    return False
            return True
        
        def completeRow(i, j):
            ''' Function that returns a boolean value whether or not the row will be filled if i, j is filled '''
            for row in range(N):
                if game_state.board.get(i, row) == SudokuBoard.empty \
                        and row != j:
                    return False
            return True
        
        def completeBlock(i, j):
            ''' Function that returns a boolean value whether or not the block will be filled if i, j is filled '''
            x = i - i % game_state.board.m
            y = j - j % game_state.board.n
            for col in range(game_state.board.m):
                for row in range(game_state.board.n):
                    if game_state.board.get(x+col, y+row) == SudokuBoard.empty \
                            and x+col != i \
                            and y+row != j:
                        return False
            return True

        def possible(i, j, value):
            ''' Function that returns a boolean value whether or not this input is a possible move '''
            return not TabooMove(i, j, value) in game_state.taboo_moves \
                    and checkColumn(i, j, value) \
                    and checkRow(i, j, value) \
                    and checkBlock(i, j, value)

        ''' List all_moves contains all the possible moves for the current game_state ''' 
        all_moves = [Move(cell[0], cell[1], value) for cell in checkEmpty(game_state.board, N)
                     for value in range(1, N+1) if possible(cell[0], cell[1], value)]

        def evaluate(move):
            ''' Function that returns a numerical score for a given game state '''
            return completeRow(move.i, move.j) + completeColumn(move.i, move.j)+ completeBlock(move.i, move.j)

        # TODO: Implement a variant of the minimax tree search algorithm
        # --> Assign the best score at the moment to self.propose_move(), and update this every time

        best_move = random.choice(all_moves)

        while True:
            if evaluate(all_moves[0]) < evaluate(best_move):
                all_moves.remove[0]
            else:
                best_move = all_moves[0]
            self.propose_move(best_move)
            time.sleep(0.2)

