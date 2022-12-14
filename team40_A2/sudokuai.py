#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import random
import time
import typing
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

import numpy as np # TODO: Look at exactly what functions we need from numpy (instead of importing everything)

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    def _init_(self):
        super()._init_()

    def compute_best_move(self, game_state: GameState) -> None:

        N = game_state.board.N    

        def checkEmpty(board) -> list[typing.Tuple[int,int]]:
            """
            Finds all the empty cells of the input board
            @param board: a SudokuBoard stored as array of N**2 entries
            """
            emptyCells = []
            for k in range(N**2):
                i,j = SudokuBoard.f2rc(board, k)
                if board.get(i,j) == SudokuBoard.empty:
                    emptyCells.append([i,j])
            return emptyCells

        def getColumn(state, i, j) -> np.array:
            """
            Gets the column where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            column = []
            for c in range(N):
                column.append(state.board.get(c,j))
            return np.asarray(column)

        def getRow(state, i, j) -> np.array:
            """
            Gets the row where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            row = []
            for r in range(N):
                row.append(state.board.get(i,r))
            return np.asarray(row)

        def getBlock(state, i, j) -> np.array:
            """
            Gets the block where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            block = []
            x = i - (i % state.board.m)
            y = j - (j % state.board.n)
            for c in range(state.board.m):
                for r in range(state.board.n):
                    block.append(state.board.get(x+c, y+r))
            return np.asarray(block)

        def getAllPossibleMoves(state) -> list[Move]:
            """
            Finds a list of all possible moves for a given game state
            @param state: a game state containing a SudokuBoard object
            """
            def possible(i, j, value) -> bool:
                """
                Checks whether the given value at position (i,j) is valid for all regions
                    and not a previously tried wrong move
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                @param value: A value in the range [1, ..., N]
                """
                def valueInColumn(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the column
                        i.e. finds if the value already exists in the column
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """
                    return value in getColumn(state, i, j)
            
                def valueInRow(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the row
                        i.e. finds if the value already exists in the row
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """            
                    return value in getRow(state, i, j)
            
                def valueInBlock(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the block
                        i.e. finds if the value already exists in the block which holds (i,j)
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """  
                    return value in getBlock(state, i, j)
 
                return not TabooMove(i, j, value) in state.taboo_moves \
                        and not valueInColumn(i, j, value) \
                        and not valueInRow(i, j, value) \
                        and not valueInBlock(i, j, value)

            return [Move(cell[0], cell[1], value) for cell in checkEmpty(state.board)
                        for value in range(1, N+1) if possible(cell[0], cell[1], value)]

        # Propose a first move that is possible for all the regions, and not a TabooMove
        # This is included such that we always propose a move in the smallest amount of time
        self.propose_move(getAllPossibleMoves(game_state)[0])

        def assignScore(move, state) -> int:
            """
            Assigns a score to a move using some heuristic
            @param move: a Move object containing a coordinate and a value
            @param state: a game state containing a SudokuBoard object
            """
            def completeColumn(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the column
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                return np.count_nonzero(getColumn(state, i, j) == SudokuBoard.empty) == 1
            
            def completeRow(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the row
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                return np.count_nonzero(getRow(state, i, j) == SudokuBoard.empty) == 1
            
            def completeBlock(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the block
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                return np.count_nonzero(getBlock(state, i, j) == SudokuBoard.empty) == 1

            # Assign a score based on how many regions are completed
            completedRegions = completeRow(move.i, move.j) + completeColumn(move.i, move.j) + completeBlock(move.i, move.j)
            
            scores = {0: 0, 1: 1, 2: 3, 3: 7}
            return scores[completedRegions]

        def usefulMoves(moves, state):
            """
            Compute a list of useful moves, i.e. moves that score at least one point
            @param moves: a list of Move objects to filter
            @param state: a game state containing a SudokuBoard object
            """
            usefulmoves = []

            for move in moves:
                if assignScore(move, state) > 0:
                    usefulmoves.append(move)
            
            if len(usefulmoves) > 0:
                return usefulmoves
            return moves

        # Propose a new first move, for which we know we will get at least one point
        # (if such a move is available, otherwise propose the same move as before)
        self.propose_move(usefulMoves(getAllPossibleMoves(game_state), game_state)[0])

        def secondToLast(move, state):
            """
            Computes whether doing the given move leaves only one empty square in any region
            i.e. finds if there are two empty cells in any region
            @param move: a Move object containing a coordinate and a value
            @param state: a game state containing a SudokuBoard object
            """
            return np.count_nonzero(getColumn(state, move.i, move.j) == SudokuBoard.empty) == 2 \
                or np.count_nonzero(getRow(state, move.i, move.j) == SudokuBoard.empty) == 2 \
                or np.count_nonzero(getBlock(state, move.i, move.j) == SudokuBoard.empty) == 2
        
        def evaluate(state) -> typing.Tuple[Move, int]:
            """
            Finds the best move for the given game state
            @param state: a game state containing a SudokuBoard object
            """
            best = (Move(0,0,0), float("-inf"))

            legalmoves = getAllPossibleMoves(state)
            moves = usefulMoves(legalmoves, state)
            
            notsecondtolast = []
            for move in moves:
                if not secondToLast(move, state):
                    notsecondtolast.append(move)
            
            moves = notsecondtolast if len(notsecondtolast) > 0 else moves

            for move in moves:
                value = assignScore(move, state)
                if value > best[1]:
                    best = (move, value)

            return best      

        def minimax(state, isMaximizingPlayer, max_depth, current_depth = 0, current_score = 0, transposition_table = {}, alpha=float("-inf"), beta=float("inf")) -> typing.Tuple[Move, int]:
            """
            Makes a tree to a given depth and returns the move a node needs to make to get a certain value
            @param state: a game state containing a SudokuBoard object
            @param isMaximizingPlayer: a boolean value which determines if the player is maximizing
            @param max_depth: a depth value which defines when to terminate the tree search
            @param current_depth: a depth value which defines the current depth
            @param current_score: a score value which defines the score of the parent node
            @param transposition_table: a dictionary storing previously computed game states
            @param alpha: the best value the maximizing player can guarantee at the current level or higher
            @param beta: the best value the minimizing player can guarantee at the current level or higher
            """
            alphaOrig = alpha

            # If we have already seen the current state of the gameboard in a previous computation
            if state in transposition_table:
                trans_move, trans_value, trans_depth, trans_alphabeta = transposition_table[state]
                if max_depth - trans_depth < max_depth - current_depth:
                    if trans_alphabeta == "EXACT":
                        return trans_move, trans_value + current_score
                    elif trans_alphabeta == "LOWERBOUND":
                        alpha = max(alpha, trans_value)
                    elif trans_alphabeta == "UPPERBOUND":
                        beta = min(beta, trans_value)
                    if alpha >= beta:
                        return trans_move, trans_value + current_score

            # If there are no possible moves (when no move is valid), return a infinite value
            if len(getAllPossibleMoves(state)) == 0:
                if isMaximizingPlayer:
                    return None, float("-inf")
                return None, float("inf")

            # If the tree is in the final leaf, resturn a move and value
            if len(getAllPossibleMoves(state)) == 1 or current_depth == max_depth:
                move, value = evaluate(state)
                transposition_table[state] = (move, value+current_score, current_depth, "EXACT")
                if isMaximizingPlayer:
                    return move, value
                return move, -value

            if isMaximizingPlayer:
                best = (Move(0,0,0), float("-inf"))
                for move in getAllPossibleMoves(state):
                    total_score = current_score + assignScore(move, state)
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, transposition_table, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value > best[1]:
                        best = (move, result_value)
                    alpha = max(alpha, best[1])
                    if beta <= alpha:
                        break
            else:
                best = (Move(0,0,0), float("-inf"))
                for move in getAllPossibleMoves(state):
                    total_score = current_score - assignScore(move, state)
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, transposition_table, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value < best[1]:
                        best = (move, result_value)
                    beta = min(beta, best[1])
                    if beta <= alpha:
                        break

            if best[1] <= alphaOrig:
                alphabeta = "UPPERBOUND"
            elif best[1] >= beta:
                alphabeta = "LOWERBOUND"
            else:
                alphabeta = "EXACT"

            transposition_table[state] = (best[0], best[1] + current_score, current_depth, alphabeta)
            return best[0], best[1] + current_score

        # Run the main minimax algorithm
        for depth in range(0, game_state.board.squares.count(SudokuBoard.empty)):
            move, value = minimax(game_state, True, depth)
            self.propose_move(move)