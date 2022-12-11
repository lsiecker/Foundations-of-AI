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


                def checkColumn(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the column
                        i.e. finds if the value already exists in the column
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """
                    for col in range(N):
                        if state.board.get(col, j) == value:
                            return False
                    return True
            
                def checkRow(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the row
                        i.e. finds if the value already exists in the row
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """            
                    for row in range(N):
                        if state.board.get(i, row) == value:
                            return False
                    return True
            
                def checkBlock(i, j, value) -> bool:
                    """
                    Checks whether the given value at position (i,j) is valid for the block
                        i.e. finds if the value already exists in the block which holds (i,j)
                    @param i: A row value in the range [0, ..., N)
                    @param j: A column value in the range [0, ..., N)
                    @param value: A value in the range [1, ..., N]
                    """  
                    x = i - (i % state.board.m)
                    y = j - (j % state.board.n)
                    for col in range(state.board.m):
                        for row in range(state.board.n):
                            if state.board.get(x+col, y+row) == value:
                                return False
                    return True
 
                return not TabooMove(i, j, value) in state.taboo_moves \
                        and checkColumn(i, j, value) \
                        and checkRow(i, j, value) \
                        and checkBlock(i, j, value)

            return [Move(cell[0], cell[1], value) for cell in checkEmpty(state.board)
                        for value in range(1, N+1) if possible(cell[0], cell[1], value)]


        def assignScore(move, state) -> int:
            """
            Assigns a score to a move using some heuristic
            @param move: a Move object containing a coordinate and a value
            """

            def completeColumn(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the column
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                for col in range(N):
                    if state.board.get(col, j) == SudokuBoard.empty \
                            and col != i:
                        return False
                return True
            
            def completeRow(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the row
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                for row in range(N):
                    if state.board.get(i, row) == SudokuBoard.empty \
                            and row != j:
                        return False
                return True
            
            def completeBlock(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the block
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                x = i - (i % state.board.m)
                y = j - (j % state.board.n)
                for col in range(state.board.m):
                    for row in range(state.board.n):
                        if state.board.get(x+col, y+row) == SudokuBoard.empty \
                                and (x+col != i or y+row != j):
                            return False
                return True

            completedRegions = completeRow(move.i, move.j) + completeColumn(move.i, move.j) + completeBlock(move.i, move.j)
            
            scores = {0: 0, 1: 1, 2: 3, 3: 7}
            return scores[completedRegions]

        
        def usefulMoves(legalmoves):
            usefulmoves = []
            non_usefulmoves = []

            for move in legalmoves:
                if assignScore(move, game_state)>0:
                    usefulmoves.append(move)
                else:
                    non_usefulmoves.append(move)
            
            if len(non_usefulmoves)>0:
                usefulmoves.append(non_usefulmoves[0])
            # print(usefulmoves)
            return usefulmoves

        def secondToLast(move):
 
            emptyrow = 0
            emptycol = 0
            emptyblock = 0

            for j in range(N):
                if game_state.board.get(move.i,j) == SudokuBoard.empty:
                    emptyrow += 1

                    if emptyrow > 2:
                        return False
            row = emptyrow == 2
            
            for i in range(N):
                if game_state.board.get(i,move.j) == SudokuBoard.empty:
                    emptycol += 1

                    if emptycol > 2:
                        return False
            col = emptycol == 2

            x = move.i - (move.i % N)
            y = move.j - (move.j % N)
            for c in range(N):
                for r in range(N):
                    if game_state.board.get(x+c, y+r) == SudokuBoard.empty:
                        emptyblock += 1

                        if emptyblock > 2:
                            return False
            block = emptyblock == 2

            if row + col + block > 0:
                return True

        
        def evaluate(state) -> typing.Tuple[Move, int]:
            """
            Finds the best Move for the given game state
            @param state: a game state containing a SudokuBoard object
            """
            best_value = -1
            ## Initalize the best move as a random possible move
            legalmoves = getAllPossibleMoves(game_state)
            moves = usefulMoves(legalmoves)

            best_move = random.choice(usefulMoves(legalmoves))
            notsecondtolast = []
            secondtolast = []
            for move in usefulMoves(legalmoves):
                if secondToLast(move):
                    secondtolast.append(move)
                else:
                    notsecondtolast.append(move)
            
            moves = notsecondtolast if len(notsecondtolast) > 0 else secondtolast

            for move in moves:
                value = assignScore(move, state)
                if value > best_value:
                    best_move = move
                    best_value = value

            return best_move, best_value
                

        def minimax(state, isMaximizingPlayer, max_depth, current_depth = 0, current_score = 0, transposition_table = {}, alpha=float("-inf"), beta=float("inf")) -> typing.Tuple[Move, int]:
            """
            Makes a tree to a given depth and returns the move a node needs to make to get a certain value
            @param state: a game state containing a SudokuBoard object
            @param isMaximizingPlayer: a boolean value which determines if the player is maximizing
            @param max_depth: a depth value which defines when to terminate the tree search
            @param current_depth: a depth value which defines the current depth
            @param current_score: a score value which defines the score of the parent node 
            """

            # print("empty cells: ", state.board.squares.count(game_state.board.empty)%2)

            if state in transposition_table:
                trans_move, trans_value, trans_depth = transposition_table[state]
                if max_depth - trans_depth < max_depth - current_depth:
                    return trans_move, trans_value + current_score

            # If there are no possible moves (when no move is valid), return a infinite value
            if len(getAllPossibleMoves(state)) == 0:
                emptyCells = state.board.squares.count(game_state.board.empty)
                if isMaximizingPlayer:
                    return None, float("-inf")
                return None, float("inf")

            # If the tree is in the final leaf, resturn a move and value
            if len(getAllPossibleMoves(state)) == 1 or current_depth == max_depth:
                move, value = evaluate(state)
                if isMaximizingPlayer:
                    return move, value
                return move, -value

            if isMaximizingPlayer:
                best = (Move(0,0,0), float("-inf"))
                for move in getAllPossibleMoves(state):
                    total_score = current_score + assignScore(move, state)
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value > best[1]:
                        best = (move, result_value)
                    alpha = max(alpha, best[1])
                    if beta <= alpha:
                        break
                transposition_table[state] = (best[0], best[1]+current_score, current_depth)
                return best[0], best[1] + current_score
            else:
                best = (Move(0,0,0), float("inf"))
                for move in getAllPossibleMoves(state):
                    total_score = current_score - assignScore(move, state)
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value < best[1]:
                        best = (move, result_value)
                    beta = min(beta, best[1])
                    if beta <= alpha:
                        break
                transposition_table[state] = (best[0], best[1]+current_score, current_depth)
                return best[0], best[1] + current_score

        #  Intialize a random possible move as return
        # (to ensure we always have a return ready on timeout)
        start = time.time()
        legalmoves = getAllPossibleMoves(game_state)
        self.propose_move(usefulMoves(legalmoves)[0])

        # Search the minimax tree with iterative deepening
        for depth in range(0, game_state.board.squares.count(SudokuBoard.empty)):
            move, value = minimax(game_state, True, depth)
            self.propose_move(move)
            # print(time.time() - start)