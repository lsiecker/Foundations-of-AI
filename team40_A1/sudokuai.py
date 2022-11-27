#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
import random
import time
import typing
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

        def checkEmpty(board) -> list[int,int]:
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
                        if game_state.board.get(col, j) == value:
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
                        if game_state.board.get(i, row) == value:
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
                    x = i - i % game_state.board.m
                    y = j - j % game_state.board.n
                    for col in range(game_state.board.m):
                        for row in range(game_state.board.n):
                            if game_state.board.get(x+col, y+row) == value:
                                return False
                    return True
 
                return not TabooMove(i, j, value) in game_state.taboo_moves \
                        and checkColumn(i, j, value) \
                        and checkRow(i, j, value) \
                        and checkBlock(i, j, value)

            return [Move(cell[0], cell[1], value) for cell in checkEmpty(state.board)
                        for value in range(1, N+1) if possible(cell[0], cell[1], value)]

        def assignScore(move) -> int:
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
                    if game_state.board.get(col, j) == SudokuBoard.empty \
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
                    if game_state.board.get(i, row) == SudokuBoard.empty \
                            and row != j:
                        return False
                return True
            
            def completeBlock(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the block
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                x = i - i % game_state.board.m
                y = j - j % game_state.board.n
                for col in range(game_state.board.m):
                    for row in range(game_state.board.n):
                        if game_state.board.get(x+col, y+row) == SudokuBoard.empty \
                                and (x+col != i or y+row != j):
                            return False
                return True

            completedRegions = completeRow(move.i, move.j) + completeColumn(move.i, move.j) + completeBlock(move.i, move.j)

            if completedRegions == 0:
                return 0
            if completedRegions == 1:
                return 1
            if completedRegions == 2:
                return 3
            if completedRegions == 3:
                return 7
        
        def evaluate(state) -> typing.Tuple[int, Move]:
            """
            Finds the best Move for the given game state
            @param state: a game state containing a SudokuBoard object
            """
            best_value = 0
            ## Initalize the best move as a random possible move
            best_move = random.choice(getAllPossibleMoves(state))
            for move in getAllPossibleMoves(state):
                value = assignScore(move)
                if value > best_value:
                    best_move = move
                    best_value = value
            return best_value, best_move

        ## TODO: Implement a variant of the minimax tree search algorithm (Iterative Deepening)
        ## --> Assign the best score at the moment to self.propose_move(), and update this every time

        def getChildren(state) -> list[GameState]:
            """
            Gets a list of all the game states that can follow from the current state
            after executing a valid move
            @param state: a game state containing a SudokuBoard object
            """
            children = []
            for move in getAllPossibleMoves(state):
                state.board.put(move.i, move.j, move.value)
                children.append(copy.deepcopy(state))
                state.board.put(move.i, move.j, SudokuBoard.empty)
            return children

        def minimax(state, max_depth, current_depth = 1, isMaximizingPlayer = True, current_score = 0) -> typing.Tuple[int, Move]:
            evaluated_score, evaluated_move = evaluate(state)
            
            # print("Current depth of node: " + str(current_depth-1))
            # print("Current score of node: " + str(current_score + evaluated_score) + " (" + str(current_score) + " + " + str(evaluated_score) + ")")
            # print("Current board: " + str(state.board))

            ## If the current depth is the target depth, evaluate that state
            if current_depth == max_depth or len(checkEmpty(state.board)) == 1:
                return current_score + evaluated_score, evaluated_move  # Returns score of the best next move for the input state

            current_depth += 1

            for child in getChildren(state):
                # print("Current player is maximizing at depth " + str(current_depth-2) + ("\n" * 2))
                best_value, best_move = minimax(child, max_depth, current_depth, not isMaximizingPlayer, current_score + evaluated_score)

            return best_value, best_move

        ## Intialize a random possible move as return
        ## (to ensure we always have a return ready on timeout)
        self.propose_move(random.choice(getAllPossibleMoves(game_state)))

        value, evaluated_move = minimax(game_state, 6)  # Give gamestate and the maximum depth of nodes

        while True:
            self.propose_move(evaluated_move)
            print("Proposed move: [" + str(evaluated_move.i) + ", " + str(evaluated_move.j) + "] => " + str(evaluated_move.value) + " | Reward: " + str(value))
            time.sleep(0.2)

