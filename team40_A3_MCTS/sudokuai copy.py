#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import array
import copy
import math
import random
import time
import typing

from numpy import argmax, asarray, count_nonzero, log, sqrt, random as np_random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class Node():
    def __init__(self, state, parent=None, move=None) -> None:
        self.state = state  	        # The gamestate of the node
        self.parent = parent            # The parent of the node
        self.move = move                # The move that leads from the parent to this node
        self.children = []              # All the possible children of this node
        self._number_of_visits = 0      # The number of times current node is visited
        self._results = []              # A dictionary with results
        self._results[1] = 0    
        self._results[-1] = 0
        self._untried_actions = None    # All possible actions
        self._untried_actions = self.untried_actions()
        self.N = state.board.N
        return

    def untried_actions(self, possible_moves) -> list[Move]:
        self._untried_actions = self.getAllPossibleMoves()
        return self._untried_actions

    def win_loss(self) -> int:
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def visits(self) -> int:
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.state.board.put(action.i, action.j, action.value)
        child = Node(next_state, parent=self, move=action)
        self.children.append(child)
        return child

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_ending(self.state):
            possible_moves = current_rollout_state.getAllPossibleMoves()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.board.put(action.i, action.j, action.value)
            # TODO: count the game result and make it per player switch signs.
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(c.win_loss() / c.visits()) + c_param * sqrt((2 * log(self.visits()) / c.visits())) for c in self.children]
        return self.children[argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np_random.randint(len(possible_moves))]

    def tree_policy(self):
        current_node = self
        while not current_node.is_game_ending(self.state):
            if not current_node.is_fully_expanded():
                return current_node.expand()
            current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulations = 100
        for i in range(simulations):
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        
        return self.best_child(c_param=0.2)

    def is_game_ending(state) -> bool:
        if state.board.squares.count(SudokuBoard.empty) == 0:
            return True
        return False

    def game_result(self):
        pass

    def checkEmpty(self, board) -> list[typing.Tuple[int, int]]:
        """
        Finds all the empty cells of the input board
        @param board: a SudokuBoard stored as array of N**2 entries
        """
        emptyCells = []
        for k in range(self.N**2):
            i, j = SudokuBoard.f2rc(board, k)
            if board.get(i, j) == SudokuBoard.empty:
                emptyCells.append([i, j])
        return emptyCells

    def getColumn(self, state, i, j) -> array:
        """
        Gets the column where the position (i,j) is in
        @param state: a game state containing a SudokuBoard object
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        """
        column = []
        for c in range(self.N):
            column.append(state.board.get(c, j))
        return asarray(column)

    def getRow(self, state, i, j) -> array:
        """
        Gets the row where the position (i,j) is in
        @param state: a game state containing a SudokuBoard object
        @param i: A row value in the range [0, ..., N)
        @param j: A column value in the range [0, ..., N)
        """
        row = []
        for r in range(self.N):
            row.append(state.board.get(i, r))
        return asarray(row)

    def getBlock(self, state, i, j) -> array:
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
        return asarray(block)

    def getAllPossibleMoves(self, state) -> list[Move]:
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
                return value in self.getColumn(state, i, j)

            def valueInRow(i, j, value) -> bool:
                """
                Checks whether the given value at position (i,j) is valid for the row
                    i.e. finds if the value already exists in the row
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                @param value: A value in the range [1, ..., N]
                """
                return value in self.getRow(state, i, j)

            def valueInBlock(i, j, value) -> bool:
                """
                Checks whether the given value at position (i,j) is valid for the block
                    i.e. finds if the value already exists in the block which holds (i,j)
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                @param value: A value in the range [1, ..., N]
                """
                return value in self.getBlock(state, i, j)

            return not TabooMove(i, j, value) in state.taboo_moves \
                and not valueInColumn(i, j, value) \
                and not valueInRow(i, j, value) \
                and not valueInBlock(i, j, value)

        return [Move(cell[0], cell[1], value) for cell in self.checkEmpty(state.board)
                for value in range(1, self.N+1) if possible(cell[0], cell[1], value)]

    def assignScore(self, move, state) -> int:
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
            return count_nonzero(self.getColumn(state, i, j) == SudokuBoard.empty) == 1
        
        def completeRow(i, j) -> bool:
            """
            Checks whether the given position (i,j) is the only empty square in the row
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            return count_nonzero(self.getRow(state, i, j) == SudokuBoard.empty) == 1
        
        def completeBlock(i, j) -> bool:
            """
            Checks whether the given position (i,j) is the only empty square in the block
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            return count_nonzero(self.getBlock(state, i, j) == SudokuBoard.empty) == 1

        # Assign a score based on how many regions are completed
        completedRegions = completeRow(move.i, move.j) + completeColumn(move.i, move.j) + completeBlock(move.i, move.j)
        
        scores = {0: 0, 1: 1, 2: 3, 3: 7}
        return scores[completedRegions]

    def usefulMoves(self, moves, state) -> list[Move]:
        """
        Compute a list of useful moves, i.e. moves that score at least one point
        @param moves: a list of Move objects to filter
        @param state: a game state containing a SudokuBoard object
        """
        usefulmoves = []

        for move in moves:
            if self.assignScore(move, state) > 0:
                usefulmoves.append(move)
        
        if len(usefulmoves) > 0:
            return usefulmoves
        return moves

    def secondToLast(self, move, state) -> bool:
        """
        Computes whether doing the given move leaves only one empty square in any region
        i.e. finds if there are two empty cells in any region
        @param move: a Move object containing a coordinate and a value
        @param state: a game state containing a SudokuBoard object
        """
        return count_nonzero(self.getColumn(state, move.i, move.j) == SudokuBoard.empty) == 2 \
            or count_nonzero(self.getRow(state, move.i, move.j) == SudokuBoard.empty) == 2 \
            or count_nonzero(self.getBlock(state, move.i, move.j) == SudokuBoard.empty) == 2

    def evaluate(self, state) -> typing.Tuple[Move, int]:
        """
        Finds the best move for the given game state
        @param state: a game state containing a SudokuBoard object
        """
        moves = self.usefulMoves(self.getAllPossibleMoves(state), state)
        best = (moves[0], 0)
        
        notsecondtolast = []
        for move in moves:
            if not self.secondToLast(move, state):
                notsecondtolast.append(move)
        
        if len(notsecondtolast) <= 0:
            notsecondtolast.extend(moves)

        for move in notsecondtolast:
            value = self.assignScore(move, state)
            if value > best[1]:
                best = (move, value)
        return best

class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self) -> None:
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        root = Node(state=game_state)
        selected_node = root.best_action()
        self.propose_move = selected_node.move

        