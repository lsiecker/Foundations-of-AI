#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import array
import copy
import math
import random
import time
import typing

from numpy import asarray, count_nonzero, unique
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class Node:
    def __init__(self, move, parent, turn) -> None:
        self.move, self.parent, self.children = move, parent, []
        self.wins, self.visits = 0, 0
        self.my_turn = True

    def expand(self, state, possibleMoves) -> None:
        if possibleMoves != None:
            for move in possibleMoves:
                child = Node(move, self, not self.my_turn)  # New child node
                self.children.append(child)

    def update(self, win) -> None:
        if win:
            self.wins += 1

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def has_parent(self) -> bool:
        if self.parent is not None:
            return True
        return False


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self) -> None:
        super().__init__()

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        def checkEmpty(board) -> list[typing.Tuple[int, int]]:
            """
            Finds all the empty cells of the input board
            @param board: a SudokuBoard stored as array of N**2 entries
            """
            emptyCells = []
            for k in range(N**2):
                i, j = SudokuBoard.f2rc(board, k)
                if board.get(i, j) == SudokuBoard.empty:
                    emptyCells.append([i, j])
            return emptyCells

        def getColumn(state, i, j) -> array:
            """
            Gets the column where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            column = []
            for c in range(N):
                column.append(state.board.get(c, j))
            return asarray(column)

        def getRow(state, i, j) -> array:
            """
            Gets the row where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            row = []
            for r in range(N):
                row.append(state.board.get(i, r))
            return asarray(row)

        def getBlock(state, i, j) -> array:
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

        def clone(state):
            row = []
            block = []
            for i in range(N):
                for j in range(N):
                    cell = state.board.get(i,j)
                    row.append(cell)
                block.append(asarray(row))
            return block

        possibleMoves = getAllPossibleMoves(game_state)
        
        certainMoves = self.load()
        certainMoves = certainMoves if not certainMoves == None else []

        # Filter out the (at most 2) moves already done by you and the opponent since last save
        certainMoves = [move for move in certainMoves if game_state.board.get(move.i, move.j) == SudokuBoard.empty]

        if len(certainMoves) > 0:
            # Propose a first move we are sure will lead to a valid solution
            self.propose_move(certainMoves[0])

            # Fill in all the moves that we are certain about in the board
            for move in certainMoves:
                game_state.board.put(move.i, move.j, move.value)

            # Find all possible moves after the certain moves are placed
            possibleMoves = getAllPossibleMoves(game_state) + certainMoves

            # Remove all certain moves to get back to the actual game state
            for move in certainMoves:
                game_state.board.put(move.i, move.j, SudokuBoard.empty)

        def storeAllCertainMoves(moves, N) -> list[Move]:
            """
            Finds a list of all certain moves and stores it for later use,
            i.e. finds positions (i,j) for which only one value is possible
            @param moves: a list of Move objects to filter
            @param N: the dimensions of the sudoku
            """
            if len(moves) <= 2:
                return moves

            # Convert every move to an index in the 1D array
            indexedMoves = list(map(lambda k: k.i * N + k.j, moves))

            # Find all indices that occur only once in the list
            number, index, count = unique(indexedMoves, return_index=True, return_counts=True)
            certainIndices = [i[1] for i in set(zip(number, index, count)) if i[2] == 1]

            # Get all moves whose index occurs only once from the input list
            certainMoves = []
            for index in certainIndices:
                certainMoves.append(moves[index])
            
            if len(certainMoves) > 0:
                self.save(certainMoves)
            return certainMoves

        certainMoves = storeAllCertainMoves(possibleMoves, N)

        def getInsolvableMoves(knownMoves, state) -> list[Move]:
            """
            Get a list of all moves that are certain to make the board insolvable,
            i.e. the compliment of the list of certain moves,
            and that are not a taboo move yet
            @param moves: a list of Move objects to filter
            @param state: a game state containing a SudokuBoard object
            """
            allMoves = getAllPossibleMoves(state)
            return [move for move in allMoves if move not in knownMoves and move not in state.taboo_moves]
        
        # Add a new taboo move to the pool of possible moves
        insolvableMoves = getInsolvableMoves(possibleMoves, game_state)
        if len(insolvableMoves) > 0:
            possibleMoves.append(insolvableMoves[0])

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
                return count_nonzero(getColumn(state, i, j) == SudokuBoard.empty) == 1
            
            def completeRow(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the row
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                return count_nonzero(getRow(state, i, j) == SudokuBoard.empty) == 1
            
            def completeBlock(i, j) -> bool:
                """
                Checks whether the given position (i,j) is the only empty square in the block
                @param i: A row value in the range [0, ..., N)
                @param j: A column value in the range [0, ..., N)
                """
                return count_nonzero(getBlock(state, i, j) == SudokuBoard.empty) == 1

            # Assign a score based on how many regions are completed
            completedRegions = completeRow(move.i, move.j) + completeColumn(move.i, move.j) + completeBlock(move.i, move.j)
            
            scores = {0: 0, 1: 1, 2: 3, 3: 7}
            return scores[completedRegions]

        def usefulMoves(moves, state) -> list[Move]:
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

        def decision_process(state, certainmoves):

            if not checkEmpty(state.board):
                return []
            
            expected_values = []
            
            state_copy = clone(state)

            possibilities = 0

            state_copy = asarray(state_copy)

            for i in range(N):
                for j in range(N):

                    for move in certainmoves:
                        state_copy[move.i][move.j] = move.value

                    if state_copy[i][j] == SudokuBoard.empty:

                        # Recursively solve the puzzle and get the expected value of the future actions
                        future_actions = decision_process(state_copy)
                        future_value = sum(future_actions)


                        # Calculate the probability of the current action leading to a solved puzzle
                        for value in certainmoves(i,j):
                            possibilities += 1
                        probability = 1 / possibilities

                        # Calculate the expected value of the current action
                        expected_value = probability * future_value

                        # Add the current action and its expected value to the list of expected values
                        expected_values.append(((i, j), expected_value))

            # Return the action with the highest expected value
            return max(expected_values, key=lambda x: x[1])

        def secondToLast(move, state) -> bool:
            """
            Computes whether doing the given move leaves only one empty square in any region
            i.e. finds if there are two empty cells in any region
            @param move: a Move object containing a coordinate and a value
            @param state: a game state containing a SudokuBoard object
            """
            return count_nonzero(getColumn(state, move.i, move.j) == SudokuBoard.empty) == 2 \
                or count_nonzero(getRow(state, move.i, move.j) == SudokuBoard.empty) == 2 \
                or count_nonzero(getBlock(state, move.i, move.j) == SudokuBoard.empty) == 2

        def evaluate(state) -> typing.Tuple[Move, int]:
            """
            Finds the best move for the given game state
            @param state: a game state containing a SudokuBoard object
            """
            moves = usefulMoves(getAllPossibleMoves(state), state)
            best = (moves[0], 0)
            
            notsecondtolast = []
            for move in moves:
                if not secondToLast(move, state):
                    notsecondtolast.append(move)
            
            if len(notsecondtolast) <= 0:
                notsecondtolast.extend(moves)

            for move in notsecondtolast:
                value = assignScore(move, state)
                if value > best[1]:
                    best = (move, value)
            return best  

        def select_child_UCT(node) -> Node:
            best_child = node.children[0]
            best_value = -9999

            for child in node.children:
                if child.visits == 0:
                    value = 9999
                else:
                    value = child.wins/child.visits
                    if not node.my_turn:
                        value = 1 - value

                    value = child.wins/child.visits + math.sqrt(2*math.log(child.parent.visits/child.visits))
                
                if best_value < value:
                    best_value = value
                    best_child = child
            return best_child

        def monte_carlo_tree_search(initial_state, iterations) -> Move:
            root = Node(None, None, True)

            for _ in range(iterations):
                node, state = root, copy.deepcopy(initial_state)

                # Select
                while not node.is_leaf():
                    # print("Select")
                    node = select_child_UCT(node)
                    state.board.put(node.move.i, node.move.j, node.move.value)

                # Expand
                # print("Expand")
                node.expand(state, usefulMoves(getAllPossibleMoves(state), state)[:10])
                # print(node.children)
                if node.children:
                    best_node = select_child_UCT(node)
                    # print("Selected child:", node.children.index(best_node))

                if node.is_leaf() and state.board.squares.count(SudokuBoard.empty) > 0:
                    break

                # Simulate
                if node.children:
                    state.board.put(best_node.move.i, best_node.move.j, best_node.move.value)
                agent_score = state.scores[0]
                opponent_score = state.scores[1]
                while len(moves := usefulMoves(getAllPossibleMoves(state), state)) > 1:
                    # print("Simulate | number of possible moves ", len(moves))
                    move = decision_process(state, certainMoves)
                    value = assignScore(move, state)
                    if best_node.my_turn:
                        agent_score += value
                    else:
                        opponent_score += value
                    state.board.put(move.i, move.j, move.value)

                # Backpropagate
                while node:
                    node.visits += 1
                    if node.has_parent():
                        # print("Backpropagate")
                        node.update(agent_score > opponent_score)
                        node = node.parent
                    else:
                        break

            # children = root.children
            # best_child = children[0]
            # best_visits = 9999
            # for child in children:
            #     if best_visits < child.visits:
            #         best_visits = child.visits
            #         best_child = child
            # return best_child.move
            return select_child_UCT(root).move

        for i in range(1,25):
            self.propose_move(monte_carlo_tree_search(game_state, i))
