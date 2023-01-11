#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import typing
import torch
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai

import numpy as np
from numpy import array, asarray, count_nonzero

class SudokuNet(torch.nn.Module):
    def _init_(self, input_size, hidden_sizes):
        super(SudokuNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = torch.nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train(self, optimizer, criterion, num_epochs, puzzle: GameState):
        for epoch in range(num_epochs):
            # Convert the puzzle to a tensor
            puzzle_tensor = torch.puzzle_to_tensor(puzzle.board.squares)

            # Zero the gradients
            optimizer.zero_grad()

            hidden_sizes = [81, 27, 9]
            input_size = (puzzle.board.N**2)

            model = SudokuNet(input_size, hidden_sizes)

            # Forward pass
            output = model.forward(puzzle_tensor)

            # Calculate the loss
            loss = criterion(output, puzzle_tensor)

            # Backward pass
            loss.backward()

            # Optimize the weights
            optimizer.step()

class Solve():
    def _init_(self):
        super().__init__()
    
    def solver(self, puzzle):
    # Clone the puzzle

        # Convert the puzzle to a tensor
        puzzle_tensor = torch.Tensor(puzzle.board.squares)

        hidden_sizes = [81, 27, 9, 3]
        input_size = (puzzle.board.N**2)

        model = SudokuNet(input_size, hidden_sizes)

        # Forward pass
        output = model.forward(puzzle_tensor)



        # Convert the output to a numpy array and reshape it to the shape of the puzzle
        # Returns a new tensor with the same data as output, but that does not require gradient computation.
        # This is useful when you want to pass a tensor to a function that is not expecting a gradient-tracking tensor.
        # Then reshapes the tensor converted into a numpy
        output_array = output.detach().np().reshape((puzzle.shape[0], puzzle.shape[0]))

        optimizer = torch.optim.Adam()
        criterion = torch.nn.CrossEntropyLoss()

        # Iterate over the cells in the puzzle
        for i in range(puzzle.shape[0]):
            for j in range(puzzle.shape[0]):
                # If the cell is empty, fill it in with the value recommended by the neural network
                if puzzle[i][j] == 0:
                    puzzle[i][j] = int(np.argmax(output_array[i][j]) + 1)

                    # Train the neural network on the updated puzzle
                    output.train(optimizer, criterion, 10, puzzle)

                    hidden_sizes = [81, 27, 9, 3]
                    input_size = torch.Tensor(output)

                    model = SudokuNet(input_size, hidden_sizes)

                    # Forward pass
                    output = model.forward(output)

                    # Convert the updated output to a numpy array and reshape it to the shape of the puzzle
                    output_array = output.detach().np().reshape((puzzle.shape[0], puzzle.shape[0]))

        # Return the solved puzzle
        return output_array


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """
    def _init_(self) -> None:
        super()._init_()



    def compute_best_move(self, game_state: GameState) -> None:

        N = game_state.board.N

        def clone(self):
            return SudokuAI([row[:] for row in self.board])

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

        def getColumn(state, i, j) -> array:
            """
            Gets the column where the position (i,j) is in
            @param state: a game state containing a SudokuBoard object
            @param i: A row value in the range [0, ..., N)
            @param j: A column value in the range [0, ..., N)
            """
            column = []
            for c in range(N):
                column.append(state.board.get(c,j))
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
                row.append(state.board.get(i,r))
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

        # Propose a new first move, for which we know we will get at least one point
        # (if such a move is available, otherwise propose the same move as before)
        self.propose_move(usefulMoves(getAllPossibleMoves(game_state), game_state)[0])

        # def get_valid_values(self, i, j):
        #     valid_values = set(range(1, game_state.board.n + 1))
        #     for row in self.board:
        #         valid_values.discard(row[j])
        #     for col in self.board[i]:
        #         valid_values.discard(col)
        #     region_size = int(game_state.board.n ** 0.5)
        #     start_i = (i // region_size) * region_size
        #     start_j = (j // region_size) * region_size
        #     for k in range(region_size):
        #         for l in range(region_size):
        #             valid_values.discard(game_state.board[start_i + k][start_j + l])
        #     return list(valid_values)



        # def decision_process(state):
        #     if not checkEmpty(state):
        #         return []
            
        #     expected_values = []

        #     for i, j in checkEmpty(state):
        #         for value in get_valid_values(i,j):
        #             state_copy = clone(state)
        #             state_copy.board[i][j] = value


        #             # Recursively solve the puzzle and get the expected value of the future actions
        #             future_actions = decision_process(state_copy)
        #             future_value = sum(a[1] for a in future_actions)


        #             # Calculate the probability of the current action leading to a solved puzzle
        #             probability = 1 / len(get_valid_values(i, j))

        #             # Calculate the expected value of the current action
        #             expected_value = probability * future_value

        #             # Add the current action and its expected value to the list of expected values
        #             expected_values.append(((i, j, value), expected_value))

        #     # Return the action with the highest expected value
        #     return max(expected_values, key=lambda x: x[1])


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
            best = (Move(0,0,0), float("-inf"))

            moves = usefulMoves(getAllPossibleMoves(state), state)
            
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

        def sortPossibleMoves(state, moves):
            """
            Returns the possible moves sorted on their assigned score
            @param state: a game state containing a SudokuBoard object
            @param moves: an array of Move objects, which contain a coordinate and a value
            """
            moves_with_score = [(move, assignScore(move, state)) for move in moves]
            moves_with_score.sort(key=lambda a: -a[1])
            return moves_with_score

        def neuralNetworkSolver(state):
            solved = Solve().solver(state)
            return solved

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
            possibleMoves = sortPossibleMoves(state, getAllPossibleMoves(state))

            # If we have already seen the current state of the gameboard in a previous computation
            if state in transposition_table:
                trans_move, trans_value, trans_depth, trans_alphabeta = transposition_table[state]
                if max_depth - trans_depth < max_depth - current_depth:
                    if trans_alphabeta == "EXACT":
                        return trans_move, trans_value + current_score
                    if trans_alphabeta == "LOWERBOUND":
                        alpha = max(alpha, trans_value)
                    if trans_alphabeta == "UPPERBOUND":
                        beta = min(beta, trans_value)
                    if beta <= alpha:
                        return trans_move, trans_value + current_score

            # If there are no possible moves (when no move is valid), return an infinite value
            if possibleMoves == None:
                if isMaximizingPlayer:
                    return None, float("-inf")
                return None, float("inf")

            # If the tree is in the final leaf, return a move and value
            if len(possibleMoves) == 1 or current_depth == max_depth:
                move, value = evaluate(state)
                transposition_table[state] = (move, value + current_score, current_depth, "EXACT")
                if isMaximizingPlayer:
                    return move, value
                return move, -value

            if isMaximizingPlayer:
                best = (Move(0,0,0), float("-inf"))
                for move, evaluation in possibleMoves:
                    total_score = current_score + evaluation
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, transposition_table, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value > best[1]:
                        best = (move, result_value)
                    # Update the value of alpha, since we are the maximizing player
                    alpha = max(alpha, best[1])
                    if beta <= alpha:
                        break
            else:
                best = (Move(0,0,0), float("inf"))
                for move, evaluation in possibleMoves:
                    total_score = current_score - evaluation
                    state.board.put(move.i, move.j, move.value)
                    result_move, result_value = minimax(state, not isMaximizingPlayer, max_depth, current_depth+1, total_score, transposition_table, alpha, beta)
                    state.board.put(move.i, move.j, SudokuBoard.empty)
                    if result_value < best[1]:
                        best = (move, result_value)
                    # Update the value of beta, since we are the minimizing player
                    beta = min(beta, best[1])
                    if beta <= alpha:
                        break

            alphabeta = "EXACT"
            if best[1] <= alphaOrig:
                alphabeta = "UPPERBOUND"
            if best[1] >= beta:
                alphabeta = "LOWERBOUND"

            # Save the current state in the transposition table
            transposition_table[state] = (best[0], best[1] + current_score, current_depth, alphabeta)
            return best[0], best[1] + current_score

        # Run the main minimax algorithm
        for depth in range(0, game_state.board.squares.count(SudokuBoard.empty)):
            move, value = minimax(game_state, True, depth)
            solved  = neuralNetworkSolver(game_state)
            print(solved.shape())
            self.propose_move(move)