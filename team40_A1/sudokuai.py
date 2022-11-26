#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import copy
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

        def checkEmpty(board):
            ''' Function that returns a list of empty cells for the current board '''
            emptyCells = []
            for k in range(N**2):
                i,j = SudokuBoard.f2rc(board, k)
                if board.get(i,j) == SudokuBoard.empty:
                    emptyCells.append([i,j])
            # print(emptyCells)
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
                            and (x+col != i or y+row != j):
                        return False
            return True

        def possible(i, j, value):
            ''' Function that returns a boolean value whether or not this input is a possible move '''
            return not TabooMove(i, j, value) in game_state.taboo_moves \
                    and checkColumn(i, j, value) \
                    and checkRow(i, j, value) \
                    and checkBlock(i, j, value)

        def get_all_moves(state):
            ''' List all_moves contains all the possible moves for the current game_state ''' 
            return [Move(cell[0], cell[1], value) for cell in checkEmpty(state.board)
                        for value in range(1, N+1) if possible(cell[0], cell[1], value)]

        def evaluate(state):
            ''' Return numerical evaluation of state '''
            ''' Check how many fillable cells are in the state'''
            cells = checkEmpty(state.board)
            best_value = 0
            best_cell = random.choice(get_all_moves(state))
            for cell in get_all_moves(state):
                value = completeRow(cell.i, cell.j) + completeColumn(cell.i, cell.j) + completeBlock(cell.i, cell.j)
                print(str(cell) + " has value: " + str(value) + " | || | " + str(completeRow(cell.i, cell.j)) + " | " + str(completeColumn(cell.i, cell.j)) + " | " + str(completeBlock(cell.i, cell.j)))
                if value > best_value:
                    best_cell = cell
                    best_value = value
                if best_value == 2:
                    best_value = 3
                elif best_value == 3:
                    best_value = 7
            # print(best_value, best_cell)
            return best_value, best_cell

        # TODO: Implement a variant of the minimax tree search algorithm (Iterative Deepening)
        # --> Assign the best score at the moment to self.propose_move(), and update this every time

        def getChildren(state):
            ''' Return list of states that follow from state '''
            children = []
            for move in get_all_moves(state):
                state.board.put(int(move.i), int(move.j), int(move.value))
                children.append(copy.deepcopy(state))
                state.board.put(int(move.i), move.j, SudokuBoard.empty)
            return children

        def minimax(state, max_depth, current_depth = 1, isMaximizingPlayer = True, current_score = 0):
            evaluated_score, evaluated_move = evaluate(state)
            
            print("Current depth of node: " + str(current_depth-1))
            print("Current score of node: " + str(current_score + evaluated_score) + " (" + str(current_score) + " + " + str(evaluated_score) + ")")
            print("Current board: " + str(state.board))

            # If the current depth is the target depth, evaluate that state.
            if current_depth == max_depth or len(checkEmpty(state.board)) == 1:
                return current_score + evaluated_score, evaluated_move  # Returns score of the best next move for the input state
            else: current_depth += 1

            for child in getChildren(state):
                if isMaximizingPlayer:
                    print("Current player is maximizing at depth " + str(current_depth-2) + ("\n" * 2))
                    best_value, best_move = minimax(child, max_depth, current_depth, False, current_score + evaluated_score)
                else:
                    print("Current player is minimizing at depth " + str(current_depth-2) + ("\n" * 2))
                    best_value, best_move = minimax(child, max_depth, current_depth, True, current_score= + evaluated_score)

            return best_value, best_move
            # if depth == 0:
            #     return evaluate(state)

            # children = getChildren(state)
            # best_move = None
            # current_value, current_move = evaluate(state)
            # print("Diepte: " + str(depth))
            # print(current_move)
            # if isMaximizingPlayer:
            #     best_value = float('-inf')
            #     for child in children:
            #         value, move = minimax(child, depth-1, False)
            #         if (value + current_value) > best_value:
            #             best_value = value + current_value
            #             best_move = move
            #             # print(best_move, best_value, depth, value, current_value, current_move)
            # else:
            #     best_value = float('inf')
            #     for child in children:
            #         value, move = minimax(child, depth-1, True)
            #         if (value + current_value) < best_value:
            #             best_value = value + current_value
            #             best_move = move
            #             # print(best_move, best_value, depth, value, current_value, current_move)
            # return best_value, best_move

        self.propose_move(random.choice(get_all_moves(game_state)))

        # print(minimax(game_state, 3, True))
        value, evaluated_move = minimax(game_state, 6)  # Give gamestate and the maximum depth of nodes

        while True:
            self.propose_move(evaluated_move)
            print("Proposed move: [" + str(evaluated_move.i) + ", " + str(evaluated_move.j) + "] => " + str(evaluated_move.value) + " | Reward: " + str(value))
            time.sleep(0.2)

