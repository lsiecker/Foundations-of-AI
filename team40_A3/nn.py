
import numpy as np
import torch

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI, torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = torch.nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x


    def train(self, net, optimizer, criterion, num_epochs, puzzle: GameState):
        for epoch in range(num_epochs):
            # Convert the puzzle to a tensor
            puzzle_tensor = torch.puzzle_to_tensor(puzzle)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = net(puzzle_tensor)

            # Calculate the loss
            loss = criterion(output, puzzle_tensor)

            # Backward pass
            loss.backward()

            # Optimize the weights
            optimizer.step()

def solve(puzzle):
    # Clone the puzzle
    puzzle = [row[:] for row in puzzle]

    # Convert the puzzle to a tensor
    puzzle_tensor = torch.puzzle_to_tensor(puzzle)

    # Get the output of the neural network
    output = SudokuAI(puzzle_tensor)


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
                output.train(SudokuAI, optimizer, criterion, 50, puzzle)

                # Get the updated output of the neural network
                output = SudokuAI(puzzle_tensor)

                # Convert the updated output to a numpy array and reshape it to the shape of the puzzle
                output_array = output.detach().np().reshape((puzzle.shape[0], puzzle.shape[0]))

    # Return the solved puzzle
    return puzzle


solver = solve(GameState)
solver