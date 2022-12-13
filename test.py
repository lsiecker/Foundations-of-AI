from io import StringIO
import sys
import os
from simulate_game import simulate_game

# First, import the ThreadPoolExecutor class from the concurrent.futures module
from concurrent.futures import ThreadPoolExecutor

# Get the number of CPU cores on the system
num_cores = os.cpu_count()

# Create the threadpool executor with the same number of workers as CPU cores
executor = ThreadPoolExecutor(max_workers=num_cores)

def main():
    results = []

    # Next, define the function that you want to run in parallel
    def run_file(player_1, player_2, time, board):
            res = os.popen('python simulate_game.py --first ' + player_1 + ' --second ' + player_2 + ' --time ' + time + ' --board boards\\' + board).read()
            return ((player_1, player_2, time, board, res.splitlines()[-1]))

    times = ["0.1", "0.5", "1", "5"]
    players = ["greedy_player", "team40_A1"]
    boards = ["easy-2x2.txt","easy-3x3.txt","empty-3x3.txt","hard-3x3.txt","random-2x3.txt","random-3x3.txt","random-3x4.txt","random-4x4.txt"]

    # Submit the function to the executor to run in parallel
    for player in players:
        for time in times:
            for board in boards:
                for i in range(10):
                    result = executor.submit(run_file, "team40_A2", player, time, board)
                    results.append(result)
                    result = executor.submit(run_file, player, "team40_A2", time, board)
                    results.append(result)

    # Open the file for writing
    with open('results.txt', 'w') as outfile:
        # Get the results from the function calls
        for result in results:
            # Write the result to the file
            outfile.write(str(result.result()) + '\n')

if __name__ == '__main__':
    main()